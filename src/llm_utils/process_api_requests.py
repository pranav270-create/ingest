import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field

import aiohttp

from src.llm_utils.api_requests import get_request_header, get_request_url
from src.llm_utils.tokenize_utils import get_chat_prompt_tokens, get_embed_input_tokens
from src.llm_utils.utils import Functionality, Provider, model_mapping


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    provider: Provider,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    # chat completion tokens
    if api_endpoint.endswith("chat/completions") or api_endpoint.endswith("chat") or api_endpoint.endswith("messages"):
        # expected number of completion tokens
        max_tokens = request_json.get("max_tokens", 512)  # TODO
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # prompt tokens
        num_tokens = len(get_chat_prompt_tokens(request_json, provider))
        return num_tokens + completion_tokens

    elif api_endpoint.endswith("embeddings") or api_endpoint.endswith("embed"):
        # embedding input tokens
        num_tokens = len(get_embed_input_tokens(request_json, provider))
        return num_tokens
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits
    result_buffer: dict = field(default_factory=dict)  # Buffer for storing results temporarily
    next_expected_task_id: int = 0  # The next task ID expected to be written to the file

    def write_buffered_results_in_order(self, save_filepath: str):
        def append_to_jsonl(data, filename: str, task_id) -> None:
            """Append a json payload to the end of a jsonl file."""
            json_string = json.dumps(data)
            save_file = f"{filename}/task_{task_id}.json"
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            with open(save_file, "a") as f:
                f.write(json_string + "\n")

        """Write results from the buffer to the file in the order of task IDs."""
        while self.next_expected_task_id in self.result_buffer:
            data = self.result_buffer.pop(self.next_expected_task_id)
            append_to_jsonl(data, save_filepath, self.next_expected_task_id)  # NOTE
            self.next_expected_task_id += 1


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api_plain(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
    ):
        """Calls the OpenAI API and returns the response."""
        async with session.post(url=request_url, headers=request_header, json=self.request_json) as response:
            return await response.json()

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with session.post(url=request_url, headers=request_header, json=self.request_json) as response:
                response = await response.json()
            if "error" in response:
                logging.warning(f"Request {self.task_id} failed with error {response['error']}")
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

            # voyage rate limits don't throw errors, need to inspect response
            if request_url.split(".")[1] == "voyageai":
                if "detail" in response:
                    logging.warning(f"Request {self.task_id} failed with error: RATE LIMIT")
                    status_tracker.num_api_errors += 1
                    error = response
                    if "rate limit" in response["detail"]:
                        status_tracker.time_of_last_rate_limit_error = time.time()
                        status_tracker.num_rate_limit_errors += 1
                        status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catch naked exceptions
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                # Add to buffer instead of writing directly
                status_tracker.result_buffer[self.task_id] = data
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = [self.request_json, response, self.metadata] if self.metadata else [self.request_json, response]
            # Add to buffer instead of writing directly
            status_tracker.result_buffer[self.task_id] = data
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} buffered for writing to {save_filepath}")

        # Attempt to write buffered results in order
        status_tracker.write_buffered_results_in_order(save_filepath)


async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    provider: Provider,
    functionality: Functionality,
    model: str,
    max_attempts: int = 5,
    logging_level: int = logging.INFO,
):
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()
    status_tracker = StatusTracker()
    next_request = None

    # initialize available capacity counts
    if provider == Provider.ANYSCALE:
        max_in_flight = model_mapping[model].limit.max_in_flight
    else:
        # mistral has no RPM rate limit, so use big number
        if provider == Provider.MISTRAL:
            max_requests_per_minute = 1000000000
        else:
            max_requests_per_minute = model_mapping[model].limit.rpm
        max_tokens_per_minute = model_mapping[model].limit.tpm
        available_request_capacity = max_requests_per_minute
        available_token_capacity = max_tokens_per_minute

    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, skip reading it
    logging.debug("Initialization complete.")

    # get request header and url
    request_header = get_request_header(provider)
    request_url = get_request_url(provider, functionality)

    # initialize file reading
    with open(requests_filepath) as file:
        requests = file.__iter__()
        logging.debug("File opened. Entering main loop")
        async with aiohttp.ClientSession() as session:
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
                    elif file_not_finished:
                        try:
                            # get new request
                            request_json = json.loads(next(requests))
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(request_json, request_url, provider=provider),
                                attempts_left=max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                        except StopIteration:
                            # if file runs out, set flag to stop reading it
                            logging.debug("Read file exhausted")
                            file_not_finished = False

                # update available capacity for rpm/tpm rate limiting
                if provider != Provider.ANYSCALE:
                    current_time = time.time()
                    seconds_since_update = current_time - last_update_time
                    available_request_capacity = min(
                        available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
                        max_requests_per_minute,
                    )
                    available_token_capacity = min(
                        available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
                        max_tokens_per_minute,
                    )
                    last_update_time = current_time

                # if enough capacity available, call API
                can_process_request = False
                if next_request:
                    if provider == Provider.ANYSCALE:
                        can_process_request = status_tracker.num_tasks_in_process < max_in_flight
                    else:
                        next_request_tokens = next_request.token_consumption
                        can_process_request = available_request_capacity >= 1 and available_token_capacity >= next_request_tokens
                # update counters
                if can_process_request:
                    if provider != Provider.ANYSCALE:
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                    next_request.attempts_left -= 1

                    # call API
                    asyncio.create_task(
                        next_request.call_api(
                            session=session,
                            request_url=request_url,
                            request_header=request_header,
                            retry_queue=queue_of_requests_to_retry,
                            save_filepath=save_filepath,
                            status_tracker=status_tracker,
                        )
                    )
                    next_request = None  # reset next_request to empty

                # if all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                if provider != Provider.ANYSCALE:
                    seconds_since_rate_limit_error = time.time() - status_tracker.time_of_last_rate_limit_error
                    if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
                        remaining_seconds_to_pause = seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
                        await asyncio.sleep(remaining_seconds_to_pause)
                        # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                        t = time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)
                        logging.warn(f"Pausing to cool down until {t}")

        # after finishing, log final status
        logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. "
                f"Errors logged to {save_filepath}."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )
