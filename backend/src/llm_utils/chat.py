import asyncio
import sys
from pathlib import Path
from typing import Any, Optional

import aiohttp
from instructor import handle_response_model

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.llm_utils.api_requests import get_request_header, get_request_url
from src.llm_utils.utils import Functionality, Provider, get_mode


def format_chat_prompt(
    model: str,
    provider: Provider,
    system_prompts: list[str],
    user_prompts: list[str],
    metadata: list[dict[str, Any]],
    response_model: Optional[str] = None,
    max_tokens: int = 512,
) -> list[dict[str, Any]]:
    """
    Formats the text to be embedded according to the API requirements.
    """
    if isinstance(user_prompts, str):
        user_prompts = [user_prompts]
        metadata = [metadata]

    # anthropic has a separate system prompt
    if provider == Provider.ANTHROPIC:
        messages = [[{"role": "user", "content": user_prompt} for user_prompt in user_prompts]]
    else:
        messages = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            for system_prompt, user_prompt in zip(system_prompts, user_prompts)
        ]

    if response_model:  # use instructor to format the prompt
        mode = get_mode(provider.value)
        prompts = [
            handle_response_model(response_model=response_model, mode=mode, messages=message, max_tokens=max_tokens)[1]
            for message in messages
        ]
    else:
        prompts = [{"messages": message, "max_tokens": max_tokens} for message in messages]

    # add system parameter if provider is anthropic
    if provider == Provider.ANTHROPIC:
        for i, prompt in enumerate(prompts):
            prompt["system"] = system_prompts[i]

    # Ensure metadata is the same length as the prompts, if not make it the same
    if len(metadata) == 0:
        metadata = [{}] * len(prompts)
    if len(metadata) != len(prompts):
        raise ValueError("Metadata must be the same length as the prompts or a single value.")
    # Include the model name and metadata in the prompt format
    prompt_formats = [{"model": model, **prompt, "metadata": meta} for prompt, meta in zip(prompts, metadata)]

    return prompt_formats


async def async_chat(
    queries: list[str],
    system_prompts: list[str],
    model: str = "gpt-4",
    provider: Provider = Provider.OPENAI,
    functionality: Functionality = Functionality.CHAT,
    max_tokens: int = 512,
    metadata: dict[str, Any] = {},
    max_attempts: int = 3,
) -> list[dict[str, Any]]:
    """
    Asynchronously process multiple chat queries and return their responses.
    """
    # Format prompts
    prompts = format_chat_prompt(
        model, provider, system_prompts, queries, metadata=[metadata] * len(queries), max_tokens=max_tokens
    )
    # get request header and url
    request_header = get_request_header(provider)
    request_url = get_request_url(provider, functionality)

    async def process_single_request(prompt: dict[str, Any]) -> dict[str, Any]:
        # metadata = prompt.pop("metadata")
        for attempt in range(max_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url=request_url, headers=request_header, json=prompt) as response:
                        return await response.json()
            except Exception as e:
                if attempt == max_attempts - 1:
                    return {"error": str(e), "prompt": prompt}
                await asyncio.sleep(2**attempt)  # Exponential backoff

    # Process all requests concurrently
    tasks = [asyncio.create_task(process_single_request(prompt)) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    return responses
