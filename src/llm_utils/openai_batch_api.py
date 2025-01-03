# modified from https://github.com/jxnl/instructor/blob/main/instructor/batch.py
import json
import uuid
from collections.abc import Iterable
from typing import Any, Literal, Optional, TypeVar, Union

from instructor.process_response import handle_response_model
from pydantic import BaseModel, Field

from src.llm_utils.utils import Provider, model_mapping

T = TypeVar("T", bound=BaseModel)

openai_models = [k for k, v in model_mapping.items() if v.provider == Provider.OPENAI]


class Function(BaseModel):
    name: str
    description: str
    parameters: Any


class Tool(BaseModel):
    type: str
    function: Function


class RequestBody(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = Field(default=1000)
    tools: Optional[list[Tool]]
    tool_choice: Optional[dict[str, Any]]


class BatchModel(BaseModel):
    custom_id: str
    method: Literal["POST"]
    url: Literal["/v1/chat/completions", "/v1/embeddings"]
    body: RequestBody


class BatchJob:
    @classmethod
    def parse_from_file(
        cls, file_path: str, url: Literal["/v1/chat/completions", "/v1/embeddings"], response_model: Optional[type[T]] = None
    ) -> tuple[Union[list[T], list[dict[str, Any]]], list[dict[Any, Any]]]:
        if url == "/v1/embeddings":
            assert response_model is None, "embedding jobs can't have a response model"

        with open(file_path) as file:
            res: Union[list[T], list[dict[str, Any]]] = []
            error_objs: list[dict[Any, Any]] = []
            for line in file:
                data = json.loads(line)
                try:
                    if url == "/v1/chat/completions":
                        message = data["response"]["body"]["choices"][0]["message"]
                        if response_model:
                            parsed_data = response_model(**json.loads(message["tool_calls"][0]["function"]["arguments"]))
                        else:
                            parsed_data = message
                    elif url == "/v1/embeddings":
                        parsed_data = data["response"]["body"]["data"]
                    res.append(parsed_data)
                except Exception:
                    error_objs.append(data)

            return res, error_objs

    @classmethod
    def create_from_messages(
        cls,
        messages_batch: Union[list[list[dict[str, Any]]], Iterable[list[dict[str, Any]]], list[str], list[list[str]]],
        model: str,
        response_model: Optional[type[BaseModel]],
        url: Literal["/v1/chat/completions", "/v1/embeddings"],
        file_path: str,
        max_tokens: int = 1000,
    ):
        if url == "/v1/embeddings":
            assert response_model is None, "embedding jobs can't have a response model"

        kwargs = {}
        if response_model:
            _, kwargs = handle_response_model(response_model=response_model)

        with open(file_path, "w") as file:
            for messages in messages_batch:
                file.write(
                    BatchModel(
                        custom_id=str(uuid.uuid4()),
                        method="POST",
                        url=url,
                        body=RequestBody(
                            model=model,
                            max_tokens=max_tokens,
                            messages=messages,
                            **kwargs,
                        ),
                    ).model_dump_json()
                    + "\n"
                )
