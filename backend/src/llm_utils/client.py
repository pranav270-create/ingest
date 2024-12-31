# very inspired by https://github.com/jxnl/instructor/blob/main/instructor/client.py

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Callable, overload

import anthropic
import google.generativeai as gemini
import openai
from pydantic import BaseModel
from typing_extensions import Self
import importlib

from src.llm_utils.utils import Provider, get_provider


class ChatResponse(BaseModel):
    content: str | BaseModel
    cost: float


class Chat:
    """
    A synchronous LLM client wrapper
    0 is unstructured, 1 is structured
    """

    client: Any | None
    create_fn: Callable[..., Any]
    provider: Provider

    def __init__(self, client: Any | None, create: dict[int, Callable[..., Any]], provider: Provider, **kwargs: Any):
        self.client = client
        self.create_fn = create
        self.provider = provider
        self.kwargs = kwargs

    @overload
    def chat(self: AsyncChat, messages: list[dict[str, str]], **kwargs: Any) -> Awaitable[ChatResponse]: ...

    @overload
    def chat(self: Self, messages: list[dict[str, str]], **kwargs: Any) -> ChatResponse: ...

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatResponse:
        kwargs = self.handle_kwargs(kwargs)

        if kwargs.get("response_format", False):
            create_fn = self.create_fn[1]
        else:
            create_fn = self.create_fn[0]

        return create_fn(messages=messages, **kwargs)

    def handle_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        for key, value in self.kwargs.items():
            if key not in kwargs:
                kwargs[key] = value
        return kwargs


class AsyncChat(Chat):
    """
    An asynchronous LLM client wrapper
    0 is unstructured, 1 is structured
    """

    client: Any | None
    create_fn: Callable[..., Any]
    provider: Provider

    def __init__(self, client: Any | None, create: dict[int, Callable[..., Any]], provider: Provider, **kwargs: Any):
        self.client = client
        self.create_fn = create
        self.provider = provider
        self.kwargs = kwargs

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatResponse:
        kwargs = self.handle_kwargs(kwargs)

        if kwargs.get("response_format", False):
            create_fn = self.create_fn[1]
        else:
            create_fn = self.create_fn[0]
        return await create_fn(messages=messages, **kwargs)


def create_openai_wrapper(chat_func: Callable):
    def wrapper(messages: list[dict[str, str]], model: str, **kwargs: Any) -> ChatResponse:
        return chat_func(messages=messages, model=model, **kwargs)

    return wrapper


@overload
def from_openai(client: openai.OpenAI, **kwargs: Any) -> Chat: ...


@overload
def from_openai(client: openai.AsyncOpenAI, **kwargs: Any) -> AsyncChat: ...


def from_openai(client: openai.OpenAI | openai.AsyncOpenAI, **kwargs: Any) -> Chat | AsyncChat:
    """
    accepts Provider.OPENAI, Provider.ANYSCALE, Provider.TOGETHER, Provider.DATABRICKS
    """
    # change provider if base_url is present
    if hasattr(client, "base_url"):
        provider = get_provider(str(client.base_url))
    else:
        provider = Provider.OPENAI

    # select correct function to call
    wrapped_chat = {
        0: create_openai_wrapper(client.chat.completions.create),
        1: create_openai_wrapper(client.beta.chat.completions.parse),
    }

    if isinstance(client, openai.OpenAI):
        return Chat(client=client, create=wrapped_chat, provider=provider, **kwargs)

    async def async_wrapped_chat(messages: list[dict[str, str]], **kwargs: Any):
        if kwargs.get("response_format", False):
            return await create_openai_wrapper(client.beta.chat.completions.parse)(messages=messages, **kwargs)
        else:
            return await create_openai_wrapper(client.chat.completions.create)(messages=messages, **kwargs)

    return AsyncChat(client=client, create={0: async_wrapped_chat, 1: async_wrapped_chat}, provider=provider, **kwargs)


def create_anthropic_wrapper(chat_func: Callable):
    def wrapper(messages: list[dict[str, str]], model: str, **kwargs: Any) -> ChatResponse:
        assert "max_tokens" in kwargs, "max_tokens must be provided for anthropic"
        return chat_func(messages=messages, model=model, **kwargs)

    return wrapper


@overload
def from_anthropic(client: anthropic.Anthropic, **kwargs: Any) -> Chat: ...


@overload
def from_anthropic(client: anthropic.AsyncAnthropic, **kwargs: Any) -> AsyncChat: ...


def from_anthropic(client: anthropic.Anthropic | anthropic.AsyncAnthropic, **kwargs: Any) -> Chat | AsyncChat:

    wrapped_chat = {
        0: create_anthropic_wrapper(client.messages.create),
        1: create_anthropic_wrapper(client.messages.create),
    }

    if isinstance(client, anthropic.Anthropic):
        return Chat(client=client, create=wrapped_chat, provider=Provider.ANTHROPIC, **kwargs)

    async def async_wrapped_chat(messages: list[dict[str, str]], **kwargs: Any):
        if kwargs.get("response_format", False):
            return await create_anthropic_wrapper(client.messages.create)(messages=messages, **kwargs)
        else:
            return await create_anthropic_wrapper(client.messages.create)(messages=messages, **kwargs)

    return AsyncChat(client=client, create={0: async_wrapped_chat, 1: async_wrapped_chat}, provider=Provider.ANTHROPIC, **kwargs)


def create_gemini_wrapper(chat_func: Callable):
    def wrapper(messages: str, **kwargs: Any) -> ChatResponse:
        kwargs.pop("use_async", None)
        return chat_func(contents=messages, **kwargs)

    return wrapper


@overload
def from_gemini(client: gemini.GenerativeModel, **kwargs: Any) -> Chat: ...


@overload
def from_gemini(client: gemini.GenerativeModel, **kwargs: Any) -> AsyncChat: ...


def from_gemini(client: gemini.GenerativeModel, use_async: bool = False, **kwargs: Any) -> Chat | AsyncChat:
    wrapped_chat = create_gemini_wrapper(client.generate_content)

    if not use_async:
        return Chat(client=client, create=wrapped_chat, provider=Provider.GEMINI, **kwargs)

    async def async_wrapped_chat(*args, **kwargs):
        return await wrapped_chat(*args, **kwargs)

    return AsyncChat(client=client, create=async_wrapped_chat, provider=Provider.GEMINI, **kwargs)
