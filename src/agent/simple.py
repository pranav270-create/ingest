from collections.abc import Awaitable
from typing import Any, Union, overload
from httpx import RequestError, HTTPStatusError

from src.agent.base import BaseAgent
from src.llm_utils.client import AsyncChat, Chat
from src.llm_utils.utils import Provider
from src.prompts.base_prompt import BasePrompt


class SimpleAgent(BaseAgent):
    def __init__(self, prompt: BasePrompt):
        super().__init__(prompt)
        self.prompt_class = prompt
        self.data_model = getattr(prompt, "DataModel", None)
        self.parse_response = prompt.parse_response

    @overload
    def call(self, client: Chat, model: str, **variables: Any) -> tuple[str, float]: ...

    @overload
    async def call(self, client: AsyncChat, model: str, **variables: Any) -> Awaitable[tuple[str, float]]: ...

    def call(
        self, client: Union[Chat, AsyncChat], model: str, retries: int = 3, **variables: Any
    ) -> Union[tuple[str, float], Awaitable[tuple[str, float]]]:
        llm_kwargs = {"model": model}
        for key in ["max_tokens", "stop_sequences", "system", "temperature"]:
            if key in variables:
                llm_kwargs[key] = variables.pop(key)

        formatted_prompt = self.prompt_class.format_prompt(**variables)

        if client.provider == Provider.OPENAI:
            llm_kwargs["messages"] = [
                {"role": "system", "content": formatted_prompt["system"]},
                {"role": "user", "content": formatted_prompt["user"]},
            ]

            if self.data_model:
                llm_kwargs["response_format"] = self.data_model

        elif client.provider == Provider.ANTHROPIC:
            llm_kwargs["system"] = formatted_prompt["system"]
            llm_kwargs["messages"] = [{"role": "user", "content": formatted_prompt["user"]}]
            if "assistant" in formatted_prompt:
                llm_kwargs["messages"].append({"role": "assistant", "content": formatted_prompt["assistant"]})

        elif client.provider == Provider.GEMINI:
            llm_kwargs["messages"] = f"{formatted_prompt['system']}\n{formatted_prompt['user']}"

        else:
            raise NotImplementedError(f"Provider {client.provider} not supported")

        async def async_call():
            attempts = 0
            while attempts < retries:
                try:
                    raw_response = await client.chat(**llm_kwargs)
                    return self.parse_response(raw_response, model)
                except Exception as e:
                    if attempts == retries - 1 or not isinstance(e, HTTPStatusError) or e.status_code != 500:
                        raise
                    attempts += 1

        def sync_call():
            attempts = 0
            while attempts < retries:
                try:
                    raw_response = client.chat(**llm_kwargs)
                    return self.parse_response(raw_response, model)
                except Exception as e:
                    if attempts == retries - 1 or not isinstance(e, HTTPStatusError) or e.status_code != 500:
                        raise
                    attempts += 1

        if isinstance(client, AsyncChat):
            return async_call()
        return sync_call()
