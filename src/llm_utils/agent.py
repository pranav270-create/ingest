from collections.abc import Awaitable
from typing import Any, Union

from litellm import acompletion, completion

from src.llm_utils.utils import Provider
from src.prompts.base_prompt import BasePrompt


class Agent:
    def __init__(self, prompt: BasePrompt):
        self.prompt_class = prompt
        self.data_model = getattr(prompt, "DataModel", None)
        self.parse_response = prompt.parse_response


    def _prepare_llm_kwargs(self, provider: Union[Provider, str], model: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Extract and prepare LLM kwargs from variables"""
        if isinstance(provider, Provider):
            provider = provider.value
        llm_kwargs = {"model": f"{provider}/{model}"}

        # Move these special keys to a constant
        LLM_SPECIAL_KEYS = ["max_tokens", "stop_sequences", "system", "temperature", "num_retries"]
        for key in LLM_SPECIAL_KEYS:
            if key in variables:
                llm_kwargs[key] = variables.pop(key)

        formatted_prompt = self.prompt_class.format_prompt(**variables)
        llm_kwargs["messages"] = [
            {"role": "system", "content": formatted_prompt["system"]},
            {"role": "user", "content": formatted_prompt["user"]},
        ]

        if self.data_model:
            llm_kwargs["response_format"] = self.data_model

        return llm_kwargs

    def call(
        self, provider: Union[Provider, str], model: str, is_async: bool = False, **variables: Any
    ) -> Union[tuple[str, float], Awaitable[tuple[str, float]]]:
        llm_kwargs = self._prepare_llm_kwargs(provider, model, variables)

        # asynchronous completion
        async def async_call():
            raw_response = await acompletion(**llm_kwargs)
            return self.parse_response(raw_response)

        if is_async:
            return async_call()

        # synchronous completion
        raw_response = completion(**llm_kwargs)
        return self.parse_response(raw_response)
