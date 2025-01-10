import base64
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, Optional, Union

from litellm import acompletion, completion

from src.llm_utils.utils import Provider
from src.prompts.base_prompt import BasePrompt


class Agent:
    def __init__(self, prompt: BasePrompt):
        self.prompt_class = prompt
        self.data_model = getattr(prompt, "DataModel", None)
        self.parse_response = prompt.parse_response


    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


    def _prepare_llm_kwargs(
        self,
        provider: Union[Provider, str],
        model: str,
        variables: dict[str, Any],
        image_path: Optional[Union[str, Path]] = None
    ) -> dict[str, Any]:
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

        # Add image to the prompt if provided
        if image_path:
            base64_image = self._encode_image(image_path)
            llm_kwargs["messages"] = [
                {"role": "system", "content": formatted_prompt["system"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": formatted_prompt["user"],
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ]
        else:
            llm_kwargs["messages"] = [
                {"role": "system", "content": formatted_prompt["system"]},
                {"role": "user", "content": formatted_prompt["user"]},
            ]

        if self.data_model:
            llm_kwargs["response_format"] = self.data_model

        return llm_kwargs

    def call(
        self,
        provider: Union[Provider, str],
        model: str,
        is_async: bool = False,
        image_path: Optional[Union[str, Path]] = None,
        **variables: Any
    ) -> Union[tuple[str, float], Awaitable[tuple[str, float]]]:
        llm_kwargs = self._prepare_llm_kwargs(provider, model, variables, image_path)

        # asynchronous completion
        async def async_call():
            raw_response = await acompletion(**llm_kwargs)
            return self.parse_response(raw_response)

        if is_async:
            return async_call()

        # synchronous completion
        raw_response = completion(**llm_kwargs)
        return self.parse_response(raw_response)
