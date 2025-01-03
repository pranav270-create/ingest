from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import Any, Union, overload

from src.llm_utils.client import AsyncChat, Chat


class BaseAgent(ABC):
    def __init__(self, prompt):
        self.prompt_class = prompt
        self.data_model = getattr(prompt, "DataModel", None)
        self.parse_response = prompt.parse_response

    @overload
    def call(self, client: Chat, model: str, **variables: Any) -> str: ...

    @overload
    async def call(self, client: AsyncChat, model: str, **variables: Any) -> str: ...

    @abstractmethod
    def call(self, client: Union[Chat, AsyncChat], **variables: Any) -> Union[str, Awaitable[str]]:
        """
        Interacts with the LLM client in both synchronous and asynchronous contexts.

        Args:
            client (Chat | AsyncChat): The chat client instance.
            **variables (Any): Variables to format the prompts.

        Returns:
            Union[str, Awaitable[str]]: Parsed response from the LLM or an awaitable.
        """
        pass
