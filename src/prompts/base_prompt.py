import sys
from abc import ABC, abstractmethod
from pathlib import Path

from litellm import ModelResponse

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import BaseModelListType


class BasePrompt(ABC):
    system: str = ""
    user: str = ""

    @classmethod
    @abstractmethod
    def format_prompt(cls, *args, **kwargs) -> list[dict[str, str]]:
        """
        Returns a list of dictionaries following the OpenAI messages format

        Example:
        [
            {"role": "system", "content": cls.system},
            {"role": "user", "content": cls.user},
        ]
        """
        pass


    @staticmethod
    @abstractmethod
    def parse_response(entries: BaseModelListType, responses: list[ModelResponse]) -> BaseModelListType:
        """
        Parses the response from the LLM and returns a list of entries
        """
        pass
