import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from litellm import ModelResponse
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import BaseModelListType


class BasePrompt(ABC):
    system: str = ""
    user: str = ""
    DataModel: Optional[type[BaseModel]] = None

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

    @classmethod
    def has_data_model(cls) -> bool:
        """
        Helper method to check if the prompt class has a DataModel defined
        """
        return cls.DataModel is not None