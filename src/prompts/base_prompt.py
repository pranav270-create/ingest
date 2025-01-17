import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Optional, TypeVar

from litellm import ModelResponse
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parents[2]))

# from src.schemas.schemas import BaseModelListType

T = TypeVar('T', bound=BaseModel)

class BasePrompt(ABC, Generic[T]):
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
    def parse_response(entry: T, response: ModelResponse) -> T:
        """
        Parses the response from the LLM and returns a new Entry
        """
        pass

    @classmethod
    def has_data_model(cls) -> bool:
        """
        Helper method to check if the prompt class has a DataModel defined
        """
        return cls.DataModel is not None