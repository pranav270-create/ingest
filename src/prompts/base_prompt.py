import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.llm_utils.utils import text_cost_parser


class BasePrompt(ABC):
    system: str = ""
    user: str = ""

    @classmethod
    @abstractmethod
    def format_prompt(cls, *args, **kwargs) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def parse_response(response: Any, model: str) -> tuple[Any, float]:
        return text_cost_parser(response, model)
