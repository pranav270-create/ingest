import re
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.prompts.parser import structured_text_cost_parser, text_cost_parser


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
