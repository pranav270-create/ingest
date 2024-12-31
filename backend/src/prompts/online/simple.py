import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).parents[2]))

from src.prompts.parser import text_cost_parser, structured_text_cost_parser 


class QABot:
    system = (
        "You are to faithfully answer the user's question based on your knowledge."
    )

    user = (
        "Based on the user's question: '{query}'\n\n"
        "Please answer the question to the best of your ability."
    )

    @classmethod
    def format_prompt(cls, query: str) -> dict:
        return {"system": cls.system, "user": cls.user.format(query=query)}

    @staticmethod
    def parse_response(response: Any, model: str) -> tuple[str, float]:
        return text_cost_parser(response, model)
