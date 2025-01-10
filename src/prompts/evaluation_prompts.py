from typing import Any

from pydantic import BaseModel, Field

from src.llm_utils.utils import text_cost_parser
from src.prompts.base_prompt import BasePrompt


class ChunkEvaluationPrompt(BasePrompt):
    system_prompt = """You are a highly capable image analysis assistant.
    Analyze the provided image and answer questions about it accurately and concisely."""

    user_prompt = """
    Rate this text chunk from 1-5 on the following criteria:
    1. Text Clarity (1-5): Is the text well-formed and readable?
    2. Coherence (1-5): Does the chunk represent a complete, coherent unit of information?
    3. Organization (1-5): Is the content well-structured within the chunk?

    Text chunk:
    {chunk}
    """

    class DataModel(BaseModel):
        text_clarity: int = Field(..., description="Rating from 1-5 for text clarity")
        coherence: int = Field(..., description="Rating from 1-5 for coherence")
        organization: int = Field(..., description="Rating from 1-5 for organization")

    @classmethod
    def format_prompt(cls, question: str) -> dict[str, str]:
        """Format the prompt with the given question."""
        return {
            "system": cls.system_prompt,
            "user": cls.user_prompt.format(question=question)
        }

    @staticmethod
    def parse_response(response: Any) -> tuple[dict[str, int], float]:
        """Parse the response into a structured output."""
        response, cost = text_cost_parser(response)
        return response.model_dump(), cost
