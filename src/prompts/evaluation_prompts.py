from typing import Any

from pydantic import BaseModel, Field

from src.llm_utils.utils import text_cost_parser
from src.prompts.base_prompt import BasePrompt
from src.prompts.registry import PromptRegistry
from src.schemas.schemas import Entry


@PromptRegistry.register("chunk_evaluation")
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
    async def format_prompt(cls, entry: BaseModel, read=None) -> dict[str, str]:
        """Format the prompt with the given question."""
        return [
            {"role": "system", "content": cls.system_prompt},
            {"role": "user", "content": cls.user_prompt.format(chunk=entry.string)},
        ]

    @staticmethod
    def parse_response(basemodels: list[Entry], responses: Any) -> tuple[dict[str, int], float]:
        """Parse the response into a structured output."""
        for basemodel, response in zip(basemodels, responses):
            parsed_response, cost = text_cost_parser(response)
            if not basemodel.added_featurization:
                basemodel.added_featurization = {}
            basemodel.added_featurization["chunk_evaluation"] = parsed_response
        return basemodels
