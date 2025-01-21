import sys
from pathlib import Path
from litellm import ModelResponse
from pydantic import BaseModel, Field, ValidationError

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.llm_utils.utils import text_cost_parser
from src.pipeline.registry.prompt_registry import PromptRegistry
from src.prompts.base_prompt import BasePrompt
from src.schemas.schemas import ChunkComparison


@PromptRegistry.register("LLM_relative_evaluation")
class LLMRelativeEvaluationPrompt(BasePrompt):
    system_prompt = (
        "You are an assistant specialized in RAG tasks."
        "The task is the following: given two chunkings of the same page, you will have to"
        "determine which better preserves the meaning and structure of the original text?"
    )

    user_prompt = """
    Chunking A:
    {chunks_a}

    Chunking B:
    {chunks_b}

    Respond with a JSON containing:
    1. winner: "A" or "B"
    2. reasoning: Brief explanation"""

    class DataModel(BaseModel):
        winner: str = Field(..., description="The winner of the chunking, either 'A' or 'B'")
        reasoning: str = Field(..., description="A brief explanation of why the winner is the way it is")

    @classmethod
    async def format_prompt(cls, base_model: ChunkComparison, read=None, **kwargs) -> tuple[str, str]:
        chunks_a = base_model.chunks_a
        chunks_a_str = "\n".join([chunk.string for chunk in chunks_a])
        chunks_b = base_model.chunks_b
        chunks_b_str = "\n".join([chunk.string for chunk in chunks_b])
        return [{"role": "system", "content": cls.system_prompt}, {"role": "user", "content": cls.user_prompt.format(chunks_a=chunks_a_str, chunks_b=chunks_b_str)}]

    @staticmethod
    def parse_response(entry: ChunkComparison, response: ModelResponse) -> ChunkComparison:
        text, _ = text_cost_parser(response)
        try:
            model_response = LLMRelativeEvaluationPrompt.DataModel.model_validate_json(text)
            entry.winner = model_response.winner
            entry.reasoning = model_response.reasoning
        except Exception as e:
            entry.winner = ""
            entry.reasoning = f"Failed to parse response: {e}"
        return entry
