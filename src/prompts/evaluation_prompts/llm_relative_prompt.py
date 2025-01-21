import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.llm_utils.utils import text_cost_parser
from src.pipeline.registry.prompt_registry import PromptRegistry
from src.schemas.schemas import ChunkComparison


@PromptRegistry.register("LLM_relative_evaluation")
class LLMRelativeEvaluationPrompt:
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
        chunks_b = base_model.chunks_b
        return cls.system_prompt, cls.user_prompt.format(chunks_a=chunks_a, chunks_b=chunks_b)

    @classmethod
    def parse_response(cls, base_models: list[ChunkComparison], parsed_items: dict[str, ChunkComparison]) -> list[ChunkComparison]:
        for i, base_model in enumerate(base_models):
            model_response, cost = text_cost_parser(parsed_items.get(i))
            print(model_response)
            try:
                model_response = cls.DataModel.model_validate_json(model_response)
            except ValidationError as e:
                print(f"Error validating model response: {e}")
                raise e
            base_model.winner = model_response.winner
            base_model.reasoning = model_response.reasoning
        return base_models
