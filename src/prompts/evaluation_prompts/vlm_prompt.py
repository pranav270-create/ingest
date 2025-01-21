import base64
from typing import Any
from litellm import ModelResponse
from pydantic import BaseModel, Field

from src.llm_utils.utils import structure_image_prompt, text_cost_parser
from src.pipeline.registry.prompt_registry import PromptRegistry
from src.prompts.base_prompt import BasePrompt
from src.schemas.schemas import ChunkEvaluation


@PromptRegistry.register("VLM_relative_evaluation")
class VLMELOPrompt(BasePrompt):
    """
    A prompt class for evaluating the winner of a VLM evaluation.
    """

    system_prompt = """
    You are an advanced language model who is excellent at evaluating the winner of a VLM evaluation.
    """

    user_prompt = """
    Please analyze the following VLM evaluation and determine the winner.
    There are two chunks, A and B.
    The winner is the chunk that is more similar to the document.
    {chunks_a}
    {chunks_b}
    """

    class DataModel(BaseModel):
        reasoning: str = Field(description="The reasoning for the winner")
        winner: str = Field(description="The winner of the VLM evaluation")

    @classmethod
    async def format_prompt(cls, vlm_evaluation: ChunkEvaluation, read=None) -> list[dict[str, Any]]:
        # Collect all unique page file paths from both chunks
        page_paths_a = [loc.page_file_path for chunk in vlm_evaluation.chunks_a for loc in chunk.chunk_locations]
        page_paths_b = [loc.page_file_path for chunk in vlm_evaluation.chunks_b for loc in chunk.chunk_locations]
        # Ensure both chunks reference the same set of pages
        if set(page_paths_a) != set(page_paths_b):
            raise ValueError("The two chunks must reference the same set of pages")
        # Get unique page paths in order
        page_paths = list(dict.fromkeys(page_paths_a))  # preserves order while removing duplicates
        # Read all images
        images = []
        for path in page_paths:
            if read is not None:
                image = await read(path)
            else:
                with open(path, "rb") as f:
                    image = f.read()
            images.append(base64.b64encode(image).decode("utf-8"))

        chunks_a_str = "\n".join([chunk.string for chunk in vlm_evaluation.chunks_a])
        chunks_b_str = "\n".join([chunk.string for chunk in vlm_evaluation.chunks_b])
        return structure_image_prompt(cls.system_prompt, cls.user_prompt.format(chunks_a=chunks_a_str, chunks_b=chunks_b_str), images)

    @staticmethod
    def parse_response(vlm_evaluation: ChunkEvaluation, response: ModelResponse) -> ChunkEvaluation:
        # extract vlm evaluation data
        text, _ = text_cost_parser(response)
        scores = VLMELOPrompt.DataModel.model_validate_json(text)
        vlm_evaluation.winner = scores.winner
        vlm_evaluation.reasoning = scores.reasoning
        return vlm_evaluation
