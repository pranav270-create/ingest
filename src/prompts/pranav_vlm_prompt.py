import base64
import json
from litellm import ModelResponse
from pydantic import BaseModel, Field

from src.llm_utils.utils import structure_image_prompt, text_cost_parser
from src.pipeline.registry.prompt_registry import PromptRegistry
from src.prompts.base_prompt import BasePrompt
from src.schemas.schemas import VLMEvaluation


@PromptRegistry.register("vlm_elo")
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
        rationale: str = Field(description="The rationale for the winner")
        winner: str = Field(description="The winner of the VLM evaluation")

    @classmethod
    async def format_prompt(cls, vlm_evaluation: VLMEvaluation, read=None):
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
                with open(path, 'rb') as f:
                    image = f.read()
            images.append(base64.b64encode(image).decode('utf-8'))

        chunks_a = vlm_evaluation.chunks_a
        chunks_b = vlm_evaluation.chunks_b
        return structure_image_prompt(
            cls.system_prompt,
            cls.user_prompt.format(chunks_a=chunks_a, chunks_b=chunks_b), 
            images
        )

    @staticmethod
    def parse_response(vlm_evaluation: VLMEvaluation, response: ModelResponse) -> VLMEvaluation:
        # extract vlm evaluation data
        text, _ = text_cost_parser(response)
        winner = json.loads(text).get('winner', '')
        rationale = json.loads(text).get('rationale', '')
        vlm_evaluation.winner = winner
        vlm_evaluation.rationale = rationale
        return vlm_evaluation
