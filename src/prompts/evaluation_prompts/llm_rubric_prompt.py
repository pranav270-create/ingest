from pydantic import BaseModel, Field
from litellm import ModelResponse
from src.llm_utils.utils import text_cost_parser
from src.pipeline.registry.prompt_registry import PromptRegistry
from src.prompts.base_prompt import BasePrompt
from src.schemas.schemas import ChunkEvaluation


@PromptRegistry.register("LLM_chunk_rubric")
class ChunkRubricPrompt(BasePrompt):
    """Prompt for evaluating individual chunk quality."""

    system_prompt = """
    You are an expert at evaluating text chunks for RAG systems.
    Your task is to rate the quality of individual text chunks.
    """

    user_prompt = """
    Rate this text chunk from 1-5 on the following criteria:
    1. Text Clarity (1-5): Is the text well-formed and readable?
    2. Coherence (1-5): Does the chunk represent a complete, coherent unit of information?
    3. Organization (1-5): Is the content well-structured within the chunk?

    Text chunk:
    {chunk}

    Provide ratings and brief explanations in JSON format.
    """

    class DataModel(BaseModel):
        text_clarity: int = Field(..., description="Rating from 1-5 on text clarity")
        coherence: int = Field(..., description="Rating from 1-5 on coherence")
        organization: int = Field(..., description="Rating from 1-5 on organization")
        explanation: str = Field(..., description="Brief explanation of ratings")

    @classmethod
    async def format_prompt(cls, entry: ChunkEvaluation, read=None) -> list[dict[str, str]]:
        """Format the prompt as a list of message dictionaries."""
        return [{"role": "system", "content": cls.system_prompt}, {"role": "user", "content": cls.user_prompt.format(chunk=entry.string)}]

    @staticmethod
    def parse_response(entry: ChunkEvaluation, response: ModelResponse) -> ChunkEvaluation:
        text, _ = text_cost_parser(response)
        try:
            scores = ChunkRubricPrompt.DataModel.model_validate_json(text)
            entry.text_clarity = scores.text_clarity
            entry.coherence = scores.coherence
            entry.organization = scores.organization
            entry.explanation = scores.explanation
            entry.score = scores.text_clarity + scores.coherence + scores.organization
        except Exception as e:
            print(f"Failed to parse response: {e}")
            entry.text_clarity = 0
            entry.coherence = 0
            entry.organization = 0
            entry.explanation = f"Failed to parse response: {e}"
            entry.score = 0
        return entry
