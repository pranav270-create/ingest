from pydantic import BaseModel, Field
from src.llm_utils.utils import text_cost_parser
from src.pipeline.registry.prompt_registry import PromptRegistry
from src.prompts.base_prompt import BasePrompt
from src.schemas.schemas import Entry

@PromptRegistry.register("chunk_evaluation")
class ChunkEvaluationPrompt(BasePrompt[Entry]):
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
    async def format_prompt(cls, entry: Entry, read=None) -> tuple[str, str]:
        return cls.system_prompt, cls.user_prompt.format(chunk=entry.string)

    @staticmethod
    def parse_response(entry: Entry, response) -> Entry:
        text, _ = text_cost_parser(response)
        try:
            scores = ChunkEvaluationPrompt.DataModel.model_validate_json(text)
            entry.evaluation_scores = {
                "text_clarity": scores.text_clarity,
                "coherence": scores.coherence,
                "organization": scores.organization,
                "explanation": scores.explanation
            }
        except Exception as e:
            entry.evaluation_scores = {
                "text_clarity": 0,
                "coherence": 0,
                "organization": 0,
                "error": f"Failed to parse response: {e}"
            }
        return entry