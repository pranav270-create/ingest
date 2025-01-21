import json
from pydantic import BaseModel, Field

from src.pipeline.registry.prompt_registry import PromptRegistry
from src.prompts.base_prompt import BasePrompt
from src.llm_utils.utils import text_cost_parser


@PromptRegistry.register("chunk_text")
class GenerativeChunkingPrompt(BasePrompt):
    """
    A prompt class for breaking down text into meaningful, self-contained chunks using an LLM.
    """

    system_prompt = """You are an expert at breaking down text into meaningful, self-contained chunks.
Your task is to analyze the provided text and break it down into coherent chunks that maintain complete thoughts and context.
Each chunk should be a meaningful, self-contained unit that can be understood on its own.

Follow these rules when creating chunks:
1. Preserve complete sentences - never break in the middle of a sentence
2. Keep related content together - maintain logical groupings of information
3. Maintain context within each chunk - ensure each chunk can be understood independently
4. Respect natural topic boundaries - start new chunks when topics change
5. Keep paragraphs together when possible - avoid splitting paragraphs unless necessary
6. Aim for consistency in chunk size - but prioritize coherence over size uniformity

Your response must be a valid JSON object with a single "chunks" key containing an array of text chunks."""

    user_prompt = """Please analyze and chunk the following text:

{text}

Remember to format your response as a JSON object:
{{"chunks": ["chunk1", "chunk2", ...]}}"""

    class DataModel(BaseModel):
        chunks: list[str] = Field(description="List of text chunks")

    @classmethod
    async def format_prompt(cls, text: str, read=None):
        """
        Format the prompt with the given text.

        Args:
            text: The text to be chunked
            read: Optional read function (not used for text chunking)

        Returns:
            Formatted prompt string
        """
        return [
            {"role": "system", "content": cls.system_prompt},
            {"role": "user", "content": cls.user_prompt.format(text=text)}
        ]

    @staticmethod
    def parse_response(response: str) -> list[str]:
        """
        Parse the LLM response into a list of chunks.

        Args:
            response: The raw response from the LLM

        Returns:
            List of text chunks
        """
        result, _ = text_cost_parser(response)
        try:
            return result.get("chunks", [])
        except json.JSONDecodeError:
            # Fallback: try to extract chunks from malformed response
            print(f"Warning: Failed to parse JSON response: {result}")
            return [result]
