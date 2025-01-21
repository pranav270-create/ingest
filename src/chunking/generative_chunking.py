from typing import Any, Optional

from src.schemas.schemas import Entry, ChunkingMethod, BoundingBox
from src.pipeline.registry.function_registry import FunctionRegistry
from src.chunking.chunk_utils import filter_entries, create_chunk_entry
from src.featurization.get_features import featurize
from src.llm_utils.utils import model_mapping
from src.prompts.generative_chunking import GenerativeChunkingPrompt


DEFAULT_CHUNKING_PROMPT = """You are an expert at breaking down text into meaningful, self-contained chunks.
Given the following text, break it down into coherent chunks that maintain complete thoughts and context.
Each chunk should be a meaningful, self-contained unit that can be understood on its own.

Rules:
1. Preserve complete sentences
2. Keep related content together
3. Maintain context within each chunk
4. Respect natural topic boundaries
5. Keep paragraphs together when possible

Format your response as a JSON array of chunks:
{"chunks": ["chunk1", "chunk2", ...]}

Text to chunk:
{text}
"""


@FunctionRegistry.register("chunk", ChunkingMethod.GENERATIVE.value)
async def generative_chunks(entries: list[Entry], **kwargs) -> list[Entry]:
    """
    Chunk entries using an LLM to create semantically meaningful chunks.

    Args:
        entries: List of entries to chunk
        kwargs:
            model_name: Name of the model to use (default: gpt-4-turbo)
            chunk_prompt: Custom prompt for chunking
            max_chunk_size: Maximum size for initial context windows
            context_overlap: Overlap between context windows
    """
    model_name = kwargs.get("model_name", "gpt-4-turbo")
    chunk_prompt = kwargs.get("chunk_prompt", DEFAULT_CHUNKING_PROMPT)
    max_chunk_size = kwargs.get("max_chunk_size", model_mapping[model_name].context_length)
    context_overlap = kwargs.get("context_overlap", 100)

    chunking_metadata = {
        "model_name": model_name,
        "max_chunk_size": max_chunk_size,
        "context_overlap": context_overlap,
        "method": "llm_semantic_chunking",
    }

    # Filter entries to get text entries and other entries
    text_entries, other_entries = filter_entries(entries)

    # First combine entries into context windows
    context_windows = create_context_windows(text_entries, max_size=max_chunk_size, overlap=context_overlap)

    # Process each context window with the LLM to get semantic chunks
    chunks = await process_context_windows(context_windows, model_name=model_name, chunk_prompt=chunk_prompt, chunking_metadata=chunking_metadata)

    # Combine chunks with other entries
    return chunks + other_entries


def create_context_windows(entries: list[Entry], max_size: int, overlap: int) -> list[tuple[str, list[Entry]]]:
    """
    Combine entries into context windows based on token limits.
    Returns list of (combined_text, contributing_entries) tuples.
    """
    windows = []
    current_text = ""
    current_entries = []
    current_size = 0

    for entry in entries:
        text = entry.string or ""
        text_size = len(text.split())  # Simple word count approximation

        # If adding this entry would exceed max_size, create new window
        if current_size + text_size > max_size and current_entries:
            windows.append((current_text, current_entries))

            # Start new window with overlap
            overlap_start = max(0, len(current_text.split()) - overlap)
            current_text = " ".join(current_text.split()[overlap_start:])
            current_entries = [entry for entry in current_entries if entry.string in current_text]
            current_size = len(current_text.split())

        current_text += " " + text
        current_entries.append(entry)
        current_size += text_size

    # Add final window if there's content
    if current_entries:
        windows.append((current_text, current_entries))

    return windows


async def process_context_windows(
    context_windows: list[tuple[str, list[Entry]]], model_name: str, chunk_prompt: str, chunking_metadata: dict
) -> list[Entry]:
    """
    Process each context window with the LLM to get semantic chunks.
    """
    all_chunks = []

    for window_text, contributing_entries in context_windows:
        try:
            # Format the prompt using the prompt class
            formatted_prompt = await GenerativeChunkingPrompt.format_prompt(window_text)

            # Call the LLM using featurize
            response = await featurize(
                formatted_prompt, "chunk_text", model_name=model_name, model_params={"temperature": 0.0, "response_format": {"type": "json_object"}}
            )

            # Parse the response using the prompt class
            chunks_text = GenerativeChunkingPrompt.parse_response(response)

            # Create new entries for each chunk
            for chunk_text in chunks_text:
                chunk_entries = find_contributing_entries(chunk_text, contributing_entries)

                if chunk_entries:
                    new_entry = create_chunk_entry(chunk_entries, chunk_text, chunking_metadata)
                    all_chunks.append(new_entry)

        except Exception as e:
            print(f"Error processing context window: {e}")
            # Fallback: treat the entire window as one chunk
            new_entry = create_chunk_entry(contributing_entries, window_text, chunking_metadata)
            all_chunks.append(new_entry)

    return all_chunks


def find_contributing_entries(chunk_text: str, entries: list[Entry]) -> list[Entry]:
    """
    Find which entries contributed to a given chunk based on text overlap.
    """
    contributing_entries = []
    chunk_words = set(chunk_text.lower().split())

    for entry in entries:
        entry_text = entry.string or ""
        entry_words = set(entry_text.lower().split())

        # If there's significant word overlap, consider this entry as contributing
        overlap = len(chunk_words & entry_words) / len(entry_words)
        if overlap > 0.5:  # More than 50% overlap
            contributing_entries.append(entry)

    return contributing_entries


def combine_bounding_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    """Combine multiple bounding boxes into one encompassing box."""
    if not boxes:
        return None

    return BoundingBox(
        left=min(box.left for box in boxes),
        top=min(box.top for box in boxes),
        width=max(box.left + box.width for box in boxes) - min(box.left for box in boxes),
        height=max(box.top + box.height for box in boxes) - min(box.top for box in boxes),
    )
