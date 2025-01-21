from functools import lru_cache

import nltk
from nltk.tokenize import TextTilingTokenizer

from src.chunking.chunk_utils import create_chunk_entry, filter_entries
from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import Entry


@FunctionRegistry.register("chunk", "topic_segmentation_chunking")
async def topic_chunks(entries: list[Entry], **kwargs) -> list[Entry]:
    """
    Chunk entries using topic segmentation, operating directly on Entry schemas.
    """
    chunking_metadata = {"tokenizer": "Text Tiling Tokenizer"}

    # Filter entries to get text entries and other entries
    text_entries, other_entries = filter_entries(entries)

    # Process text entries directly to generate chunks
    chunks = topic_segmentation_chunking_entries(text_entries, chunking_metadata=chunking_metadata)

    # Combine chunks with other entries
    return chunks + other_entries


@lru_cache(maxsize=1)
def load_text_tiling_tokenizer():
    """Load and cache the TextTilingTokenizer."""
    try:
        return TextTilingTokenizer()
    except LookupError:
        nltk.download("punkt")
        return TextTilingTokenizer()


def topic_segmentation_chunking_entries(entries: list[Entry], chunking_metadata: dict = {}) -> list[Entry]:
    """
    Chunk the entries using topic segmentation, operating directly on Entry objects.
    Each chunk will be a new Entry object with preserved metadata.
    """
    tokenizer = load_text_tiling_tokenizer()
    if tokenizer is None:
        print("TextTilingTokenizer not initialized.")
        return []

    # Concatenate all entries' text into a single string, keeping track of positions and entries
    concatenated_text = ""
    entry_positions = []  # List of tuples: (start_pos, end_pos, entry)
    current_pos = 0

    for entry in entries:
        text = entry.string or ""
        start_pos = current_pos
        end_pos = current_pos + len(text)
        concatenated_text += text
        entry_positions.append((start_pos, end_pos, entry))
        current_pos = end_pos

    # Get topic segments using TextTilingTokenizer
    segmented_topics = tokenizer.tokenize(concatenated_text)

    # Function to find contributing entries for each chunk
    def find_contributing_entries(chunk_start: int, chunk_end: int) -> list[Entry]:
        contributing_entries = []
        for start_pos, end_pos, entry in entry_positions:
            if chunk_start < end_pos and chunk_end > start_pos:
                contributing_entries.append(entry)
        return contributing_entries

    # Create chunks from the segmented topics
    chunks = []
    current_pos = 0
    for topic in segmented_topics:
        chunk_start = current_pos
        chunk_end = chunk_start + len(topic)

        # Find entries that contribute to this chunk
        contributing_entries = find_contributing_entries(chunk_start, chunk_end)

        topic = topic.strip()
        if topic and contributing_entries:
            # Create a new Entry for the chunk
            new_entry = create_chunk_entry(contributing_entries, topic, chunking_metadata)
            chunks.append(new_entry)

        current_pos = chunk_end + 1  # +1 for the delimiter

    return chunks
