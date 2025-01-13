from functools import lru_cache, partial
from typing import Any
import sys
from pathlib import Path
from nltk.tokenize import TextTilingTokenizer
import nltk

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import Entry
from src.pipeline.registry.function_registry import FunctionRegistry
from src.chunking.chunk_utils import entries_to_content, chunks_to_entries


@FunctionRegistry.register("chunk", "topic_segmentation_chunking")
async def topic_chunks(entries: list[Entry], **kwargs) -> list[Entry]:
    chunking_metadata = {
        "tokenizer": "Text Tiling Tokenizer"
    }
    content = entries_to_content(entries)
    chunks = topic_segmentation_chunking(content)
    formatted_entries = chunks_to_entries(entries, chunks, "topic_segmentation", chunking_metadata)
    entries = formatted_entries
    return entries


@lru_cache(maxsize=1)
def load_text_tiling_tokenizer():
    try:
        return TextTilingTokenizer()
    except LookupError:
        nltk.download('punkt')
        return TextTilingTokenizer()


def topic_segmentation_chunking(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokenizer = load_text_tiling_tokenizer()
    if tokenizer is None:
        print("TextTilingTokenizer not initialized.")
        return []

    # Concatenate all pages with delimiters and track page boundaries
    concatenated_text = ""
    page_boundaries: list[tuple[int, int, list[int]]] = []
    current_pos = 0

    for page in content:
        pages = page["pages"]
        text = page["text"]
        start_idx = current_pos
        end_idx = current_pos + len(text)
        page_boundaries.append((start_idx, end_idx, pages))
        concatenated_text += text
        current_pos = len(concatenated_text)

    segmented_topics = tokenizer.tokenize(concatenated_text)

    def find_chunk_pages(chunk_start: int, chunk_end: int) -> list[int]:
        chunk_pages = set()
        for start, end, pages in page_boundaries:
            if chunk_start < end and chunk_end > start:
                chunk_pages.update(pages)
        return sorted(list(chunk_pages))

    chunks = []
    current_pos = 0
    for topic in segmented_topics:
        chunk_start = current_pos
        chunk_end = chunk_start + len(topic)
        pages = find_chunk_pages(chunk_start, chunk_end)

        topic = topic.strip()
        if topic:
            chunks.append({"text": topic, "pages": pages})
        current_pos = chunk_end + 1  # +1 for the delimiter

    return chunks
