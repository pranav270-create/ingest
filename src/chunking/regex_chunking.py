import re
from typing import Any

from src.schemas.schemas import ChunkingMethod, Document, Entry, Index
from src.pipeline.registry import FunctionRegistry
from src.chunking.chunk_utils import document_to_content, chunks_to_entries


@FunctionRegistry.register("chunk", "regex_chunking")
async def regex_chunks(document: list[Document], **kwargs) -> list[Entry]:
    patterns = kwargs.get("patterns", None)
    if patterns is None:
        patterns = [r"\n \n", r"\n\n"]  # Default split pattern
    chunking_metadata = {
        "patterns": patterns
    }
    new_docs = []
    for doc in document:
        content = document_to_content(doc)
        chunks = regex_chunking(content)
        formatted_entries = chunks_to_entries(doc, chunks, "regex", chunking_metadata)
        doc.entries = formatted_entries
        new_docs.append(doc)
    return new_docs

def regex_chunking(content: list[dict[str, Any]], patterns=None) -> list[dict[str, Any]]:
    if patterns is None:
        patterns = [r"\n \n", r"\n\n"]  # Default split pattern
    
    chunks = []
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

    pattern = "|".join(patterns)
    split_positions = [m.start() for m in re.finditer(pattern, concatenated_text)]
    split_positions = [0] + split_positions + [len(concatenated_text)]

    def find_chunk_pages(chunk_start: int, chunk_end: int) -> list[int]:
        chunk_pages = set()
        for start, end, pages in page_boundaries:
            if chunk_start < end and chunk_end > start:
                chunk_pages.update(pages)
        return sorted(list(chunk_pages))

    for i in range(len(split_positions) - 1):
        chunk_start = split_positions[i]
        chunk_end = split_positions[i + 1]
        chunk_text = concatenated_text[chunk_start:chunk_end].strip()

        if not chunk_text:
            continue

        pages = find_chunk_pages(chunk_start, chunk_end)
        chunks.append({"text": chunk_text, "pages": pages})

    return chunks
