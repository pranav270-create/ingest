from typing import Any
import re

from src.schemas.schemas import ChunkingMethod, Document, Entry, Index
from src.pipeline.registry import FunctionRegistry
from src.chunking.chunk_utils import document_to_content, chunks_to_entries
from src.llm_utils.tokenize_utils import detokenize_embed_input, tokenize_embed_input
from src.llm_utils.utils import Provider


@FunctionRegistry.register("chunk", "sliding_window_chunking")
async def sliding_chunks(document: list[Document], **kwargs) -> list[Entry]:
    chunk_size = kwargs.get("chunk_size", 500)
    overlap = kwargs.get("overlap", 50)
    provider = kwargs.get("provider", None)
    model = kwargs.get("model", None)
    chunking_metadata = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "provider": provider,
        "model": model
    }
    new_docs = []
    for doc in document:
        content = document_to_content(doc)
        chunks = sliding_window_chunking(content, chunk_size=chunk_size, overlap=overlap, provider=provider, model=model)
        formatted_entries = chunks_to_entries(doc, chunks, "sliding_window", chunking_metadata)
        doc.entries = formatted_entries
        new_docs.append(doc)
    return new_docs

def sliding_window_chunking(content: list[dict[str, Any]], chunk_size=500, overlap=50, provider=None, model=None) -> list[dict[str, Any]]:
    """
    Chunk the content using a sliding window of words.
    Each chunk will have 'text' and 'pages'.
    """
    tokenizer, detokenizer = tokenize_embed_input, detokenize_embed_input
    if provider is not None:
        if isinstance(provider, str):
            provider = Provider(provider)
        elif not isinstance(provider, Provider):
            raise ValueError(f"Invalid provider type: {type(provider)}. Expected str or Provider.")

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

    def find_chunk_pages(chunk_start: int, chunk_end: int) -> list[int]:
        chunk_pages = set()
        for start, end, pages in page_boundaries:
            if chunk_start < end and chunk_end > start:
                chunk_pages.update(pages)
        return sorted(list(chunk_pages))

    chunks = []
    if provider:
        tokens = tokenizer(provider, model, concatenated_text)
        num_tokens = len(tokens)
        step = chunk_size - overlap
        for i in range(0, num_tokens, step):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = detokenizer(provider, model, chunk_tokens)

            # Find the starting index of the chunk_text in concatenated_text
            chunk_start = concatenated_text.find(chunk_text)
            if chunk_start == -1:
                # Fallback if chunk_text is not found
                chunk_start = 0
            chunk_end = chunk_start + len(chunk_text)

            chunk_pages = find_chunk_pages(chunk_start, chunk_end)

            chunks.append({"text": chunk_text, "pages": chunk_pages})
    else:
        word_positions = [(m.start(), m.end()) for m in re.finditer(r"\S+", concatenated_text)]
        step = chunk_size - overlap

        for i in range(0, len(word_positions), step):
            chunk_start = word_positions[i][0]
            if i + chunk_size < len(word_positions):
                chunk_end = word_positions[i + chunk_size - 1][1]
            else:
                chunk_end = word_positions[-1][1]

            chunk_text = concatenated_text[chunk_start:chunk_end]
            chunk_pages = find_chunk_pages(chunk_start, chunk_end)

            chunks.append({"text": chunk_text, "pages": chunk_pages})
    return chunks
