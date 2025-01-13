import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.chunking.chunk_utils import chunks_to_entries, entries_to_content, spacy_tokenize
from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import Entry


@FunctionRegistry.register("chunk", "distance_chunking")
async def distance_chunks(entries: list[Entry], **kwargs) -> list[Entry]:
    chunk_size = kwargs.get("chunk_size", None)
    threshold = kwargs.get("threshold", None)
    embedding_model = kwargs.get("embedding_model", "all-MiniLM-L6-v2")
    percentile_threshold = kwargs.get("percentile_threshold", 95)

    chunking_metadata = {
        "chunk_size": chunk_size,
        "threshold": threshold,
        "embedding_model": embedding_model,
        "percentile_threshold": percentile_threshold,
        "similarity_metric": "cosine",
    }

    content = entries_to_content(entries)
    chunks, similarity_data = distance_chunking(  # noqa
        content,
        chunk_size=chunk_size,
        threshold=threshold,
        embedding_model=embedding_model,
        percentile_threshold=percentile_threshold,
    )
    # Convert chunks to entries and add to flat list
    entries = chunks_to_entries(entries, chunks, "distance", chunking_metadata)
    metadata = {}
    metadata['sentence_indices'] = similarity_data['sentence_indices']
    metadata['similarities'] = similarity_data['similarities']
    return entries


@lru_cache(maxsize=1)
def load_sentence_transformer_model(embedding_model: str) -> SentenceTransformer:
    return SentenceTransformer(embedding_model)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10))


def distance_chunking(
    content: list[dict[str, Any]],
    chunk_size: Optional[int] = None,
    threshold: Optional[float] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    percentile_threshold: int = 95,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = load_sentence_transformer_model(embedding_model)

    # Combine all text while tracking page boundaries
    concatenated_text = ""
    page_boundaries: list[tuple[int, int, list[int]]] = []
    current_pos = 0

    for page in content:
        pages = page["pages"]
        text = page["text"]
        if concatenated_text:
            concatenated_text += "\n"
            current_pos += 1
        start_idx = current_pos
        end_idx = current_pos + len(text)
        page_boundaries.append((start_idx, end_idx, pages))
        concatenated_text += text
        current_pos = len(concatenated_text)

    # Get initial chunks (either fixed-size or sentences)
    if chunk_size:
        words = concatenated_text.split()
        initial_chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
    else:
        initial_chunks = spacy_tokenize(concatenated_text)

    # Get embeddings and calculate similarities
    embeddings = model.encode(initial_chunks, convert_to_numpy=True)
    similarities = [cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]

    # If no threshold provided, calculate based on percentile
    if threshold is None:
        distances = [1 - sim for sim in similarities]
        threshold = float(np.percentile(distances, percentile_threshold))

    # Find page numbers for each chunk
    def find_chunk_pages(chunk_start: int, chunk_end: int) -> list[int]:
        chunk_pages = set()
        for start, end, pages in page_boundaries:
            if chunk_start < end and chunk_end > start:
                chunk_pages.update(pages)
        return sorted(list(chunk_pages))

    # Create final chunks
    final_chunks = []
    current_chunk = []
    current_pages = set()
    current_pos = 0

    for i, chunk in enumerate(initial_chunks):
        chunk_start = concatenated_text.find(chunk, current_pos)
        chunk_end = chunk_start + len(chunk)
        current_pos = chunk_end

        if not current_chunk:
            current_chunk.append(chunk)
            current_pages.update(find_chunk_pages(chunk_start, chunk_end))
            continue

        if i < len(similarities) and (1 - similarities[i - 1]) > threshold:
            # Save current chunk and start new one
            final_chunks.append({"text": " ".join(current_chunk), "pages": sorted(list(current_pages))})
            current_chunk = [chunk]
            current_pages = set(find_chunk_pages(chunk_start, chunk_end))
        else:
            current_chunk.append(chunk)
            current_pages.update(find_chunk_pages(chunk_start, chunk_end))

    # Add the last chunk
    if current_chunk:
        final_chunks.append({"text": " ".join(current_chunk), "pages": sorted(list(current_pages))})

    similarity_data = {
        "similarities": similarities,
        "threshold_used": threshold,
        "chunk_boundaries": [i for i, sim in enumerate(similarities) if (1 - sim) > threshold],
    }

    return final_chunks, similarity_data
