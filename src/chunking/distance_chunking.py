import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.chunking.chunk_utils import create_chunk_entry, filter_entries, load_spacy_model, spacy_tokenize
from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import Entry


@FunctionRegistry.register("chunk", "distance_chunking")
async def distance_chunks(entries: list[Entry], **kwargs) -> list[Entry]:
    """
    Chunk the given entries using a distance-based approach,
    operating directly on Entry schemas.
    """
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

    # Filter entries to get text entries and other entries
    text_entries, other_entries = filter_entries(entries)

    # Process text entries directly to generate chunks
    chunks, similarity_data = distance_chunking_entries(
        text_entries,
        chunk_size=chunk_size,
        threshold=threshold,
        embedding_model=embedding_model,
        percentile_threshold=percentile_threshold,
        chunking_metadata=chunking_metadata
    )

    # Combine chunks with other entries
    return chunks + other_entries


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
        initial_chunks = [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]
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


def distance_chunking_entries(
    entries: list[Entry],
    chunk_size: Optional[int] = None,
    threshold: Optional[float] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    percentile_threshold: int = 95,
    chunking_metadata: dict = {},
) -> tuple[list[Entry], dict[str, Any]]:
    """
    Chunk the entries based on distance (similarity) between chunks,
    operating directly on Entry objects.
    """
    model = load_sentence_transformer_model(embedding_model)

    # Step 1: Concatenate all text from entries, keeping track of positions and entries
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

    # Step 2: Get initial chunks (either fixed-size or sentences), keeping positions
    initial_chunks = []
    chunk_positions = []
    current_pos = 0

    if chunk_size:
        words = concatenated_text.split()
        word_positions = []
        pos = 0
        for word in words:
            start = concatenated_text.find(word, pos)
            end = start + len(word)
            word_positions.append((start, end))
            pos = end

        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)

            chunk_start = word_positions[i][0]
            if i + chunk_size - 1 < len(word_positions):
                chunk_end = word_positions[i + chunk_size - 1][1]
            else:
                chunk_end = word_positions[-1][1]
            initial_chunks.append(chunk_text)
            chunk_positions.append((chunk_start, chunk_end))
    else:
        # Use spacy to tokenize into sentences, getting positions
        nlp = load_spacy_model()
        doc = nlp(concatenated_text)
        for sent in doc.sents:
            chunk_text = sent.text
            chunk_start = sent.start_char
            chunk_end = sent.end_char
            initial_chunks.append(chunk_text)
            chunk_positions.append((chunk_start, chunk_end))

    # Step 3: Get embeddings of initial chunks
    embeddings = model.encode(initial_chunks, convert_to_numpy=True)

    # Step 4: Compute similarities between adjacent chunks
    similarities = [
        cosine_similarity(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ]

    # Step 5: Determine threshold if not provided
    if threshold is None:
        distances = [1 - sim for sim in similarities]
        threshold = float(np.percentile(distances, percentile_threshold))

    # Step 6: Group initial chunks into final chunks based on similarities
    final_chunks = []
    current_chunk_texts = []
    current_chunk_entries = set()
    chunk_boundaries = []
    similarity_data = {}
    current_chunk_start = None
    current_chunk_end = None

    for i, (chunk_text, (chunk_start, chunk_end)) in enumerate(zip(initial_chunks, chunk_positions)):
        if current_chunk_start is None:
            current_chunk_start = chunk_start
        current_chunk_texts.append(chunk_text)
        current_chunk_end = chunk_end

        # Identify entries contributing to this chunk
        chunk_entries = [
            entry for start_pos, end_pos, entry in entry_positions
            if chunk_start < end_pos and chunk_end > start_pos
        ]
        current_chunk_entries.update(chunk_entries)

        # Check if we need to split here
        if i < len(similarities) and (1 - similarities[i]) > threshold:
            # Record the boundary
            chunk_boundaries.append(i)
            # Create final chunk
            final_chunk_text = ' '.join(current_chunk_texts)
            final_chunk_entries = list(current_chunk_entries)
            # Create a new Entry
            new_entry = create_chunk_entry(
                final_chunk_entries,
                final_chunk_text,
                chunking_metadata
            )
            final_chunks.append(new_entry)
            # Reset current chunk
            current_chunk_texts = []
            current_chunk_entries = set()
            current_chunk_start = None
            current_chunk_end = None

    # Add the last chunk
    if current_chunk_texts:
        final_chunk_text = ' '.join(current_chunk_texts)
        final_chunk_entries = list(current_chunk_entries)
        # Create a new Entry
        new_entry = create_chunk_entry(
            final_chunk_entries,
            final_chunk_text,
            chunking_metadata
        )
        final_chunks.append(new_entry)

    # Prepare similarity data
    similarity_data = {
        "similarities": similarities,
        "threshold_used": threshold,
        "chunk_boundaries": chunk_boundaries,
    }

    return final_chunks, similarity_data

