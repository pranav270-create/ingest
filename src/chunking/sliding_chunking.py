import re

from src.chunking.chunk_utils import create_chunk_entry, filter_entries
from src.llm_utils.tokenize_utils import detokenize_embed_input, tokenize_embed_input
from src.llm_utils.utils import Provider
from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import ChunkingMethod, Entry


@FunctionRegistry.register("chunk", ChunkingMethod.SLIDING_WINDOW.value)
async def sliding_chunks(entries: list[Entry], **kwargs) -> list[Entry]:
    """
    Chunk the given entries using a sliding window approach, operating directly on Entry schemas.
    """
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
    text_entries, other_entries = filter_entries(entries)

    # Process text entries directly to generate chunks
    chunks = sliding_window_chunking_entries(
        text_entries,
        chunk_size=chunk_size,
        overlap=overlap,
        provider=provider,
        model=model,
        chunking_metadata=chunking_metadata
    )

    # Combine chunks with other entries
    return chunks + other_entries


def sliding_window_chunking_entries(
    entries: list[Entry],
    chunk_size=500,
    overlap=50,
    provider=None,
    model=None,
    chunking_metadata: dict = {}
) -> list[Entry]:
    """
    Chunk the entries using a sliding window, operating directly on Entry objects.
    Each chunk will be a new Entry object with preserved metadata.
    """
    tokenizer, detokenizer = tokenize_embed_input, detokenize_embed_input
    if provider is not None:
        if isinstance(provider, str):
            provider = Provider(provider)
        elif not isinstance(provider, Provider):
            raise ValueError(
                f"Invalid provider type: {type(provider)}. Expected str or Provider."
            )

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

    chunks = []
    if provider:
        tokens = tokenizer(provider, model, concatenated_text)
        num_tokens = len(tokens)
        step = chunk_size - overlap
        for i in range(0, num_tokens, step):
            chunk_tokens = tokens[i: i + chunk_size]
            chunk_text = detokenizer(provider, model, chunk_tokens)

            # Find the starting index of the chunk_text in concatenated_text
            chunk_start = concatenated_text.find(chunk_text)
            if chunk_start == -1:
                # Fallback if chunk_text is not found
                chunk_start = 0
            chunk_end = chunk_start + len(chunk_text)

            # Find the entries that contribute to this chunk
            chunk_entries = [
                entry for start_pos, end_pos, entry in entry_positions
                if chunk_start < end_pos and chunk_end > start_pos
            ]

            # Create a new Entry for the chunk
            new_entry = create_chunk_entry(
                chunk_entries,
                chunk_text,
                chunking_metadata=chunking_metadata
            )
            chunks.append(new_entry)
    else:
        # Tokenize into words
        word_positions = [(m.start(), m.end()) for m in re.finditer(r"\S+", concatenated_text)]
        num_words = len(word_positions)
        step = chunk_size - overlap

        for i in range(0, num_words, step):
            chunk_word_positions = word_positions[i: i + chunk_size]

            if not chunk_word_positions:
                break

            chunk_start = chunk_word_positions[0][0]
            chunk_end = chunk_word_positions[-1][1]

            chunk_text = concatenated_text[chunk_start:chunk_end]

            # Find the entries that contribute to this chunk
            chunk_entries = [
                entry for start_pos, end_pos, entry in entry_positions
                if chunk_start < end_pos and chunk_end > start_pos
            ]

            # Create a new Entry for the chunk
            new_entry = create_chunk_entry(
                chunk_entries,
                chunk_text,
                chunking_metadata=chunking_metadata
            )
            chunks.append(new_entry)

    return chunks
