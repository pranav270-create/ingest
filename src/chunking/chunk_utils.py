import sys
import uuid
from functools import lru_cache
from pathlib import Path

import spacy

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import ChunkingMethod, Entry, ExtractedFeatureType
from src.utils.datetime_utils import get_current_utc_datetime


@lru_cache(maxsize=1)
def load_spacy_model(model: str = "en_core_web_sm"):
    nlp = spacy.load(model)
    return nlp


def spacy_tokenize(text):
    nlp = load_spacy_model()
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


def filter_entries(entries: list[Entry]) -> tuple[list[Entry], list[Entry]]:
    """
    Filter entries into text entries (with consolidated_feature_type COMBINED_TEXT) and others.

    Args:
        entries: List of entries to filter

    Returns:
        Tuple of (text_entries, other_entries)
    """
    combined_text_entries = []
    other_entries = []
    for entry in entries:
        if entry.consolidated_feature_type == ExtractedFeatureType.combined_text:
            combined_text_entries.append(entry)
        else:
            other_entries.append(entry)
    return combined_text_entries, other_entries


def create_chunk_entry(
    contributing_entries: list[Entry],
    chunk_text: str,
    chunking_metadata: dict = {}
) -> Entry:
    """
    Create a new Entry object for a chunk, preserving necessary metadata from contributing entries.

    Args:
        contributing_entries: List of entries that contributed to this chunk
        chunk_text: The text content of the chunk
        chunking_metadata: Additional metadata about the chunking process

    Returns:
        A new Entry object containing the chunk and combined metadata
    """
    # Combine index_numbers from contributing entries
    index_numbers = []
    for entry in contributing_entries:
        if entry.index_numbers:
            index_numbers.extend(entry.index_numbers)

    # Remove duplicate index_numbers while preserving order
    index_numbers = list({(idx.primary, idx.secondary): idx for idx in index_numbers}.values())
    # Sort index_numbers by primary then secondary
    index_numbers.sort(key=lambda x: (x.primary, x.secondary))
    # Get base entry for metadata
    base_entry = contributing_entries[0] if contributing_entries else Entry()
    # Combine keywords from all contributing entries
    keywords = set()
    for entry in contributing_entries:
        if entry.keywords:
            keywords.update(entry.keywords)
    # Combine citations from all contributing entries
    citations = []
    seen_citations = set()
    for entry in contributing_entries:
        if entry.citations:
            for citation in entry.citations:
                citation_key = (citation.doi, citation.url, citation.text)
                if citation_key not in seen_citations:
                    citations.append(citation)
                    seen_citations.add(citation_key)
    # Set up ingestion metadata
    ingestion = base_entry.ingestion
    if ingestion:
        ingestion.chunking_method = ChunkingMethod.DISTANCE
        ingestion.chunking_date = get_current_utc_datetime()
        ingestion.chunking_metadata = chunking_metadata
    # Combine chunk_locations if present
    chunk_locations = []
    for entry in contributing_entries:
        if entry.chunk_locations:
            chunk_locations.extend(entry.chunk_locations)
    # Calculate min and max primary index
    min_primary_index = min((idx.primary for idx in index_numbers), default=None)
    max_primary_index = max((idx.primary for idx in index_numbers), default=None)
    # Combine added_featurization dictionaries
    added_featurization = {}
    for entry in contributing_entries:
        if entry.added_featurization:
            added_featurization.update(entry.added_featurization)
    # Create new Entry with the chunk text and combined metadata
    new_entry = Entry(
        uuid=str(uuid.uuid4()),
        string=chunk_text,
        ingestion=ingestion,
        index_numbers=index_numbers,
        consolidated_feature_type=base_entry.consolidated_feature_type,
        embedded_feature_type=base_entry.embedded_feature_type,
        embedding_date=base_entry.embedding_date,
        embedding_model=base_entry.embedding_model,
        embedding_dimensions=base_entry.embedding_dimensions,
        entry_title=base_entry.entry_title,
        keywords=list(keywords) if keywords else None,
        added_featurization=added_featurization if added_featurization else None,
        citations=citations if citations else None,
        chunk_locations=chunk_locations if chunk_locations else None,
        min_primary_index=min_primary_index,
        max_primary_index=max_primary_index,
        chunk_index=None,  # This would need to be set by the calling function if needed
        table_number=base_entry.table_number,
        figure_number=base_entry.figure_number
    )
    return new_entry
