from collections import defaultdict
from datetime import datetime
from typing import Any
import sys
import re
import spacy
from functools import lru_cache
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import Entry, Index, ChunkingMethod, BoundingBox
from src.utils.datetime_utils import get_current_utc_datetime


def custom_sent_tokenize(text):
    # Split on periods followed by space or newline, question marks, or exclamation points
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in sentences if s.strip()]


@lru_cache(maxsize=1)
def load_spacy_model(model: str = "en_core_web_sm"):
    nlp = spacy.load(model)
    return nlp


def spacy_tokenize(text):
    nlp = load_spacy_model()
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


def entries_to_content(entries: list[Entry]) -> list[dict[str, Any]]:
    """Convert document entries to content format for chunking."""
    content = []
    for entry in entries:
        if entry.string:  # Only process entries with text content
            # Ensure we capture all parsed feature types
            feature_types = []
            if entry.parsed_feature_type:
                feature_types = [
                    ft.value if hasattr(ft, 'value') else ft 
                    for ft in entry.parsed_feature_type
                ]
            # Get bounding boxes if they exist
            bounding_boxes = []
            if entry.bounding_box:
                bounding_boxes = entry.bounding_box
            content.append({
                "text": entry.string,
                "pages": [idx.primary for idx in entry.index_numbers] if entry.index_numbers else [],
                "feature_types": feature_types,
                "bounding_boxes": bounding_boxes
            })
    return content


def create_index_numbers(chunks: list[dict[str, Any]]) -> list[list[Index]]:
    page_chunk_count = defaultdict(int)
    index_numbers = []
    for chunk in chunks:
        chunk_indices = []
        for page in chunk["pages"]:
            page_chunk_count[page] += 1
            chunk_indices.append(Index(primary=page, secondary=page_chunk_count[page], tertiary=None))
        index_numbers.append(chunk_indices)
    return index_numbers


def chunks_to_entries(entries: list[Entry], chunks: list[dict[str, Any]], strategy_type: str, chunking_metadata: dict = {}) -> list[Entry]:
    """
    Convert chunks back to Entry objects, preserving feature types and handling ingestion metadata.
    
    Args:
        document: Source document containing entries
        chunks: List of chunk dictionaries containing text, pages, and feature types
        strategy_type: Chunking strategy/method used
        chunking_metadata: Additional metadata about the chunking process
    """
    # Handle ingestion metadata
    for entry in entries:
        if entry.ingestion:
            ingestion = entry.ingestion
            ingestion.chunking_method = ChunkingMethod(strategy_type)
        ingestion.chunking_date = get_current_utc_datetime()
        ingestion.chunking_metadata = chunking_metadata

    entries = []
    for i, chunk in enumerate(chunks):
        # Create a new Entry for each chunk
        entry = Entry(
            string=chunk["text"],
            index_numbers=[Index(primary=page, secondary=i) for page in chunk["pages"]],
            ingestion=ingestion,
            parsed_feature_type=chunk.get("feature_types", []),  # Preserve feature types
            bounding_box=[BoundingBox(left=0, top=0, width=0, height=0)]  # Default bounding box
        )
        entries.append(entry)
    return entries
