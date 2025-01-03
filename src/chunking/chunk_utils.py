from collections import defaultdict
from datetime import datetime
from typing import Any
import sys
import re
import spacy
from functools import lru_cache
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import Document, Entry, Index, ChunkingMethod
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

def document_to_content(document: Document) -> list[dict[str, Any]]:
    if document.entries[0].index_numbers is None:
        content = [{"text": entry.string, "pages": [1]} for entry in document.entries]
    else:
        content = [
            {"text": entry.string, "pages": [int(index.primary) for index in entry.index_numbers]}
            for entry in document.entries
        ]
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

def chunks_to_entries(document: Document, chunks: list[dict[str, Any]], strategy_type: str, chunking_metadata: dict = {}) -> list[Entry]:
    index_numbers = create_index_numbers(chunks)
    ingestion = document.entries[0].ingestion
    ingestion.total_length = sum([len(chunk["text"]) for chunk in chunks])
    ingestion.chunking_method = ChunkingMethod(strategy_type)
    ingestion.chunking_date = get_current_utc_datetime()
    ingestion.chunking_metadata = chunking_metadata
    return [
        Entry(ingestion=ingestion, string=chunk["text"], index_numbers=indices, citations=None)
        for chunk, indices in zip(chunks, index_numbers)
    ]
