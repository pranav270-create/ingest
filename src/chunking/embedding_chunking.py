from typing import Any
from functools import lru_cache
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
import sys
import re
import spacy
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import ChunkingMethod, Document, Entry, Index
from src.pipeline.registry import FunctionRegistry
from src.chunking.chunk_utils import document_to_content, chunks_to_entries, spacy_tokenize
from src.llm_utils.tokenize_utils import detokenize_embed_input, tokenize_embed_input


@FunctionRegistry.register("chunk", "embedding_chunking")
async def embedding_chunks(document: list[Document], **kwargs) -> tuple[list[Entry], dict[str, Any]]:
    threshold = kwargs.get("threshold", 0.5)
    embedding_model = kwargs.get("embedding_model", "all-MiniLM-L6-v2")
    chunking_metadata = {
        "threshold": threshold,
        "embedding_model": embedding_model,
        "similarity_metric": "cosine"
    }
    new_docs = []
    for doc in document:
        content = document_to_content(doc)
        chunks, similarity_data = embedding_chunking(content, threshold=threshold, embedding_model=embedding_model)
        formatted_entries = chunks_to_entries(doc, chunks, "embedding", chunking_metadata)
        doc.entries = formatted_entries
        metadata = {}
        metadata['sentence_indices'] = similarity_data['sentence_indices']
        metadata['similarities'] = similarity_data['similarities']
        doc.metadata = metadata
        new_docs.append(doc)
    return new_docs

@lru_cache(maxsize=1)
def load_sentence_transformer_model(embedding_model: str) -> SentenceTransformer:
    return SentenceTransformer(embedding_model)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)


def embedding_chunking(content: list[dict[str, Any]], threshold: int=0.5, embedding_model: str="all-MiniLM-L6-v2") -> list[dict[str, Any]]:
    model = load_sentence_transformer_model(embedding_model)
    concatenated_text = ""
    page_boundaries: list[tuple[int, int, list[int]]] = []
    current_pos = 0

    for page in content:
        pages = page["pages"]
        text = page["text"]
        # Add a newline character between pages to help with sentence tokenization
        if concatenated_text:
            concatenated_text += "\n"
            current_pos += 1
        start_idx = current_pos
        end_idx = current_pos + len(text)
        page_boundaries.append((start_idx, end_idx, pages))
        concatenated_text += text
        current_pos = len(concatenated_text)

    sentences = spacy_tokenize(concatenated_text)

    sentence_positions = []
    current_pos = 0
    for sentence in sentences:
        start = current_pos
        end = start + len(sentence)
        sentence_positions.append((start, end))
        current_pos = end + 1

    def find_sentence_pages(sent_start: int, sent_end: int) -> list[int]:
        sentence_pages = set()
        for start, end, pages in page_boundaries:
            if sent_start < end and sent_end > start:
                sentence_pages.update(pages)
        return sorted(list(sentence_pages))

    sentences = [s.strip() for s in sentences if s.strip()]
    embeddings = model.encode(sentences, convert_to_numpy=True)

    chunks = []
    current_chunk = []
    current_pages = set()
    current_embedding = None

    # New dictionary to store similarity data
    similarity_data = {
        "sentence_indices": [],
        "similarities": []
    }

    for idx, (sentence, embedding, (sent_start, sent_end)) in enumerate(zip(sentences, embeddings, sentence_positions)):
        if current_embedding is None:
            current_embedding = embedding
            current_chunk.append(sentence)
            current_pages.update(find_sentence_pages(sent_start, sent_end))
            continue

        similarity = np.dot(current_embedding, embedding)
        
        # Store similarity data
        similarity_data["sentence_indices"].append(idx)
        similarity_data["similarities"].append(float(similarity))  # Convert to float for JSON serialization

        if similarity > threshold:
            chunk_text = " ".join(current_chunk)
            chunks.append({"text": chunk_text, "pages": sorted(list(current_pages))})
            current_chunk = [sentence]
            current_pages = set(find_sentence_pages(sent_start, sent_end))
            current_embedding = embedding
        else:
            current_chunk.append(sentence)
            current_embedding = (current_embedding * (len(current_chunk) - 1) + embedding) / len(current_chunk)
            current_embedding /= np.linalg.norm(current_embedding) + 1e-10
            current_pages.update(find_sentence_pages(sent_start, sent_end))

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({"text": chunk_text, "pages": sorted(list(current_pages))})

    return chunks, similarity_data


if __name__ == "__main__":
    import sys
    import json
    import asyncio
    with open("/private/tmp/pipeline_storage/10_ingest_youtube_to_transcript_1.json") as f:
        documents = json.load(f)

    for document in documents:
        document = Document(**document)
        entries = asyncio.run(embedding_chunks([document]))
