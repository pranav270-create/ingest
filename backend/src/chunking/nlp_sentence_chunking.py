from typing import Any

from src.schemas.schemas import ChunkingMethod, Document, Entry, Index
from src.pipeline.registry import FunctionRegistry
from src.chunking.chunk_utils import document_to_content, chunks_to_entries, spacy_tokenize


@FunctionRegistry.register("chunk", "nlp_sentence_chunking")
async def sentence_chunks(document: list[Document], **kwargs) -> list[Entry]:
    min_sentences = kwargs.get("min_sentences", 5)
    chunking_metadata = {
        "min_sentences": min_sentences,
        "tokenizer": "nltk_punkt"
    }
    new_docs = []
    for doc in document:
        content = document_to_content(doc)
        chunks = nlp_sentence_chunking(content, min_sentences=min_sentences)
        formatted_entries = chunks_to_entries(doc, chunks, "nlp_sentence", chunking_metadata)
        doc.entries = formatted_entries
        new_docs.append(doc)
    return new_docs

def nlp_sentence_chunking(content: list[dict[str, Any]], min_sentences=5) -> list[dict[str, Any]]:
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

    sentences = spacy_tokenize(concatenated_text)

    def find_sentence_pages(sent_start: int, sent_end: int) -> list[int]:
        sentence_pages = set()
        for start, end, pages in page_boundaries:
            if sent_start < end and sent_end > start:
                sentence_pages.update(pages)
        return sorted(list(sentence_pages))

    sentence_chunks = []
    current_pos = 0
    for sentence in sentences:
        sent_start = current_pos
        sent_end = sent_start + len(sentence)
        pages = find_sentence_pages(sent_start, sent_end)

        sentence = sentence.strip()
        if sentence:
            sentence_chunks.append({"text": sentence, "pages": pages})
        current_pos = sent_end + 1

    final_chunks = []
    current_chunk = {"text": [], "pages": set()}
    for sentence in sentence_chunks:
        current_chunk["text"].append(sentence["text"])
        current_chunk["pages"].update(sentence["pages"])

        if len(current_chunk["text"]) >= min_sentences:
            final_chunks.append({"text": " ".join(current_chunk["text"]), "pages": sorted(list(current_chunk["pages"]))})
            current_chunk = {"text": [], "pages": set()}

    if current_chunk["text"]:
        final_chunks.append({"text": " ".join(current_chunk["text"]), "pages": sorted(list(current_chunk["pages"]))})

    return final_chunks