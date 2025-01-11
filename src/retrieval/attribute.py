import re
from typing import Any

import nltk

from src.prompts.online.citation_prompts import TextChunkWithAttribution
from src.schemas.schemas import FormattedScoredPoints

nltk.data.find("tokenizers/punkt")
sentence_tokenizer = nltk.sent_tokenize


def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into actual sentences using NLTK's sentence tokenizer.
    """
    sentences = sentence_tokenizer(text)
    return [sentence.strip() for sentence in sentences]


def remove_square_bracket_numbers(text: str) -> str:
    return re.sub(r"\[\d+\]", "", text)


def _index_documents(documents: list[FormattedScoredPoints]) -> tuple[list[str], dict[int, int]]:
    """
    Index the documents and split them into sentences.
    """
    indexed_documents = {}
    indexed_sentences = []
    i = 1
    for doc in documents:
        indexed_documents[i] = doc.id
        formatted_sentence = f"SOURCE {i}:\n==========\n" + doc.raw_text + "\n"
        indexed_sentences.append(formatted_sentence)
        i += 1
    return indexed_sentences, indexed_documents


def index_documents(documents: list[FormattedScoredPoints]) -> tuple[list[str], dict[int, int]]:
    indexed_documents = {}
    indexed_sentences = []
    i = 1
    for doc in documents:
        sentences = split_into_sentences(doc.raw_text)
        for sentence in sentences:
            cleaned_sentence = remove_square_bracket_numbers(sentence)
            formatted_sentence = f"SOURCE {i}:\n==========\n" + cleaned_sentence + "\n"
            indexed_documents[i] = doc.id
            indexed_sentences.append(formatted_sentence)
            i += 1
    return indexed_sentences, indexed_documents


def group_citations(
    sentence_plan: list[TextChunkWithAttribution],
    indexed_documents: dict[int, int],
    reranked_context: list[FormattedScoredPoints],
) -> tuple[list[dict[str, Any]], list[str]]:
    citations = []
    citation_lookup = {}
    annotated_sentences = []

    for sentence in sentence_plan:
        sorted_indices = sorted(sentence.attribution)
        current_group = []
        current_doc_id = None
        sentence_citations = []

        for idx in sorted_indices:
            doc_id = indexed_documents.get(idx)

            if current_doc_id is None:
                current_doc_id = doc_id
                current_group = [idx]
            elif doc_id == current_doc_id and idx == current_group[-1] + 1:
                current_group.append(idx)
            else:
                # Save the current group and start a new one
                if current_group:
                    doc = next(d for d in reranked_context if d.id == current_doc_id)
                    all_sentences = split_into_sentences(doc.raw_text)

                    # Calculate the actual sentence indices and highlight ranges
                    start_idx = min(current_group) - 2
                    end_idx = max(current_group) + 2
                    relevant_sentences = all_sentences[start_idx:end_idx]
                    citation_text = " ".join(relevant_sentences)

                    # Generate highlight ranges
                    highlight_ranges = []
                    current_pos = 0
                    for i, sent in enumerate(all_sentences):
                        if i >= start_idx and i < end_idx:
                            # Find exact position in original text
                            pos = doc.raw_text.find(sent.strip(), current_pos)
                            if pos != -1:
                                highlight_ranges.append({
                                    'start': pos,
                                    'end': pos + len(sent.strip())
                                })
                        # Move position pointer
                        pos = doc.raw_text.find(sent.strip(), current_pos)
                        if pos != -1:
                            current_pos = pos + len(sent.strip())

                    # Check if this citation already exists
                    citation_key = (current_doc_id, citation_text)
                    if citation_key in citation_lookup:
                        citation_idx = citation_lookup[citation_key]
                    else:
                        citations.append({
                            "text": citation_text,
                            "document_id": current_doc_id,
                            "title": doc.title,
                            "date": doc.date,
                            "metadata": {
                                "full_chunk": doc.raw_text,
                                "highlights": highlight_ranges
                            },
                            "public_url": doc.ingestion.public_url
                        })
                        citation_idx = len(citations)
                        citation_lookup[citation_key] = citation_idx

                    sentence_citations.append(citation_idx)
                current_doc_id = doc_id
                current_group = [idx]

        # Handle the last group (same logic as above)
        if current_group:
            doc = next(d for d in reranked_context if d.id == current_doc_id)
            all_sentences = split_into_sentences(doc.raw_text)

            start_idx = min(current_group)
            end_idx = max(current_group) + 1
            relevant_sentences = all_sentences[start_idx:end_idx]
            citation_text = " ".join(relevant_sentences)

            # Generate highlight ranges
            highlight_ranges = []
            current_pos = 0
            for i, sent in enumerate(all_sentences):
                if i >= start_idx and i < end_idx:
                    pos = doc.raw_text.find(sent.strip(), current_pos)
                    if pos != -1:
                        highlight_ranges.append({
                            'start': pos,
                            'end': pos + len(sent.strip())
                        })
                pos = doc.raw_text.find(sent.strip(), current_pos)
                if pos != -1:
                    current_pos = pos + len(sent.strip())

            citation_key = (current_doc_id, citation_text)
            if citation_key in citation_lookup:
                citation_idx = citation_lookup[citation_key]
            else:
                citations.append({
                    "text": citation_text,
                    "document_id": current_doc_id,
                    "title": doc.title,
                    "date": doc.date,
                    "metadata": {
                        "full_chunk": doc.raw_text,
                        "highlights": highlight_ranges
                    },
                    "public_url": doc.ingestion.public_url
                })
                citation_idx = len(citations)
                citation_lookup[citation_key] = citation_idx

            sentence_citations.append(citation_idx)

        annotated_sentences.append(f"{sentence} [{']['.join(str(i) for i in sentence_citations)}]")

    return citations, annotated_sentences
