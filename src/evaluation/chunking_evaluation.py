from datetime import datetime, timezone
from enum import Enum

import numpy as np
from scipy.spatial import ConvexHull
from sentence_transformers import SentenceTransformer

from src.extraction.textract import textract_parse
from src.extraction.ocr_service import main_ocr
from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import ChunkingMethod, ContentType, Entry, FileType, Ingestion, ExtractionMethod, Scope
import src.pipeline.pipeline

class ExtractionMethod(Enum):
    TEXTRACT = "textract"
    OCR = "ocr"


async def evaluate_extraction_chunking(
    pdf_path: str,
    extraction_method: ExtractionMethod,
    chunking_method: ChunkingMethod,
    chunk_size: int = 1000,
    scope: Scope = Scope.EXTERNAL,
    content_type: ContentType = ContentType.OTHER_ARTICLES,
    **kwargs,
) -> tuple[list[Entry], dict[str, float]]:
    """Run extraction + chunking and return chunks with metrics."""

    # Extract
    if extraction_method == ExtractionMethod.TEXTRACT:
        document = textract_parse(pdf_path, scope=scope, content_type=content_type)
    else:
        # Create Ingestion object for OCR
        ingestion = Ingestion(
            file_path=pdf_path,
            file_type=FileType.PDF,
            scope=scope,
            content_type=content_type,
            creator_name="evaluation_pipeline",
            ingestion_method="local_file",
            ingestion_date=datetime.now(timezone.utc).isoformat(),
            parsing_method=ExtractionMethod.OCR2_0,
        )
        documents = await main_ocr([ingestion])
        document = documents[0]
        if not document:
            raise ValueError("OCR parsing failed to return a document")

    # Chunk
    chunking_func = FunctionRegistry.get("chunk", chunking_method.value)
    chunked_docs = await chunking_func([document], chunk_size=chunk_size, **kwargs)

    # Calculate metrics
    metrics = calculate_chunk_metrics(chunked_docs[0].entries)

    return chunked_docs[0].entries, metrics


def calculate_chunk_metrics(chunks: list[Entry]) -> dict[str, float]:
    """Calculate diversity metrics for chunks."""
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Get embeddings for all chunks
    texts = [chunk.string for chunk in chunks if chunk.string]
    embeddings = model.encode(texts)

    metrics = {}

    # Calculate convex hull volume as diversity metric
    if len(embeddings) > 3:  # Need at least 4 points for 3D hull
        try:
            hull = ConvexHull(embeddings)
            metrics["embedding_diversity"] = hull.volume
        except Exception:
            metrics["embedding_diversity"] = 0

    # Calculate average pairwise distance
    distances = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            distances.append(dist)
    metrics["avg_pairwise_distance"] = np.mean(distances) if distances else 0

    return metrics
