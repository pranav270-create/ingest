import asyncio

import numpy as np

from src.evaluation.experimental.chunking_evaluation import ExtractionMethod
from src.schemas.schemas import ChunkingMethod

"""
Utility functions and helper classes for chunk evaluation.
Contains shared functionality used across different evaluation methods:
- Scoring utilities
- Data processing helpers
- Common evaluation metrics
"""


def calculate_chunk_comparison_score(comp_a: int, comp_b: int) -> float:
    """Calculate comparison score between two chunk sets.
    Returns score between 0 and 1, where higher favors set A.
    """
    if comp_a == comp_b:
        return 0.5
    elif comp_a == 0 and comp_b == 0:
        return 0.5
    else:
        # Normalize to 0-1 range
        total = comp_a + comp_b
        if total == 0:
            return 0.5
        return comp_a / total


def get_chunk_metrics(chunks: list) -> dict:
    """Calculate metrics for a chunk set."""
    if not chunks:
        return {"count": 0, "avg_length": 0}

    lengths = [len(chunk.string) for chunk in chunks]
    return {
        "count": len(chunks),
        "avg_length": np.mean(lengths),
        "std_length": np.std(lengths),
    }


async def evaluate_single_pipeline(pdf_path: str, extraction: ExtractionMethod, chunking: ChunkingMethod, **kwargs):
    """Evaluate a single pipeline's chunk quality."""
    chunks, metrics = await evaluate_extraction_chunking(pdf_path=pdf_path, extraction_method=extraction, chunking_method=chunking, **kwargs)

    quality_scores = []
    batch_size = 5

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        tasks = [evaluate_chunk_quality(chunk) for chunk in batch]
        batch_scores = await asyncio.gather(*tasks)
        quality_scores.extend(batch_scores)

        if i + batch_size < len(chunks):
            await asyncio.sleep(1)

    return {
        "extraction": extraction.value,
        "chunking": chunking.value,
        "metrics": metrics,
        "quality_scores": quality_scores,
        "num_chunks": len(chunks),
    }
