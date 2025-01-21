from pathlib import Path
import numpy as np

from src.schemas.schemas import Entry

"""
Utility functions and helper classes for chunk evaluation.
Contains shared functionality used across different evaluation methods:
- Scoring utilities
- Data processing helpers
- Common evaluation metrics
"""


def can_use_vlm(chunks_a: list[Entry], chunks_b: list[Entry]) -> bool:
    """Check if VLM evaluation is possible for these chunks."""
    # Check if chunks have locations
    has_locations_a = all(chunk.chunk_locations for chunk in chunks_a)
    has_locations_b = all(chunk.chunk_locations for chunk in chunks_b)

    if not (has_locations_a and has_locations_b):
        return False

    return True


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


def get_chunk_metrics(chunks: list[Entry]) -> dict:
    """Calculate metrics for a chunk set."""
    if not chunks:
        return {"count": 0, "avg_length": 0}

    lengths = [len(chunk.string) for chunk in chunks]
    return {
        "count": len(chunks),
        "avg_length": np.mean(lengths),
        "std_length": np.std(lengths),
    }
