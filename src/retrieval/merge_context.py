import sys
from itertools import groupby
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import (
    FormattedScoredPoints,
)


def merge_overlapping_context(
    points: list[FormattedScoredPoints],
) -> list[FormattedScoredPoints]:
    """
    Merges overlapping context from FormattedScoredPoints by removing duplicate text
    from lower-ranked results when they share the same source and are consecutive.
    """

    def get_doc_id(point: FormattedScoredPoints) -> str:
        """Helper to get unique document identifier"""
        if point.ingestion and point.ingestion.file_path:
            return point.ingestion.file_path
        elif point.ingestion and point.ingestion.public_url:
            return point.ingestion.public_url
        return ""

    def are_neighbors(p1: FormattedScoredPoints, p2: FormattedScoredPoints) -> bool:
        """Check if two points are neighbors based on their indices"""
        if not (p1.index and p2.index):
            return False

        # Same primary index - check secondary indices
        if p1.index.primary == p2.index.primary:
            # If both have secondary indices, they must be consecutive
            if p1.index.secondary is not None and p2.index.secondary is not None:
                return abs(p1.index.secondary - p2.index.secondary) == 1
            return True  # If either doesn't have secondary, treat as neighbors

        # Consecutive primary indices
        if abs(p1.index.primary - p2.index.primary) == 1:
            # If no secondary indices, they're definitely neighbors
            if not p1.index.secondary and not p2.index.secondary:
                return True

            # If both have secondary indices, check if there must be a chunk between
            if p1.index.secondary is not None and p2.index.secondary is not None:
                # For ascending primary indices (p1 -> p2)
                if p1.index.primary < p2.index.primary:
                    # If p1's secondary is too low or p2's is too high, there's a gap
                    return not (p1.index.secondary < p2.index.secondary)
                # For descending primary indices (p2 -> p1)
                else:
                    # If p2's secondary is too low or p1's is too high, there's a gap
                    return not (p2.index.secondary < p1.index.secondary)

            # If only one has secondary index, we can't be certain
            return True
        return False

    def remove_overlap(p1: FormattedScoredPoints, p2: FormattedScoredPoints):
        """Remove overlapping text from the lower-scored point"""
        # Find the point with lower score
        lower_point = p1 if p1.rerank_score < p2.rerank_score else p2
        higher_point = p2 if p1.rerank_score < p2.rerank_score else p1
        # Find the longest common suffix/prefix between the two texts
        min_len = min(len(lower_point.raw_text), len(higher_point.raw_text))
        overlap_len = 0
        # Start with a reasonable minimum overlap length (e.g., 10 chars)
        for i in range(10, min_len + 1):
            # Look for overlap between end of first chunk and start of second chunk
            suffix = p1.raw_text[-i:]
            if p2.raw_text.startswith(suffix):
                overlap_len = i
        # If significant overlap found
        if overlap_len > 10:  # Minimum threshold to avoid tiny matches
            # Remove overlap from lower-scored point
            if lower_point == p1:
                lower_point.raw_text = lower_point.raw_text[:-overlap_len].strip()
            else:
                lower_point.raw_text = lower_point.raw_text[overlap_len:].strip()

    # Group points by document ID
    points.sort(key=get_doc_id)
    for doc_id, group in groupby(points, key=get_doc_id):
        if not doc_id:  # Skip if no valid document ID
            continue
        # Convert group to list and sort by primary index
        group_points = sorted(
            list(group), key=lambda x: x.index.primary if x.index else 0
        )
        # Check each pair of consecutive points
        for i in range(len(group_points) - 1):
            if are_neighbors(group_points[i], group_points[i + 1]):
                remove_overlap(group_points[i], group_points[i + 1])

    return points
