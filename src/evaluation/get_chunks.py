from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from sql_db.etl_model import Entry
from sql_db.database_simple import get_async_db_session


@dataclass
class ChunkComparison:
    content_hash: str
    page_range: Tuple[int, int]
    pipeline_a_chunk: Optional[Entry]
    pipeline_b_chunk: Optional[Entry]


async def get_pipeline_entries(pipeline_id: str) -> List[Entry]:
    """Fetch all entries associated with a pipeline ID."""
    async for session in get_async_db_session():
        stmt = (
            select(Entry)
            .options(selectinload(Entry.ingest))
            .where(Entry.pipeline_id == pipeline_id)
        )
        result = await session.execute(stmt)
        return result.scalars().all()


def group_entries_by_document(entries: List[Entry]) -> Dict[str, List[Entry]]:
    """Group entries by their document hash (from associated Ingest record)."""
    grouped = defaultdict(list)
    for entry in entries:
        if entry.ingest:  # Check if there's an associated ingest record
            grouped[entry.ingest.hash].append(entry)
    return grouped


def find_overlapping_chunks(
    chunks_a: List[Entry], chunks_b: List[Entry]
) -> List[ChunkComparison]:
    """Find overlapping chunks between two lists of entries from the same document."""
    comparisons = []

    # Create lookup dictionaries for quick access
    chunks_b_dict = {
        (chunk.min_primary_index, chunk.max_primary_index): chunk for chunk in chunks_b
    }

    for chunk_a in chunks_a:
        range_a = (chunk_a.min_primary_index, chunk_a.max_primary_index)
        matching_chunk_b = None

        # Look for exact matches first
        if range_a in chunks_b_dict:
            matching_chunk_b = chunks_b_dict[range_a]
        else:
            # Look for overlapping ranges
            for (min_b, max_b), chunk_b in chunks_b_dict.items():
                # Check if ranges overlap
                if not (
                    max_b < chunk_a.min_primary_index
                    or min_b > chunk_a.max_primary_index
                ):
                    matching_chunk_b = chunk_b
                    break

        comparisons.append(
            ChunkComparison(
                content_hash=chunk_a.content_hash,
                page_range=(chunk_a.min_primary_index, chunk_a.max_primary_index),
                pipeline_a_chunk=chunk_a,
                pipeline_b_chunk=matching_chunk_b,
            )
        )

    return comparisons


async def compare_pipeline_chunks(
    pipeline_id_a: str, pipeline_id_b: str
) -> Dict[str, List[ChunkComparison]]:
    """Compare chunks between two pipelines, grouping by document."""
    # Get entries for both pipelines
    entries_a = await get_pipeline_entries(pipeline_id_a)
    entries_b = await get_pipeline_entries(pipeline_id_b)

    print(f"Pipeline {pipeline_id_a} has {len(entries_a)} entries")
    print(f"Pipeline {pipeline_id_b} has {len(entries_b)} entries")

    # Group entries by document
    docs_a = group_entries_by_document(entries_a)
    docs_b = group_entries_by_document(entries_b)

    print(f"Pipeline {pipeline_id_a} has {len(docs_a)} documents")
    print(f"Pipeline {pipeline_id_b} has {len(docs_b)} documents")

    # Compare chunks for each document that appears in both pipelines
    comparisons = {}
    for content_hash in set(docs_a.keys()) & set(docs_b.keys()):
        comparisons[content_hash] = find_overlapping_chunks(
            docs_a[content_hash], docs_b[content_hash]
        )

    return comparisons


async def main():
    # Example pipeline IDs
    pipeline_a = 2
    pipeline_b = 1

    comparisons = await compare_pipeline_chunks(pipeline_a, pipeline_b)

    # Print results
    for content_hash, chunk_comparisons in comparisons.items():
        print(f"\nDocument hash: {content_hash}")
        for comp in chunk_comparisons:
            print(f"Page range: {comp.page_range}")
            print(f"Pipeline A chunk present: {comp.pipeline_a_chunk is not None}")
            print(f"Pipeline B chunk present: {comp.pipeline_b_chunk is not None}")
            print("---")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
