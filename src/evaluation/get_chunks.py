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
    document_title: str
    page_range: Tuple[int, int]
    pipeline_a_chunks: list[Entry]
    pipeline_b_chunks: list[Entry]


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
    document_title = chunks_a[0].ingest.document_title if chunks_a else "Unknown"

    # Create dictionaries grouping chunks by page range
    chunks_b_dict = defaultdict(list)
    for chunk in chunks_b:
        page_range = (chunk.min_primary_index, chunk.max_primary_index)
        chunks_b_dict[page_range].append(chunk)

    # Group chunks_a by page range
    chunks_a_dict = defaultdict(list)
    for chunk in chunks_a:
        page_range = (chunk.min_primary_index, chunk.max_primary_index)
        chunks_a_dict[page_range].append(chunk)

    # Create comparisons for all unique page ranges
    all_page_ranges = set(chunks_a_dict.keys()) | set(chunks_b_dict.keys())
    for page_range in all_page_ranges:
        comparisons.append(
            ChunkComparison(
                content_hash=chunks_a[0].content_hash if chunks_a else chunks_b[0].content_hash,
                document_title=document_title,
                page_range=page_range,
                pipeline_a_chunks=chunks_a_dict[page_range],
                pipeline_b_chunks=chunks_b_dict[page_range],
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
        print(f"Document title: {chunk_comparisons[0].document_title}")
        print(f"Document path: {chunk_comparisons[0].pipeline_a_chunks[0].ingest.file_path}")

        for comp in chunk_comparisons:
            print(f"\nPage range: {comp.page_range}")

            # Count text or combined_text chunks for Pipeline A
            combined_text_a = sum(1 for chunk in comp.pipeline_a_chunks if chunk.consolidated_feature_type in ['combined_text', 'text'])
            print(f"Pipeline A chunks: {len(comp.pipeline_a_chunks)} entries")
            print(f"\033[94mCombined text A: {combined_text_a} entries\033[0m")  # Print in blue
            if comp.pipeline_a_chunks:
                print("Feature types A:", [chunk.consolidated_feature_type for chunk in comp.pipeline_a_chunks])

            # Count text or combined_text chunks for Pipeline B
            combined_text_b = sum(1 for chunk in comp.pipeline_b_chunks if chunk.consolidated_feature_type in ['combined_text', 'text'])
            print(f"Pipeline B chunks: {len(comp.pipeline_b_chunks)} entries")
            print(f"\033[94mCombined text B: {combined_text_b} entries\033[0m")  # Print in blue
            if comp.pipeline_b_chunks:
                print("Feature types B:", [chunk.consolidated_feature_type for chunk in comp.pipeline_b_chunks])
            print("---")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
