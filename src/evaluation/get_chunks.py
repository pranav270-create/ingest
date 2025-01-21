import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

sys.path.append(str(Path(__file__).parents[2]))

from src.schemas.schemas import Entry, Ingestion
from src.sql_db.database import get_async_db_session
from src.sql_db.etl_model import Entry as DBEntry


@dataclass
class ChunkComparison:
    document_title: str
    page_range: tuple[int, int]
    pipeline_a_chunks: list[Entry]
    pipeline_b_chunks: list[Entry]


async def get_pipeline_entries(session: AsyncSession, pipeline_id: str) -> list[Entry]:
    """Get all entries for a given pipeline ID."""
    pipeline_id = int(pipeline_id)

    query = select(DBEntry).where(DBEntry.pipeline_id == pipeline_id).options(
        selectinload(DBEntry.ingest)
    )

    result = await session.execute(query)
    db_entries = result.scalars().all()

    # Debug logging
    print(f"Found {len(db_entries)} entries for pipeline {pipeline_id}")

    entries = []
    for db_entry in db_entries:
        try:
            # First create the Ingestion object
            ingest = Ingestion(
                document_hash=db_entry.ingest.document_hash,
                document_title=db_entry.ingest.document_title,
                file_path=db_entry.ingest.file_path,
                creator_name=db_entry.ingest.creator_name,
                creation_date=db_entry.ingest.creation_date.isoformat() if db_entry.ingest.creation_date else None,
                file_type=db_entry.ingest.file_type,
                ingestion_method=db_entry.ingest.ingestion_method,
                ingestion_date=db_entry.ingest.ingestion_date.isoformat() if db_entry.ingest.ingestion_date else None,
                scope=db_entry.ingest.scope
            )

            # Then create the Entry with the Ingestion object
            entry = Entry(
                uuid=db_entry.uuid,
                string=db_entry.string,
                consolidated_feature_type=db_entry.consolidated_feature_type,
                pipeline_id=db_entry.pipeline_id,
                min_primary_index=db_entry.min_primary_index,
                max_primary_index=db_entry.max_primary_index,
                ingestion=ingest  # Use ingestion instead of ingest
            )
            entries.append(entry)

            # Debug verification
            print(f"Created entry {entry.uuid} with ingestion: {entry.ingestion is not None}")

        except Exception as e:
            print(f"Error creating entry: {e}")
            continue

    return entries


def group_entries_by_document(entries: List[Entry]) -> Dict[str, List[Entry]]:
    """Group entries by document hash."""
    grouped = defaultdict(list)
    for entry in entries:
        try:
            if hasattr(entry, 'ingestion') and entry.ingestion and entry.ingestion.document_hash:
                grouped[entry.ingestion.document_hash].append(entry)
            else:
                print(f"Warning: Entry {entry.uuid} has no valid ingestion information")
        except AttributeError as e:
            print(f"Warning: Could not process entry {entry}: {e}")
            continue
    return dict(grouped)


def find_overlapping_chunks(entries_a: List[Entry], entries_b: List[Entry]) -> List[ChunkComparison]:
    """Find chunks that cover the same page ranges across two sets of entries."""
    comparisons = []
    document_title = entries_a[0].ingestion.document_title if entries_a else "Unknown"

    # Create dictionaries grouping entries by page range
    entries_b_dict = defaultdict(list)
    for entry in entries_b:
        page_range = (entry.min_primary_index, entry.max_primary_index)
        entries_b_dict[page_range].append(entry)

    entries_a_dict = defaultdict(list)
    for entry in entries_a:
        page_range = (entry.min_primary_index, entry.max_primary_index)
        entries_a_dict[page_range].append(entry)

    # Create comparisons for all unique page ranges
    all_page_ranges = set(entries_a_dict.keys()) | set(entries_b_dict.keys())
    for page_range in all_page_ranges:
        comparisons.append(
            ChunkComparison(
                document_title=document_title,
                page_range=page_range,
                pipeline_a_chunks=entries_a_dict[page_range],
                pipeline_b_chunks=entries_b_dict[page_range],
            )
        )

    return comparisons


async def compare_pipeline_chunks(pipeline_a: str, pipeline_b: str) -> Dict[str, List[ChunkComparison]]:
    """Compare chunks between two pipelines."""
    async for session in get_async_db_session():
        entries_a = await get_pipeline_entries(session, pipeline_a)
        entries_b = await get_pipeline_entries(session, pipeline_b)

        docs_a = group_entries_by_document(entries_a)
        docs_b = group_entries_by_document(entries_b)

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
        print(f"Document path: {chunk_comparisons[0].pipeline_a_chunks[0].ingestion.file_path}")

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
