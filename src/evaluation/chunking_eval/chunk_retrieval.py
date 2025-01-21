import asyncio
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

sys.path.append(str(Path(__file__).parents[2]))

from src.schemas.schemas import Entry, Ingestion, ChunkComparison
from src.sql_db.database import get_async_db_session
from src.sql_db.etl_model import Entry as DBEntry


async def get_pipeline_entries(session: AsyncSession, pipeline_id: str) -> list[Entry]:
    """Get all entries for a given pipeline ID."""
    pipeline_id = int(pipeline_id)

    query = select(DBEntry).where(DBEntry.pipeline_id == pipeline_id).options(selectinload(DBEntry.ingest))

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
                scope=db_entry.ingest.scope,
            )

            # Then create the Entry with the Ingestion object
            entry = Entry(
                uuid=db_entry.uuid,
                string=db_entry.string,
                consolidated_feature_type=db_entry.consolidated_feature_type,
                pipeline_id=db_entry.pipeline_id,
                min_primary_index=db_entry.min_primary_index,
                max_primary_index=db_entry.max_primary_index,
                ingestion=ingest,  # Use ingestion instead of ingest
            )
            entries.append(entry)

            # Debug verification
            print(f"Created entry {entry.uuid} with ingestion: {entry.ingestion is not None}")

        except Exception as e:
            print(f"Error creating entry: {e}")
            continue

    return entries


def group_entries_by_document(entries: list[Entry]) -> Dict[str, list[Entry]]:
    """Group entries by document hash."""
    grouped = defaultdict(list)
    for entry in entries:
        try:
            if hasattr(entry, "ingestion") and entry.ingestion and entry.ingestion.document_hash:
                grouped[entry.ingestion.document_hash].append(entry)
            else:
                print(f"Warning: Entry {entry.uuid} has no valid ingestion information")
        except AttributeError as e:
            print(f"Warning: Could not process entry {entry}: {e}")
            continue
    return dict(grouped)


def find_overlapping_chunks(entries_a: list[Entry], entries_b: list[Entry]) -> list[ChunkComparison]:
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
                chunks_a=entries_a_dict[page_range],
                chunks_b=entries_b_dict[page_range],
            )
        )

    return comparisons


async def compare_pipeline_chunks(pipeline_a: str, pipeline_b: str) -> Dict[str, list[ChunkComparison]]:
    """Compare chunks between two pipelines."""
    async for session in get_async_db_session():
        entries_a = await get_pipeline_entries(session, pipeline_a)
        entries_b = await get_pipeline_entries(session, pipeline_b)

        docs_a = group_entries_by_document(entries_a)
        docs_b = group_entries_by_document(entries_b)

        comparisons = {}
        for content_hash in set(docs_a.keys()) & set(docs_b.keys()):
            comparisons[content_hash] = find_overlapping_chunks(docs_a[content_hash], docs_b[content_hash])

        return comparisons


async def get_single_pipeline_entries(pipeline_id: str) -> list[Entry]:
    """Get entries from a single pipeline."""
    async for session in get_async_db_session():
        pipeline_id = int(pipeline_id)
        query = select(DBEntry).where(DBEntry.pipeline_id == pipeline_id).options(selectinload(DBEntry.ingest))
        result = await session.execute(query)
        db_entries = result.scalars().all()

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
                    scope=db_entry.ingest.scope,
                )

                # Then create the Entry with the Ingestion object
                entry = Entry(
                    uuid=db_entry.uuid,
                    string=db_entry.string,
                    consolidated_feature_type=db_entry.consolidated_feature_type,
                    pipeline_id=db_entry.pipeline_id,
                    min_primary_index=db_entry.min_primary_index,
                    max_primary_index=db_entry.max_primary_index,
                    ingestion=ingest,
                )
                entries.append(entry)

            except Exception as e:
                print(f"Error creating entry: {e}")
                continue

        return entries


async def inspect_pipeline_chunks(pipeline_id: str):
    """Inspect chunks from a pipeline for VLM-related attributes."""
    async for session in get_async_db_session():
        pipeline_id = int(pipeline_id)
        query = (
            select(DBEntry)
            .options(
                selectinload(DBEntry.ingest)  # Eagerly load the ingestion relationship
            )
            .where(DBEntry.pipeline_id == pipeline_id)
        )
        result = await session.execute(query)
        entries = result.scalars().all()

        print(f"\nPipeline {pipeline_id} Inspection:")
        print(f"Total chunks: {len(entries)}")

        if entries:
            sample_entry = entries[0]
            print("\nSample Entry Attributes:")
            print(f"Has chunk_locations: {hasattr(sample_entry, 'chunk_locations')}")
            if hasattr(sample_entry, "chunk_locations"):
                print(f"Locations data: {sample_entry.chunk_locations}")
            print(f"Feature type: {sample_entry.consolidated_feature_type}")
            if sample_entry.ingest:
                print(f"File path: {sample_entry.ingest.file_path}")


async def main():
    """Run inspection for multiple pipelines."""
    pipeline_ids = ["49", "50", "51"]
    for pid in pipeline_ids:
        await inspect_pipeline_chunks(pid)


if __name__ == "__main__":
    # Set up the event loop explicitly
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()

"""
Module for retrieving and managing document chunks from different sources.
Handles the loading, filtering, and preprocessing of chunks before evaluation.
Supports multiple chunk storage backends and formats.
"""
