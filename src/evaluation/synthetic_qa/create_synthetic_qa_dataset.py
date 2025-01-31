import asyncio
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.featurization.get_features import featurize
from src.sql_db.database import get_async_session
from src.sql_db.etl_model import Entry


async def get_entries(
    session: AsyncSession,
    collection_name: str,
    pipeline_id: int = None,
    force_rerun: bool = False
) -> list[dict]:
    """Get entries from SQL database filtered by collection_name and optionally pipeline_id"""
    query = select(
        Entry.id,
        Entry.uuid,
        Entry.string,
        Entry.synthetic_questions
    ).where(
        Entry.collection_name == collection_name
    )
    if pipeline_id is not None:
        query = query.where(Entry.pipeline_id == pipeline_id)

    # Add filter for entries without synthetic questions unless force_rerun is True
    if not force_rerun:
        query = query.where(
            (Entry.synthetic_questions.is_(None)) |
            (Entry.synthetic_questions == [])
        )

    result = session.execute(query)
    return [
        {
            "id": row.id,
            "input": row.string,
            "synthetic_questions": row.synthetic_questions or []
        }
        for row in result.fetchall()
    ]


async def update_synthetic_questions(
    session: AsyncSession,
    entry_updates: list[dict],
    existing_entries: list[dict]
) -> None:
    """Update synthetic questions for entries in the database"""
    # Create lookup for existing questions
    existing_questions = {
        entry["id"]: entry["synthetic_questions"]
        for entry in existing_entries
    }

    for entry in entry_updates:
        stmt = (
            select(Entry)
            .where(Entry.id == entry["id"])
        )
        result = session.execute(stmt)
        db_entry = result.scalar_one()

        # Combine existing and new questions
        existing = existing_questions.get(entry["id"], [])
        new_questions = entry.get("synthetic_questions", [])

        # Merge questions, avoiding duplicates
        all_questions = existing + [
            q for q in new_questions
            if q not in existing
        ]

        db_entry.synthetic_questions = all_questions

    session.commit()
    print(f"Successfully updated {len(entry_updates)} entries with synthetic questions")


async def main(force_rerun: bool = False):
    model = "gpt-4o-mini"
    provider = "openai"
    functionality = "chat"
    max_tokens = 2000
    collection_name = "demo_10_20_24"
    pipeline_id = 3

    session = None
    try:
        for session in get_async_session():
            # Get entries from SQL database
            entries = await get_entries(
                session,
                collection_name=collection_name,
                pipeline_id=pipeline_id,
                force_rerun=force_rerun
            )

            if not entries:
                print("No entries found that need processing")
                return

            print(f"Processing {len(entries)} entries")

            # Generate synthetic QA pairs
            new_entries = await featurize(
                basemodels=entries,
                prompt_name="synthetic_qa_pair",
                update_metadata=False,
                model=model,
                provider=provider,
                functionality=functionality,
                max_tokens=max_tokens
            )

            # Update the database with new synthetic questions
            await update_synthetic_questions(
                session,
                new_entries,
                entries
            )

            break
    finally:
        if session:
            session.close()


if __name__ == "__main__":
    # Set force_rerun to True if you want to reprocess all entries
    asyncio.run(main(force_rerun=False))
