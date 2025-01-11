import asyncio
import sys
from pathlib import Path

from sqlalchemy import select

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import ContentType
from src.sql_db.database_simple import get_async_session
from src.sql_db.etl_model import Ingest
from src.vector_db.qdrant_utils import async_get_qdrant_client, async_update_payload_by_filter


async def update_content_type(collection_name: str, old_type: ContentType, new_type: ContentType):
    """
    Update the content_type in Entry's ingestion data from old_type to new_type,
    both in SQL database and Qdrant.
    """
    print(f"Starting update of content_type from {old_type} to {new_type}")

    # Get Qdrant client
    qdrant_client = await async_get_qdrant_client()

    try:
        async with get_async_session() as session:
            try:
                print("Updating entries in SQL database...")
                # Query Ingest directly
                query = (
                    select(Ingest)
                    .where(Ingest.content_type == old_type.value)
                )
                result = session.execute(query)
                ingests = result.scalars().all()

                print(f"Found {len(ingests)} entries to update")

                # Update content_type directly on Ingest objects
                for ingest in ingests:
                    ingest.content_type = new_type.value

                session.commit()
                print(f"Successfully updated {len(ingests)} entries in SQL")

                # Update Qdrant payloads
                print("Updating entries in Qdrant...")
                await async_update_payload_by_filter(
                    client=qdrant_client,
                    collection=collection_name,
                    filter_key="content_type",
                    filter_value=old_type.value,
                    payload_key="content_type",
                    payload_value=new_type.value
                )
                print("Successfully updated entries in Qdrant")

            except Exception as e:
                session.rollback()
                print(f"Error updating entries: {str(e)}")
                raise
    finally:
        await qdrant_client.close()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=ResourceWarning)

    collection = "demo_10_20_24"
    old_type = "rss"
    new_type = "one_pager"

    asyncio.run(update_content_type(
        collection,
        ContentType(old_type),
        ContentType(new_type)
    ))
