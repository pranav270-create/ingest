import argparse
import asyncio
import sys
import warnings
from pathlib import Path

from sqlalchemy import delete, select

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import ContentType
from src.sql_db.database_simple import get_async_session
from src.sql_db.etl_model import Entry, Ingest, ProcessingPipeline, ProcessingStep, ingest_pipeline
from src.upsert.qdrant_utils import async_get_qdrant_client, async_update_payload_by_filter, get_qdrant_client, remove_points_by_filter


async def remove_pipeline_data(collection_name: str, pipeline_id: int):
    """
    Remove all data associated with a pipeline ID from both SQL and vector databases.
    """
    print(f"Starting removal process for pipeline {pipeline_id} in collection {collection_name}")
    # Initialize Qdrant client
    qdrant_client = get_qdrant_client()
    for session in get_async_session():
        # 1. First delete all entries associated with this pipeline
        try:
            print("Removing entries from SQL database...")
            delete_entries = delete(Entry).where(Entry.pipeline_id == pipeline_id)
            session.execute(delete_entries)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error removing entries from SQL database: {str(e)}")
            raise

        # 2. Remove from Qdrant
        try:
            print("Removing entries from Qdrant...")
            remove_points_by_filter(client=qdrant_client, collection=collection_name, key="pipeline_id", value=pipeline_id)
        except Exception as e:
            print(f"Error removing entries from Qdrant: {str(e)}")
            raise

        # 3. Delete processing steps associated with this pipeline
        try:
            print("Removing processing steps...")
            delete_steps = delete(ProcessingStep).where(ProcessingStep.pipeline_id == pipeline_id)
            session.execute(delete_steps)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error removing processing steps: {str(e)}")
            raise

        # 4. Delete the association in ingest_pipeline table
        try:
            print("Removing pipeline associations...")
            delete_assoc = delete(ingest_pipeline).where(ingest_pipeline.c.pipeline_id == pipeline_id)
            session.execute(delete_assoc)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error removing pipeline associations: {str(e)}")
            raise

        # 5. Finally delete the pipeline itself
        try:
            print("Removing pipeline from SQL database...")
            delete_pipeline = delete(ProcessingPipeline).where(ProcessingPipeline.id == pipeline_id)
            session.execute(delete_pipeline)
            session.commit()
            print(f"Successfully removed pipeline {pipeline_id} and all related data")
        except Exception as e:
            session.rollback()
            print(f"Error removing pipeline from SQL database: {str(e)}")
            raise


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

    warnings.filterwarnings("ignore", category=ResourceWarning)

    parser = argparse.ArgumentParser(description="Remove pipeline data from SQL and vector databases")
    parser.add_argument("--collection", type=str, required=True, help="Qdrant collection name")
    parser.add_argument("--pipeline-id", type=int, required=True, help="Pipeline ID to remove")

    args = parser.parse_args()

    asyncio.run(remove_pipeline_data(args.collection, args.pipeline_id))


    # Example 2: Modify content type
    warnings.filterwarnings("ignore", category=ResourceWarning)

    collection = "demo_10_20_24"
    old_type = "rss"
    new_type = "one_pager"

    asyncio.run(update_content_type(
        collection,
        ContentType(old_type),
        ContentType(new_type)
    ))

