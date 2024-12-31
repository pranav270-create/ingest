import asyncio
import sys
from pathlib import Path
from sqlalchemy import delete

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.sql_db.database import get_async_session
from src.sql_db.etl_model import Entry, Ingest, ProcessingPipeline, ProcessingStep, ingest_pipeline
from src.vector_db.qdrant_utils import remove_points_by_filter
from src.vector_db.qdrant_utils import get_qdrant_client


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
            remove_points_by_filter(
                client=qdrant_client,
                collection=collection_name,
                key="pipeline_id",
                value=pipeline_id
            )
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


if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore", category=ResourceWarning)

    parser = argparse.ArgumentParser(description="Remove pipeline data from SQL and vector databases")
    parser.add_argument("--collection", type=str, required=True, help="Qdrant collection name")
    parser.add_argument("--pipeline-id", type=int, required=True, help="Pipeline ID to remove")

    args = parser.parse_args()

    asyncio.run(remove_pipeline_data(args.collection, args.pipeline_id))
