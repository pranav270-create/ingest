import sys
from pathlib import Path
from typing import Union

sys.path.append(str(Path(__file__).parents[2]))

from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import Embedding, Entry, Upsert
from src.sql_db.database import get_async_db_session
from src.sql_db.etl_crud import create_entries


@FunctionRegistry.register("upsert", "upsert_entries")
async def upsert_entries(
    results: Union[list[Entry], list[Upsert], list[Embedding]],
    collection_name: str,
    version: str,
    update_on_collision: bool = False,
    read: callable = None, # noqa
    write: callable = None, # noqa
    **kwargs, # noqa
):
    async for session in get_async_db_session():
        # Perform the upsert operation
        created_entries = await create_entries(
            session=session,
            data_list=results,
            collection_name=collection_name,
            version=version,
            update_on_collision=update_on_collision
        )

        # Print the results from EntryCreationResult
        print(f"Total Processed: {created_entries.total_processed}")
        print(f"New Entries: {created_entries.new_entries}")
        print(f"Skipped Duplicates: {created_entries.skipped_duplicates}")
        print(f"Updated Entries: {created_entries.updated_entries}")
        print(f"Failed Entries: {created_entries.failed_entries}")
        print(f"Error Messages: {created_entries.error_messages}")

        # Return the original results to maintain pipeline consistency
        # This allows the pipeline to track what was actually upserted
        return results
