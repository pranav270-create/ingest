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
):
    async for session in get_async_db_session():

        await create_entries(
            session=session,
            data_list=results,
            collection_name=collection_name,
            version=version,
            update_on_collision=update_on_collision
        )

