import json
import os
import sys
from pathlib import Path
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parents[2]))

from src.vector_db.qdrant_utils import async_create_hybrid_collection, async_get_qdrant_client, async_upsert_embed
from src.pipeline.registry import FunctionRegistry


@FunctionRegistry.register("upsert", "upsert_embeddings")
async def upsert_embeddings(
    embeddings,
    collection_name: str,
    dense_model_name: str,
    sparse_model_name: str,
    dimensions: int,
    batch_size: int = 1000,
    read: callable = None,
    write: callable = None,
    **kwargs,
):
    client = await async_get_qdrant_client(timeout=1000)
    await async_create_hybrid_collection(
        client=client,
        collection_name=collection_name,
        embedding_model_name=dense_model_name,
        embedding_model_dim=dimensions,
        text_model_name=sparse_model_name,
    )
    all_upserts = await async_upsert_embed(
        client=client,
        embeddings=embeddings,
        collection=collection_name,
        dense_model_name=dense_model_name,
        sparse_model_name=sparse_model_name,
        batch_size=batch_size,
    )
    return all_upserts
