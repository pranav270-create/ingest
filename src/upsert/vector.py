import sys
from pathlib import Path
from typing import Optional, Any
sys.path.append(str(Path(__file__).parents[2]))

from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import Embedding
from src.upsert.qdrant_utils import async_create_hybrid_collection, async_get_qdrant_client, async_upsert_embed
from src.utils.filter_utils import filter_basemodels


@FunctionRegistry.register("upsert", "upsert_embeddings")
async def upsert_embeddings(
    embeddings: list[Embedding],
    collection_name: str,
    dense_model_name: str,
    sparse_model_name: str,
    dimensions: int,
    batch_size: int = 1000,
    filter_params: Optional[dict[str, Any]] = None,
    read: callable = None, # noqa
    write: callable = None, # noqa
    **kwargs, # noqa
):
    # Filter embeddings if filter params provided
    embeddings_to_process = embeddings
    unfiltered_embeddings = []
    if filter_params:
        embeddings_to_process, unfiltered_embeddings = filter_basemodels(embeddings, filter_params)
        if not embeddings_to_process:
            return unfiltered_embeddings

    # Separate entries with and without embeddings
    embeddings_with_vectors = []
    embeddings_without_vectors = []
    for embedding in embeddings_to_process:
        if embedding.embedding is not None:
            embeddings_with_vectors.append(embedding)
        else:
            embeddings_without_vectors.append(embedding)

    if not embeddings_with_vectors:
        return embeddings_without_vectors + unfiltered_embeddings

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
        embeddings=embeddings_with_vectors,
        collection=collection_name,
        dense_model_name=dense_model_name,
        sparse_model_name=sparse_model_name,
        batch_size=batch_size,
    )
    return all_upserts + embeddings_without_vectors + unfiltered_embeddings
