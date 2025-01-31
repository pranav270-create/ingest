import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import UpdateStatus

sys.path.append(str(Path(__file__).parents[2]))

from src.llm_utils.utils import model_mapping
from src.schemas.schemas import Embedding, Upsert


# Clients
def get_qdrant_client(timeout: int = 10) -> QdrantClient:
    """
    Creates and returns a QdrantClient instance with the specified URL and API key from environment variables.

    Returns:
        QdrantClient: An instance of the QdrantClient connected to the specified Qdrant service.
    """
    client = QdrantClient(
        url=f'{os.getenv("QDRANT_API_URL")}:6333', api_key=os.getenv("QDRANT_API_KEY"), timeout=timeout
    )
    return client


async def async_get_qdrant_client(timeout: int = 10) -> QdrantClient:
    """
    Creates and returns a QdrantClient instance with the specified URL and API key from environment variables.

    Returns:
        QdrantClient: An instance of the QdrantClient connected to the specified Qdrant service.
    """
    client = AsyncQdrantClient(
        url=f'{os.getenv("QDRANT_API_URL")}:6333', api_key=os.getenv("QDRANT_API_KEY"), timeout=timeout
    )
    return client


# collection is a set of points (vector + payload) that you can search over
# they recommed just using 1 collection
def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 1024,
    distance: models.Distance = models.Distance.COSINE,
    datatype: models.Datatype = models.Datatype.FLOAT16,
):
    """
    Creates a new collection in Qdrant with the specified name and vector configuration.

    Args:
        client: The QdrantClient instance.
        collection_name (str): The name of the collection to create.
    """
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=distance,
            datatype=datatype
        ),
        sparse_vectors_config={
            "text": models.SparseVectorParams(
                index=models.SparseIndexParams(datatype=datatype)
            ),
        },
    )


async def async_create_hybrid_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    embedding_model_name: str,
    embedding_model_dim: int,
    text_model_name: str = "bm25",
    distance: models.Distance = models.Distance.COSINE,
    modifier: models.Modifier = models.Modifier.IDF,
    datatype: models.Datatype = models.Datatype.FLOAT16,
):
    """
    creates a new collection that supports hybrid search
    """
    try:
        result = await async_check_collection(client, collection_name)
        return result
    except Exception:
        response = await client.create_collection(
            collection_name=collection_name,
            vectors_config={
                embedding_model_name: models.VectorParams(
                    size=embedding_model_dim,
                    distance=distance,
                    datatype=datatype,
                ),
            },
            sparse_vectors_config={
                text_model_name: models.SparseVectorParams(
                    modifier=modifier,
                    index=models.SparseIndexParams(
                        datatype=datatype
                    )
                )
            },
        )
        if response:
            print("Collection created successfully")


async def async_check_collection(client: AsyncQdrantClient, collection_name: str):
    """
    Checks if a collection exists in Qdrant and returns its configuration if it does.
    """
    return await client.get_collection(collection_name=collection_name)


def check_collection(client: QdrantClient, collection_name: str):
    """
    Checks if a collection exists in Qdrant and returns its configuration if it does.
    """
    return client.get_collection(collection_name=collection_name)


def delete_collection(client: QdrantClient, collection_name: str):
    """
    Deletes a collection from Qdrant if it exists.

    Args:
        client: The QdrantClient instance.
        collection_name (str): The name of the collection to delete.
    """
    client.delete_collection(collection_name=collection_name)


# Index thresholds
async def async_change_index_threshold(client: AsyncQdrantClient, collection_name: str, threshold: float) -> None:
    """
    Updates the indexing threshold of a specified collection in Qdrant. Useful when bulk upserting vectors

    Args:
        client: The client used to interact with Qdrant.
        collection: The name of the collection to update.
        threshold: The new indexing threshold value.
    """
    await client.update_collection(
        collection_name=collection_name, optimizer_config=models.OptimizersConfigDiff(indexing_threshold=threshold)
    )


def change_index_threshold(client: QdrantClient, collection_name: str, threshold: float) -> None:
    """
    Updates the indexing threshold of a specified collection in Qdrant. Useful when bulk upserting vectors

    Args:
        client: The client used to interact with Qdrant.
        collection: The name of the collection to update.
        threshold: The new indexing threshold value.
    """
    client.update_collection(
        collection_name=collection_name, optimizer_config=models.OptimizersConfigDiff(indexing_threshold=threshold)
    )


# upserting vectors
@lru_cache(maxsize=1)
def get_bm25_model(sparse_model_name: str):
    """
    Returns a Bm25 model initialized with the specified sparse model name.
    """
    return SparseTextEmbedding(model_name=f"Qdrant/{sparse_model_name}")


def batch_update_points(
    client: QdrantClient, collection: str, points: list[models.PointStruct], records: list[dict], save_path: str
) -> bool:
    """
    Perform a batch update of points in a Qdrant collection.

    Attempts to upsert a batch of points into the specified collection.
    If successful, saves the records to a file. Returns True on success,
    False otherwise.
    """
    try:
        response = client.batch_update_points(
            collection_name=collection, update_operations=[models.UpsertOperation(upsert=models.PointsBatch(points=points))]
        )
        if response.status == UpdateStatus.COMPLETED:
            with open(save_path, "a", encoding="utf-8") as output_file:
                for record in records:
                    json.dump(record, output_file)
                    output_file.write("\n")
            return True
        else:
            logging.warning(f"Upsert batch not completed. Status: {response.status}")
            return False
    except UnexpectedResponse as e:
        logging.error(f"Unexpected response during upsert: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Error upserting batch: {str(e)}")
        return False


def create_payload(embedding: Embedding) -> dict:
    """
    Returns a dictionary dump of the Upsert schema, which is a flattened Entry and Ingestion
    """
    payload = {}

    # Add Entry fields first (excluding vector-related fields)
    entry_fields = embedding.model_dump(exclude={
        'schema__',
        'embedding',
        'tokens',
        'ingestion',  # Handle ingestion separately
        # Fields from ENTRY we dont want to store in the payload
        # 'string',
        # 'added_featurization',
        # 'citations',
    })
    payload.update({k: v for k, v in entry_fields.items() if v is not None})

    # Add Ingestion fields if present
    if embedding.ingestion:
        ingestion_fields = embedding.ingestion.model_dump(exclude={
            'schema__',
            # Fields from INGESTION we dont want to store in the payload
            # 'document_summary'
            # 'unprocessed_citations',
            # 'citations'
        })
        payload.update({k: v for k, v in ingestion_fields.items() if v is not None})

    return payload


async def async_upsert_embed(
    client: AsyncQdrantClient,
    embeddings: list[Embedding],
    collection: str,
    dense_model_name: str,
    sparse_model_name: str,
    batch_size: int = 1000,
) -> list[str]:
    """
    Upserts data into a Qdrant collection from specified directories.
    Saves a record of the upserted points in a jsonl file
    Removes and re-applies index threshold to ensure faster indexing
    """

    all_upserts = []

    async def upsert_batch(points: list, records: list[Upsert]):
        """
        efficient Qdrant upsert
        """
        response = await client.upsert(collection_name=collection, points=points)

        # save a record of successful upserts
        if response.status == UpdateStatus.COMPLETED:
            all_upserts.extend(records)
            return len(points)
        else:
            raise Exception(f"Upsert batch not completed. Status: {response.status}")

    await async_change_index_threshold(client, collection, threshold=0) # makes upsert faster

    try:
        points_batch, records_batch = [], []
        total_usage, total_points = 0, 0

        # iterate over files in folder
        for embedding in embeddings:
            if embedding.embedding:
                # make vector
                vector = {dense_model_name: embedding.embedding}
                sparse_vector_upsert = []
                if sparse_model_name:
                    sparse_embedding_model = get_bm25_model(sparse_model_name)
                    sparse_embedding = list(sparse_embedding_model.passage_embed(embedding.string))[0].as_object()
                    vector[sparse_model_name] = sparse_embedding
                    sparse_vector_upsert = sparse_embedding['values'].tolist()

                # make payload
                payload = create_payload(embedding)

                # make Upsert record
                upsert_record = Upsert(
                    dense_vector=embedding.embedding,
                    sparse_vector={sparse_model_name: sparse_vector_upsert} if sparse_model_name in vector else {},
                    **payload  # Include all payload fields in the Upsert record
                )

                # make point
                point = models.PointStruct(id=embedding.uuid, vector=vector, payload=payload)
                points_batch.append(point)

                # save some records, don't save the vector
                records_batch.append(upsert_record)

                # upsert batch of points
                if len(points_batch) >= batch_size:
                    total_points += await upsert_batch(points_batch, records_batch)
                    points_batch.clear()
                    records_batch.clear()

                total_usage += embedding.tokens

        # upsert remaining points
        if points_batch:
            total_points += await upsert_batch(points_batch, records_batch)

        print(f"Upserted {total_points}")
        print(f"Total cost of embedding with {dense_model_name}: ${total_usage * model_mapping[dense_model_name].cost.input:.6f}")

        return all_upserts

    except Exception as e:
        print(e)
        raise Exception("Upsert failed") from None

    finally:
        await async_change_index_threshold(client, collection, 2000) # reset to default


# remove points
def remove_points_by_id(client: QdrantClient, collection: str, path_to_ids: str):
    """
    Remove points from a Qdrant collection based on IDs listed in a file.
    give the function a jsonl and it'll grab the ids and remove them from the vector db
    """
    ids_to_remove = []
    with open(path_to_ids, encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            ids_to_remove.append(data["id"])

    response = client.delete(
        collection_name=collection,
        points_selector=models.PointIdsList(
            points=ids_to_remove,
        ),
    )
    print(response)


def remove_points_by_filter(client, collection, key, value):
    """
    Remove points from a Qdrant collection by satisfying some metadata filter.
    """
    response = client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    ),
                ],
            )
        ),
    )
    print(f"Qdrant Delete Response: {response}")


async def retrieve_points_by_filter(client: AsyncQdrantClient, collection, key, value):
    response = await client.scroll(
        collection_name=collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key=key, match=models.MatchValue(value=value)),
            ]
        ),
        limit=200,
        with_payload=True,
        with_vectors=False,
    )
    return response


def validate_upsert(client, collection, key, value):
    response = client.scroll(
        collection_name=collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key=key, match=models.MatchValue(value=value)),
            ]
        ),
        limit=200,
        with_payload=True,
        with_vectors=False,
    )
    if response[0]:
        print(f"{value}: {response}")
    else:
        print(f"{value}: Not in VDB")


def retrieve_points(client, collection, ids: list[int]):
    return client.retrieve(
        collection_name=collection,
        ids=ids,
        with_vectors=True,
    )


def get_single_point(client, collection, idx):
    response = client.get(collection_name=collection, point_id=idx)
    return response


async def async_update_payload_by_filter(
    client: AsyncQdrantClient,
    collection: str,
    filter_key: str,
    filter_value: str,
    payload_key: str,
    payload_value: any,
) -> None:
    """
    Update a specific payload field for all points matching a filter condition.

    Args:
        client: AsyncQdrantClient instance
        collection: Name of the collection
        filter_key: Key to filter points by
        filter_value: Value to match in the filter
        payload_key: Key of the payload field to update
        payload_value: New value for the payload field
    """
    try:
        response = await client.set_payload(
            collection_name=collection,
            payload={
                payload_key: payload_value
            },
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key=filter_key,
                        match=models.MatchValue(value=filter_value),
                    ),
                ],
            ),
        )
        print(f"Qdrant Payload Update Response: {response}")
    except Exception as e:
        print(f"Error updating payload in Qdrant: {str(e)}")
        raise


if __name__ == "__main__":
    client = get_qdrant_client()

    # Delete specified collections
    collections_to_delete = ["abilene_full"]

    for collection in collections_to_delete:
        response = check_collection(client, collection)
        if response:
            print(f"Deleting collection: {collection}")
            delete_collection(client, collection)  # You may need to implement this function if it doesn't exist
        else:
            print(f"Collection {collection} does not exist.")
