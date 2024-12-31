import json
import logging
import os
import sys
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Optional

import aiofiles
import numpy as np
from fastembed.sparse.bm25 import Bm25
from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import UpdateStatus

sys.path.append(str(Path(__file__).parents[2]))

from src.llm_utils.utils import model_mapping
from src.schemas.schemas import Upsert


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
def create_collection(client: QdrantClient, collection_name: str):
    """
    Creates a new collection in Qdrant with the specified name and vector configuration.

    Args:
        client: The QdrantClient instance.
        collection_name (str): The name of the collection to create.
    """
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
    )


async def async_create_hybrid_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    embedding_model_name: str,
    embedding_model_dim: int,
    text_model_name: str = "bm25",
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
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                text_model_name: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
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
    return Bm25(f"Qdrant/{sparse_model_name}")


def process_response(file_path: str, dense_model_name: str, sparse_model_name: Optional[str]) -> Optional[dict]:
    """
    Generate a Qdrant compatible vector from an async embedding model response.

    Reads a JSON file, extracts relevant information, and creates a vector
    representation using dense and optionally sparse embeddings.
    Returns a dictionary with vector data or None if an error occurs.
    """
    try:
        # read json embedding response
        with open(file_path, encoding="utf-8") as file:
            data = json.load(file)
        prompt, response, metadata = data[0], data[1], data[2]

        if isinstance(response, list):
            logging.error(f"Error in {file_path}: {response}")
            return None

        # build the dense and sparse vector
        vector = {dense_model_name: response["data"][0]["embedding"]}

        if sparse_model_name:
            bm25_embedding_model = get_bm25_model(sparse_model_name)
            sparse_embedding = bm25_embedding_model.passage_embed(prompt["input"])
            vector[sparse_model_name] = list(sparse_embedding)[0].as_object()

        # build the payload
        payload = {key: value for key, value in metadata.items() if value is not None}

        return {
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": payload,
            "input": prompt["input"],
            "tokens": response["usage"]["total_tokens"],
        }
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return None


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


async def async_upsert_embed_response_files(
    client: AsyncQdrantClient,
    collection: str,
    dense_model_name: str,
    sparse_model_name: str,
    response_base_dir: str,
    vdb_record_base_dir: str,
    batch_size: int = 1000,
) -> list[str]:
    """
    Upserts data into a Qdrant collection from specified directories.
    Saves a record of the upserted points in a jsonl file
    Removes and re-applies index threshold to ensure faster indexing
    """

    async def upsert_batch(points, records):
        response = await client.upsert(collection_name=collection, points=points)
        if response.status == UpdateStatus.COMPLETED:
            async with aiofiles.open(save_path, "a", encoding="utf-8") as f:
                for record in records:
                    serializable_record = json.loads(
                        json.dumps(record, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                    )
                    await f.write(json.dumps(serializable_record) + "\n")
            return len(points)
        else:
            logging.warning(f"Upsert batch to {save_path} not completed. Status: {response.status}")
            return 0

    await async_change_index_threshold(client, collection, threshold=0)
    try:
        points_batch, records_batch = [], []
        total_usage, total_points = 0, 0

        # create a save path using the base dir and the basename of the source directory
        folder_name = os.path.basename(response_base_dir)
        save_path = os.path.join(vdb_record_base_dir, folder_name, "records.jsonl")

        # iterate over files in folder
        for json_file in filter(lambda f: f.endswith(".json"), os.listdir(response_base_dir)):
            file_path = os.path.join(response_base_dir, json_file)
            processed = process_response(file_path, dense_model_name, sparse_model_name)

            if processed:
                # make point
                points_batch.append(
                    models.PointStruct(id=processed["id"], vector=processed["vector"], payload=processed["payload"])
                )

                # save some records, don't save the vector
                records_batch.append(processed.pop("vector", None))
                total_usage += processed["tokens"]

                # upsert
                if len(points_batch) >= batch_size:
                    total_points += await upsert_batch(points_batch, records_batch)
                    points_batch.clear()
                    records_batch.clear()

        if points_batch:
            total_points += await upsert_batch(points_batch, records_batch)

        print(f"Upserted {total_points}")
        print(f"Total cost of embedding with {dense_model_name}: ${total_usage * model_mapping[dense_model_name].cost.input:.6f}")

    finally:
        await async_change_index_threshold(client, collection, 2000)


def process_embedding(embedding: dict, dense_model_name: str, sparse_model_name: Optional[str]) -> Optional[dict]:
    """
    Generate a Qdrant compatible vector from an embedding dictionary.

    Extracts relevant information and creates a vector representation using dense and optionally sparse embeddings.
    Returns a dictionary with vector data or None if an error occurs.
    """
    # build the dense and sparse vector
    vector = {dense_model_name: embedding["embedding"]}

    if sparse_model_name:
        bm25_embedding_model = get_bm25_model(sparse_model_name)
        sparse_embedding = bm25_embedding_model.passage_embed(embedding["string"])
        vector[sparse_model_name] = list(sparse_embedding)[0].as_object()

    # build the payload based on the Upsert schema
    vdb_payload = {
        key: value for key, value in embedding["ingestion"].items() if value is not None and key in Upsert.__fields__.keys()
    }
    vdb_payload.pop("schema__", None)  # Use None to avoid KeyError if "schema__" is not present
    upsert_payload = vdb_payload.copy()
    upsert_payload.update(
        {
            "string": embedding["string"],
            "context_summary_string": embedding.get("context_summary_string", None),
            "added_featurization": embedding.get("added_featurization", None),
            "keywords": embedding.get("keywords", None),
            "index_numbers": embedding.get("index_numbers", None),
            "uuid": str(uuid.uuid4()),
            "sparse_vector": {
                "values": vector[sparse_model_name]["values"].tolist(),
                "indices": vector[sparse_model_name]["indices"].tolist(),
            },
            "dense_vector": vector[dense_model_name].tolist()
            if isinstance(vector[dense_model_name], np.ndarray)
            else vector[dense_model_name],
            "schema__": "Upsert",
        }
    )
    upsert = Upsert(**upsert_payload)
    return {
        "upsert": upsert,
        "vdb_payload": vdb_payload,
        "vector": vector,
        "tokens": embedding["tokens"],
    }


async def async_upsert_embed(
    client: AsyncQdrantClient,
    embeddings: list[dict],
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

    async def upsert_batch(points, records):
        response = await client.upsert(collection_name=collection, points=points)
        if response.status == UpdateStatus.COMPLETED:
            all_upserts.extend(records)
            return len(points)
        else:
            raise Exception(f"Upsert batch not completed. Status: {response.status}")

    await async_change_index_threshold(client, collection, threshold=0)
    try:
        points_batch, records_batch = [], []
        total_usage, total_points = 0, 0

        # iterate over files in folder
        for embedding in embeddings:
            processed = process_embedding(embedding.model_dump(), dense_model_name, sparse_model_name)
            if processed:
                upsert = processed["upsert"]
                # make point
                points_batch.append(
                    models.PointStruct(id=upsert.uuid, vector=processed["vector"], payload=processed["vdb_payload"])
                )
                # save some records, don't save the vector
                records_batch.append(upsert)
                total_usage += processed["tokens"]
                # upsert
                if len(points_batch) >= batch_size:
                    total_points += await upsert_batch(points_batch, records_batch)
                    points_batch.clear()
                    records_batch.clear()
        if points_batch:
            total_points += await upsert_batch(points_batch, records_batch)

        print(f"Upserted {total_points}")
        print(f"Total cost of embedding with {dense_model_name}: ${total_usage * model_mapping[dense_model_name].cost.input:.6f}")

        return all_upserts
    except Exception as e:
        print(e)
        raise Exception("Upsert failed")
    finally:
        await async_change_index_threshold(client, collection, 2000)


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
