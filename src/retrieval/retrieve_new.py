import json
import sys
from itertools import chain
from pathlib import Path
from typing import Optional, Union

from fastembed import SparseEmbedding
from httpx import ReadTimeout
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import FieldCondition, Filter, MatchValue, QueryRequest, ScoredPoint, SearchRequest, SparseVector
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.llm_utils.utils import Provider
from src.retrieval.embed_text import async_embed_text
from src.schemas.schemas import EmbeddedFeatureType, FormattedScoredPoints, Ingestion
from src.sql_db.database_simple import get_async_session
from src.sql_db.etl_model import Entry
from src.vector_db.qdrant_utils import get_bm25_model


def assign_citation_numbers(formatted_context: list[list[FormattedScoredPoints]]) -> dict[str, int]:
    citation_dict = {}
    current_number = 1

    for section_context in formatted_context:
        for item in section_context:
            source_id = item.id
            if source_id not in citation_dict:
                citation_dict[source_id] = current_number
                current_number += 1

    return citation_dict


async def format_scored_points(
    session: AsyncSession,
    search_results: tuple[list[list[ScoredPoint]], list[list[ScoredPoint]]],
    threshold: float = 0.0,
    citations=False,
    flatten: bool = False,
) -> Union[list[FormattedScoredPoints], list[list[FormattedScoredPoints]]]:  # Updated return type
    """
    Parse search results into their data models and fetch raw text from the SQL database in a single batch
    """
    dummy = citations  # noqa
    vector_results_list, full_text_results_list = search_results

    all_unique_ids = set()
    points_per_query = []
    all_points_dict = {}

    for vector_results, full_text_results in zip(vector_results_list, full_text_results_list):
        unique_points = set()
        points = []
        for result in chain(vector_results, full_text_results):
            idx = result.id
            if idx not in unique_points:
                score = result.score
                if score < threshold:
                    continue
                try:
                    ingestion = Ingestion(**result.payload)
                except Exception:
                    continue
                unique_points.add(idx)
                all_unique_ids.add(idx)
                date = ingestion.ingestion_date
                title = ingestion.document_title or ingestion.content_type.value
                point = FormattedScoredPoints(id=idx, ingestion=ingestion, score=score, date=date, title=title)
                points.append(point)
                all_points_dict[idx] = point
        points_per_query.append(points)

    # Fetch all raw texts in a single query
    stmt = select(Entry.unique_identifier, Entry.string, Entry.index_numbers).where(Entry.unique_identifier.in_(all_unique_ids))
    result = session.execute(stmt)
    id_to_raw_text = {
        row.unique_identifier: {
            "text": row.string,
            "index_numbers": json.loads(row.index_numbers) if row.index_numbers else [],
        }
        for row in result.fetchall()
    }
    # Second pass: add raw text and index numbers to points
    filtered_points_per_query = []
    for points in points_per_query:
        filtered_points = []
        for point in points:
            raw_data = id_to_raw_text.get(point.id)
            if raw_data:
                point.raw_text = raw_data["text"]
                point.index = raw_data["index_numbers"]  # Update the index from SQL
                filtered_points.append(point)
                all_points_dict[point.id].raw_text = raw_data["text"]
                all_points_dict[point.id].index = raw_data["index_numbers"]  # Update in the dict too
        if filtered_points:
            filtered_points_per_query.append(filtered_points)

    return list(all_points_dict.values()) if flatten else filtered_points_per_query


async def hybrid_retrieval(
    client: AsyncQdrantClient,
    query: Union[str, list[str]],
    embed_client,
    dense_model,
    provider: Provider,
    dimensions,
    sparse_model_name,
    qdrant_collection,
    feature_types,
    filters: Optional[list[list[FieldCondition]]] = None,
    limit: int = 10,
) -> tuple[tuple[list[list[ScoredPoint]], list[list[ScoredPoint]]], float]:
    embeddings, cost = await async_embed_text(embed_client, dense_model, provider, query, dimensions=dimensions)
    bm25_embedding_model = get_bm25_model(sparse_model_name)
    sparse_embeddings = list(bm25_embedding_model.passage_embed(query))

    # Search the vector database
    search_results = await search_vector_db(
        client=client,
        collection=qdrant_collection,
        dense_model=dense_model,
        sparse_model=sparse_model_name,
        embeddings=embeddings,
        sparse_embeddings=sparse_embeddings,
        feature_types=feature_types,
        filters=filters,
        limit=limit,
    )
    return search_results, cost


async def get_scored_points(
    client: AsyncQdrantClient, collection: str, qdrant_requests: Union[list[SearchRequest], list[QueryRequest]]
) -> list[list[ScoredPoint]]:
    """
    supports query_batch_points and search_batch Qdrant API
    """
    request_type = type(qdrant_requests[0])
    try:
        if request_type == SearchRequest:
            results = await client.search_batch(collection_name=collection, requests=qdrant_requests)
        elif request_type == QueryRequest:
            results = await client.query_batch_points(collection_name=collection, requests=qdrant_requests)
        if not results:
            raise ValueError("Qdrant search returned empty results")
        return results
    except (ResponseHandlingException, ReadTimeout) as e:
        error_message = f"Error during Qdrant search: {str(e)}"
        raise ValueError(error_message) from e
    except ValueError as e:
        raise ValueError(str(e)) from e


async def search_vector_db(
    client: AsyncQdrantClient,
    collection: str,
    dense_model: str,
    sparse_model: str,
    embeddings: list[list[float]],
    sparse_embeddings: Optional[list[SparseEmbedding]],
    feature_types: list[EmbeddedFeatureType],
    filters: Optional[list[list[FieldCondition]]] = None,
    limit: int = 10,
) -> Union[list[list[ScoredPoint]], tuple[list[list[ScoredPoint]], list[list[ScoredPoint]]]]:
    """
    conduct a vector db search for each type of feature
    [# queries x dim embed] -> [# queries x # features * # results]
    """
    requests = []
    # [# queries x dim embed] -> [# queries * # features]
    for i, feature_type in enumerate(feature_types):  # want top 'limit' hits for each feature type
        if filters:
            must = [FieldCondition(key="embedded_feature_type", match=MatchValue(value=feature_type.value))] + filters[i]
        else:
            must = [FieldCondition(key="embedded_feature_type", match=MatchValue(value=feature_type.value))]
        if not sparse_embeddings:
            for embed in embeddings:
                requests.append(
                    SearchRequest(vector=embed, using=dense_model, filter=Filter(must=must), with_payload=True, limit=limit)
                )
        else:
            for dense_embed, sparse_embed in zip(embeddings, sparse_embeddings):
                requests.extend(
                    [
                        QueryRequest(
                            query=dense_embed, using=dense_model, filter=Filter(must=must), with_payload=True, limit=limit
                        ),
                        QueryRequest(
                            query=SparseVector(**sparse_embed.as_object()),
                            using=sparse_model,
                            filter=Filter(must=must),
                            with_payload=True,
                            limit=limit,
                        ),
                    ]
                )

    # [# queries * # features] -> [# queries * # features x # results]
    results = await get_scored_points(client=client, collection=collection, qdrant_requests=requests)
    num_queries = len(embeddings)

    def reshape_results(results_list):
        grouped = [results_list[i::num_queries] for i in range(num_queries)]
        return [[points for response in query_set for points in response.points if response.points] for query_set in grouped]

    # [# queries * # features x # results] -> [# queries x # features * # results]
    if not sparse_embeddings:
        grouped = [results[i::num_queries] for i in range(num_queries)]
        return [list(chain.from_iterable(group)) for group in grouped]

    # [# queries * # features x # results] -> 2 x [# queries x # features * # results]
    else:
        vector_results = results[::2]
        full_text_results = results[1::2]
        return reshape_results(vector_results), reshape_results(full_text_results)


async def test(questions: list[str]):
    import os

    from openai import AsyncOpenAI

    from src.vector_db.qdrant_utils import async_get_qdrant_client

    qdrant_client = await async_get_qdrant_client(timeout=1000)
    embed_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    dense_model = "text-embedding-3-large"
    provider = Provider.OPENAI
    embedding_dimensions = 512
    sparse_model_name = "bm25"
    qdrant_collection = "bridgewater_full"
    feature_types = [EmbeddedFeatureType.TEXT]
    filters = None
    limit = 3
    search_results, _ = await hybrid_retrieval(
        qdrant_client,
        questions,
        embed_client,
        dense_model,
        provider,
        embedding_dimensions,
        sparse_model_name,
        qdrant_collection,
        feature_types,
        filters,
        limit,
    )
    for session in get_async_session():
        formatted_context = await format_scored_points(session, search_results, threshold=0.4)
    return formatted_context


if __name__ == "__main__":
    import asyncio

    questions = ["what is the best protein power to be taking?"]
    formatted_context = asyncio.run(test(questions))
    print(formatted_context)
