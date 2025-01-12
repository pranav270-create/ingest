import cohere
from rerankers import Reranker

from src.schemas.schemas import FormattedScoredPoints


def rerank_local(ranker: Reranker, query: str, points: list[FormattedScoredPoints], threshold: int = 0) -> list[FormattedScoredPoints]:
    results = ranker.rank(query=query, docs=[point.raw_text for point in points], doc_ids=[point.id for point in points])

    point_map = {point.id: point for point in points}
    for result in results:
        if result.score < threshold:
            continue
        point = point_map.get(result.document.doc_id)
        if point:
            point.rerank_score = result.score

    return points


async def rerank_remote(
        cohere_client: cohere.Client, rerank_model: str, query: str, context: list[FormattedScoredPoints]
        ) -> list[FormattedScoredPoints]:
    try:
        documents = [doc.raw_text for doc in context]
        results = await cohere_client.rerank(model=rerank_model, query=query, documents=documents, top_n=10)
        results = [{'index': result.index, 'relevance_score': result.relevance_score} for result in results.results]
        # Sort the results by index and extract scores
        sorted_results = sorted(results, key=lambda x: x['index'])
        relevance_scores = [result['relevance_score'] for result in sorted_results]
        for i, c in enumerate(context):
            if i < len(relevance_scores):
                c.rerank_score = relevance_scores[i]
            else:
                c.rerank_score = 0
        return context
    except Exception as e:
        print(f"Reranking failed: {e}")
        return context