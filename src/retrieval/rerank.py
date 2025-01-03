from rerankers import Reranker

from src.schemas.schemas import FormattedScoredPoints


def rerank(ranker: Reranker, query: str, points: list[FormattedScoredPoints], threshold: int = 0) -> list[FormattedScoredPoints]:
    results = ranker.rank(query=query, docs=[point.raw_text for point in points], doc_ids=[point.id for point in points])

    point_map = {point.id: point for point in points}
    for result in results:
        if result.score < threshold:
            continue
        point = point_map.get(result.document.doc_id)
        if point:
            point.rerank_score = result.score

    return points
