import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.llm_utils.utils import Provider
from src.retrieval.retrieve_new import hybrid_retrieval
from src.schemas.schemas import EmbeddedFeatureType
from src.sql_db.database import get_async_session
from src.upsert.qdrant_utils import async_get_qdrant_client


async def evaluate_batch(
    vdb_client: AsyncQdrantClient, embed_client: AsyncOpenAI, session: AsyncSession, questions: list[str], feature_paths: list[int]
) -> tuple[list[dict[str, Any]], float]:
    """
    Compare the feature_path of the returned entries to those used to formulate questions
    Calculate precision and recall for top-k vector, full-text, and hybrid retrieval
    """
    responses, cost = await hybrid_retrieval(
        vdb_client,
        questions,
        embed_client,
        dense_model="text-embedding-3-large",
        provider=Provider.OPENAI,
        dimensions=512,
        sparse_model_name="bm25",
        qdrant_collection="research_etl",
        feature_types=[EmbeddedFeatureType.TEXT],
        filters=[],
        limit=10,
    )

    results = []
    if isinstance(responses, tuple):
        vector_results, full_text_results = responses
        for question, vector_result, full_text_result, expected_feature_path in zip(
            questions, vector_results, full_text_results, feature_paths
        ):
            print(vector_result[0].payload)
            vector_paths = [vr.payload["ingestion"]["embedded_feature_path"] for vr in vector_result]
            full_text_paths = [fr.payload["ingestion"]["embedded_feature_path"] for fr in full_text_result]
            vector_scores = [vr.score for vr in vector_result]
            full_text_scores = [fr.score for fr in full_text_result]

            result = {
                "question": question,
                "expected_feature_path": expected_feature_path,
                "vector_top_1": expected_feature_path in vector_paths[:1],
                "vector_top_3": expected_feature_path in vector_paths[:3],
                "vector_top_5": expected_feature_path in vector_paths[:5],
                "vector_top_10": expected_feature_path in vector_paths[:10],
                "full_text_top_1": expected_feature_path in full_text_paths[:1],
                "full_text_top_3": expected_feature_path in full_text_paths[:3],
                "full_text_top_5": expected_feature_path in full_text_paths[:5],
                "full_text_top_10": expected_feature_path in full_text_paths[:10],
                "hybrid_top_1": expected_feature_path in vector_paths[:1] or expected_feature_path in full_text_paths[:1],
                "hybrid_top_3": expected_feature_path in vector_paths[:3] or expected_feature_path in full_text_paths[:3],
                "hybrid_top_5": expected_feature_path in vector_paths[:5] or expected_feature_path in full_text_paths[:5],
                "hybrid_top_10": expected_feature_path in vector_paths[:10] or expected_feature_path in full_text_paths[:10],
                "vector_scores": vector_scores,
                "full_text_scores": full_text_scores,
            }

            # Calculate individual statistics
            result["vector_score_avg"] = np.mean(vector_scores)
            result["vector_score_std"] = np.std(vector_scores)
            result["full_text_score_avg"] = np.mean(full_text_scores)
            result["full_text_score_std"] = np.std(full_text_scores)

            # Calculate individual precision and recall
            for search_type in ["vector", "full_text", "hybrid"]:
                for k in [1, 3, 5, 10]:
                    result[f"{search_type}_precision_{k}"] = result[f"{search_type}_top_{k}"] / k
                    result[f"{search_type}_recall_{k}"] = result[f"{search_type}_top_{k}"]

            results.append(result)

    return results, cost


def compute_precision_recall(results_df: pd.DataFrame) -> dict[str, float]:
    metrics = {}
    for search_type in ["vector", "full_text", "hybrid"]:
        for k in [1, 3, 5, 10]:
            true_positives = results_df[f"{search_type}_top_{k}"].sum()
            precision = true_positives / (len(results_df) * k)
            recall = true_positives / len(results_df)
            metrics[f"{search_type}_precision_{k}"] = precision
            metrics[f"{search_type}_recall_{k}"] = recall
    return metrics


async def main(df: pd.DataFrame):
    qdrant_client = await async_get_qdrant_client(timeout=300)
    embed_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    questions = df["question"].tolist()[:10]
    feature_paths = df["feature_path"].tolist()[:10]

    batch_size = 10
    all_results = []
    total_cost = 0

    for session in get_async_session():
        tasks = []
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_feature_paths = feature_paths[i:i + batch_size]
            tasks.append(evaluate_batch(
                qdrant_client,
                embed_client,
                session,
                batch_questions,
                batch_feature_paths
            ))

        results = await asyncio.gather(*tasks)

        for batch_results, cost in tqdm(results, desc="Processing batches"):
            all_results.extend(batch_results)
            total_cost += cost

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Calculate average and standard deviation for scores
    vector_scores = np.array(results_df["vector_scores"].tolist())
    full_text_scores = np.array(results_df["full_text_scores"].tolist())

    score_stats = {
        "vector_score_avg": np.mean(vector_scores),
        "vector_score_std": np.std(vector_scores),
        "full_text_score_avg": np.mean(full_text_scores),
        "full_text_score_std": np.std(full_text_scores),
    }

    # Compute precision and recall
    precision_recall_metrics = compute_precision_recall(results_df)

    # Combine all metrics
    all_metrics = {**score_stats, **precision_recall_metrics, "total_cost": total_cost}

    # Create a single DataFrame with all results and metrics
    detailed_results = results_df.drop(["vector_scores", "full_text_scores"], axis=1)
    metrics_df = pd.DataFrame([all_metrics])

    # Combine detailed results and metrics
    combined_results = pd.concat([detailed_results, metrics_df], axis=0, ignore_index=True)

    # Save combined results to a single CSV file
    output_file_path = "evaluation_results.csv"
    print(f"Saving evaluation results to {output_file_path}")
    combined_results.to_csv(output_file_path, index=False)

    # Print summary statistics
    print("Evaluation Results:")
    for metric, value in all_metrics.items():
        if metric != "total_cost":
            print(f"{metric}: {value:.4f}")

    print(f"Total cost: \033[31m${total_cost:.2f}\033[0m")


if __name__ == "__main__":
    csv_file_path = "tmp/synthetic_qa_sample.csv"
    print(f"Running evaluation on {csv_file_path}")
    df = pd.read_csv(csv_file_path, index_col=False)

    asyncio.run(main(df))
