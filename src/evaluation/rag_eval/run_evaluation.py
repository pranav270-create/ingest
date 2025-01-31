"""
Main entry point for running comparative evaluations between different chunking pipelines.
This module orchestrates the evaluation process by:
- Loading and comparing chunks from different pipelines
- Running both LLM and VLM evaluations
- Aggregating results and updating ELO ratings
- Generating evaluation reports
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
import os
from typing import Dict, List, Any
import yaml
from pydantic import BaseModel
from openai import AsyncOpenAI

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.pipeline.storage_backend import StorageFactory
from src.pipeline.registry.prompt_registry import PromptRegistry
from src.retrieval.retrieve_new import hybrid_retrieval
from src.upsert.qdrant_utils import async_get_qdrant_client
from src.llm_utils.utils import Provider
from src.schemas.schemas import EmbeddedFeatureType
from qdrant_client.models import FieldCondition, MatchValue

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PromptRegistry.autodiscover("src.prompts.evaluation_prompts")


async def run_evaluation_pipeline(config_path: str):
    """Run evaluation pipeline based on config file."""
    # Resolve config path relative to project root
    config_path = PROJECT_ROOT / "src" / "config" / f"{config_path}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    results = await run_rag_evaluation(config)

    if config.get("output", {}).get("save_results"):
        save_evaluation_results(results, config.get("output", {}))


def calculate_rag_score(results: list[Any], document_ids: list[Any]) -> float:
    """Calculate the RAG score for a list of vector database search results and the actual document ids (correct answers)"""
    # TODO: Implement NDCG score, Precision and Recall @3, 5, 10?
    pass


async def run_rag_evaluation(config: dict) -> List[Dict]:
    """Run individual chunk quality evaluation."""
    pipeline_config = config.get("pipeline_evaluation", {})
    pipeline_id = pipeline_config.get("pipeline_id")

    # Get questions from dataset and the actual document ids (correct answers)
    dataset = config.get("dataset", {})
    questions = dataset.get("questions", [])
    document_ids = dataset.get("document_ids", [])

    qdrant_client = await async_get_qdrant_client(timeout=1000)
    embed_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    dense_model = "text-embedding-3-large"
    provider = Provider.OPENAI
    embedding_dimensions = 512
    sparse_model_name = "bm25"
    qdrant_collection = "test_collection_pranav"
    feature_types = [EmbeddedFeatureType.TEXT]
    filters = [[FieldCondition(key="pipeline_id", match=MatchValue(value=pipeline_id))]]
    limit = 10
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
    all_results = []
    for question in questions:
        # Query the vector database for the top K results
        top_k = pipeline_config.get("top_k", 10)
        results = await hybrid_retrieval(qdrant_client,
                                         question,
                                         embed_client,
                                         dense_model,
                                         provider,
                                         embedding_dimensions,
                                         sparse_model_name,
                                         qdrant_collection,
                                         feature_types,
                                         filters,
                                         limit,
                                         question,
                                         top_k,
                                        )
        all_results.append(results)

    # Calculate RAG score
    rag_score = calculate_rag_score(all_results, document_ids)

    # Print summary statistics
    if rag_score:
        print("\nEvaluation Summary:")
    return rag_score


def save_evaluation_results(results: list[BaseModel], output_config: dict):
    """Save evaluation results to file."""
    output_dir = Path(output_config.get("output_dir", "evaluation_results"))
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"evaluation_results_{timestamp}.json"

    with open(output_file, "w") as f:
        for result in results:
            json.dump(result.model_dump(mode="json"), f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to evaluation config file")
    args = parser.parse_args()
    asyncio.run(run_evaluation_pipeline(args.config))
