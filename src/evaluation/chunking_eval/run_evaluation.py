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
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import umap
from scipy.spatial import ConvexHull
import yaml
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.evaluation.chunking_eval.chunk_retrieval import compare_pipeline_chunks, get_single_pipeline_entries
from src.evaluation.chunking_eval.elo_system import run_elo_analysis
from src.schemas.schemas import ChunkEvaluation, Entry
from src.featurization.get_features import featurize
from src.evaluation.chunking_eval.evaluation_utils import can_use_vlm
from src.evaluation.chunking_eval.elo_system import ELOSystem
from src.pipeline.storage_backend import StorageFactory
from src.pipeline.registry.prompt_registry import PromptRegistry

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

    if config.get("pipeline_evaluation", {}).get("evaluation_mode") == "comparative":
        results = await run_comparative_evaluation(config)
    elif config.get("pipeline_evaluation", {}).get("evaluation_mode") == "individual":
        results = await run_individual_evaluation(config)
    else:
        raise ValueError(f"Invalid evaluation mode: {config.get('pipeline_evaluation', {}).get('evaluation_mode')}")

    if config.get("output", {}).get("save_results"):
        save_evaluation_results(results, config.get("output", {}))


def calculate_chunk_metrics(chunks: list[Entry]) -> dict[str, float]:
    """Calculate diversity metrics for chunks."""
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Get embeddings for all chunks
    texts = [chunk.string for chunk in chunks if chunk.string]
    embeddings = model.encode(texts)

    metrics = {}

    # For large datasets, reduce dimensionality with UMAP first
    if len(embeddings) > 100:  # Threshold for using UMAP
        try:
            # Configure UMAP for dimensionality reduction
            reducer = umap.UMAP(
                n_neighbors=min(int((len(embeddings) - 1) ** 0.5), 50),
                n_components=min(20, len(embeddings) - 1),
                min_dist=0,
                metric="cosine",
                random_state=42
            )
            embeddings = reducer.fit_transform(embeddings)
        except Exception as e:
            print(f"Warning: UMAP reduction failed with error: {str(e)}. Using original embeddings.")

    # Calculate convex hull volume as diversity metric
    if len(embeddings) > 3:  # Need at least 4 points for 3D hull
        try:
            hull = ConvexHull(embeddings)
            metrics["embedding_diversity"] = hull.volume
        except Exception:
            metrics["embedding_diversity"] = 0

    # Calculate average pairwise distance
    distances = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            distances.append(dist)
    metrics["avg_pairwise_distance"] = np.mean(distances) if distances else 0

    return metrics


async def run_individual_evaluation(config: dict) -> List[Dict]:
    """Run individual chunk quality evaluation."""
    pipeline_config = config.get("pipeline_evaluation", {})
    pipeline_id = pipeline_config.get("pipeline_id")
    # Get entries from pipeline
    entries = await get_single_pipeline_entries(str(pipeline_id), filter_params=config.get("pipeline_evaluation", {}).get("filter_params", {}))
    # Cast Entry to ChunkEvaluation type
    chunk_evaluations = [ChunkEvaluation.model_validate(entry.model_dump()) for entry in entries]
    # Featurize chunks
    print(f"\nEvaluating {len(chunk_evaluations)} chunks from pipeline {pipeline_id}")
    results = await featurize(chunk_evaluations, "LLM_chunk_rubric", config.get("model_name", "gpt-4o"), model_params=config.get("model_params", {}))

    # Print summary statistics
    if results:
        print("\nEvaluation Summary:")
        avg_clarity = sum(r.text_clarity for r in results) / len(results)
        avg_coherence = sum(r.coherence for r in results) / len(results)
        avg_organization = sum(r.organization for r in results) / len(results)
        total_score = avg_clarity + avg_coherence + avg_organization
        print(f"Average Text Clarity: {avg_clarity:.2f}/5.00")
        print(f"Average Coherence: {avg_coherence:.2f}/5.00")
        print(f"Average Organization: {avg_organization:.2f}/5.00")
        print(f"Total Score: {total_score:.2f}/15.00")
    return results


async def run_comparative_evaluation(config: dict):
    """Run comparative evaluation between pipelines."""
    pipeline_config = config.get("pipeline_evaluation", {})
    pipeline_configs = pipeline_config.get("pipeline_ids", [])
    evaluation_type = pipeline_config.get("evaluation_type", "LLM")
    elo_system = ELOSystem()

    if evaluation_type == "VLM":
        storage_config = config.get("storage", {})
        storage_client = StorageFactory.create(**storage_config)

    for i, pipeline_a in enumerate(pipeline_configs):
        for pipeline_b in pipeline_configs[i + 1 :]:
            print(f"\nComparing Pipeline {pipeline_a['id']} vs Pipeline {pipeline_b['id']}")

            # Get chunks from both pipelines
            comparisons = await compare_pipeline_chunks(str(pipeline_a["id"]), str(pipeline_b["id"]), filter_params=config.get("pipeline_evaluation", {}).get("filter_params", {}))
            # Flatten the dictionary of lists into a [ChunkComparison]
            comparisons_list = [comp for comps in comparisons.values() for comp in comps]

            actual_evaluation_type = "VLM" if evaluation_type == "VLM" and all([can_use_vlm(comp.chunks_a, comp.chunks_b) for comp in comparisons_list]) else "LLM"
            print(f"Actual evaluation type: {actual_evaluation_type}")
            if actual_evaluation_type == "VLM":
                all_results = await featurize(comparisons_list, "VLM_relative_evaluation", config.get("model_name", "gpt-4o"), model_params=config.get("model_params", {}), read=storage_client.read)
            else:
                all_results = await featurize(comparisons_list, "LLM_relative_evaluation", config.get("model_name", "gpt-4o"), model_params=config.get("model_params", {}))

            for result in all_results:
                pipeline_a = result.chunks_a[0].ingestion.pipeline_id
                pipeline_b = result.chunks_b[0].ingestion.pipeline_id
                elo_score = 1.0 if result.winner == "A" else 0.0
                elo_system.update_ratings(pipeline_a, pipeline_b, elo_score)
                run_elo_analysis([pipeline_a, pipeline_b])

    # Run final ELO analysis
    if config.get("elo", {}).get("enabled"):
        elo_analysis = run_elo_analysis([str(p["id"]) for p in pipeline_configs])
        print("\nFinal ELO Ratings:")
        for pipeline_id, rating in elo_analysis["current_ratings"].items():
            print(f"Pipeline {pipeline_id}: {rating}")

    return all_results


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
