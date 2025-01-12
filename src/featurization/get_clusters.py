import asyncio
import os
from collections import defaultdict
from typing import Any

import numpy as np
import umap
from openai import AsyncOpenAI
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from src.llm_utils.agent import Agent
from src.llm_utils.utils import Provider
from src.pipeline.registry import FunctionRegistry
from src.prompts.offline.etl_featurization import ExtractClaimsPrompt, LabelClustersPrompt
from src.retrieval.embed_text import async_embed_text
from src.schemas.schemas import Entry
from src.utils.datetime_utils import get_current_utc_datetime


@FunctionRegistry.register("featurize", "get_clusters")
async def get_clusters(entries: list[Entry], **kwargs) -> list[Entry]:
    """Extract claims from entries, cluster them, and add keywords back to entries."""

    # Configuration
    model = kwargs.get("model", "gpt-4o-mini")
    embedding_model = kwargs.get("embedding_model", "text-embedding-3-small")
    dimensions = kwargs.get("dimensions", 1536)
    criterion = kwargs.get("criterion", "bic")
    provider = Provider.OPENAI
    embedding_provider = Provider.OPENAI

    assert "gpt" in model, "method only supports openai models"
    assert "text-embedding" in embedding_model, "method only supports openai embedding models"
    assert criterion in ["bic", "silhouette"], "criterion must be either 'bic' or 'silhouette'"

    # Initialize a map to track which claims belong to which labels for each entry
    entry_label_info_map = defaultdict(lambda: defaultdict(lambda: {"claims": [], "description": ""}))

    # Process all entries
    all_claims = []
    entry_to_claims_map = defaultdict(list)  # Track which claims belong to which entry

    for entry_id, entry in enumerate(entries):
        claims = await extract_claims(entry.string, model=model, provider=provider)
        entry.added_featurization = {"claims": claims}
        all_claims.extend(claims)
        for claim in claims:
            entry_to_claims_map[claim].append(entry_id)

    # Get embeddings for all claims
    embeddings = await get_claim_embeddings(all_claims, model=embedding_model, provider=embedding_provider, dimensions=dimensions)

    # Cluster the claims
    reduced_embeddings = run_umap(embeddings)
    cluster_assignments, _ = cluster_data(reduced_embeddings, criterion=criterion)

    # Organize claims into clusters
    clusters = organize_clusters(all_claims, cluster_assignments)

    # Get labels for clusters
    cluster_labels = await label_clusters(provider=provider, model=model, prompt=LabelClustersPrompt, clusters=list(clusters.values()))

    # Track all unique cluster labels
    all_cluster_labels = set()

    # Build the label-claims mapping for each entry
    for claim, entry_ids in entry_to_claims_map.items():
        for cluster_id, cluster_items in clusters.items():
            if claim in cluster_items:
                label_info = cluster_labels[int(cluster_id)]
                subtopic = label_info["subtopic"]
                description = label_info["description"]
                all_cluster_labels.add(subtopic)

                # Add claim and description to the appropriate label for each entry
                for entry_id in entry_ids:
                    entry_label_info_map[entry_id][subtopic]["claims"].append(claim)
                    entry_label_info_map[entry_id][subtopic]["description"] = description

    # Update entries with the new structure
    for entry_id, entry in enumerate(entries):
        if entry_id in entry_label_info_map:
            # Store the label info mapping in added_featurization
            entry.added_featurization = {
                "labeled_claims": {
                    subtopic: {"claims": info["claims"], "description": info["description"]}
                    for subtopic, info in entry_label_info_map[entry_id].items()
                }
            }
            # Store just the labels in keywords
            entry.keywords = list(entry_label_info_map[entry_id].keys())

        # Add all unique cluster labels to the shared Ingestion object
        if entry.ingestion and not entry.ingestion.keywords:
            entry.ingestion.keywords = list(all_cluster_labels)
            entry.ingestion.feature_dates = (
                [get_current_utc_datetime()]
                if entry.ingestion.feature_dates is None
                else entry.ingestion.feature_dates.append(get_current_utc_datetime())
            )  # noqa

    return entries


async def label_clusters(provider: Provider, model: str, prompt, clusters: list[list[str]], **kwargs) -> list[dict[str, str]]:
    """
    label the clusters
    """

    async def label_cluster(claims: list[str]) -> dict[str, str]:
        agent = Agent(prompt)
        labels, _ = await agent.call(provider=provider, model=model, claims=claims, **kwargs)
        return labels

    tasks = [label_cluster(c) for c in clusters]
    return await asyncio.gather(*tasks)


def cluster_data(embeddings: list[list[float]], n_clusters=20, criterion="bic") -> tuple[list[list[int]], int]:
    covariance_type = "full"
    threshold = 0.1
    min_clusters = 5

    scores = []
    cluster_range = range(min_clusters, n_clusters + 1)

    for n in tqdm(cluster_range):
        gmm = GaussianMixture(n_components=n, random_state=42, covariance_type=covariance_type)
        gmm.fit(embeddings)

        if criterion == "bic":
            scores.append(gmm.bic(embeddings))
        else:  # silhouette
            cluster_labels = gmm.predict(embeddings)
            scores.append(silhouette_score(embeddings, cluster_labels))

    # For BIC, we want to minimize; for silhouette, we want to maximize
    optimal_idx = np.argmin(scores) if criterion == "bic" else np.argmax(scores)
    optimal_n_clusters = cluster_range[optimal_idx]
    print(f"Optimal number of clusters based on {criterion}: {optimal_n_clusters}")

    clustering_model = GaussianMixture(n_components=optimal_n_clusters, covariance_type=covariance_type, random_state=42)
    clustering_model.fit(embeddings)
    probabilities = clustering_model.predict_proba(embeddings)
    cluster_assignment = [np.where(p > threshold)[0] for p in probabilities]

    return cluster_assignment, optimal_n_clusters


def organize_clusters(data: list[dict[str, Any]], cluster_assignment: list[list[int]]) -> dict[str, list[dict[str, Any]]]:
    clusters = defaultdict(list)
    for idx, cluster_ids in enumerate(cluster_assignment):
        for cluster_id in cluster_ids:
            clusters[cluster_id].append(data[idx])

    sorted_clusters = dict(sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True))
    return {str(k): v for k, v in sorted_clusters.items()}


async def extract_claims(text: str, model: str, provider: Provider) -> list[str]:
    """Extract claims from text using LLM."""
    agent = Agent(ExtractClaimsPrompt)
    claims, _ = await agent.call(provider=provider, model=model, text=text)
    return claims


async def get_claim_embeddings(claims: list[str], model: str, provider: Provider, dimensions: int) -> list[list[float]]:
    """Get embeddings for claims."""
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embeddings, _ = await async_embed_text(client, model, provider, claims, dimensions=dimensions)
    return embeddings


def run_umap(embeddings: list[list[float]], n_neighbors: int = 50, n_components: int = 20) -> np.ndarray:
    """Reduce dimensionality of embeddings."""
    vector_matrix = np.array(embeddings)
    # If we have very few samples, return the original embeddings
    if len(vector_matrix) <= n_components:
        print(f"Warning: Number of samples ({len(vector_matrix)}) is less than or equal to target dimensions ({n_components}). Skipping UMAP reduction.") # noqa
        return vector_matrix
    try:
        n_neighbors = min(int((len(vector_matrix) - 1) ** 0.5), n_neighbors)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=min(n_components, len(vector_matrix) - 1),  # Ensure n_components is valid
            min_dist=0,
            metric="cosine",
            random_state=42
        )
        return reducer.fit_transform(vector_matrix)
    except Exception as e:
        print(f"Warning: UMAP reduction failed with error: {str(e)}. Falling back to original embeddings.")
        return vector_matrix