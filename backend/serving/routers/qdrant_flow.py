import json
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from pathlib import Path
import sys
import logging
from datetime import datetime, timedelta
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http.models import CollectionInfo, VectorParams, SparseVectorParams
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.vector_db.qdrant_utils import get_qdrant_client
from serving.crud.qdrant_flow import fetch_collection_metadata

router = APIRouter()

base_path = Path(__file__).parent.parent


async def gather_qdrant_stats(output_file: str):
    """
    Gathers statistics from Qdrant and writes them to a JSON file.
    """
    client = get_qdrant_client()
    stats = {
        "timestamp": datetime.now().isoformat(),
        "collections": await fetch_collection_metadata(client)
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    print(f"Qdrant statistics saved to {output_file}")


@router.get("/qdrant_update")
async def get_qdrant_stats(force: bool = False):
    """Retrieves Qdrant statistics, generating them if needed or if force=true"""
    stats_file = base_path / "qdrant_stats.json"
    try:
        # Try to read from cache first, unless force refresh is requested
        if not force and stats_file.exists():
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                    # Check if stats are less than 1 hour old
                    timestamp = datetime.fromisoformat(stats["timestamp"])
                    if datetime.now() - timestamp < timedelta(hours=1):
                        return stats
            except json.JSONDecodeError:
                # If the file is corrupted, we'll regenerate the stats
                pass
        # Generate new stats if needed
        await gather_qdrant_stats(str(stats_file))
        # Read and return the newly generated stats
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
            return stats
    except Exception as e:
        error_msg = f"Error in get_qdrant_stats: {str(e)}"
        logging.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@router.delete("/qdrant_stats/collection/{collection_name}")
async def delete_collection(collection_name: str):
    """Deletes a collection from Qdrant"""
    client = get_qdrant_client()
    client.delete_collection(collection_name=collection_name)
    return {"status": "success", "message": f"Collection {collection_name} deleted"}


########################################################


# NOTE: This needs an index to work, which we don't have in the current setup
async def facet_fetch_collection_metadata(client: QdrantClient) -> Dict[str, Any]:
    """
    Fetches metadata for each collection in Qdrant using faceting for efficient counting.
    """
    collections = client.get_collections().collections
    stats = {}

    for collection in collections:
        collection_name = collection.name
        stats[collection_name] = {}

        try:
            # Get total points in collection
            collection_info: CollectionInfo = client.get_collection(collection_name)
            total_points = collection_info.points_count

            # Use faceting to get counts efficiently
            content_types_facet = client.facet(
                collection_name=collection_name,
                key="content_type",
            ).counters

            scopes_facet = client.facet(
                collection_name=collection_name,
                key="scope",
            ).counters

            creation_dates_facet = client.facet(
                collection_name=collection_name,
                key="creation_date",
            ).counters

            embedding_dates_facet = client.facet(
                collection_name=collection_name,
                key="embedding_date",
            ).counters

            pipeline_ids_facet = client.facet(
                collection_name=collection_name,
                key="pipeline_id",
            ).counters

            # Convert facet results to the same format as before
            content_types = {str(f.value): f.count for f in content_types_facet}
            scopes = {str(f.value or "Unknown"): f.count for f in scopes_facet}
            creation_dates = {str(f.value or "Unknown"): f.count for f in creation_dates_facet}
            embedding_dates = {str(f.value or "Unknown"): f.count for f in embedding_dates_facet}
            pipeline_ids = {str(f.value or "Unknown"): f.count for f in pipeline_ids_facet if f.value != "Unknown"}

            # Store all statistics
            stats[collection_name].update({
                'content_types': content_types,
                'scopes': scopes,
                'creation_dates': creation_dates,
                'embedding_dates': embedding_dates,
                'pipeline_distribution': pipeline_ids,
                'vector_count': total_points,
                'pipeline_coverage': {
                    'with_pipeline': sum(pipeline_ids.values()),
                    'total': total_points,
                    'percentage': round(sum(pipeline_ids.values()) / total_points * 100, 2) if total_points > 0 else 0
                }
            })

            print(f"Completed processing collection {collection_name}")

        except Exception as e:
            stats[collection_name]['error'] = f"Error fetching metadata: {str(e)}"
            print(f"Error processing collection {collection_name}: {str(e)}")

    return stats
