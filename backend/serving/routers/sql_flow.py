import json
from fastapi import HTTPException, APIRouter
from sqlalchemy import text
from typing import Dict, Any
from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.sql_db.database import get_async_session
from crud.sql_flow import remove_pipeline_data, sql_get_collection_pipeline_stats, generate_distribution_stats

router = APIRouter()

base_path = Path(__file__).parent.parent

@router.get("/sql_stats/overview")
async def get_overview_stats() -> Dict[str, Any]:
    """
    Provides a high-level overview of the database contents.
    First tries to read from cache, falls back to live query if needed.
    """
    try:
        # Try to read from cache first
        cache_path = base_path / "overview_stats.json"
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to read from cache, falling back to live query: {e}")

    try:
        # Raw SQL queries instead of ORM
        queries = {
            'ingest_count': "SELECT COUNT(*)from ingest",
            'pipeline_count': "SELECT COUNT(*) FROM processing_pipelines",
            'step_count': "SELECT COUNT(*) FROM processing_steps",
            'entry_count': "SELECT COUNT(*) FROM entries"
        }
        overview = {}
        for session in get_async_session():
            for key, query in queries.items():
                result = session.execute(text(query))
                overview[key] = result.scalar()
        return overview
    except Exception as e:
        print(f"Error in get_overview_stats: {str(e)}", flush=True)
        logging.error(f"Error in get_overview_stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@router.delete("/sql_stats/pipeline/{collection_name}/{pipeline_id}")
async def delete_pipeline(collection_name: str, pipeline_id: int):
    """
    Delete a pipeline and all its associated data from both SQL and vector databases.

    Args:
        collection_name (str): Name of the Qdrant collection
        pipeline_id (int): ID of the pipeline to delete

    Returns:
        dict: Status message indicating success or failure
    """
    try:
        for session in get_async_session():
            await remove_pipeline_data(collection_name, pipeline_id, session)
            return {
                "status": "success",
                "message": f"Successfully removed pipeline {pipeline_id} and all related data from collection {collection_name}"
            }
    except Exception as e:
        print(f"Error deleting pipeline: {str(e)}", flush=True)
        logging.error(f"Error deleting pipeline: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete pipeline: {str(e)}"
        )


@router.get("/sql_stats/pipelines")
async def get_pipelines_stats() -> Dict[str, Any]:
    """Always retrieves live pipeline data (no caching)"""
    try:
        for session in get_async_session():
            # First get all pipelines
            pipelines_query = """
                SELECT id, version, description, created_at
            FROM processing_pipelines
                ORDER BY created_at DESC
            """
            pipelines_result = session.execute(text(pipelines_query))
            pipelines = pipelines_result.all()

            pipelines_data = []
            for pipeline in pipelines:
                # For each pipeline, get its steps
                steps_query = """
                    SELECT id, "order", step_type, status, date, output_path
                    FROM processing_steps
                    WHERE pipeline_id = :pipeline_id
                    ORDER BY "order" ASC
                """
                steps_result = session.execute(
                    text(steps_query),
                    {"pipeline_id": pipeline.id}
                )
                steps = steps_result.all()

                steps_data = [{
                    'step_id': step.id,
                    'order': step.order,
                    'step_type': step.step_type,
                    'status': step.status,
                    'date': step.date.isoformat(),
                    'output_path': step.output_path,
                } for step in steps]

                pipelines_data.append({
                    'pipeline_id': pipeline.id,
                    'version': pipeline.version,
                    'description': pipeline.description,
                    'created_at': pipeline.created_at.isoformat(),
                    'steps': steps_data,
                })

            return {'pipelines': pipelines_data}
    except Exception as e:
        logging.error(f"Error in get_pipelines_stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@router.get("/sql_stats/distributions")
async def get_distributions_stats() -> Dict[str, Any]:
    """Retrieves distribution stats, generating them if needed"""
    cache_path = base_path / "distribution_stats.json"
    try:
        # Try to read from cache first
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                stats = json.load(f)
                # Add timestamp check here if you want to validate freshness
                return stats
        # Generate new stats if cache missing
        for session in get_async_session():
            distributions = await generate_distribution_stats(session)
            return distributions
    except Exception as e:
        logging.error(f"Error in get_distributions_stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sql_stats/collection_pipelines")
async def get_collection_pipeline_stats() -> Dict[str, Any]:
    """
    Provides detailed information about pipeline distribution across collections.
    First tries to read from cache, falls back to live query if needed.
    """
    try:
        # Try to read from cache first
        cache_path = base_path / "pipeline_stats.json"
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to read from cache, falling back to live query: {e}")

    # Fall back to live query if cache fails or doesn't exist
    try:
        for session in get_async_session():
            return await sql_get_collection_pipeline_stats(session)
    except Exception as e:
        logging.error(f"Error in get_collection_pipeline_stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

