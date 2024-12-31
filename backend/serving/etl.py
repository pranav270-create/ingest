import json
from fastapi import FastAPI, Depends, HTTPException, APIRouter
from fastapi_utils.tasks import repeat_every
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select, text
from sqlalchemy.orm import selectinload
from collections import Counter
from typing import Dict, Any
from sqlalchemy.orm import Session
import sys
import logging
import os
from datetime import datetime, timedelta
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http.models import CollectionInfo, VectorParams, SparseVectorParams
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.sql_db.database import get_async_session
from src.vector_db.qdrant_utils import remove_points_by_filter


router = APIRouter()


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


def get_qdrant_client(timeout: int = 10) -> QdrantClient:
    """
    Creates and returns a QdrantClient instance with the specified URL and API key from environment variables.
    """
    client = QdrantClient(
        url=f'{os.getenv("QDRANT_API_URL")}:6333',
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=timeout
    )
    return client


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


async def fetch_collection_metadata(client: QdrantClient) -> Dict[str, Any]:
    """
    Fetches metadata for each collection in Qdrant with proper pagination, including pipeline information.
    """
    collections = client.get_collections().collections
    stats = {}
    
    for collection in collections:
        collection_name = collection.name
        stats[collection_name] = {}
        
        try:
            # Initialize counters
            content_types = Counter()
            scopes = Counter()
            creation_dates = Counter()
            embedding_dates = Counter()
            pipeline_ids = Counter()  # New counter for pipeline IDs
            
            # Get total points in collection
            collection_info: CollectionInfo = client.get_collection(collection_name)
            total_points = collection_info.points_count
            
            # Initialize pagination
            batch_size = 10000
            total_processed = 0
            next_page_offset = None
            
            while total_processed < total_points:
                # Fetch batch with pagination
                points, next_page_offset = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=None,
                    with_payload=True,
                    with_vectors=False,
                    limit=batch_size,
                    offset=next_page_offset
                )
                
                # Break if no more results
                if not points:
                    break
                    
                # Process batch
                for point in points:
                    ingestion_data = point.payload
                    if ingestion_data:
                        # Handle standard fields with defaults
                        content_types[ingestion_data.get("content_type", "other")] += 1
                        scopes[ingestion_data.get("scope", "Unknown")] += 1
                        creation_dates[ingestion_data.get("creation_date", "Unknown")] += 1
                        embedding_dates[ingestion_data.get("embedding_date", "Unknown")] += 1
                        
                        # Handle pipeline information
                        pipeline_id = ingestion_data.get("pipeline_id", "Unknown")
                        if pipeline_id != "Unknown":
                            pipeline_ids[str(pipeline_id)] += 1  # Convert to string for JSON serialization
                
                total_processed += len(points)
                
                # Log progress for large collections
                if total_processed % 50000 == 0:
                    print(f"Processed {total_processed}/{total_points} records for collection {collection_name}")
                
                # Break if no next page
                if next_page_offset is None:
                    break
            
            # Store all statistics
            stats[collection_name].update({
                'content_types': dict(content_types),
                'scopes': dict(scopes),
                'creation_dates': dict(creation_dates),
                'embedding_dates': dict(embedding_dates),
                'pipeline_distribution': dict(pipeline_ids),  # New pipeline statistics
                'vector_count': total_points,
                'pipeline_coverage': {
                    'with_pipeline': sum(pipeline_ids.values()),
                    'total': total_points,
                    'percentage': round(sum(pipeline_ids.values()) / total_points * 100, 2) if total_points > 0 else 0
                }
            })
            
            print(f"Completed processing {total_processed}/{total_points} total records for collection {collection_name}")
            
        except Exception as e:
            stats[collection_name]['error'] = f"Error fetching metadata: {str(e)}"
            print(f"Error processing collection {collection_name}: {str(e)}")
    
    return stats


async def gather_qdrant_stats(output_file: str):
    """
    Gathers statistics from Qdrant and writes them to a JSON file.
    """
    client = get_qdrant_client()
    stats = {
        "timestamp": datetime.now().isoformat(),
        "collections": await fetch_collection_metadata(client)
    }

    # Ensure directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    print(f"Qdrant statistics saved to {output_file}")


@router.get("/qdrant_update")
async def get_qdrant_stats(force: bool = False):
    """Retrieves Qdrant statistics, generating them if needed or if force=true"""
    stats_file = Path(__file__).parent / "qdrant_stats.json"
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


@router.get("/sql_stats/overview")
async def get_overview_stats() -> Dict[str, Any]:
    """
    Provides a high-level overview of the database contents.
    First tries to read from cache, falls back to live query if needed.
    """
    try:
        # Try to read from cache first
        cache_path = Path(__file__).parent / "overview_stats.json"
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


async def remove_pipeline_data(collection_name: str, pipeline_id: int, session: AsyncSession):
    """
    Remove all data associated with a pipeline ID from both SQL and vector databases.
    """
    print(f"Starting removal process for pipeline {pipeline_id} in collection {collection_name}")
    # Initialize Qdrant client
    qdrant_client = get_qdrant_client()

    # 1. First delete all entries associated with this pipeline
    try:
        print("Removing entries from SQL database...")
        delete_entries_query = text("""
            DELETE FROM entries
            WHERE pipeline_id = :pipeline_id
        """)
        session.execute(delete_entries_query, {"pipeline_id": pipeline_id})
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error removing entries from SQL database: {str(e)}")
        raise

    # 2. Remove from Qdrant
    try:
        print("Removing entries from Qdrant...")
        remove_points_by_filter(
            client=qdrant_client,
            collection=collection_name,
            key="pipeline_id",
            value=pipeline_id
        )
    except Exception as e:
        print(f"Error removing entries from Qdrant: {str(e)}")
        raise

    # 3. Delete processing steps associated with this pipeline
    try:
        print("Removing processing steps...")
        delete_steps_query = text("""
            DELETE FROM processing_steps
            WHERE pipeline_id = :pipeline_id
        """)
        session.execute(delete_steps_query, {"pipeline_id": pipeline_id})
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error removing processing steps: {str(e)}")
        raise

    # 4. Delete the association in ingest_pipeline table
    try:
        print("Removing pipeline associations...")
        delete_assoc_query = text("""
            DELETE FROM ingest_pipeline
            WHERE pipeline_id = :pipeline_id
        """)
        session.execute(delete_assoc_query, {"pipeline_id": pipeline_id})
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error removing pipeline associations: {str(e)}")
        raise

    # 5. Finally delete the pipeline itself
    try:
        print("Removing pipeline from SQL database...")
        delete_pipeline_query = text("""
            DELETE FROM processing_pipelines
            WHERE id = :pipeline_id
        """)
        session.execute(delete_pipeline_query, {"pipeline_id": pipeline_id})
        session.commit()
        print(f"Successfully removed pipeline {pipeline_id} and all related data")
    except Exception as e:
        session.rollback()
        print(f"Error removing pipeline from SQL database: {str(e)}")
        raise


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


async def generate_distribution_stats(session):
    """Generate distribution statistics and save to JSON file."""
    try:
        distributions = {}

        # Entries per Pipeline distribution
        entries_per_pipeline_query = """
            SELECT pipeline_id, COUNT(id) as entry_count
            FROM entries
            GROUP BY pipeline_id
        """
        result = session.execute(text(entries_per_pipeline_query))
        entries_per_pipeline = result.all()

        # Create histogram buckets for entries per pipeline
        pipeline_entry_counts = [count for _, count in entries_per_pipeline]
        pipeline_histogram = {
            '0-100': len([c for c in pipeline_entry_counts if 0 <= c <= 100]),
            '101-500': len([c for c in pipeline_entry_counts if 101 <= c <= 500]),
            '501-1000': len([c for c in pipeline_entry_counts if 501 <= c <= 1000]),
            '1001+': len([c for c in pipeline_entry_counts if c > 1000])
        }
        distributions['entries_per_pipeline_histogram'] = pipeline_histogram

        # Entries per Ingest distribution
        entries_per_ingest_query = """
            SELECT i.id, COUNT(DISTINCT e.id) as entry_count
           from ingest i
            JOIN entries e ON i.id = e.ingestion_id
            GROUP BY i.id
        """
        result = session.execute(text(entries_per_ingest_query))
        entries_per_ingest = result.all()

        ingest_entry_counts = [count for _, count in entries_per_ingest]
        ingest_histogram = {
            '0-10': len([c for c in ingest_entry_counts if 0 <= c <= 10]),
            '11-50': len([c for c in ingest_entry_counts if 11 <= c <= 50]),
            '51-100': len([c for c in ingest_entry_counts if 51 <= c <= 100]),
            '101+': len([c for c in ingest_entry_counts if c > 100])
        }
        distributions['entries_per_ingest_histogram'] = ingest_histogram

        # Pipelines per Ingest distribution
        pipelines_per_ingest_query = """
            SELECT i.id, COUNT(DISTINCT pp.id) as pipeline_count
           from ingest i
            JOIN ingest_pipeline ipa ON i.id = ipa.ingest_id
            JOIN processing_pipelines pp ON ipa.pipeline_id = pp.id
            GROUP BY i.id
        """
        result = session.execute(text(pipelines_per_ingest_query))
        pipelines_per_ingest = result.all()

        ingest_pipeline_counts = [count for _, count in pipelines_per_ingest]
        pipeline_per_ingest_histogram = {
            '1': len([c for c in ingest_pipeline_counts if c == 1]),
            '2-3': len([c for c in ingest_pipeline_counts if 2 <= c <= 3]),
            '4-5': len([c for c in ingest_pipeline_counts if 4 <= c <= 5]),
            '6+': len([c for c in ingest_pipeline_counts if c > 5])
        }
        distributions['pipelines_per_ingest_histogram'] = pipeline_per_ingest_histogram

        # Add query for pipeline versions
        pipeline_versions_query = """
            SELECT version, COUNT(*) as count
            FROM processing_pipelines
            GROUP BY version
            ORDER BY version
        """
        result = session.execute(text(pipeline_versions_query))
        distributions['pipeline_versions'] = {str(row.version): row.count for row in result}

        # Add query for entry collections
        collections_query = """
            SELECT collection_name, COUNT(*) as count
            FROM entries
            GROUP BY collection_name
            ORDER BY count DESC
        """
        result = session.execute(text(collections_query))
        distributions['entry_collections'] = {str(row.collection_name): row.count for row in result}

        # Add query for ingest creation dates
        ingest_dates_query = """
            SELECT DATE(creation_date)::text as date, COUNT(*) as count
            FROM ingest
            GROUP BY DATE(creation_date)
            ORDER BY DATE(creation_date)
        """
        result = session.execute(text(ingest_dates_query))
        distributions['ingest_creation_dates'] = {str(row.date): row.count for row in result}

        # Add query for content types by collection - corrected to use ingest table
        content_types_query = """
            SELECT
                e.collection_name,
                i.content_type,
                COUNT(*) as count
            FROM entries e
            JOIN ingest i ON e.ingestion_id = i.id
            GROUP BY e.collection_name, i.content_type
            ORDER BY e.collection_name, count DESC
        """
        result = session.execute(text(content_types_query))
        content_types_by_collection = {}
        for row in result:
            if row.collection_name not in content_types_by_collection:
                content_types_by_collection[row.collection_name] = {}
            content_types_by_collection[row.collection_name][str(row.content_type)] = row.count

        distributions['content_types_by_collection'] = content_types_by_collection

        # Add query for processing step types
        step_types_query = """
            SELECT step_type, COUNT(*) as count
            FROM processing_steps
            GROUP BY step_type
            ORDER BY count DESC
        """
        result = session.execute(text(step_types_query))
        distributions['processing_step_types'] = {str(row.step_type): row.count for row in result}

        # Save to file
        stats_path = Path(__file__).parent / "distribution_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(distributions, f)

        return distributions
    except Exception as e:
        print(f"Error generating distribution stats: {e}")
        return None


@router.get("/sql_stats/distributions")
async def get_distributions_stats() -> Dict[str, Any]:
    """Retrieves distribution stats, generating them if needed"""
    cache_path = Path(__file__).parent / "distribution_stats.json"
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
        cache_path = Path(__file__).parent / "pipeline_stats.json"
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to read from cache, falling back to live query: {e}")

    # Fall back to live query if cache fails or doesn't exist
    try:
        query = """
            SELECT 
                e.collection_name,
                e.pipeline_id,
                pp.version,
                pp.description,
                COUNT(e.id) as entry_count
            FROM entries e
            LEFT JOIN processing_pipelines pp ON e.pipeline_id = pp.id
            GROUP BY 
                e.collection_name,
                e.pipeline_id,
                pp.version,
                pp.description
            ORDER BY 
                e.collection_name,
                e.pipeline_id
        """

        for session in get_async_session():
            results = session.execute(text(query))
            
            # Organize results by collection
            collection_stats = {}
            for row in results:
                collection_name = row.collection_name
                if collection_name not in collection_stats:
                    collection_stats[collection_name] = []
                
                pipeline_info = {
                    'pipeline_id': row.pipeline_id,
                    'version': row.version,
                    'description': row.description,
                    'entry_count': row.entry_count
                }
                collection_stats[collection_name].append(pipeline_info)

            return {
                'collection_pipeline_stats': collection_stats,
                'total_collections': len(collection_stats)
            }
    except Exception as e:
        logging.error(f"Error in get_collection_pipeline_stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@repeat_every(seconds=60 * 60)  # Run every hour
async def periodic_task():
    """Run ETL tasks every hour"""
    try:
        # Get a new session for the periodic task
        for session in get_async_session():
            # Update Qdrant stats
            stats_file = Path(__file__).parent / "qdrant_stats.json"
            await gather_qdrant_stats(str(stats_file))
            
            # Update distribution stats
            await generate_distribution_stats(session)
            
            # Update overview stats
            overview_stats = await get_overview_stats(session)
            overview_path = Path(__file__).parent / "overview_stats.json"
            with open(overview_path, 'w') as f:
                json.dump(overview_stats, f, indent=4)
                
            print("Scheduled ETL tasks completed successfully")
    except Exception as e:
        logging.error(f"Error in scheduled ETL tasks: {e}", exc_info=True)


@router.post("/trigger-update")
async def trigger_update():
    """Manually trigger the ETL tasks"""
    await periodic_task()
    return {"status": "success", "message": "ETL tasks completed"}
