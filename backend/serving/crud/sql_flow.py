import json
from fastapi import HTTPException, APIRouter
from sqlalchemy import text
from typing import Dict, Any
from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.sql_db.database import get_async_session
from src.vector_db.qdrant_utils import remove_points_by_filter, get_qdrant_client

base_path = Path(__file__).parent.parent


async def remove_pipeline_data(collection_name: str, pipeline_id: int, session):
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


async def sql_get_collection_pipeline_stats(session):
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
        stats_path = base_path / "distribution_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(distributions, f)

        return distributions
    except Exception as e:
        print(f"Error generating distribution stats: {e}")
        return None
