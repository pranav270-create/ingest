from typing import Dict, Any
from collections import Counter
from datetime import datetime, timedelta
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http.models import CollectionInfo, VectorParams, SparseVectorParams


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
