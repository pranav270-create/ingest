import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import Ingestion


def update_ingestion_with_metadata(ingestion: Ingestion, added_metadata: Optional[dict]) -> Ingestion:
    """
    Update an Ingestion object with additional metadata, handling both direct field updates and metadata dictionary entries.

    Args:
        ingestion: The Ingestion object to update
        added_metadata: Dictionary containing new metadata key-value pairs to add

    Returns:
        The updated Ingestion object

    For each key-value pair in added_metadata:
    - If the key matches an existing Ingestion field (that isn't itself a BaseModel),
      updates that field directly
    - Otherwise, adds the key-value pair to ingestion.metadata dictionary
    - If added_metadata is None, returns the original ingestion unchanged
    """
    if added_metadata is None:
        return ingestion

    for key, value in added_metadata.items():
        if hasattr(ingestion, key):
            setattr(ingestion, key, value)
        else:
            ingestion.metadata.setdefault(key, value)
    return ingestion
