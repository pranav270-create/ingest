import sys
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from requests import Session
from datetime import datetime
import re

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import Ingestion

def update_ingestion_with_metadata(ingestion: Ingestion, added_metadata: Optional[dict]) -> Ingestion:
    """
    Update the Ingestion object with additional metadata.
    Updates fields that exist in the Ingestion object directly,
    and adds other fields to the metadata dictionary.
    """
    if added_metadata is None:
        return ingestion
    for key, value in added_metadata.items():
        if hasattr(ingestion, key) and not isinstance(getattr(ingestion, key), BaseModel):
            setattr(ingestion, key, value)
        else:
            ingestion.metadata[key] = value
    return ingestion
