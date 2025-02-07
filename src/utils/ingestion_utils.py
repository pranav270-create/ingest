import sys
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parents[2]))


from src.schemas.schemas import ChunkingMethod, ContentType, ExtractionMethod, FileType, Ingestion, IngestionMethod, Scope


def update_ingestion_with_metadata(ingestion: Ingestion, added_metadata: Optional[dict]) -> Ingestion:
    """
    Update an Ingestion object with additional metadata, handling both direct field updates and metadata dictionary entries.

    For each key-value pair in added_metadata:
    - If the key matches an existing Ingestion field (that isn't itself a BaseModel),
      updates that field directly
    - Otherwise, adds the key-value pair to ingestion.document_metadata dictionary
    - If added_metadata is None, returns the original ingestion unchanged
    """
    if added_metadata is None:
        return ingestion

    # Map of field names to their enum classes
    enum_fields = {
        "content_type": ContentType,
        "file_type": FileType,
        "ingestion_method": IngestionMethod,
        "extraction_method": ExtractionMethod,
        "scope": Scope,
        "chunking_method": ChunkingMethod,
    }

    for key, value in added_metadata.items():
        if hasattr(ingestion, key):
            # If the field is an enum field, convert string to enum member
            if key in enum_fields and isinstance(value, str):
                enum_class = enum_fields[key]
                try:
                    # Find the enum member whose value matches our input string
                    matching_member = next(member for member in enum_class if member.value == value.lower())
                    value = matching_member
                except StopIteration:
                    valid_values = [member.value for member in enum_class]
                    raise ValueError(f"Invalid value '{value}' for {key}. Must be one of {valid_values}")
            setattr(ingestion, key, value)
        else:
            ingestion.document_metadata.setdefault(key, value)
    return ingestion
