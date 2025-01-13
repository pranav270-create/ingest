import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import Entry, ChunkingMethod, BoundingBox
from src.pipeline.registry.function_registry import FunctionRegistry
from src.chunking.chunk_utils import entries_to_content, chunks_to_entries


@FunctionRegistry.register("chunk", ChunkingMethod.TEXTRACT.value)
async def textract_chunks(entries: list[Entry], **kwargs) -> list[Entry]:
    """
    Chunks documents while preserving textract-specific features (tables, forms, etc.)
    """
    chunk_size = kwargs.get("chunk_size", 1000)
    chunking_metadata = {
        "chunk_size": chunk_size,
        "method": "textract_preserve_features"
    }
    
    content = entries_to_content(entries)
    chunks = textract_chunking(content, chunk_size=chunk_size)
    formatted_entries = chunks_to_entries(entries, chunks, ChunkingMethod.TEXTRACT, chunking_metadata)
    entries = formatted_entries
    return entries


# TODO: This should be used and only not used when the chunks cross pages
def combine_bounding_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    """Combine multiple bounding boxes into one encompassing box."""
    if not boxes:
        return None
    
    return BoundingBox(
        left=min(box.left for box in boxes),
        top=min(box.top for box in boxes),
        width=max(box.left + box.width for box in boxes) - min(box.left for box in boxes),
        height=max(box.top + box.height for box in boxes) - min(box.top for box in boxes)
    )


def textract_chunking(content: list[dict[str, Any]], chunk_size: int = 1000) -> list[dict[str, Any]]:
    """
    Chunks content while preserving textract-specific features and parent-child relationships.
    """
    chunks = []
    current_chunk = []
    current_length = 0
    current_pages = set()
    current_feature_types = set()
    current_bounding_boxes = []
    
    for item in content:
        text = item["text"]
        pages = item["pages"]
        feature_types = item.get("feature_types", [])
        bounding_boxes = item.get("bounding_boxes", [])
        
        # Always keep container elements as single chunks
        if any(ft in ['table', 'figure', 'form'] for ft in feature_types):
            if current_chunk:
                chunks.append({
                    "text": " ".join(current_chunk),
                    "pages": sorted(list(current_pages)),
                    "feature_types": list(current_feature_types),
                    "bounding_boxes": current_bounding_boxes
                })
                current_chunk = []
                current_length = 0
                current_pages = set()
                current_feature_types = set()
                current_bounding_boxes = []
            
            chunks.append({
                "text": text,
                "pages": pages,
                "feature_types": feature_types,
                "bounding_boxes": bounding_boxes
            })
            continue
        
        if current_length + len(text) > chunk_size and current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "pages": sorted(list(current_pages)),
                "feature_types": list(current_feature_types),
                "bounding_boxes": current_bounding_boxes
            })
            current_chunk = []
            current_length = 0
            current_pages = set()
            current_feature_types = set()
            current_bounding_boxes = []
        
        current_chunk.append(text)
        current_length += len(text)
        current_pages.update(pages)
        current_feature_types.update(feature_types)
        current_bounding_boxes.extend(bounding_boxes)
    
    if current_chunk:
        chunks.append({
            "text": " ".join(current_chunk),
            "pages": sorted(list(current_pages)),
            "feature_types": list(current_feature_types),
            "bounding_boxes": current_bounding_boxes
        })
    
    return chunks


if __name__ == "__main__":
    pass