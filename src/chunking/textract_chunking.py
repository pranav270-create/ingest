import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import Entry, Index, ChunkingMethod, ExtractedFeatureType, BoundingBox
from src.utils.datetime_utils import get_current_utc_datetime
from src.pipeline.registry import FunctionRegistry
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
    import asyncio
    import argparse
    import json
    from src.extraction.textract import textract_parse
    from src.schemas.schemas import Scope, ContentType
    
    parser = argparse.ArgumentParser(description='Chunk a PDF document using textract')
    parser.add_argument('--pdf_path', type=str, required=True, help='Path to the PDF file')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Size of text chunks')
    parser.add_argument('--output_path', type=str, help='Path to save chunked output (optional)')
    parser.add_argument('--scope', type=str, default='EXTERNAL', choices=['EXTERNAL', 'INTERNAL'], help='Document scope')
    parser.add_argument('--content_type', type=str, default='OTHER_ARTICLES', help='Content type')
    
    async def main(args):
        # Parse document using textract
        document = textract_parse(
            args.pdf_path, 
            Scope[args.scope], 
            ContentType[args.content_type]
        )
        
        # Chunk the document
        chunked_docs = await textract_chunks([document], chunk_size=args.chunk_size)
        total_chunks = sum(len(doc.entries) for doc in chunked_docs)
        print(f"Created {total_chunks} chunks")
        
        # Save chunks if output path is provided
        if args.output_path:
            # dump the document
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(document.model_dump(), f, indent=2, ensure_ascii=False)
            print(f"Saved document to {args.output_path}")
    
    args = parser.parse_args()
    asyncio.run(main(args))