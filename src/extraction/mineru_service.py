import sys
from pathlib import Path
import modal
import io
import json
import uuid
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import (
    Entry,
    ChunkLocation,
    Index,
    Ingestion,
    ExtractedFeatureType,
    ExtractionMethod,
)
from src.utils.datetime_utils import get_current_utc_datetime


def _map_mineru_type(category_id: int) -> ExtractedFeatureType:
    """Maps MinerU CategoryType to our common ExtractedFeatureType enum"""
    # CategoryType mapping to our ExtractedFeatureType
    mineru_to_common = {
        0: ExtractedFeatureType.section_header,  # title
        1: ExtractedFeatureType.text,           # plain_text
        2: ExtractedFeatureType.other,          # abandon (headers, footers, etc)
        3: ExtractedFeatureType.figure,         # figure
        4: ExtractedFeatureType.caption,        # figure_caption
        5: ExtractedFeatureType.table,          # table
        6: ExtractedFeatureType.caption,        # table_caption
        7: ExtractedFeatureType.footnote,       # table_footnote
        8: ExtractedFeatureType.equation,       # isolate_formula
        9: ExtractedFeatureType.caption,        # formula_caption
        13: ExtractedFeatureType.textinlinemath, # embedding (inline formula)
        14: ExtractedFeatureType.equation,      # isolated (block formula)
        15: ExtractedFeatureType.text,          # text (OCR result)
    }
    output = mineru_to_common.get(category_id, ExtractedFeatureType.other)
    if output == ExtractedFeatureType.other:
        print(f"Unknown category_id: {category_id}")
    return output


def _convert_poly_to_bbox(poly: list[float], page_width: float, page_height: float) -> dict:
    """Convert MinerU polygon format to our bbox format"""
    # Poly format is [x1,y1,x2,y2,x3,y3,x4,y4] representing corners
    x_coords = [poly[i] for i in range(0, len(poly), 2)]
    y_coords = [poly[i] for i in range(1, len(poly), 2)]
    
    return {
        "left": min(x_coords),
        "top": min(y_coords),
        "width": max(x_coords) - min(x_coords),
        "height": max(y_coords) - min(y_coords),
        "page_width": page_width,
        "page_height": page_height,
    }


async def _process_inference_result(
    result: ObjectInferenceResult,
    page_num: int,
    secondary_idx: int,
    ingestion: Ingestion,
    page_file_path: str,
    chunk_idx: int,
    page_width: float,
    page_height: float,
) -> tuple[list[Entry], int, int]:
    """Process a single inference result from MinerU output"""
    entries = []
    feature_type = _map_mineru_type(result.category_id)
    
    # Get content based on type
    content = result.html if result.html else result.latex if result.latex else ""
    
    if content:
        location = ChunkLocation(
            index=Index(
                primary=page_num + 1,
                secondary=secondary_idx + 1,
            ),
            extracted_feature_type=feature_type,
            page_file_path=page_file_path,
            bounding_box=_convert_poly_to_bbox(
                result.poly,
                page_width,
                page_height
            ),
        )
        
        entry = Entry(
            uuid=str(uuid.uuid4()),
            ingestion=ingestion,
            string=content,
            consolidated_feature_type=feature_type,
            chunk_locations=[location],
            min_primary_index=page_num + 1,
            max_primary_index=page_num + 1,
            chunk_index=chunk_idx + 1,
        )
        entries.append(entry)
        chunk_idx += 1
        secondary_idx += 1
    
    return entries, chunk_idx, secondary_idx


@FunctionRegistry.register("parse", "mineru")
async def main_mineru(
    ingestions: list[Ingestion],
    write=None,
    read=None,
    mode="by_page",
    visualize=False,
    **kwargs,
) -> list[Entry]:
    """Parse documents using the MinerU library."""
    all_entries = []
    cls = modal.Cls.lookup("mineru-modal", "MinerU")
    obj = cls()
    
    for ingestion in ingestions:
        ingestion.extraction_method = ExtractionMethod.MINERU
        ingestion.extraction_date = get_current_utc_datetime()
        
        if not ingestion.extracted_document_file_path:
            base_path = os.path.splitext(ingestion.file_path)[0]
            ingestion.extracted_document_file_path = f"{base_path}_mineru.json"
        
        # Read and process file
        file_content = await read(ingestion.file_path) if read else open(ingestion.file_path, "rb").read()
        result = await obj.process_pdf(file_content)
        
        # Create output directories
        base_dir = os.path.dirname(ingestion.extracted_document_file_path)
        pages_dir = os.path.join(base_dir, "pages")
        
        # Initialize counters
        chunk_idx = 0
        
        # Process each page
        for page_result in result["inference_result"]:
            secondary_idx = 0
            page_num = page_result.page_info.page_no
            page_file_path = f"{pages_dir}/page_{page_num + 1}.jpg"
            
            # Process each detection on the page
            for det in page_result.layout_dets:
                entries, chunk_idx, secondary_idx = await _process_inference_result(
                    result=det,
                    page_num=page_num,
                    secondary_idx=secondary_idx,
                    ingestion=ingestion,
                    page_file_path=page_file_path,
                    chunk_idx=chunk_idx,
                    page_width=page_result.page_info.width,
                    page_height=page_result.page_info.height,
                )
                all_entries.extend(entries)
                
                # Link captions to their parent elements
                if det.category_id in [4, 6, 9]:  # figure_caption, table_caption, formula_caption
                    # Find the most recent parent element
                    parent_type = {
                        4: ExtractedFeatureType.figure,
                        6: ExtractedFeatureType.table,
                        9: ExtractedFeatureType.equation
                    }[det.category_id]
                    
                    parent_entry = next(
                        (e for e in reversed(all_entries) 
                         if e.consolidated_feature_type == parent_type),
                        None
                    )
                    
                    if parent_entry and entries:
                        caption_entry = entries[-1]
                        caption_entry.citations = {parent_entry.uuid: f"{parent_type}_caption"}
    
    return all_entries


if __name__ == "__main__":
    from src.pipeline.storage_backend import LocalStorageBackend
    from src.schemas.schemas import Scope, IngestionMethod, FileType
    import asyncio
    
    storage_client = LocalStorageBackend(base_path="/tmp/mineru_service")
    test_ingestions = [
        Ingestion(
            scope=Scope.INTERNAL,
            creator_name="Test User",
            ingestion_method=IngestionMethod.LOCAL_FILE,
            file_type=FileType.PDF,
            ingestion_date="2024-03-20T12:00:00Z",
            file_path="/path/to/test.pdf",
        ),
    ]
    output = asyncio.run(main_mineru(test_ingestions, mode="by_page", visualize=True))
    with open("mineru_output.json", "w") as f:
        for entry in output:
            f.write(json.dumps(entry.model_dump(), indent=4))
            f.write("\n")