import sys
from pathlib import Path
import modal
import json
import os
import uuid
from PIL import Image, ImageDraw, ImageFont
import io
import fitz

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import (
    Entry,
    ChunkLocation,
    Index,
    Ingestion,
    ExtractedFeatureType,
    ExtractionMethod,
    RelationshipType,
    Citation,
)
from src.utils.datetime_utils import get_current_utc_datetime
from src.utils.extraction_utils import convert_to_pdf
from src.utils.visualize_utils import visualize_page_results, group_entries_by_page


def _map_mineru_type(
    block_type: str, parent_type: str = None
) -> tuple[ExtractedFeatureType, str]:
    """Maps MinerU BlockType to our common ExtractedFeatureType enum and tracks parent relationships

    Args:
        block_type: The MinerU block type
        parent_type: The parent block type if any

    Returns:
        Tuple of (ExtractedFeatureType, parent_relationship)
    """
    # BlockType mapping to our ExtractedFeatureType
    mineru_to_common = {
        # Figure-related types
        "image": ExtractedFeatureType.figure,
        "image_body": ExtractedFeatureType.figure,
        "image_caption": ExtractedFeatureType.caption,
        "image_footnote": ExtractedFeatureType.footnote,
        # Table-related types
        "table": ExtractedFeatureType.table,
        "table_body": ExtractedFeatureType.table,
        "table_caption": ExtractedFeatureType.caption,
        "table_footnote": ExtractedFeatureType.footnote,
        # Other types
        "text": ExtractedFeatureType.text,
        "title": ExtractedFeatureType.section_header,
        "interline_equation": ExtractedFeatureType.equation,
        "footnote": ExtractedFeatureType.footnote,
        "discarded": ExtractedFeatureType.other,
        "list": ExtractedFeatureType.list,
        "index": ExtractedFeatureType.other,
    }

    # Determine parent relationship
    parent_relationship = None
    if block_type in ["image_caption", "image_footnote"] or parent_type == "image":
        parent_relationship = "figure"
    elif block_type in ["table_caption", "table_footnote"] or parent_type == "table":
        parent_relationship = "table"

    feature_type = mineru_to_common.get(block_type, ExtractedFeatureType.other)
    if feature_type == ExtractedFeatureType.other:
        print(f"Unknown block_type: {block_type}")

    return feature_type, parent_relationship


def _convert_bbox(bbox: list[float], page_width: float, page_height: float) -> dict:
    """Convert MinerU bbox format to our bbox format"""
    # MinerU bbox format is [left, top, right, bottom]
    return {
        "left": bbox[0],
        "top": bbox[1],
        "width": bbox[2] - bbox[0],  # right - left
        "height": bbox[3] - bbox[1],  # bottom - top
        "page_width": page_width,
        "page_height": page_height,
    }


async def _extract_region(
    bbox: list[float],
    page_num: int,
    secondary_index: int,
    page_file_path: str,
    page_width: float,
    page_height: float,
    extracts_dir: str,
    feature_type: ExtractedFeatureType,
    read=None,
    write=None,
) -> str:
    """Extract a region from a page image and save it."""
    # Open the image
    if read:
        img_bytes = await read(page_file_path)
        img = Image.open(io.BytesIO(img_bytes))
    else:
        img = Image.open(page_file_path)

    # Check if coordinates are already absolute (>1)
    if any(coord > 1 for coord in bbox):
        left, top, right, bottom = bbox
    else:
        # Convert relative coordinates to absolute
        left = bbox[0] * page_width
        top = bbox[1] * page_height
        right = bbox[2] * page_width
        bottom = bbox[3] * page_height

    # Ensure coordinates are within image bounds
    left = max(0, min(left, page_width))
    top = max(0, min(top, page_height))
    right = max(0, min(right, page_width))
    bottom = max(0, min(bottom, page_height))

    # Ensure valid box dimensions
    if right <= left or bottom <= top:
        print(
            f"Warning: Invalid box dimensions for {feature_type} on page {page_num + 1}"
        )
        img.close()
        return None

    try:
        # Crop the region
        region = img.crop((left, top, right, bottom))

        # Save extracted region
        extracted_path = (
            f"{extracts_dir}/{feature_type.value}_{page_num + 1}_{secondary_index}.jpg"
        )
        img_byte_arr = io.BytesIO()
        region.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        if write:
            await write(extracted_path, img_byte_arr)
        else:
            os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
            with open(extracted_path, "wb") as f:
                f.write(img_byte_arr)

        return extracted_path
    except Exception as e:
        print(f"Error extracting region: {e}")
        return None
    finally:
        img.close()


async def _process_para_blocks(
    blocks: list,
    page_num: int,
    ingestion: Ingestion,
    page_file_path: str,
    chunk_idx: int,
    counters: dict,
    page_width: float,
    page_height: float,
    extracts_dir: str,
    mode: str = "by_page",
    write=None,
    read=None,
) -> tuple[list[Entry], int, list[str], list[ChunkLocation]]:
    """Recursively process para_blocks to extract entries."""
    entries = []
    current_chunk_text = []
    current_chunk_locations = []

    # Track parent elements for citations
    parent_elements = {
        ExtractedFeatureType.figure: None,
        ExtractedFeatureType.table: None,
    }

    for block in blocks:
        parent_type = block.get("parent_type")  # Get parent type if available
        feature_type, parent_relationship = _map_mineru_type(block["type"], parent_type)

        content = ""
        extracted_path = None

        # Handle visual elements (tables and figures)
        if feature_type in [ExtractedFeatureType.figure, ExtractedFeatureType.table]:
            # Extract region
            extracted_path = await _extract_region(
                bbox=block.get("bbox", []),
                page_num=page_num,
                secondary_index=len(current_chunk_locations),
                page_file_path=page_file_path,
                page_width=page_width,
                page_height=page_height,
                extracts_dir=extracts_dir,
                feature_type=feature_type,
                read=read,
                write=write,
            )

            # Create element entry
            element_uuid = str(uuid.uuid4())
            location = ChunkLocation(
                index=Index(
                    primary=page_num + 1,
                    secondary=len(current_chunk_locations) + 1,
                ),
                extracted_feature_type=feature_type,
                extracted_file_path=extracted_path,
                page_file_path=page_file_path,
                bounding_box=_convert_bbox(
                    block.get("bbox", []), page_width, page_height
                ),
            )

            if feature_type == ExtractedFeatureType.figure:
                counters["figure_count"] += 1
                figure_number = counters["figure_count"]
                parent_elements[ExtractedFeatureType.figure] = element_uuid
                element_entry = Entry(
                    uuid=element_uuid,
                    ingestion=ingestion,
                    string="",  # Empty string as content is visual
                    consolidated_feature_type=feature_type,
                    chunk_locations=[location],
                    chunk_index=chunk_idx + 1,
                    min_primary_index=page_num + 1,
                    max_primary_index=page_num + 1,
                    figure_number=figure_number,
                )
            elif feature_type == ExtractedFeatureType.table:
                counters["table_count"] += 1
                table_number = counters["table_count"]
                parent_elements[ExtractedFeatureType.table] = element_uuid
                element_entry = Entry(
                    uuid=element_uuid,
                    ingestion=ingestion,
                    string=block.get("html", ""),
                    consolidated_feature_type=feature_type,
                    chunk_locations=[location],
                    chunk_index=chunk_idx + 1,
                    min_primary_index=page_num + 1,
                    max_primary_index=page_num + 1,
                    table_number=table_number,
                )

            entries.append(element_entry)
            chunk_idx += 1
            current_chunk_text.append(block.get("html", ""))
            current_chunk_locations.append(location)

        # Handle captions and footnotes
        elif feature_type in [
            ExtractedFeatureType.caption,
            ExtractedFeatureType.footnote,
        ]:
            parent_uuid = None
            relationship_type = None

            if parent_relationship == "figure":
                parent_uuid = parent_elements.get(ExtractedFeatureType.figure)
                relationship_type = (
                    RelationshipType.FIGURE_CAPTION
                    if feature_type == ExtractedFeatureType.caption
                    else RelationshipType.FIGURE_FOOTNOTE
                )
            elif parent_relationship == "table":
                parent_uuid = parent_elements.get(ExtractedFeatureType.table)
                relationship_type = (
                    RelationshipType.TABLE_CAPTION
                    if feature_type == ExtractedFeatureType.caption
                    else RelationshipType.TABLE_FOOTNOTE
                )

            if parent_uuid and relationship_type:
                child_uuid = str(uuid.uuid4())
                citation = Citation(
                    relationship_type=relationship_type,
                    target_uuid=parent_uuid,
                    source_uuid=child_uuid,
                )

                location = ChunkLocation(
                    index=Index(
                        primary=page_num + 1,
                        secondary=len(current_chunk_locations) + 1,
                    ),
                    extracted_feature_type=feature_type,
                    page_file_path=page_file_path,
                    bounding_box=_convert_bbox(
                        block.get("bbox", []), page_width, page_height
                    ),
                )

                child_entry = Entry(
                    uuid=child_uuid,
                    ingestion=ingestion,
                    string=block.get("text", ""),
                    consolidated_feature_type=feature_type,
                    chunk_locations=[location],
                    chunk_index=chunk_idx + 1,
                    min_primary_index=page_num + 1,
                    max_primary_index=page_num + 1,
                    citations=[citation],
                )
                entries.append(child_entry)
                chunk_idx += 1

        # Handle text content
        else:
            # Process text content from lines or spans
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        content += f"{span['content']} "
            elif "spans" in block:
                for span in block["spans"]:
                    content += f"{span['content']} "

            content = content.strip()
            if content:
                location = ChunkLocation(
                    index=Index(
                        primary=page_num + 1,
                        secondary=len(current_chunk_locations) + 1,
                    ),
                    extracted_feature_type=feature_type,
                    page_file_path=page_file_path,
                    bounding_box=_convert_bbox(
                        block.get("bbox", []), page_width, page_height
                    ),
                )

                # For by_title mode, create new chunk at section headers
                if (
                    mode == "by_title"
                    and feature_type == ExtractedFeatureType.section_header
                ):
                    if current_chunk_text and current_chunk_locations:
                        combined_text = " ".join(current_chunk_text)
                        chunk_entry = Entry(
                            uuid=str(uuid.uuid4()),
                            ingestion=ingestion,
                            string=combined_text,
                            consolidated_feature_type=ExtractedFeatureType.combined_text,
                            chunk_locations=current_chunk_locations.copy(),
                            min_primary_index=page_num + 1,
                            max_primary_index=page_num + 1,
                            chunk_index=chunk_idx + 1,
                        )
                        entries.append(chunk_entry)
                        chunk_idx += 1
                        current_chunk_text = []
                        current_chunk_locations = []

                current_chunk_text.append(content)
                current_chunk_locations.append(location)

        # Recursively process nested blocks
        if "blocks" in block:
            nested_entries, chunk_idx, nested_text, nested_locations = (
                await _process_para_blocks(
                    block["blocks"],
                    page_num,
                    ingestion,
                    page_file_path,
                    chunk_idx,
                    counters,
                    page_width,
                    page_height,
                    extracts_dir,
                    mode,
                    write,
                    read,
                )
            )
            entries.extend(nested_entries)
            current_chunk_text.extend(nested_text)
            current_chunk_locations.extend(nested_locations)

    # After processing all blocks, create an entry for accumulated text if any exists
    if current_chunk_text and current_chunk_locations:
        combined_text = " ".join(current_chunk_text)
        text_entry = Entry(
            uuid=str(uuid.uuid4()),
            ingestion=ingestion,
            string=combined_text,
            consolidated_feature_type=ExtractedFeatureType.combined_text,
            chunk_locations=current_chunk_locations.copy(),
            chunk_index=chunk_idx + 1,
            min_primary_index=page_num + 1,
            max_primary_index=page_num + 1,
        )
        entries.append(text_entry)
        chunk_idx += 1

    return entries, chunk_idx, current_chunk_text, current_chunk_locations


@FunctionRegistry.register("extract", "mineru")
async def main_mineru(
    ingestions: list[Ingestion],
    write=None,
    read=None,
    visualize=False,
    mode="by_page",
    **kwargs,
) -> list[Entry]:
    """Parse documents using the MinerU library.

    Args:
        ingestions: List of Ingestion objects
        write: Optional async write function
        read: Optional async read function
        visualize: If True, saves annotated PDFs with bounding boxes
        **kwargs: Additional arguments

    Returns:
        List of Entry objects
    """
    file_bytes = []
    ingestion_map = {}  # Map to track file_bytes index to ingestion

    cls = modal.Cls.lookup("mineru-modal", "MinerU")
    obj = cls()

    # First pass: collect all file bytes
    for idx, ingestion in enumerate(ingestions):
        ingestion.extraction_method = ExtractionMethod.MINERU
        ingestion.extraction_date = get_current_utc_datetime()

        if not ingestion.extracted_document_file_path:
            base_path = os.path.splitext(ingestion.file_path)[0]
            ingestion.extracted_document_file_path = f"{base_path}_mineru.json"

        file_content = (
            await read(ingestion.file_path)
            if read
            else open(ingestion.file_path, "rb").read()
        )
        pdf_content = convert_to_pdf(
            file_content, Path(ingestion.file_path).suffix.lower()
        )
        file_bytes.append(pdf_content)
        ingestion_map[idx] = ingestion

    all_entries = []
    # Process files in parallel using async map
    idx = 0
    async for result in obj.process_pdf.map.aio(file_bytes, return_exceptions=True):
        if isinstance(result, Exception):
            print(f"Error processing document: {result}")
            continue

        current_ingestion = ingestion_map[idx]

        # Create output directories
        base_dir = os.path.dirname(current_ingestion.extracted_document_file_path)
        base_dir = os.path.dirname(Path(__file__).resolve())  # current file directory
        pages_dir = os.path.join(base_dir, "pages")
        extracts_dir = os.path.join(base_dir, "extracts")

        # Convert PDF pages to images if visualize is True
        page_image_paths = {}
        if visualize:
            with fitz.open(stream=io.BytesIO(pdf_content), filetype="pdf") as doc:
                for page_num in range(len(result["pdf_info"])):
                    # Convert page to image
                    pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(1, 1))
                    img_bytes = pix.tobytes("jpg")
                    page_image_path = f"{pages_dir}/page_{page_num + 1}.jpg"

                    if write:
                        await write(page_image_path, img_bytes)
                    else:
                        os.makedirs(os.path.dirname(page_image_path), exist_ok=True)
                        with open(page_image_path, "wb") as f:
                            f.write(img_bytes)
                    page_image_paths[page_num] = page_image_path

        # Initialize counters
        chunk_idx = 0
        counters = {"figure_count": 0, "table_count": 0}

        # Process each page
        for page_result in result["pdf_info"]:
            page_num = page_result["page_idx"]
            page_width, page_height = page_result["page_size"]
            page_file_path = f"{pages_dir}/page_{page_num + 1}.jpg"

            # Process para_blocks
            entries, chunk_idx, current_chunk_text, current_chunk_locations = (
                await _process_para_blocks(
                    blocks=page_result.get("para_blocks", []),
                    page_num=page_num,
                    ingestion=current_ingestion,
                    page_file_path=page_file_path,
                    chunk_idx=chunk_idx,
                    counters=counters,
                    page_width=page_width,
                    page_height=page_height,
                    extracts_dir=extracts_dir,
                    write=write,
                )
            )
            all_entries.extend(entries)

            # Visualize if requested
            if visualize:
                output_dir = os.path.join(base_dir, "annotated")
                entries_by_page = group_entries_by_page(all_entries)

                for page_num, elements in entries_by_page.items():
                    if not elements:
                        continue

                    output_path = f"{output_dir}/page_{page_num + 1}_annotated.png"
                    annotated_image = await visualize_page_results(
                        page_image_paths[page_num], elements, read
                    )

                    if write:
                        await write(output_path, annotated_image)
                    else:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        with open(output_path, "wb") as f:
                            f.write(annotated_image)

        # Write combined entries to file for this specific ingestion
        if write:
            await write(
                current_ingestion.extracted_document_file_path,
                json.dumps([entry.model_dump() for entry in all_entries], indent=4),
            )
        else:
            with open(
                current_ingestion.extracted_document_file_path, "w", encoding="utf-8"
            ) as f:
                json.dump([entry.model_dump() for entry in all_entries], f, indent=4)

        idx += 1
    return all_entries


if __name__ == "__main__":
    from src.schemas.schemas import Scope, IngestionMethod, FileType
    import asyncio

    test_ingestions = [
        # Ingestion(
        #     scope=Scope.INTERNAL,  # Required
        #     creator_name="Test User",  # Required
        #     ingestion_method=IngestionMethod.LOCAL_FILE,  # Required
        #     file_type=FileType.PDF,
        #     ingestion_date="2024-03-20T12:00:00Z",  # Required
        #     file_path="/Users/pranaviyer/Downloads/TR0722-315a Appendix A.pdf",
        # ),
        Ingestion(
            scope=Scope.INTERNAL,
            creator_name="Test User",
            ingestion_method=IngestionMethod.LOCAL_FILE,
            ingestion_date="2024-03-20T12:00:00Z",
            file_type=FileType.PDF,
            file_path="/Users/pranaviyer/Desktop/AstralisData/E5_Paper.pdf",
        )
    ]
    output = asyncio.run(main_mineru(test_ingestions, mode="by_page", visualize=True))
    with open("mineru_output.json", "w") as f:
        for entry in output:
            f.write(json.dumps(entry.model_dump(), indent=4))
            f.write("\n")
