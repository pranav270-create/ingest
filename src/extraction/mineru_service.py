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


def _map_mineru_type(block_type: str) -> ExtractedFeatureType:
    """Maps MinerU BlockType to our common ExtractedFeatureType enum"""
    # BlockType mapping to our ExtractedFeatureType
    mineru_to_common = {
        "image": ExtractedFeatureType.figure,
        "image_body": ExtractedFeatureType.figure,
        "image_caption": ExtractedFeatureType.caption,
        "image_footnote": ExtractedFeatureType.footnote,
        "table": ExtractedFeatureType.table,
        "table_body": ExtractedFeatureType.table,
        "table_caption": ExtractedFeatureType.caption,
        "table_footnote": ExtractedFeatureType.footnote,
        "text": ExtractedFeatureType.text,
        "title": ExtractedFeatureType.section_header,
        "interline_equation": ExtractedFeatureType.equation,
        "footnote": ExtractedFeatureType.footnote,
        "discarded": ExtractedFeatureType.other,
        "list": ExtractedFeatureType.list,
        "index": ExtractedFeatureType.other,
    }
    output = mineru_to_common.get(block_type, ExtractedFeatureType.other)
    if output == ExtractedFeatureType.other:
        print(f"Unknown block_type: {block_type}")
    return output


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

    # Convert relative coordinates to absolute
    left = bbox[0] * page_width
    top = bbox[1] * page_height
    right = bbox[2] * page_width
    bottom = bbox[3] * page_height

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

    img.close()
    return extracted_path


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
        feature_type = _map_mineru_type(block["type"])
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
            else:  # Table
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

        # Handle captions
        elif feature_type == ExtractedFeatureType.caption:
            # Determine parent element type based on context
            parent_type = None
            parent_uuid = None
            if block.get("parent_type") == "figure":
                parent_type = RelationshipType.FIGURE_CAPTION
                parent_uuid = parent_elements.get(ExtractedFeatureType.figure)
            elif block.get("parent_type") == "table":
                parent_type = RelationshipType.TABLE_CAPTION
                parent_uuid = parent_elements.get(ExtractedFeatureType.table)

            if parent_uuid and parent_type:
                caption_uuid = str(uuid.uuid4())
                citation = Citation(
                    relationship_type=parent_type,
                    target_uuid=parent_uuid,
                    source_uuid=caption_uuid,
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

                caption_entry = Entry(
                    uuid=caption_uuid,
                    ingestion=ingestion,
                    string=block.get("text", ""),
                    consolidated_feature_type=feature_type,
                    chunk_locations=[location],
                    chunk_index=chunk_idx + 1,
                    min_primary_index=page_num + 1,
                    max_primary_index=page_num + 1,
                    citations=[citation],
                )
                entries.append(caption_entry)
                chunk_idx += 1
                current_chunk_text.append(block.get("text", ""))
                current_chunk_locations.append(location)

        # Handle text content
        else:
            # Process text content from lines or spans
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        if span["type"] == "text":
                            content += f"{span['content']} "
                        elif span["type"] == "inline_equation":
                            content += f"{span.get('latex', '')} "
            elif "spans" in block:
                for span in block["spans"]:
                    if span["type"] == "text":
                        content += f"{span['content']} "
                    elif span["type"] == "inline_equation":
                        content += f"{span.get('latex', '')} "

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

    return entries, chunk_idx, current_chunk_text, current_chunk_locations


async def __process_para_blocks(
    blocks: list,
    page_num: int,
    ingestion: Ingestion,
    page_file_path: str,
    chunk_idx: int,
    counters: dict,
    page_width: float,
    page_height: float,
    extracts_dir: str,
    write=None,
) -> tuple[list[Entry], int]:
    """Recursively process para_blocks to extract entries."""
    entries = []

    for block in blocks:
        feature_type = _map_mineru_type(block["type"])

        # Initialize content
        content = ""
        extracted_path = None

        # Recursively process nested lines or spans
        if "lines" in block:
            for line in block["lines"]:
                for span in line.get("spans", []):
                    if span["type"] == "text":
                        content += f"{span['content']} "
                    elif span["type"] == "inline_equation":
                        content += f"{span.get('latex', '')} "
                    elif span["type"] == "table":
                        # Handle table content
                        content += f"{span.get('html', '')} "
                        counters["table_count"] += 1
                        # Optionally extract table image if available
                        if "image_path" in span:
                            extracted_path = (
                                f"{extracts_dir}/table_{counters['table_count']}.jpg"
                            )
                    elif span["type"] == "figure":
                        # Handle figure content
                        counters["figure_count"] += 1
                        extracted_path = (
                            f"{extracts_dir}/figure_{counters['figure_count']}.jpg"
                        )
                    else:
                        # Other types can be added here
                        continue

        elif "spans" in block:
            for span in block["spans"]:
                if span["type"] == "text":
                    content += f"{span['content']} "
                elif span["type"] == "inline_equation":
                    content += f"{span.get('latex', '')} "
                # Add other span types if necessary

        # Clean up content
        content = content.strip()

        if content:
            location = ChunkLocation(
                index=Index(
                    primary=page_num + 1,
                    secondary=chunk_idx + 1,
                ),
                extracted_feature_type=feature_type,
                extracted_file_path=extracted_path,
                page_file_path=page_file_path,
                bounding_box=_convert_bbox(
                    block.get("bbox", []), page_width, page_height
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

        # Recursively process nested blocks if any
        if "blocks" in block:
            nested_entries, chunk_idx = await __process_para_blocks(
                block["blocks"],
                page_num,
                ingestion,
                page_file_path,
                chunk_idx,
                counters,
                page_width,
                page_height,
                extracts_dir,
                write,
            )
            entries.extend(nested_entries)

    return entries, chunk_idx


@FunctionRegistry.register("parse", "mineru")
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
    cls = modal.Cls.lookup("mineru-modal", "MinerU")
    obj = cls()

    # First pass: collect all file bytes
    for ingestion in ingestions:
        ingestion.extraction_method = ExtractionMethod.MINERU
        ingestion.extraction_date = get_current_utc_datetime()

        if not ingestion.extracted_document_file_path:
            base_path = os.path.splitext(ingestion.file_path)[0]
            ingestion.extracted_document_file_path = f"{base_path}_mineru.json"

        # Read and process file
        file_content = (
            await read(ingestion.file_path)
            if read
            else open(ingestion.file_path, "rb").read()
        )

        # Convert to PDF if necessary
        file_extension = Path(ingestion.file_path).suffix.lower()
        pdf_content = convert_to_pdf(file_content, file_extension)
        file_bytes.append(pdf_content)

    all_entries = []
    # # Process files in parallel using async map
    # async for result in obj.process_pdf.map.aio(file_bytes, return_exceptions=True):
    #     if isinstance(result, Exception):
    #         print(f"Error processing document: {result}")
    #         continue

    for ingestion in ingestions:
        # Read the JSON content from the file
        with open("mineru_result.json", "r", encoding="utf-8") as f:
            result = json.load(f)

        # with open("mineru_result.json", "w") as f:
        #     json.dump(result, f, indent=4)

        # Create output directories
        base_dir = os.path.dirname(ingestion.extracted_document_file_path)
        # base_dir = os.path.dirname(Path(__file__).resolve())  # current file directory
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
            entries, chunk_idx = await _process_para_blocks(
                blocks=page_result.get("para_blocks", []),
                page_num=page_num,
                ingestion=ingestion,
                page_file_path=page_file_path,
                chunk_idx=chunk_idx,
                counters=counters,
                page_width=page_width,
                page_height=page_height,
                extracts_dir=extracts_dir,
                write=write,
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

        # Write combined entries to file
        if write:
            await write(
                json.dumps([entry.model_dump() for entry in all_entries], indent=4),
                ingestion.extracted_document_file_path,
            )
        else:
            with open(
                ingestion.extracted_document_file_path, "w", encoding="utf-8"
            ) as f:
                json.dump([entry.model_dump() for entry in all_entries], f, indent=4)

    return all_entries


if __name__ == "__main__":
    from src.schemas.schemas import Scope, IngestionMethod, FileType
    import asyncio

    test_ingestions = [
        Ingestion(
            scope=Scope.INTERNAL,  # Required
            creator_name="Test User",  # Required
            ingestion_method=IngestionMethod.LOCAL_FILE,  # Required
            file_type=FileType.PDF,
            ingestion_date="2024-03-20T12:00:00Z",  # Required
            file_path="/Users/pranaviyer/Downloads/TR0722-315a Appendix A.pdf",
        ),
        # Ingestion(
        #     scope=Scope.INTERNAL,
        #     creator_name="Test User",
        #     ingestion_method=IngestionMethod.LOCAL_FILE,
        #     ingestion_date="2024-03-20T12:00:00Z",
        #     file_type=FileType.PDF,
        #     file_path="/Users/pranaviyer/Desktop/AstralisData/E5_Paper.pdf",
        # )
    ]
    output = asyncio.run(main_mineru(test_ingestions, mode="by_page", visualize=True))
    with open("mineru_output.json", "w") as f:
        for entry in output:
            f.write(json.dumps(entry.model_dump(), indent=4))
            f.write("\n")
