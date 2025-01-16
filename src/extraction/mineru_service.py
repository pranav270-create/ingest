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
)
from src.utils.datetime_utils import get_current_utc_datetime
from src.utils.extraction_utils import convert_to_pdf


async def _visualize_page_results(
    image_path: str, parsed_elements: list, read=None
) -> bytes:
    """
    Visualize the parsed elements by drawing bounding boxes over the original image.
    Returns the annotated image as bytes.
    """
    # Colors for different types of elements (based on MinerU CategoryType)
    colors = {
        0: "purple",  # title
        1: "green",  # plain_text
        2: "gray",  # abandon (headers, footers, etc)
        3: "red",  # figure
        4: "darkgreen",  # figure_caption
        5: "blue",  # table
        6: "darkblue",  # table_caption
        7: "pink",  # table_footnote
        8: "brown",  # isolate_formula
        9: "darkbrown",  # formula_caption
        13: "teal",  # embedding (inline formula)
        14: "orange",  # isolated (block formula)
        15: "green",  # text (OCR result)
    }

    # Read the image using the provided read function or directly
    if read:
        img_bytes = await read(image_path, mode="rb")
        image = Image.open(io.BytesIO(img_bytes))
    else:
        image = Image.open(image_path)

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except Exception as e:
        print(f"Error loading font: {e}")
        font = ImageFont.load_default()

    page_width, page_height = image.size

    for element in parsed_elements:
        # Convert polygon coordinates to bbox
        x_coords = [element["polygon"][i] for i in range(0, len(element["polygon"]), 2)]
        y_coords = [element["polygon"][i] for i in range(1, len(element["polygon"]), 2)]

        left, top = min(x_coords), min(y_coords)
        right, bottom = max(x_coords), max(y_coords)

        color = colors.get(element["parsed_feature_type"], "white")
        draw.rectangle([left, top, right, bottom], outline=color, width=2)

        # Add label with type
        label = str(element["parsed_feature_type"])
        draw.rectangle(
            [left, max(0, top - 10), left + 10, max(0, top - 10) + 10], fill="white"
        )
        draw.text((left, max(0, top - 10)), label, fill=color, font=font)

    # Save to bytes instead of file
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    image.close()
    return img_byte_arr.getvalue()


def _map_mineru_type(category_id: int) -> ExtractedFeatureType:
    """Maps MinerU CategoryType to our common ExtractedFeatureType enum"""
    # CategoryType mapping to our ExtractedFeatureType
    mineru_to_common = {
        0: ExtractedFeatureType.section_header,  # title
        1: ExtractedFeatureType.text,  # plain_text
        2: ExtractedFeatureType.other,  # abandon (headers, footers, etc)
        3: ExtractedFeatureType.figure,  # figure
        4: ExtractedFeatureType.caption,  # figure_caption
        5: ExtractedFeatureType.table,  # table
        6: ExtractedFeatureType.caption,  # table_caption
        7: ExtractedFeatureType.footnote,  # table_footnote
        8: ExtractedFeatureType.equation,  # isolate_formula
        9: ExtractedFeatureType.caption,  # formula_caption
        13: ExtractedFeatureType.textinlinemath,  # embedding (inline formula)
        14: ExtractedFeatureType.equation,  # isolated (block formula)
        15: ExtractedFeatureType.text,  # text (OCR result)
    }
    output = mineru_to_common.get(category_id, ExtractedFeatureType.other)
    if output == ExtractedFeatureType.other:
        print(f"Unknown category_id: {category_id}")
    return output


def _convert_poly_to_bbox(
    poly: list[float], page_width: float, page_height: float
) -> dict:
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


def _process_para_blocks(
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
                page_file_path=page_file_path,
                bounding_box=_convert_poly_to_bbox(
                    block.get("poly", []), page_width, page_height
                ),
                extracted_image_path=None,  # Update if extracting images
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
            nested_entries, chunk_idx = _process_para_blocks(
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
    # Process files in parallel using async map
    async for result in obj.process_pdf.map.aio(file_bytes, return_exceptions=True):
        if isinstance(result, Exception):
            print(f"Error processing document: {result}")
            continue

        # Create output directories
        base_dir = os.path.dirname(ingestion.extracted_document_file_path)
        pages_dir = os.path.join(base_dir, "pages")
        extracts_dir = os.path.join(base_dir, "extracts")

        # Convert PDF pages to images if visualize is True
        page_image_paths = {}
        if visualize:
            with fitz.open(stream=io.BytesIO(pdf_content), filetype="pdf") as doc:
                for page_num in range(len(result["middle_json"]["pdf_info"])):
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
        for page_result in result["middle_json"]["pdf_info"]:
            page_num = page_result["page_idx"]
            page_width, page_height = page_result["page_size"]
            page_file_path = f"{pages_dir}/page_{page_num + 1}.jpg"

            # Process para_blocks recursively
            entries, chunk_idx = _process_para_blocks(
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
