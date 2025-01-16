import sys
from pathlib import Path
import modal
import io
import pandas as pd
from docx import Document as DocxDocument
from PIL import Image, ImageDraw, ImageFont
import fitz
import os
import json
import uuid
import re

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import (
    Entry,
    ChunkLocation,
    Index,
    Ingestion,
    ExtractedFeatureType,
    ExtractionMethod,
    Scope,
    IngestionMethod,
    FileType,
)
from src.utils.datetime_utils import get_current_utc_datetime


def convert_to_pdf(file_content: bytes, file_extension: str) -> bytes:
    pdf_bytes = io.BytesIO()
    if file_extension in [".xlsx", ".xls"]:
        # Convert Excel to PDF
        df = pd.read_excel(io.BytesIO(file_content))
        df.to_pdf(pdf_bytes)
    elif file_extension in [".docx", ".doc"]:
        # Convert Word to PDF
        doc = DocxDocument(io.BytesIO(file_content))
        # This is a placeholder. You'll need a library like python-docx2pdf for actual conversion
        # For now, we'll just extract text
        full_text = "\n".join([para.text for para in doc.paragraphs])
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((50, 50), full_text)
        pdf.save(pdf_bytes)
        pdf.close()
    elif file_extension in [".png", ".jpg", ".jpeg"]:
        # Convert Image to PDF
        image = Image.open(io.BytesIO(file_content))
        pdf = fitz.open()
        page = pdf.new_page(width=image.width, height=image.height)
        page.insert_image(page.rect, stream=file_content)
        pdf.save(pdf_bytes)
        pdf.close()
    else:
        # Assume it's already a PDF
        return file_content
    return pdf_bytes.getvalue()


def _map_marker_type(block_type: str) -> ExtractedFeatureType:
    """Maps Marker block types to our common ExtractedFeatureType enum"""
    block_type_lower = block_type.lower()
    # Common type mappings
    marker_to_common = {
        # Common content types
        "text": ExtractedFeatureType.text,
        "word": ExtractedFeatureType.word,
        "line": ExtractedFeatureType.line,
        "image": ExtractedFeatureType.image,
        "table": ExtractedFeatureType.table,
        "tablegroup": ExtractedFeatureType.tablegroup,  # Map to common table type
        "figure": ExtractedFeatureType.figure,
        "figuregroup": ExtractedFeatureType.figuregroup,  # Map to common figure type
        "code": ExtractedFeatureType.code,
        "equation": ExtractedFeatureType.equation,
        "form": ExtractedFeatureType.form,
        "pageheader": ExtractedFeatureType.header,
        "pagefooter": ExtractedFeatureType.footer,
        "sectionheader": ExtractedFeatureType.section_header,
        "listitem": ExtractedFeatureType.list,
        "listgroup": ExtractedFeatureType.listgroup,
        "pagenumber": ExtractedFeatureType.page_number,
        # Marker-specific types
        "span": ExtractedFeatureType.span,
        "picturegroup": ExtractedFeatureType.picturegroup,
        "picture": ExtractedFeatureType.picture,
        "page": ExtractedFeatureType.page,
        "caption": ExtractedFeatureType.caption,
        "footnote": ExtractedFeatureType.footnote,
        "handwriting": ExtractedFeatureType.handwriting,
        "textinlinemath": ExtractedFeatureType.textinlinemath,
        "tableofcontents": ExtractedFeatureType.tableofcontents,
        "document": ExtractedFeatureType.document,
        "complexregion": ExtractedFeatureType.complexregion,
    }
    output = marker_to_common.get(block_type_lower, ExtractedFeatureType.other)
    if output == ExtractedFeatureType.other:
        print(f"Unknown block type: {block_type_lower}")
    return output


def _polygon_to_bbox(polygon, page_width, page_height):
    """Convert polygon coordinates to bounding box format."""
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]

    return {
        "left": min(x_coords),
        "top": min(y_coords),
        "width": max(x_coords) - min(x_coords),
        "height": max(y_coords) - min(y_coords),
        "page_width": page_width,
        "page_height": page_height,
    }


async def _visualize_page_results(
    image_path: str, parsed_elements: list, read=None
) -> bytes:
    """
    Visualize the parsed elements by drawing bounding boxes over the original image.
    Returns the annotated image as bytes.
    """
    # Colors for different types of elements
    colors = {
        "Figure": "red",
        "FigureGroup": "red",
        "Picture": "red",
        "PictureGroup": "red",
        "Table": "blue",
        "TableGroup": "blue",
        "Text": "green",
        "Page": "yellow",
        "PageHeader": "orange",
        "PageFooter": "orange",
        "SectionHeader": "purple",
        "ListGroup": "cyan",
        "ListItem": "magenta",
        "Footnote": "pink",
        "Equation": "brown",
        "TextInlineMath": "teal",
        "Line": "gray",
        "Span": "lightgray",
        "Caption": "darkgreen",
        "Code": "darkblue",
        "Form": "darkred",
        "Handwriting": "violet",
        "TableOfContents": "navy",
        "Document": "white",
        "ComplexRegion": "olive",
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
        bbox = _polygon_to_bbox(element["polygon"], page_width, page_height)
        left, top = bbox["left"], bbox["top"]
        right = left + bbox["width"]
        bottom = top + bbox["height"]

        color = colors.get(element["parsed_feature_type"], "white")
        draw.rectangle([left, top, right, bottom], outline=color, width=2)

        label = element["parsed_feature_type"]
        draw.rectangle(
            [left, max(0, top - 10), left + 10, max(0, top - 10) + 10], fill="white"
        )
        draw.text((left, max(0, top - 10)), label, fill=color, font=font)

    # Save to bytes instead of file
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    image.close()
    return img_byte_arr.getvalue()


async def _extract_region(
    child: dict,
    page_num: int,
    secondary_index: int,
    page_image_path: str,
    page_dimensions: tuple[int, int],
    extracts_dir: str,
    feature_type: ExtractedFeatureType,
    read=None,
    write=None,
) -> tuple[str, dict]:
    """
    Extract a region from a page image and save it.

    Args:
        child: The child element containing polygon information
        page_num: Page number
        secondary_index: Index within the page
        page_image_path: Path to the page image
        page_dimensions: Tuple of (width, height) of the page
        extracts_dir: Directory to save extracted regions
        feature_type: Type of feature being extracted
        read: Optional async read function
        write: Optional async write function

    Returns:
        tuple[str, dict]: (Path to extracted region, bounding box dictionary)
    """
    page_width, page_height = page_dimensions
    bbox = _polygon_to_bbox(child["polygon"], page_width, page_height)

    # Open the image
    if read:
        img_bytes = await read(page_image_path)
        img = Image.open(io.BytesIO(img_bytes))
    else:
        img = Image.open(page_image_path)

    # Crop the region
    left, top = bbox["left"], bbox["top"]
    right = left + bbox["width"]
    bottom = top + bbox["height"]
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
    return extracted_path, bbox


async def _process_child_element(
    child: dict,
    parent_feature_type: ExtractedFeatureType,
    page_num: int,
    secondary_idx: int,
    ingestion: Ingestion,
    page_image_paths: dict,
    page_dimensions: tuple[int, int],
    extracts_dir: str,
    chunk_idx: int,
    counters: dict,
    read=None,
    write=None,
) -> tuple[list[Entry], int, int, dict, str]:
    """
    Process a child element and its immediate children (depth 2).
    Args:
        child: The child element containing polygon information
        parent_feature_type: Type of the parent feature
        page_num: Page number
        secondary_idx: Index within the page
        ingestion: Ingestion object
        page_image_paths: Dictionary of page image paths
        page_dimensions: Tuple of (width, height) of the page
        extracts_dir: Directory to save extracted regions
        chunk_idx: Index within the chunk
        counters: Dictionary tracking figure and table counts
        read: Optional async read function
        write: Optional async write function
    Returns:
        tuple[list[Entry], int, int, dict, str]: (entries, chunk_idx, secondary_idx, counters, html_content)
    """
    entries = []
    page_width, page_height = page_dimensions

    if "html" not in child:
        return entries, chunk_idx, secondary_idx, counters

    parent_content = child["html"]

    # Handle visual elements
    if parent_feature_type in [
        ExtractedFeatureType.figure,
        ExtractedFeatureType.figuregroup,
        ExtractedFeatureType.picture,
        ExtractedFeatureType.picturegroup,
        ExtractedFeatureType.complexregion,
        ExtractedFeatureType.table,
        ExtractedFeatureType.tablegroup,
    ]:
        parent_extracted_path, parent_bbox = await _extract_region(
            child=child,
            page_num=page_num,
            secondary_index=secondary_idx,
            page_image_path=page_image_paths[page_num],
            page_dimensions=(page_width, page_height),
            extracts_dir=extracts_dir,
            feature_type=parent_feature_type,
            read=read,
            write=write,
        )

        # Simple case: Individual figure/picture/table
        if parent_feature_type in [
            ExtractedFeatureType.figure,
            ExtractedFeatureType.picture,
            ExtractedFeatureType.table,
        ] and parent_content:
            if parent_feature_type in [ExtractedFeatureType.figure, ExtractedFeatureType.picture]:
                counters["figure_count"] += 1
                element_number = counters["figure_count"]
                number_type = "figure_number"
            elif parent_feature_type == ExtractedFeatureType.table:
                counters["table_count"] += 1
                element_number = counters["table_count"]
                number_type = "table_number"
            else:
                element_number = None
                number_type = None

            entry_data = {
                "uuid": str(uuid.uuid4()),
                "ingestion": ingestion,
                "string": parent_content,
                "consolidated_feature_type": parent_feature_type,
                "chunk_locations": [
                    ChunkLocation(
                        index=Index(primary=page_num + 1, secondary=secondary_idx + 1),
                        extracted_feature_type=parent_feature_type,
                        page_file_path=page_image_paths[page_num],
                        extracted_file_path=parent_extracted_path,
                        bounding_box=parent_bbox,
                    )
                ],
                "min_primary_index": page_num + 1,
                "max_primary_index": page_num + 1,
                "chunk_index": chunk_idx + 1,
            }
            if number_type:
                entry_data[number_type] = element_number
            chunk_idx += 1
            secondary_idx += 1
            entries.append(Entry(**entry_data))
            return entries, chunk_idx, secondary_idx, counters, parent_content

        # Complex case: Groups with multiple elements
        children = child.get("children", []) or []
        html_content = ""

        # Track parent elements for citation linking
        parent_elements = {
            ExtractedFeatureType.figure: None,
            ExtractedFeatureType.table: None,
            ExtractedFeatureType.picture: None,
        }

        # First pass: Process figures and tables
        for sub_child in children:
            block_type = sub_child.get("block_type", "").lower()
            child_feature_type = _map_marker_type(block_type)

            # Skip captions in first pass
            if child_feature_type == ExtractedFeatureType.caption:
                continue

            child_extracted_path, child_bbox = await _extract_region(
                child=sub_child,
                page_num=page_num,
                secondary_index=secondary_idx,
                page_image_path=page_image_paths[page_num],
                page_dimensions=(page_width, page_height),
                extracts_dir=extracts_dir,
                feature_type=child_feature_type,
                read=read,
                write=write,
            )

            child_chunk_location = ChunkLocation(
                index=Index(
                    primary=page_num + 1,
                    secondary=secondary_idx + 1,
                ),
                extracted_feature_type=child_feature_type,
                page_file_path=page_image_paths[page_num],
                extracted_file_path=child_extracted_path,
                bounding_box=child_bbox,
            )

            # Process figures and tables
            if child_feature_type in [
                ExtractedFeatureType.figure,
                ExtractedFeatureType.picture,
            ]:
                figure_uuid = str(uuid.uuid4())
                figure_entry = Entry(
                    uuid=figure_uuid,
                    ingestion=ingestion,
                    string="",
                    consolidated_feature_type=child_feature_type,
                    chunk_locations=[child_chunk_location],
                    chunk_index=chunk_idx + 1,
                    min_primary_index=child_chunk_location.index.primary,
                    max_primary_index=child_chunk_location.index.primary,
                    figure_number=counters["figure_count"],
                )
                chunk_idx += 1
                counters["figure_count"] += 1
                entries.append(figure_entry)

                llm_caption_entry = Entry(
                    uuid=str(uuid.uuid4()),
                    ingestion=ingestion,
                    string=sub_child["html"],
                    consolidated_feature_type=ExtractedFeatureType.caption,
                    chunk_locations=[child_chunk_location],
                    chunk_index=chunk_idx + 1,
                    min_primary_index=child_chunk_location.index.primary,
                    max_primary_index=child_chunk_location.index.primary,
                    citations={figure_uuid: "figure_caption"},
                )
                chunk_idx += 1
                entries.append(llm_caption_entry)
                parent_elements[child_feature_type] = figure_uuid
                html_content += sub_child["html"]

            elif child_feature_type == ExtractedFeatureType.table:
                table_uuid = str(uuid.uuid4())
                table_entry = Entry(
                    uuid=table_uuid,
                    ingestion=ingestion,
                    string=sub_child["html"],
                    consolidated_feature_type=child_feature_type,
                    chunk_locations=[child_chunk_location],
                    chunk_index=chunk_idx + 1,
                    min_primary_index=child_chunk_location.index.primary,
                    max_primary_index=child_chunk_location.index.primary,
                    table_number=counters["table_count"],
                )
                chunk_idx += 1
                counters["table_count"] += 1
                entries.append(table_entry)
                parent_elements[child_feature_type] = table_uuid
                html_content += sub_child["html"]
            secondary_idx += 1

        # Second pass: Process captions and llm_captions and link to parent elements
        for sub_child in children:
            if not isinstance(sub_child, dict):
                continue

            block_type = sub_child.get("block_type", "").lower()
            child_feature_type = _map_marker_type(block_type)

            if child_feature_type != ExtractedFeatureType.caption:
                continue

            child_chunk_location = ChunkLocation(
                index=Index(
                    primary=page_num + 1,
                    secondary=secondary_idx + 1,
                ),
                extracted_feature_type=child_feature_type,
                page_file_path=page_image_paths[page_num],
                bounding_box=_polygon_to_bbox(sub_child["polygon"], page_width, page_height),
            )

            # Determine which parent to link to
            parent_uuid = None
            citation_type = None
            if parent_feature_type in [ExtractedFeatureType.figuregroup, ExtractedFeatureType.picturegroup]:
                parent_uuid = parent_elements.get(ExtractedFeatureType.figure) or parent_elements.get(ExtractedFeatureType.picture)
                citation_type = "figure_caption"
            elif parent_feature_type == ExtractedFeatureType.tablegroup:
                parent_uuid = parent_elements.get(ExtractedFeatureType.table)
                citation_type = "table_caption"

            if parent_uuid and sub_child.get("html"):
                caption_entry = Entry(
                    uuid=str(uuid.uuid4()),
                    ingestion=ingestion,
                    string=sub_child["html"],
                    consolidated_feature_type=ExtractedFeatureType.caption,
                    chunk_locations=[child_chunk_location],
                    chunk_index=chunk_idx + 1,
                    min_primary_index=child_chunk_location.index.primary,
                    max_primary_index=child_chunk_location.index.primary,
                    citations={parent_uuid: citation_type},
                )
                chunk_idx += 1
                entries.append(caption_entry)
                secondary_idx += 1
                html_content += sub_child["html"]

    return entries, chunk_idx, secondary_idx, counters, html_content


async def _process_title_chunk(
    accumulated_text: list,
    chunk_locations: list,
    ingestion: Ingestion,
    chunk_idx: int
) -> tuple[Entry, int]:
    """
    Process accumulated text and locations into a single chunk Entry.
    
    Args:
        accumulated_text: List of text strings in the chunk
        chunk_locations: List of ChunkLocation objects
        ingestion: Ingestion object
        chunk_idx: Current chunk index
    
    Returns:
        tuple[Entry, int]: (Created entry, incremented chunk index)
    """
    if not accumulated_text or not chunk_locations:
        return None, chunk_idx

    combined_text = " ".join(accumulated_text)
    entry = Entry(
        uuid=str(uuid.uuid4()),
        ingestion=ingestion,
        string=combined_text,
        consolidated_feature_type=ExtractedFeatureType.combined_text,
        chunk_locations=chunk_locations,
        min_primary_index=min(loc.index.primary for loc in chunk_locations),
        max_primary_index=max(loc.index.primary for loc in chunk_locations),
        chunk_index=chunk_idx + 1,
    )
    return entry, chunk_idx + 1


@FunctionRegistry.register("parse", "datalab")
async def main_datalab(
    ingestions: list[Ingestion],
    write=None,
    read=None,
    mode="by_page",
    visualize=False,
    **kwargs,
) -> list[Entry]:
    """
    Parse documents using the Marker library.

    Args:
        ingestions: List of Ingestion objects
        write: Optional async write function
        read: Optional async read function
        mode: "by_page" or "by_section" parsing mode
        visualize: If True, saves annotated PDFs with bounding boxes
        **kwargs: Additional arguments
    """
    file_bytes = []
    cls = modal.Cls.lookup("marker-modal", "Model")
    obj = cls()
    for ingestion in ingestions:
        ingestion.extraction_method = ExtractionMethod.MARKER
        ingestion.extraction_date = get_current_utc_datetime()

        if not ingestion.extracted_document_file_path:
            base_path = os.path.splitext(ingestion.file_path)[0]
            ingestion.extracted_document_file_path = f"{base_path}_marker.json"

        # Update file reading to handle async
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
    async for ret in obj.parse_document.map.aio(file_bytes, return_exceptions=True):
        if isinstance(ret, Exception):
            print(f"Error processing document: {ret}")
            continue

        parsed_data = json.loads(ret)
        # Create output directories (these should be virtual paths for cloud storage)
        base_dir = os.path.dirname(ingestion.extracted_document_file_path)
        pages_dir = os.path.join(base_dir, "pages")
        extracts_dir = os.path.join(base_dir, "extracts")

        # Convert PDF pages to images and store paths
        page_image_paths = {}
        with fitz.open(stream=io.BytesIO(pdf_content), filetype="pdf") as doc:
            for page_num, page in enumerate(parsed_data["children"]):
                # Convert page to image
                pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(1, 1))
                img_bytes = pix.tobytes("jpg")  # Convert to jpg bytes
                page_image_path = f"{pages_dir}/page_{page_num + 1}.jpg"
                if write:
                    await write(page_image_path, img_bytes)
                else:
                    os.makedirs(os.path.dirname(page_image_path), exist_ok=True)
                    with open(page_image_path, "wb") as f:
                        f.write(img_bytes)
                page_image_paths[page_num] = page_image_path

                if visualize:
                    output_dir = os.path.join(base_dir, "annotated")
                    # Convert page to image for visualization
                    pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(1, 1))
                    temp_bytes = pix.tobytes("png")
                    temp_image_path = f"{output_dir}/temp_page_{page_num + 1}.png"

                    # Save temporary image
                    if write:
                        await write(temp_image_path, temp_bytes)
                    else:
                        os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
                        with open(temp_image_path, "wb") as f:
                            f.write(temp_bytes)

                    # Prepare elements for visualization
                    elements = [
                        {
                            "parsed_feature_type": page["block_type"],
                            "polygon": page["polygon"],
                        }
                    ]
                    if page.get("children"):
                        elements.extend(
                            [
                                {
                                    "parsed_feature_type": child["block_type"],
                                    "polygon": child["polygon"],
                                }
                                for child in page["children"]
                            ]
                        )

                    # Create and save annotated version
                    output_path = f"{output_dir}/page_{page_num + 1}_annotated.png"
                    annotated_image = await _visualize_page_results(
                        temp_image_path, elements, read
                    )
                    if write:
                        await write(output_path, annotated_image)
                    else:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        with open(output_path, "wb") as f:
                            f.write(annotated_image)

        all_text = {}
        if mode == "by_page":
            chunk_idx = 0
            # Initialize counters dictionary
            counters = {"figure_count": 0, "table_count": 0}

            for page_num, page in enumerate(parsed_data["children"]):
                page_text = []
                all_chunk_locations = []
                if not page.get("children"):
                    continue

                # Get page dimensions
                if read:
                    img_bytes = await read(page_image_paths[page_num])
                    img = Image.open(io.BytesIO(img_bytes))
                    page_width, page_height = img.size
                    img.close()
                else:
                    with Image.open(page_image_paths[page_num]) as img:
                        page_width, page_height = img.size

                # Process all children of the Page
                secondary_idx = 0
                for child in page["children"]:
                    block_type = child["block_type"]
                    feature_type = _map_marker_type(block_type)

                    # Process visual elements
                    if feature_type in [
                        ExtractedFeatureType.figure,
                        ExtractedFeatureType.figuregroup,
                        ExtractedFeatureType.picture,
                        ExtractedFeatureType.picturegroup,
                        ExtractedFeatureType.complexregion,
                        ExtractedFeatureType.table,
                        ExtractedFeatureType.tablegroup,
                    ]:
                        # Process visual elements using _process_child_element
                        entries, chunk_idx, secondary_idx, counters, html_content = (
                            await _process_child_element(
                                child=child,
                                parent_feature_type=feature_type,
                                page_num=page_num,
                                secondary_idx=secondary_idx,
                                ingestion=ingestion,
                                page_image_paths=page_image_paths,
                                page_dimensions=(page_width, page_height),
                                extracts_dir=extracts_dir,
                                chunk_idx=chunk_idx,
                                counters=counters,
                                read=read,
                                write=write,
                            )
                        )
                        all_entries.extend(entries)
                        page_text.append(html_content)
                    elif feature_type == ExtractedFeatureType.listgroup:
                        # Handle nested list structures
                        if "children" in child:
                            for list_child in child["children"]:
                                if "html" in list_child:
                                    page_text.append(list_child["html"])
                                    location = ChunkLocation(
                                        index=Index(
                                            primary=page_num + 1,
                                            secondary=secondary_idx + 1,
                                        ),
                                        extracted_feature_type=_map_marker_type(list_child["block_type"]),
                                        page_file_path=page_image_paths[page_num],
                                        bounding_box=_polygon_to_bbox(
                                            list_child["polygon"], page_width, page_height
                                        ),
                                    )
                                    all_chunk_locations.append(location)
                                    secondary_idx += 1
                    else:
                        # Collect regular text chunks
                        if "html" in child:
                            page_text.append(child["html"])
                            location = ChunkLocation(
                                index=Index(
                                    primary=page_num + 1,
                                    secondary=secondary_idx + 1,
                                ),
                                extracted_feature_type=feature_type,
                                page_file_path=page_image_paths[page_num],
                                bounding_box=_polygon_to_bbox(
                                    child["polygon"], page_width, page_height
                                ),
                            )
                            all_chunk_locations.append(location)
                            secondary_idx += 1

                # After processing all children, create combined text Entry
                if page_text:
                    combined_page_text = " ".join(page_text)
                    page_entry_uuid = str(uuid.uuid4())
                    page_entry = Entry(
                        uuid=page_entry_uuid,
                        ingestion=ingestion,
                        string=combined_page_text,
                        consolidated_feature_type=ExtractedFeatureType.combined_text,
                        chunk_locations=all_chunk_locations,
                        min_primary_index=min(
                            loc.index.primary for loc in all_chunk_locations
                        ),
                        max_primary_index=max(
                            loc.index.primary for loc in all_chunk_locations
                        ),
                        chunk_index=chunk_idx + 1,
                    )
                    chunk_idx += 1
                    all_entries.append(page_entry)
                    all_text[page_num + 1] = combined_page_text

        elif mode == "by_title":
            chunk_idx = 0
            counters = {"figure_count": 0, "table_count": 0}
            
            # Variables for title-based chunking
            current_chunk_text = []
            current_chunk_locations = []
            all_text = {}
            
            for page_num, page in enumerate(parsed_data["children"]):
                if not page.get("children"):
                    continue

                # Get page dimensions
                if read:
                    img_bytes = await read(page_image_paths[page_num])
                    img = Image.open(io.BytesIO(img_bytes))
                    page_width, page_height = img.size
                    img.close()
                else:
                    with Image.open(page_image_paths[page_num]) as img:
                        page_width, page_height = img.size

                secondary_idx = 0
                for child in page["children"]:
                    block_type = child["block_type"]
                    feature_type = _map_marker_type(block_type)

                    # Process visual elements the same way as by_page mode
                    if feature_type in [
                        ExtractedFeatureType.figure,
                        ExtractedFeatureType.figuregroup,
                        ExtractedFeatureType.picture,
                        ExtractedFeatureType.picturegroup,
                        ExtractedFeatureType.complexregion,
                        ExtractedFeatureType.table,
                        ExtractedFeatureType.tablegroup,
                    ]:
                        entries, chunk_idx, secondary_idx, counters, html_content = (
                            await _process_child_element(
                                child=child,
                                parent_feature_type=feature_type,
                                page_num=page_num,
                                secondary_idx=secondary_idx,
                                ingestion=ingestion,
                                page_image_paths=page_image_paths,
                                page_dimensions=(page_width, page_height),
                                extracts_dir=extracts_dir,
                                chunk_idx=chunk_idx,
                                counters=counters,
                                read=read,
                                write=write,
                            )
                        )
                        all_entries.extend(entries)
                        current_chunk_text.append(html_content)
                    else:
                        # Check if we hit a new section header or page header
                        if feature_type in [ExtractedFeatureType.section_header, ExtractedFeatureType.header]:
                            # Process previous chunk if it exists
                            if current_chunk_text and current_chunk_locations:
                                entry, chunk_idx = await _process_title_chunk(
                                    current_chunk_text,
                                    current_chunk_locations,
                                    ingestion,
                                    chunk_idx
                                )
                                if entry:
                                    all_entries.append(entry)
                                    all_text[entry.min_primary_index] = entry.string
                                
                                # Reset accumulators
                                current_chunk_text = []
                                current_chunk_locations = []

                        # Add current text to chunk
                        if "html" in child:
                            current_chunk_text.append(child["html"])
                            location = ChunkLocation(
                                index=Index(
                                    primary=page_num + 1,
                                    secondary=secondary_idx + 1,
                                ),
                                extracted_feature_type=feature_type,
                                page_file_path=page_image_paths[page_num],
                                bounding_box=_polygon_to_bbox(
                                    child["polygon"], page_width, page_height
                                ),
                            )
                            current_chunk_locations.append(location)
                            secondary_idx += 1

                # Process any remaining text at the end of each page
                if current_chunk_text and current_chunk_locations:
                    entry, chunk_idx = await _process_title_chunk(
                        current_chunk_text,
                        current_chunk_locations,
                        ingestion,
                        chunk_idx
                    )
                    if entry:
                        all_entries.append(entry)
                        all_text[entry.min_primary_index] = entry.string

                    # Reset accumulators
                    current_chunk_text = []
                    current_chunk_locations = []

        # write the combined text to the file
        if write:
            await write(
                json.dumps(all_text, indent=4), ingestion.extracted_document_file_path
            )
        else:
            with open(
                ingestion.extracted_document_file_path, "w", encoding="utf-8"
            ) as f:
                f.write(json.dumps(all_text, indent=4))
    return all_entries


if __name__ == "__main__":
    from src.pipeline.storage_backend import LocalStorageBackend
    import asyncio
    import json

    storage_client = LocalStorageBackend(base_path="/tmp/marker_service")
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
    output = asyncio.run(main_datalab(test_ingestions, mode="by_page", visualize=True))
    with open("datalab_output.json", "w") as f:
        for entry in output:
            f.write(json.dumps(entry.model_dump(), indent=4))
            f.write("\n")

    title_output = asyncio.run(main_datalab(test_ingestions, mode="by_title", visualize=True))
    with open("title_output.json", "w") as f:
        for entry in title_output:
            f.write(json.dumps(entry.model_dump(), indent=4))
            f.write("\n")
