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
        "tablegroup": ExtractedFeatureType.table,  # Map to common table type
        "figure": ExtractedFeatureType.figure,
        "figuregroup": ExtractedFeatureType.figure,  # Map to common figure type
        "code": ExtractedFeatureType.code,
        "equation": ExtractedFeatureType.equation,
        "form": ExtractedFeatureType.form,
        "pageheader": ExtractedFeatureType.header,
        "pagefooter": ExtractedFeatureType.footer,
        "sectionheader": ExtractedFeatureType.section_header,
        "listitem": ExtractedFeatureType.list,
        "listgroup": ExtractedFeatureType.list,
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
    # Return common type if exists, otherwise use the original type
    return marker_to_common.get(block_type_lower, ExtractedFeatureType.other)


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


def _clean_html_text(html: str) -> str:
    """Clean HTML text for basic text content types"""
    # Remove all HTML tags and their content if they're refs
    text = (
        html.replace("<content-ref", "\n")  # Start new line for content refs
        .replace("</content-ref>", "\n")    # End content ref
        # Remove common HTML tags
        .replace("<p>", "")
        .replace("</p>", "")
        .replace("<h1>", "")
        .replace("</h1>", "")
        .replace("<h2>", "")
        .replace("</h2>", "")
        .replace("<h3>", "")
        .replace("</h3>", "")
        .replace("<h4>", "")
        .replace("</h4>", "")
    )
    # Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Clean up whitespace
    text = (
        text.replace('\u00a0', ' ')  # Replace non-breaking spaces
        .replace('\u200b', '')       # Remove zero-width spaces
    )
    # Normalize whitespace: multiple spaces/newlines to single
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _process_special_block(child, feature_type):
    """Helper to process special blocks like tables and figures with their captions"""
    caption_text = ""
    main_content = ""
    # If this is a group with children, process them
    if child.get("children"):
        for sub_child in child["children"]:
            sub_type = _map_marker_type(sub_child["block_type"])
            if sub_type == ExtractedFeatureType.caption:
                caption_text = _clean_html_text(sub_child["html"])
            elif sub_type in [ExtractedFeatureType.table, ExtractedFeatureType.list]:
                # Preserve original HTML for structured content
                main_content = sub_child["html"]
            else:
                # For other types, clean the text
                main_content = _clean_html_text(sub_child["html"])
    else:
        # If no children, process based on type
        if feature_type in [
            ExtractedFeatureType.table,
            ExtractedFeatureType.tablegroup,
            ExtractedFeatureType.list,
            ExtractedFeatureType.listgroup
        ]:
            main_content = child["html"]  # Preserve structure
        elif feature_type != ExtractedFeatureType.page:  # Ignore page type
            main_content = _clean_html_text(child["html"])
    return main_content, caption_text


@FunctionRegistry.register("parse", "datalab")
async def main_datalab(
    ingestions: list[Ingestion],
    write=None,
    read=None,
    mode="simple",
    visualize=False,
    **kwargs,
) -> list[Entry]:
    """
    Parse documents using the Marker library.

    Args:
        ingestions: List of Ingestion objects
        write: Optional async write function
        read: Optional async read function
        mode: "simple" or "structured" parsing mode
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
        with open("parsed_data.json", "w") as f:
            json.dump(parsed_data, f, indent=4)
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
        if mode == "simple":
            chunk_idx = 0
            figure_count = 0  # Track figure numbers
            table_count = 0   # Track table numbers
            
            for page_num, page in enumerate(parsed_data["children"]):
                page_text = []
                all_chunk_locations = []
                if page.get("children"):
                    # Get page dimensions by reading the image
                    if read:
                        img_bytes = await read(page_image_paths[page_num])
                        img = Image.open(io.BytesIO(img_bytes))
                        page_width, page_height = img.size
                        img.close()
                    else:
                        with Image.open(page_image_paths[page_num]) as img:
                            page_width, page_height = img.size

                    for secondary_index, child in enumerate(page["children"]):
                        if "html" in child:
                            text = (
                                child["html"]
                                .replace("<p>", "")
                                .replace("</p>", "")
                                .replace("<h1>", "")
                                .replace("</h1>", "")
                                .replace("<h2>", "")
                                .replace("</h2>", "")
                                .replace("<h4>", "")
                                .replace("</h4>", "")
                            ).strip()
                            
                            if text.strip():
                                page_text.append(text)
                                extracted_path = None
                                feature_type = _map_marker_type(child["block_type"])

                                if feature_type in [
                                    ExtractedFeatureType.table,
                                    ExtractedFeatureType.tablegroup,
                                    ExtractedFeatureType.figure,
                                    ExtractedFeatureType.figuregroup,
                                    ExtractedFeatureType.picture,
                                    ExtractedFeatureType.picturegroup,
                                    ExtractedFeatureType.complexregion,
                                ]:
                                    extracted_path, bbox = await _extract_region(
                                        child=child,
                                        page_num=page_num,
                                        secondary_index=secondary_index,
                                        page_image_path=page_image_paths[page_num],
                                        page_dimensions=(page_width, page_height),
                                        extracts_dir=extracts_dir,
                                        feature_type=feature_type,
                                        read=read,
                                        write=write,
                                    )

                                    main_content, caption_text = _process_special_block(
                                        child, feature_type
                                    )

                                    if feature_type in [ExtractedFeatureType.figure, ExtractedFeatureType.figuregroup]:
                                        figure_count += 1
                                        entry = Entry(
                                            uuid=str(uuid.uuid4()),
                                            ingestion=ingestion,
                                            string="",  # Empty for figures
                                            description=caption_text or main_content,  # Use caption if available, else content
                                            figure_number=figure_count,
                                            consolidated_feature_type=feature_type,
                                            chunk_locations=[
                                                ChunkLocation(
                                                    index=Index(
                                                        primary=page_num + 1,
                                                        secondary=secondary_index + 1,
                                                    ),
                                                    extracted_feature_type=feature_type,
                                                    page_file_path=page_image_paths[page_num],
                                                    extracted_file_path=extracted_path,
                                                    bounding_box=bbox,
                                                )
                                            ],
                                            chunk_index=chunk_idx + 1,
                                            title=f"Figure {figure_count}"
                                        )
                                    elif feature_type in [ExtractedFeatureType.table, ExtractedFeatureType.tablegroup]:
                                        table_count += 1
                                        entry = Entry(
                                            uuid=str(uuid.uuid4()),
                                            ingestion=ingestion,
                                            string=main_content,  # Keep table content in string
                                            description=caption_text,  # Add caption as description
                                            table_number=table_count,
                                            consolidated_feature_type=feature_type,
                                            chunk_locations=[
                                                ChunkLocation(
                                                    index=Index(
                                                        primary=page_num + 1,
                                                        secondary=secondary_index + 1,
                                                    ),
                                                    extracted_feature_type=feature_type,
                                                    page_file_path=page_image_paths[page_num],
                                                    extracted_file_path=extracted_path,
                                                    bounding_box=bbox,
                                                )
                                            ],
                                            chunk_index=chunk_idx + 1,
                                            title=f"Table {table_count}"
                                        )
                                    else:
                                        # Handle other special blocks
                                        entry = Entry(
                                            uuid=str(uuid.uuid4()),
                                            ingestion=ingestion,
                                            string=text,
                                            consolidated_feature_type=feature_type,
                                            chunk_locations=[
                                                ChunkLocation(
                                                    index=Index(
                                                        primary=page_num + 1,
                                                        secondary=secondary_index + 1,
                                                    ),
                                                    extracted_feature_type=feature_type,
                                                    page_file_path=page_image_paths[page_num],
                                                    extracted_file_path=extracted_path,
                                                    bounding_box=bbox,
                                                )
                                            ],
                                            chunk_index=chunk_idx + 1,
                                        )
                                    
                                    chunk_idx += 1
                                    all_entries.append(entry)
                                else:
                                    all_chunk_locations.append(
                                        ChunkLocation(
                                            index=Index(
                                                primary=page_num + 1,
                                                secondary=secondary_index + 1,
                                            ),
                                            extracted_feature_type=feature_type,
                                            page_file_path=page_image_paths[page_num],
                                            extracted_file_path=extracted_path,
                                            bounding_box=_polygon_to_bbox(
                                                child["polygon"],
                                                page_width,
                                                page_height,
                                            ),
                                        )
                                    )

                # Create one Entry per page other than special blocks
                combined_page_text = " ".join(page_text)
                if all_chunk_locations:  # Only create page entry if there are text chunks
                    page_entry = Entry(
                        uuid=str(uuid.uuid4()),
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

        elif mode == "title":
            current_section = []
            current_section_locations = []
            current_header = None
            chunk_idx = 0
            figure_count = 0  # Track figure numbers
            table_count = 0   # Track table numbers
            max_chunk_size = kwargs.get("max_chunk_size", 1000)

            def create_section_chunks(
                text: str, locations: list, header: str | None
            ) -> list[Entry]:
                """Helper function to create chunks from section text"""
                if not text.strip() or not locations:
                    return []

                words = text.split()
                chunks = []
                current_chunk = []
                current_size = 0

                for word in words:
                    new_size = current_size + len(word) + (1 if current_chunk else 0)
                    if new_size > max_chunk_size and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [word]
                        current_size = len(word)
                    else:
                        current_chunk.append(word)
                        current_size = new_size

                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                nonlocal chunk_idx
                entries = []
                for chunk_text in chunks:
                    chunk_idx += 1
                    entries.append(
                        Entry(
                            uuid=str(uuid.uuid4()),
                            ingestion=ingestion,
                            string=chunk_text,
                            consolidated_feature_type=ExtractedFeatureType.section_text,
                            chunk_locations=locations,
                            min_primary_index=min(
                                loc.index.primary for loc in locations
                            ),
                            max_primary_index=max(
                                loc.index.primary for loc in locations
                            ),
                            title=header,
                            chunk_index=chunk_idx,
                        )
                    )
                return entries

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

                for secondary_index, child in enumerate(page["children"]):
                    if "html" not in child:
                        continue

                    # Clean text and get feature type
                    text = (
                        child["html"]
                        .replace("<p>", "")
                        .replace("</p>", "")
                        .replace("<h1>", "")
                        .replace("</h1>", "")
                        .replace("<h2>", "")
                        .replace("</h2>", "")
                        .replace("<h4>", "")
                        .replace("</h4>", "")
                    ).strip()

                    if not text:
                        continue

                    feature_type = _map_marker_type(child["block_type"])

                    # Handle special blocks first
                    if feature_type in [
                        ExtractedFeatureType.table,
                        ExtractedFeatureType.tablegroup,
                        ExtractedFeatureType.figure,
                        ExtractedFeatureType.figuregroup,
                        ExtractedFeatureType.picture,
                        ExtractedFeatureType.picturegroup,
                        ExtractedFeatureType.complexregion,
                    ]:
                        extracted_path, bbox = await _extract_region(
                            child=child,
                            page_num=page_num,
                            secondary_index=secondary_index,
                            page_image_path=page_image_paths[page_num],
                            page_dimensions=(page_width, page_height),
                            extracts_dir=extracts_dir,
                            feature_type=feature_type,
                            read=read,
                            write=write,
                        )

                        main_content, caption_text = _process_special_block(
                            child, feature_type
                        )

                        chunk_idx += 1
                        # Increment counters and set specific fields
                        if feature_type in [ExtractedFeatureType.figure, ExtractedFeatureType.figuregroup]:
                            figure_count += 1
                            entry = Entry(
                                uuid=str(uuid.uuid4()),
                                ingestion=ingestion,
                                string="",  # Empty for figures
                                description=caption_text or main_content,  # Use caption if available, else content
                                figure_number=figure_count,
                                consolidated_feature_type=feature_type,
                                chunk_locations=[
                                    ChunkLocation(
                                        index=Index(
                                            primary=page_num + 1,
                                            secondary=secondary_index + 1,
                                        ),
                                        extracted_feature_type=feature_type,
                                        page_file_path=page_image_paths[page_num],
                                        extracted_file_path=extracted_path,
                                        bounding_box=bbox,
                                    )
                                ],
                                chunk_index=chunk_idx,
                                title=f"Figure {figure_count}"  # Add figure title
                            )
                        elif feature_type in [ExtractedFeatureType.table, ExtractedFeatureType.tablegroup]:
                            table_count += 1
                            entry = Entry(
                                uuid=str(uuid.uuid4()),
                                ingestion=ingestion,
                                string=main_content,  # Keep table content in string
                                description=caption_text,  # Add caption as description
                                table_number=table_count,
                                consolidated_feature_type=feature_type,
                                chunk_locations=[
                                    ChunkLocation(
                                        index=Index(
                                            primary=page_num + 1,
                                            secondary=secondary_index + 1,
                                        ),
                                        extracted_feature_type=feature_type,
                                        page_file_path=page_image_paths[page_num],
                                        extracted_file_path=extracted_path,
                                        bounding_box=bbox,
                                    )
                                ],
                                chunk_index=chunk_idx,
                                title=f"Table {table_count}"  # Add table title
                            )
                        else:
                            # Handle other special blocks
                            entry = Entry(
                                uuid=str(uuid.uuid4()),
                                ingestion=ingestion,
                                string=text,
                                consolidated_feature_type=feature_type,
                                chunk_locations=[
                                    ChunkLocation(
                                        index=Index(
                                            primary=page_num + 1,
                                            secondary=secondary_index + 1,
                                        ),
                                        extracted_feature_type=feature_type,
                                        page_file_path=page_image_paths[page_num],
                                        extracted_file_path=extracted_path,
                                        bounding_box=bbox,
                                    )
                                ],
                                chunk_index=chunk_idx
                            )
                        
                        all_entries.append(entry)
                        continue

                    # Handle headers and text
                    is_header = feature_type in [
                        ExtractedFeatureType.section_header,
                        ExtractedFeatureType.header,
                    ]

                    if is_header:
                        # Process previous section before starting new one
                        if current_section:
                            section_entries = create_section_chunks(
                                " ".join(current_section),
                                current_section_locations,
                                current_header,
                            )
                            all_entries.extend(section_entries)

                        # Update header and reset section
                        current_header = text
                        current_section = []
                        current_section_locations = []
                    else:
                        # Add text to current section
                        current_section.append(text)
                        current_section_locations.append(
                            ChunkLocation(
                                index=Index(
                                    primary=page_num + 1,
                                    secondary=secondary_index + 1,
                                ),
                                extracted_feature_type=ExtractedFeatureType.text,
                                page_file_path=page_image_paths[page_num],
                                bounding_box=_polygon_to_bbox(
                                    child["polygon"],
                                    page_width,
                                    page_height,
                                ),
                            )
                        )

                        # Check if current section exceeds max size
                        current_text = " ".join(current_section)
                        if len(current_text) > max_chunk_size:
                            section_entries = create_section_chunks(
                                current_text, current_section_locations, current_header
                            )
                            all_entries.extend(section_entries)
                            current_section = []
                            current_section_locations = []

            # Process final section if it exists
            if current_section:
                final_entries = create_section_chunks(
                    " ".join(current_section), current_section_locations, current_header
                )
                all_entries.extend(final_entries)
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
    output = asyncio.run(main_datalab(test_ingestions, mode="simple", visualize=True))
    with open("datalab_output.json", "w") as f:
        for entry in output:
            f.write(json.dumps(entry.model_dump(), indent=4))
            f.write("\n")
