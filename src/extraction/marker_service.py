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
import matplotlib.pyplot as plt
import uuid

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import (
    Entry,
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


def _polygon_to_bbox(polygon):
    """Convert polygon coordinates to bounding box format."""
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]

    return {
        "left": min(x_coords),
        "top": min(y_coords),
        "width": max(x_coords) - min(x_coords),
        "height": max(y_coords) - min(y_coords),
    }


def _visualize_page_results(
    image_path: str, parsed_elements: list, output_path: str
) -> None:
    """
    Visualize the parsed elements by drawing bounding boxes over the original image.
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

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except Exception as e:
        print(f"Error loading font: {e}")
        font = ImageFont.load_default()

    for element in parsed_elements:
        bbox = _polygon_to_bbox(element["polygon"])
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

    image.save(output_path)


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
            ingestion.extracted_document_file_path = f"{base_path}_datalab.txt"

        # Update file reading to handle async
        file_content = (
            await read(ingestion.file_path, mode="rb")
            if read
            else open(ingestion.file_path, "rb").read()
        )

        # Convert to PDF if necessary
        file_extension = Path(ingestion.file_path).suffix.lower()
        pdf_content = convert_to_pdf(file_content, file_extension)
        file_bytes.append(pdf_content)

        # Extract metadata
        with fitz.open(stream=io.BytesIO(pdf_content), filetype="pdf") as pdf:
            ingestion.document_metadata = pdf.metadata

    all_entries = []
    async for ret in obj.parse_document.map.aio(file_bytes, return_exceptions=True):
        if isinstance(ret, Exception):
            print(f"Error processing document: {ret}")
            continue

        parsed_data = json.loads(ret)
        with open("parsed_data.json", "w") as f:
            f.write(json.dumps(parsed_data, indent=4))

        # If visualization is requested, process each page
        if visualize:
            output_dir = os.path.join(
                os.path.dirname(ingestion.extracted_document_file_path), "annotated"
            )
            os.makedirs(output_dir, exist_ok=True)

            # Convert PDF to images and annotate them
            with fitz.open(stream=io.BytesIO(pdf_content), filetype="pdf") as doc:
                for page_num, page in enumerate(parsed_data["children"]):
                    # Convert page to image
                    pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(1, 1))
                    temp_image = f"{output_dir}/temp_page_{page_num + 1}.png"
                    pix.save(temp_image)

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

                    # Create annotated version
                    output_path = f"{output_dir}/page_{page_num + 1}_annotated.png"
                    _visualize_page_results(temp_image, elements, output_path)
                    os.remove(temp_image)  # Clean up temporary image

        all_text = []
        if mode == "simple":
            # Process each page and combine all text into a single Entry
            for page_num, page in enumerate(parsed_data["children"]):
                page_text = []
                if page.get("children"):
                    for child in page["children"]:
                        if "html" in child:
                            # Extract text from HTML, removing HTML tags
                            text = child["html"].replace("<p>", "").replace("</p>", "").replace("<h1>", "").replace("</h1>", "").replace("<h2>", "").replace("</h2>", "").replace("<h4>", "").replace("</h4>", "")
                            if text.strip():  # Only add non-empty text
                                page_text.append(text)

                # Create one Entry per page with all text combined
                combined_page_text = " ".join(page_text)
                page_entry = Entry(
                    uuid=str(uuid.uuid4()),
                    ingestion=ingestion,
                    string=combined_page_text,
                    index_numbers=[Index(primary=page_num + 1)],
                    metadata={
                        "page_number": page_num + 1,
                        "block_type": "Page"
                    }
                )
                all_entries.append(page_entry)
                all_text.append(combined_page_text)

        else:  # mode == "structured"
            # New behavior: create entries based on semantic blocks
            for page_num, page in enumerate(parsed_data["children"]):
                for child in page.get("children", []):
                    block_type = child.get("block_type")
                    if "text" in child:
                        entry = Entry(
                            uuid=str(uuid.uuid4()),
                            ingestion=ingestion,
                            string=child["text"],
                            index_numbers=[Index(primary=page_num + 1)],
                            metadata={
                                "block_type": block_type,
                                "polygon": child.get("polygon"),
                                "confidence": child.get("confidence"),
                            },
                        )
                        all_entries.append(entry)
                        all_text.append(child["text"])

        # Save combined text to file
        combined_text = "\n\n=== PAGE BREAK ===\n\n".join(all_text)
        if write:
            await write(ingestion.extracted_document_file_path, combined_text, mode="w")
        else:
            with open(ingestion.extracted_document_file_path, "w", encoding="utf-8") as f:
                f.write(combined_text)

    return all_entries


if __name__ == "__main__":
    # Create sample ingestions with only required fields
    test_ingestions = [
        Ingestion(
            scope=Scope.INTERNAL,  # Required
            creator_name="Test User",  # Required
            ingestion_method=IngestionMethod.LOCAL_FILE,  # Required
            file_type=FileType.PDF,
            ingestion_date="2024-03-20T12:00:00Z",  # Required
            file_path="/Users/pranaviyer/Downloads/TR0722-315a Appendix A.pdf"
        ),
        # Ingestion(
        #     scope=Scope.INTERNAL,
        #     creator_name="Test User",
        #     ingestion_method=IngestionMethod.LOCAL_FILE,
        #     ingestion_date="2024-03-20T12:00:00Z",
        #     file_path="/Users/pranaviyer/Desktop/AstralisData/TR0722-315a Appendix A.pdf"
        # )
    ]
    # Run the batch OCR
    import asyncio
    output = asyncio.run(main_datalab(test_ingestions, mode="simple", visualize=True))
    import json
    with open("datalab_output.json", "w") as f:
        for entry in output:
            f.write(json.dumps(entry.model_dump(), indent=4))
            f.write("\n")
