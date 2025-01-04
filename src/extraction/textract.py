import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from datetime import datetime

import boto3
import fitz
import trp

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.schemas.schemas import BoundingBox, ContentType, FileType, Index, Ingestion, Scope, Entry, IngestionMethod


def get_ingestion_data(pdf_path: str, scope: Scope, content_type: ContentType) -> dict[str, Any]:
    """
    Extract Ingestion data from the PDF file.
    """
    with fitz.open(pdf_path) as doc:
        metadata = doc.metadata
        total_length = sum(len(page.get_text()) for page in doc)

    ingestion_data = {
        "document_title": metadata.get("title", "Untitled"),
        "scope": scope,
        "content_type": content_type,
        "creator_name": metadata.get("author", "Unknown"),
        "file_type": FileType.PDF,
        "file_path": pdf_path.split("/")[-1],  # Get the filename from the path
        "total_length": total_length,
        "metadata": metadata,
        "ingestion_method": IngestionMethod.LOCAL_FILE,
        "ingestion_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    ingestion = Ingestion(**ingestion_data)
    return ingestion


def convert_pdf_to_images(pdf_path: str, output_dir: str) -> list[str]:
    """
    Convert a PDF file to images.
    """
    image_paths = []
    os.makedirs(output_dir, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase resolution
            image_path = f"{output_dir}/page_{i+1}.png"
            pix.save(image_path)
            image_paths.append(image_path)

    return image_paths


def aws_extract_page_content(image_path: str, page_number: int) -> dict[str, Any]:
    """
    Extract content from a single page image using AWS Textract.
    """
    textract_client = boto3.client(
        "textract",
        region_name=os.environ.get("AWS_REGION"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    response = textract_client.analyze_document(Document={"Bytes": image_bytes}, FeatureTypes=["TABLES", "LAYOUT"])

    doc = trp.Document(response)
    doc_pages, block_map = doc._parseDocumentPagesAndBlockMap()
    parsed_elements = []

    for page in doc.pages:
        combined_text = ""
        element_index = 1
        for block in page.blocks:
            bbox = block["Geometry"]["BoundingBox"]
            if block["BlockType"] == "LINE":
                combined_text += block["Text"] + " "
            elif block["BlockType"] == "LAYOUT_FIGURE":
                parsed_elements.append(
                    Entry(
                        string="figure",
                        index_numbers=[Index(primary=page_number, secondary=element_index)],
                        bounding_box=BoundingBox(left=bbox["Left"], top=bbox["Top"], width=bbox["Width"], height=bbox["Height"]),
                    )
                )
                element_index += 1
            elif block["BlockType"] == "LAYOUT_TABLE":
                parsed_elements.append(
                    Entry(
                        string="table",
                        index_numbers=[Index(primary=page_number, secondary=element_index)],
                        bounding_box=BoundingBox(left=bbox["Left"], top=bbox["Top"], width=bbox["Width"], height=bbox["Height"]),
                    )
                )
                element_index += 1
        parsed_elements.append(
            Entry(
                string=combined_text,
                index_numbers=[Index(primary=page_number, secondary=element_index)],
                # make this a random bounding box that is always first
                bounding_box=BoundingBox(left=0, top=0, width=0, height=0),
            )
        )
    return parsed_elements


def textract_parse(pdf_path: str, scope: Scope, content_type: ContentType) -> dict[str, Any]:
    """
    Process a PDF file: extract Ingestion data, convert to images, and extract content.
    """
    ingestion_data = get_ingestion_data(pdf_path, scope, content_type)
    image_paths = convert_pdf_to_images(pdf_path, "output")

    page_contents = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(aws_extract_page_content, image_path, i + 1) for i, image_path in enumerate(image_paths)]
        for future in futures:
            page_contents.extend(future.result())
    return ingestion_data, page_contents


if __name__ == "__main__":
    pdf_path = "ColbertV2.pdf"
    scope = Scope.EXTERNAL
    content_type = ContentType.OTHER_ARTICLES
    ingestion_data, page_contents = textract_parse(pdf_path, scope, content_type)
    print(ingestion_data)
    print(page_contents)
