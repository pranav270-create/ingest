import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from datetime import datetime
import json
import boto3
import fitz
import trp
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.schemas.schemas import BoundingBox, ContentType, FileType, Index, Ingestion, Scope, Entry, IngestionMethod, Document, ParsedFeatureType


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


def visualize_page_results(image_path: str, parsed_elements: list[Entry], output_path: str = None) -> None:
    """
    Visualize the parsed elements by drawing bounding boxes over the original image.
    """
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Get image dimensions
    width, height = image.size
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Colors for different types of elements
    colors = {
        "figure": "red",
        "table": "blue",
        "text": "green",
        "word": "yellow",
        "line": "orange",
        "title": "purple",
        "header": "cyan",
        "footer": "magenta",
        "section_header": "pink",
        "page_number": "brown",
        "list": "teal",
        "key_value": "lime",
        "combined_text": "white"  # Won't be visible since bounding box is 0
    }
    
    for element in parsed_elements:
        # Skip elements with zero bounding box (combined text)
        if element.bounding_box[0].width == 0 and element.bounding_box[0].height == 0:
            continue
            
        # Convert relative coordinates to absolute pixels
        left = element.bounding_box[0].left * width
        top = element.bounding_box[0].top * height
        right = left + (element.bounding_box[0].width * width)
        bottom = top + (element.bounding_box[0].height * height)
        
        # Get color based on block_type, default to white if not found
        color = colors.get(element.parsed_feature_type[0], "white")
            
        # Draw rectangle
        draw.rectangle([left, top, right, bottom], outline=color, width=2)
        
        # Add label with block type and text
        label = element.parsed_feature_type[0]
        draw.text((left, max(0, top-20)), label, fill=color, font=font)
    
    # Display or save the result
    if output_path:
        image.save(output_path)
    else:
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        plt.axis('off')
        plt.show()


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

    response = textract_client.analyze_document(Document={"Bytes": image_bytes}, FeatureTypes=["LAYOUT"])

    doc = trp.Document(response)
    doc_pages, block_map = doc._parseDocumentPagesAndBlockMap()
    # dump to json for debugging
    with open(f"output_textract/output_{page_number}.json", "w") as f:
        json.dump(doc_pages, f, indent=2)
    
    parsed_elements = []
    
    for page in doc.pages:
        combined_text = ""
        element_index = 1
        
        for block in page.blocks:
            bbox = block["Geometry"]["BoundingBox"]
            block_type = block["BlockType"].lower()  # Convert to lowercase for consistency
            
            # Handle different block types
            if block_type in ["line", "word"]:
                parsed_elements.append(
                    Entry(
                        string=block["Text"],
                        index_numbers=[Index(primary=page_number, secondary=element_index)],
                        bounding_box=[BoundingBox(
                            left=bbox["Left"], 
                            top=bbox["Top"], 
                            width=bbox["Width"], 
                            height=bbox["Height"]
                        )],
                        parsed_feature_type=[block_type]
                    )
                )
                if block_type == "line":
                    combined_text += block["Text"] + " "
                element_index += 1
            
            # Handle all LAYOUT types
            elif block_type.startswith("layout_"):
                layout_type = block_type.replace("layout_", "")
                display_text = block.get("Text", layout_type.upper())  # Use block text if available, otherwise type
                
                parsed_elements.append(
                    Entry(
                        string=display_text,
                        index_numbers=[Index(primary=page_number, secondary=element_index)],
                        bounding_box=[BoundingBox(
                            left=bbox["Left"], 
                            top=bbox["Top"], 
                            width=bbox["Width"], 
                            height=bbox["Height"]
                        )],
                        parsed_feature_type=[layout_type]
                    )
                )
                element_index += 1
        
        # Add combined text as a special entry
        parsed_elements.append(
            Entry(
                string=combined_text.strip(),
                index_numbers=[Index(primary=page_number, secondary=0)],
                bounding_box=[BoundingBox(left=0, top=0, width=0, height=0)],
                parsed_feature_type=[ParsedFeatureType.COMBINED_TEXT]
            )
        )
    
    # Visualize results
    output_path = f"output_textract/page_{page_number}_annotated.png"
    visualize_page_results(image_path, parsed_elements, output_path)
    
    return parsed_elements


def textract_parse(pdf_path: str, scope: Scope, content_type: ContentType) -> Document:
    """
    Process a PDF file: extract Ingestion data, convert to images, and extract content.
    Returns a Document object containing the ingestion metadata and parsed entries.
    """
    ingestion_data = get_ingestion_data(pdf_path, scope, content_type)
    image_paths = convert_pdf_to_images(pdf_path, "output_textract")

    entries = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(aws_extract_page_content, image_path, i + 1) for i, image_path in enumerate(image_paths)]
        for future in futures:
            page_entries = future.result()
            # Add ingestion data to each entry
            for entry in page_entries:
                entry.ingestion = ingestion_data
            entries.extend(page_entries)

    return Document(entries=entries)


if __name__ == "__main__":
    pdf_path = "/Users/kiaghods/Desktop/Academics/Princeton/Internships/astralis/astralisData/LegalRAG.pdf"
    pdf_path = "/Users/pranaviyer/Downloads/real-estate-market-update-data-centers-summer-2024.pdf"
    scope = Scope.EXTERNAL
    content_type = ContentType.OTHER_ARTICLES
    document = textract_parse(pdf_path, scope, content_type)
