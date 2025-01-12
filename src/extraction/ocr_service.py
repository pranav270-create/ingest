import sys
from pathlib import Path
import fitz
import modal
import io
from PIL import Image
import os

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry import FunctionRegistry
from src.schemas.schemas import Entry, Index, Ingestion, ParsedFeatureType, ParsingMethod, Scope, IngestionMethod, FileType
from src.utils.datetime_utils import get_current_utc_datetime, parse_pdf_date


def pdf_to_images(pdf_content):
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


@FunctionRegistry.register("parse", "ocr2_0")
async def main_ocr(ingestions: list[Ingestion], write=None, read=None, **kwargs) -> list[Entry]:
    cls = modal.Cls.lookup("ocr-modal", "Model")
    obj = cls()
    all_entries = []
    for ingestion in ingestions:
        # check if the ingestion is a PDF or an image
        if ingestion.file_type != FileType.PDF and ingestion.file_type not in [FileType.JPG, FileType.JPEG, FileType.PNG]:
            continue
        ingestion.parsing_method = ParsingMethod.OCR2_0
        ingestion.parsing_date = get_current_utc_datetime()
        ingestion.parsed_feature_type = [ParsedFeatureType.TEXT]

        # Set parsed_file_path (similar to simple.py)
        if not ingestion.parsed_file_path:
            base_path = os.path.splitext(ingestion.file_path)[0]
            ingestion.parsed_file_path = f"{base_path}_ocr.txt"

        file_content = await read(ingestion.file_path, mode="rb") if read else open(ingestion.file_path, "rb").read()
        # Determine if it's a PDF or an image
        is_pdf = file_content.startswith(b'%PDF')
        if is_pdf:
            with fitz.open(stream=io.BytesIO(file_content), filetype="pdf") as pdf:
                ingestion.metadata = pdf.metadata
                # Try multiple metadata fields for date
                date_fields = ['creationDate', 'modDate', 'created', 'modified']
                for field in date_fields:
                    if pdf.metadata.get(field):
                        parsed_date = parse_pdf_date(pdf.metadata[field])
                        if parsed_date:
                            ingestion.creation_date = parsed_date
                            break
            images = pdf_to_images(file_content)
        else:
            images = [Image.open(io.BytesIO(file_content))]
        # Process each image
        image_bytes_list = []
        for img in images:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            image_bytes_list.append((img_byte_arr.getvalue(), "format"))
        # Process images with OCR
        all_text = []  # Collect all text for saving to file
        count = 0
        async for ret in obj.ocr_document.map.aio(image_bytes_list, return_exceptions=True):
            if isinstance(ret, Exception):
                print(f"Error processing page: {ret}")
                continue
            page = Entry(
                ingestion=ingestion,
                string=ret,
                index_numbers=[Index(primary=count)]
            )
            count += 1
            all_entries.append(page)
            all_text.append(ret)

        # Save combined text to file
        combined_text = "\n\n=== PAGE BREAK ===\n\n".join(all_text)
        if write:
            await write(ingestion.parsed_file_path, combined_text, mode="w")
        else:
            with open(ingestion.parsed_file_path, "w", encoding='utf-8') as f:
                f.write(combined_text)
    return all_entries


@FunctionRegistry.register("parse", "ocr2_0_batch")
async def batch_ocr(ingestions: list[Ingestion], write=None, read=None, **kwargs) -> list[Entry]:
    cls = modal.Cls.lookup("ocr-modal", "Model")
    obj = cls()

    # Prepare all images and track their document mapping
    all_image_bytes = []
    image_to_doc_mapping = []  # List of tuples (ingestion_idx, page_number)
    all_entries = []

    # First pass: collect all images and build mapping
    for ing_idx, ingestion in enumerate(ingestions):
        if ingestion.file_type != FileType.PDF and ingestion.file_type not in [FileType.JPG, FileType.JPEG, FileType.PNG]:
            continue

        # Set up ingestion metadata
        ingestion.parsing_method = ParsingMethod.OCR2_0
        ingestion.parsing_date = get_current_utc_datetime()
        ingestion.parsed_feature_type = [ParsedFeatureType.TEXT]

        if not ingestion.parsed_file_path:
            base_path = os.path.splitext(ingestion.file_path)[0]
            ingestion.parsed_file_path = f"{base_path}_ocr.txt"

        # Get file content and process
        file_content = await read(ingestion.file_path, mode="rb") if read else open(ingestion.file_path, "rb").read()
        is_pdf = file_content.startswith(b'%PDF')

        if is_pdf:
            # Handle PDF metadata
            with fitz.open(stream=io.BytesIO(file_content), filetype="pdf") as pdf:
                ingestion.metadata = pdf.metadata
                for field in ['creationDate', 'modDate', 'created', 'modified']:
                    if pdf.metadata.get(field):
                        parsed_date = parse_pdf_date(pdf.metadata[field])
                        if parsed_date:
                            ingestion.creation_date = parsed_date
                            break
            images = pdf_to_images(file_content)
        else:
            images = [Image.open(io.BytesIO(file_content))]

        # Convert images to bytes and track mapping
        for page_num, img in enumerate(images):
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            all_image_bytes.append((img_byte_arr.getvalue(), "format"))
            image_to_doc_mapping.append((ing_idx, page_num))

    # Process all images in one batch
    document_texts = {ing_idx: [] for ing_idx in range(len(ingestions))}
    idx = 0
    async for ret in obj.ocr_document.map.aio(all_image_bytes, return_exceptions=True):
        ing_idx, page_num = image_to_doc_mapping[idx]
        if isinstance(ret, Exception):
            print(f"Error processing document {ing_idx}, page {page_num}: {ret}")
            idx += 1
            continue

        # Create entry and add to correct document
        entry = Entry(
            ingestion=ingestions[ing_idx],
            string=ret,
            index_numbers=[Index(primary=page_num)]
        )
        all_entries.append(entry)
        document_texts[ing_idx].append(ret)
        idx += 1

    # Save files and return
    for ing_idx, ingestion in enumerate(ingestions):
        if document_texts[ing_idx]:  # Only save if we have processed text
            combined_text = "\n\n=== PAGE BREAK ===\n\n".join(document_texts[ing_idx])
            if write:
                await write(ingestion.parsed_file_path, combined_text, mode="w")
            else:
                with open(ingestion.parsed_file_path, "w", encoding='utf-8') as f:
                    f.write(combined_text)
    return all_entries


if __name__ == "__main__":
    pass
