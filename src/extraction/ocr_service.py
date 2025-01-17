import sys
from pathlib import Path
import fitz
import modal
import io
from PIL import Image
import os
import uuid

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import (
    Entry,
    Index,
    Ingestion,
    ExtractionMethod,
    Scope,
    IngestionMethod,
    FileType,
    ChunkLocation,
    ExtractedFeatureType
)
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


@FunctionRegistry.register("extract", "ocr2_0")
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
        ingestion.extraction_method = ExtractionMethod.OCR2_0
        ingestion.extraction_date = get_current_utc_datetime()

        if not ingestion.extracted_document_file_path:
            base_path = os.path.splitext(ingestion.file_path)[0]
            ingestion.extracted_document_file_path = f"{base_path}_ocr.txt"

        # Get file content and process
        file_content = await read(ingestion.file_path) if read else open(ingestion.file_path, "rb").read()
        is_pdf = file_content.startswith(b'%PDF')

        if is_pdf:
            # Handle PDF metadata
            with fitz.open(stream=io.BytesIO(file_content), filetype="pdf") as pdf:
                ingestion.document_metadata = pdf.metadata
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
            uuid=str(uuid.uuid4()),
            ingestion=ingestions[ing_idx],
            string=ret,
            consolidated_feature_type=ExtractedFeatureType.text,
            chunk_locations=[ChunkLocation(
                index=Index(primary=page_num + 1),
                page_file_path=ingestions[ing_idx].file_path,
                extracted_feature_type=ExtractedFeatureType.text
            )],
            min_primary_index=page_num + 1,
            max_primary_index=page_num + 1,
            chunk_index=idx + 1,
        )
        all_entries.append(entry)
        document_texts[ing_idx].append(ret)
        idx += 1

    # Save files and return
    for ing_idx, ingestion in enumerate(ingestions):
        if document_texts[ing_idx]:  # Only save if we have processed text
            combined_text = "\n\n=== PAGE BREAK ===\n\n".join(document_texts[ing_idx])
            if write:
                await write(ingestion.extracted_document_file_path, combined_text)
            else:
                with open(ingestion.extracted_document_file_path, "w", encoding='utf-8') as f:
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
    output = asyncio.run(batch_ocr(test_ingestions))
    import json
    with open("output.json", "w") as f:
        for entry in output:
            f.write(json.dumps(entry.model_dump(), indent=4))
            f.write("\n")
