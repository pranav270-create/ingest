import io
import json
import os
import sys
import uuid
from pathlib import Path

import fitz

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import (
    ChunkLocation,
    EmbeddedFeatureType,
    Entry,
    ExtractedFeatureType,
    ExtractionMethod,
    FileType,
    Index,
    Ingestion,
)
from src.utils.datetime_utils import get_current_utc_datetime, parse_pdf_date


async def process_pdf(
    file_content: bytes, ingestion: Ingestion, pages_dir: str, write=None
) -> tuple[list[Entry], dict[str, str]]:
    all_entries = []
    all_text = {}  # Changed to dict to store page number -> text mapping

    with fitz.open(stream=io.BytesIO(file_content), filetype="pdf") as pdf:
        # Set document metadata
        ingestion.document_metadata = pdf.metadata

        # Try multiple metadata fields for date
        date_fields = ["creationDate", "modDate", "created", "modified"]
        for field in date_fields:
            if pdf.metadata.get(field):
                parsed_date = parse_pdf_date(pdf.metadata[field])
                if parsed_date:
                    ingestion.creation_date = parsed_date
                    break

        # Set document title from metadata if available
        if pdf.metadata.get("title"):
            ingestion.document_title = pdf.metadata["title"]

        for i in range(pdf.page_count):
            page = pdf.load_page(i)
            page_text = page.get_text("text")
            all_text[str(i + 1)] = page_text  # Store text with page number as key

            # Generate and save page image
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
            img_bytes = pix.tobytes("jpg")
            page_image_path = f"{pages_dir}/page_{i + 1}.jpg"

            if write:
                # just sent the relative path
                page_image_path = os.path.relpath(page_image_path, pages_dir)
                await write(page_image_path, img_bytes)
            else:
                os.makedirs(os.path.dirname(page_image_path), exist_ok=True)
                with open(page_image_path, "wb") as f:
                    f.write(img_bytes)

            # Create proper chunk location with index and page image path
            chunk_location = ChunkLocation(
                index=Index(primary=i + 1),
                extracted_feature_type=ExtractedFeatureType.text,
                page_file_path=page_image_path,
            )

            entry = Entry(
                uuid=str(uuid.uuid4()),
                ingestion=ingestion,
                string=page_text,
                chunk_locations=[chunk_location],
                consolidated_feature_type=ExtractedFeatureType.text,
                min_primary_index=i + 1,
                max_primary_index=i + 1,
                chunk_index=i + 1,
                embedded_feature_type=EmbeddedFeatureType.TEXT,
                citations=[],
            )
            all_entries.append(entry)
    return all_entries, all_text


@FunctionRegistry.register("extract", "simple")
async def main_simple(
    ingestions: list[Ingestion], write=None, read=None, **kwargs
) -> list[Entry]:
    all_entries = []
    for ingestion in ingestions:
        if ingestion.file_type != FileType.PDF:
            continue

        print(f"Processing ingestion: {ingestion.file_path}")
        # Set required ingestion fields
        ingestion.extraction_method = ExtractionMethod.SIMPLE
        ingestion.extraction_date = get_current_utc_datetime()

        # Setup file paths
        base_name = os.path.basename(ingestion.file_path).replace(".pdf", "")
        pages_dir = f"pages/{base_name}"
        ingestion.extracted_document_file_path = f"{base_name}_parsed.json"

        # Process the PDF
        file_content = (
            await read(ingestion.file_path)
            if read
            else open(ingestion.file_path, "rb").read()
        )
        entries, all_text = await process_pdf(file_content, ingestion, pages_dir, write)

        # Write extracted text as JSON
        if write:
            await write(
                ingestion.extracted_document_file_path, json.dumps(all_text, indent=4)
            )
        else:
            with open(
                ingestion.extracted_document_file_path, "w", encoding="utf-8"
            ) as f:
                f.write(json.dumps(all_text, indent=4))

        all_entries.extend(entries)
    return all_entries
