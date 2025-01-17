import sys
from pathlib import Path
import fitz
import io
import os

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import (
    Entry,
    Index,
    Ingestion,
    ExtractedFeatureType,
    ExtractionMethod,
    FileType,
    ChunkLocation,
    EmbeddedFeatureType,
)
from src.utils.datetime_utils import get_current_utc_datetime, parse_pdf_date


def process_pdf(file_content: bytes, ingestion: Ingestion) -> tuple[list[Entry], str]:
    all_entries = []
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

        all_text = ""
        for i in range(pdf.page_count):
            page = pdf.load_page(i)
            page_text = page.get_text("text")
            all_text += page_text + "\n"

            # Create proper chunk location with index
            chunk_location = ChunkLocation(
                index=Index(primary=i + 1),
                extracted_feature_type=ExtractedFeatureType.text,
            )

            entry = Entry(
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

        # Set required ingestion fields
        ingestion.extraction_method = ExtractionMethod.SIMPLE
        ingestion.extraction_date = get_current_utc_datetime()
        ingestion.parsed_feature_type = [ExtractedFeatureType.text]
        ingestion.extracted_document_file_path = os.path.basename(
            ingestion.file_path
        ).replace(".pdf", "_parsed.txt")

        # Process the PDF
        file_content = (
            await read(ingestion.file_path, mode="rb")
            if read
            else open(ingestion.file_path, "rb").read()
        )
        entries, all_text = process_pdf(file_content, ingestion)

        # Write extracted text
        if write:
            await write(ingestion.extracted_document_file_path, all_text, mode="w")
        else:
            with open(ingestion.extracted_document_file_path, "w") as f:
                f.write(all_text)

        all_entries.extend(entries)
    return all_entries
