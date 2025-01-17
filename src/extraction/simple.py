import sys
from pathlib import Path
import fitz
import io
import os

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import Entry, Index, Ingestion, ExtractedFeatureType, ExtractionMethod, FileType
from src.utils.datetime_utils import get_current_utc_datetime, parse_pdf_date


def process_pdf(file_content: bytes, ingestion: Ingestion) -> tuple[list[Entry], str]:
    all_entries = []
    with fitz.open(stream=io.BytesIO(file_content), filetype="pdf") as pdf:
        ingestion.document_metadata = pdf.metadata
        # Try multiple metadata fields for date
        date_fields = ['creationDate', 'modDate', 'created', 'modified']
        for field in date_fields:
            if pdf.metadata.get(field):
                parsed_date = parse_pdf_date(pdf.metadata[field])
                if parsed_date:
                    ingestion.creation_date = parsed_date
                    break
        all_text = ""
        for i in range(pdf.page_count):
            page = pdf.load_page(i)
            page_text = page.get_text("text")
            all_text += page_text + "\n"
            entry = Entry(
                ingestion=ingestion,
                string=page_text,
                index_numbers=[Index(primary=i + 1)],  # Adjusting for 1-based indexing
                citations=None,
            )
            all_entries.append(entry)
    return all_entries, all_text


@FunctionRegistry.register("extract", "simple")
async def main_simple(ingestions: list[Ingestion], write=None, read=None, **kwargs) -> list[Entry]:
    all_entries = []
    for ingestion in ingestions:
        if ingestion.file_type != FileType.PDF:
            continue
        ingestion.extraction_method = ExtractionMethod.SIMPLE
        ingestion.extraction_date = get_current_utc_datetime()
        ingestion.parsed_feature_type = [ExtractedFeatureType.TEXT]
        ingestion.extracted_document_file_path = os.path.basename(ingestion.file_path).replace(".pdf", "_parsed.txt")
        file_content = await read(ingestion.file_path, mode="rb") if read else open(ingestion.file_path, "rb").read()
        entries, all_text = process_pdf(file_content, ingestion)
        if write:
            await write(ingestion.extracted_document_file_path, all_text, mode="w")
        else:
            with open(ingestion.extracted_document_file_path, "w") as f:
                f.write(all_text)
        all_entries.extend(entries)
    return all_entries
