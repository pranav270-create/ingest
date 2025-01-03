import sys
from pathlib import Path
import fitz
import modal
import io
import os
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry import FunctionRegistry
from src.schemas.schemas import Document, Entry, Index, Ingestion, ParsedFeatureType, ParsingMethod, Scope, IngestionMethod, FileType
from src.utils.datetime_utils import get_current_utc_datetime, parse_pdf_date


def process_pdf(file_content, ingestion):
    document = Document(entries=[])
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
        total_length = 0
        all_text = ""
        for i in range(pdf.page_count):
            page = pdf.load_page(i)
            page_text = page.get_text("text")
            total_length += len(page_text)
            all_text += page_text + "\n"
            entry = Entry(
                ingestion=ingestion,
                string=page_text,
                index_numbers=[Index(primary=i + 1)],  # Adjusting for 1-based indexing
                citations=None,
            )
            document.entries.append(entry)
    for entries in document.entries:
        entries.ingestion.total_length = total_length
    return document, all_text


@FunctionRegistry.register("parse", "simple")
async def main_simple(ingestions: list[Ingestion], write=None, read=None, **kwargs) -> list[Document]:
    all_documents = []
    for ingestion in ingestions:
        if ingestion.file_type != FileType.PDF:
            continue
        ingestion.parsing_method = ParsingMethod.SIMPLE
        ingestion.parsing_date = get_current_utc_datetime()
        ingestion.parsed_feature_type = ParsedFeatureType.TEXT
        ingestion.parsed_file_path = os.path.basename(ingestion.file_path).replace(".pdf", "_parsed.txt")
        file_content = await read(ingestion.file_path, mode="rb") if read else open(ingestion.file_path, "rb").read()
        document, all_text = process_pdf(file_content, ingestion)
        if write:
            await write(ingestion.parsed_file_path, all_text, mode="w")
        else:
            with open(ingestion.parsed_file_path, "w") as f:
                f.write(all_text)
        all_documents.append(document)
    return all_documents
