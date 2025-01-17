import mimetypes
import os
import sys
from pathlib import Path
import io
import fitz  # PyMuPDF
import hashlib
from typing import Optional

sys.path.append(str(Path(__file__).parents[3]))

from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import FileType, Ingestion, IngestionMethod, Scope
from src.utils.datetime_utils import get_current_utc_datetime, parse_datetime
from src.utils.ingestion_utils import update_ingestion_with_metadata

DEFAULT_CREATOR = "Astralis"


def get_file_type(file_path: str) -> FileType:
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        main_type, sub_type = mime_type.split("/")
        if main_type == "text":
            return FileType.TXT
        elif main_type == "image":
            return FileType[sub_type.upper()]
        elif main_type == "application":
            if sub_type == "pdf":
                return FileType.PDF
            elif sub_type in ["msword", "vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                return FileType.DOCX
            elif sub_type in ["vnd.ms-powerpoint", "vnd.openxmlformats-officedocument.presentationml.presentation"]:
                return FileType.PPTX
            elif sub_type in ["vnd.ms-excel", "vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                return FileType.XLSX
    return FileType.TXT  # Default to TXT if unable to determine


async def create_ingestion(file_path: str, write=None) -> Ingestion:
    file_type = get_file_type(file_path)
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    # Calculate document hash
    with open(file_path, 'rb') as f:
        file_content = f.read()
        document_hash = hashlib.sha256(file_content).hexdigest()

    # Extract PDF metadata if applicable
    document_metadata = {}
    if file_type == FileType.PDF:
        with fitz.open(stream=io.BytesIO(file_content), filetype="pdf") as pdf:
            document_metadata = pdf.metadata

    # Handle file upload if write function provided
    if write:
        await write(file_name, file_content)
    else:
        file_name = file_path  # Use local path as cloud path

    return Ingestion(
        document_hash=document_hash,
        document_title=file_name,
        scope=Scope.INTERNAL,
        content_type=None,  # Will be inferred later in pipeline or updated in added_metadata
        creator_name=document_metadata.get('author', DEFAULT_CREATOR),
        creation_date=parse_datetime(os.path.getctime(file_path)),
        file_type=file_type,
        file_path=file_name,
        file_size=file_size,
        public_url=None,
        ingestion_method=IngestionMethod.LOCAL_FILE,
        ingestion_date=get_current_utc_datetime(),
        document_metadata=document_metadata,
        # Fields that will be populated later in pipeline
        document_summary=None,
        document_keywords=None,
        extraction_method=None,
        extraction_date=None,
        extracted_document_file_path=None,
        chunking_method=None,
        chunking_metadata=None,
        chunking_date=None,
        feature_models=None,
        feature_dates=None,
        feature_types=None,
        unprocessed_citations=None
    )


@FunctionRegistry.register("ingest", "local")
async def ingest_local_files(directory_path: str, added_metadata: dict = None, write=None, **kwargs) -> list[Ingestion]:
    if not os.path.isabs(directory_path):
        raise ValueError("The provided path must be an absolute path.")

    all_ingestions = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.startswith("."):
                continue
            file_path = os.path.join(root, file)
            ingestion = await create_ingestion(file_path, write)
            ingestion = update_ingestion_with_metadata(ingestion, added_metadata)
            all_ingestions.append(ingestion)
    return all_ingestions


if __name__ == "__main__":
    import asyncio
    import sys
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
        asyncio.run(ingest_local_files(directory_path))
    else:
        print("Please provide an absolute directory path as an argument.")
