"""
Author: Pranav Iyer
Date: 2024-10-02
Description: This script demonstrates how to ingest local files from a directory.
"""
import mimetypes
import os
import sys
from pathlib import Path
from rapidfuzz import process

sys.path.append(str(Path(__file__).parents[3]))

from src.schemas.schemas import ContentType, FileType, Ingestion, IngestionMethod, Scope
from src.pipeline.registry import FunctionRegistry
from src.utils.datetime_utils import get_current_utc_datetime, parse_datetime
from src.utils.ingestion_utils import update_ingestion_with_metadata


def get_content_type(file_path: str) -> ContentType:
    """
    Maps Google Drive folder name to ContentType based on file path.
    Uses fuzzy matching to find the best match for the folder name.
    """
    path_components = file_path.split(os.path.sep)
    try:
        content_index = path_components.index("Content")
        if len(path_components) > content_index + 1:
            target_folder = path_components[content_index + 1]
            # Normalize the target folder name: lowercase and replace spaces/special chars
            target_folder = target_folder.lower().replace(" ", "_").replace("-", "_")
            content_type_values = [ct.value for ct in ContentType]
            best_match, score, _ = process.extractOne(target_folder, content_type_values)
            print(f"Best match for {target_folder}: {best_match} with score {score}")
            if score >= 50:  # Increased threshold for more accurate matching
                return ContentType(best_match)
    except Exception:
        pass
    return ContentType.OTHER  # Explicit return for clarity


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
    content_type = get_content_type(file_path)
    # Generate cloud path (you might want to adjust this pattern)
    file_name = os.path.basename(file_path)
    # Read and upload file content if write function is provided
    if write:
        with open(file_path, 'rb') as f:
            file_content = f.read()
            await write(file_name, file_content)
    else:
        file_name = file_path  # Use the local path as the cloud path

    return Ingestion(
        document_title=file_name,
        scope=Scope.INTERNAL,
        content_type=content_type,
        file_type=file_type,
        file_path=file_name,  # Use the cloud path instead of local path
        public_url=None,
        creator_name="Astralis",
        total_length=os.path.getsize(file_path),
        metadata={},
        creation_date=parse_datetime(os.path.getctime(file_path)),
        ingestion_date=get_current_utc_datetime(),
        ingestion_method=IngestionMethod.LOCAL_FILE,
    )


@FunctionRegistry.register("ingest", "local")
async def ingest_local_files(directory_path: str, added_metadata: dict = {},  write=None, **kwargs) -> list[Ingestion]:
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
    import sys
    import asyncio
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
        asyncio.run(ingest_local_files(directory_path))
    else:
        print("Please provide an absolute directory path as an argument.")
