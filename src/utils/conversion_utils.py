import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from src.schemas.schemas import ContentType, Entry, ExtractionMethod, FileType, Ingestion, IngestionMethod, Scope


def convert_txt_to_entries(corpus_directory: str) -> List[dict]:
    """
    Convert text files to Entry objects with Ingestion metadata, recursively through subdirectories
    """
    entries = []

    # Walk through all subdirectories
    for root, _, files in os.walk(corpus_directory):
        for filename in files:
            if not filename.endswith(".txt"):
                continue

            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, corpus_directory)

            # Read the text content
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Get dataset name from subdirectory
            dataset_name = Path(root).relative_to(corpus_directory).parts[0]

            # Create Ingestion metadata
            ingestion = Ingestion(
                document_title=filename,
                scope=Scope.EXTERNAL,
                content_type=ContentType.OTHER_LAW,
                creator_name=f"LegalBench-{dataset_name}",
                file_type=FileType.TXT,
                file_path=relative_path,  # Store relative path to maintain structure
                ingestion_method=IngestionMethod.LOCAL_FILE,
                ingestion_date=datetime.now().isoformat(),
                extraction_method=ExtractionMethod.SIMPLE,
            )

            # Create Entry object
            entry = Entry(uuid=str(uuid.uuid4()), ingestion=ingestion, string=content)

            # Convert to dict for JSON serialization
            entries.append(entry.model_dump())

    return entries


async def convert_and_upload_to_s3(corpus_directory: str, bucket_name: str, prefix: str = "legal_bench") -> None:
    """
    Convert text files to JSON and upload to S3, preserving directory structure
    """
    from src.utils.s3_utils import upload_folder_async

    # Create temporary directory for JSON files
    temp_dir = "temp_json"
    os.makedirs(temp_dir, exist_ok=True)

    # Convert files
    entries = convert_txt_to_entries(corpus_directory)

    # Group entries by dataset
    entries_by_dataset = {}
    for entry in entries:
        dataset = Path(entry["ingestion"]["file_path"]).parts[0]
        if dataset not in entries_by_dataset:
            entries_by_dataset[dataset] = []
        entries_by_dataset[dataset].append(entry)

    # Save JSONs by dataset
    for dataset, dataset_entries in entries_by_dataset.items():
        dataset_dir = os.path.join(temp_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        json_path = os.path.join(dataset_dir, f"{dataset}_entries.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(dataset_entries, f, indent=2, ensure_ascii=False)

    # Upload to S3
    await upload_folder_async(folder_path=temp_dir, bucket_name=bucket_name, prefix=prefix)

    # Cleanup
    for root, dirs, files in os.walk(temp_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(temp_dir)
