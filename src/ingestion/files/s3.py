import mimetypes
import os
import sys
from pathlib import Path
import io
import fitz  # PyMuPDF
import hashlib
import aioboto3
from botocore.exceptions import ClientError
from typing import Union

sys.path.append(str(Path(__file__).parents[3]))

from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import FileType, Ingestion, IngestionMethod, Scope
from src.utils.datetime_utils import get_current_utc_datetime, parse_datetime
from src.utils.ingestion_utils import update_ingestion_with_metadata

DEFAULT_CREATOR = "Astralis"
aws_access_key_id = os.environ.get("AWS_DATA_ACCESS_KEY")
aws_secret_access_key = os.environ.get("AWS_DATA_SECRET_ACCESS_KEY")
region_name = os.environ.get("S3_DATA_REGION", "us-west-1")


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
            elif sub_type in [
                "msword",
                "vnd.openxmlformats-officedocument.wordprocessingml.document",
            ]:
                return FileType.DOCX
            elif sub_type in [
                "vnd.ms-powerpoint",
                "vnd.openxmlformats-officedocument.presentationml.presentation",
            ]:
                return FileType.PPTX
            elif sub_type in [
                "vnd.ms-excel",
                "vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ]:
                return FileType.XLSX
    return FileType.TXT  # Default to TXT if unable to determine


async def create_ingestion_from_s3(
    session, bucket_name: str, s3_key: str, write=None
) -> Ingestion:
    # Get file info
    file_name = os.path.basename(s3_key)
    file_type = get_file_type(file_name)

    async with session.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    ) as s3:
        # Get object metadata
        response = await s3.head_object(Bucket=bucket_name, Key=s3_key)
        file_size = response["ContentLength"]

        # Download file content
        response = await s3.get_object(Bucket=bucket_name, Key=s3_key)
        async with response["Body"] as stream:
            file_content = await stream.read()

        # Handle file upload if write function provided
        if write:
            await write(file_name, file_content)

    # Calculate document hash
    document_hash = hashlib.sha256(file_content).hexdigest()

    # Extract PDF metadata if applicable
    document_metadata = {}
    if file_type == FileType.PDF:
        with fitz.open(stream=io.BytesIO(file_content), filetype="pdf") as pdf:
            document_metadata = pdf.metadata

    return Ingestion(
        document_hash=document_hash,
        document_title=file_name,
        scope=Scope.INTERNAL,
        content_type=None,
        creator_name=document_metadata.get("author", DEFAULT_CREATOR),
        creation_date=parse_datetime(response["LastModified"].timestamp()),
        file_type=file_type,
        file_path=s3_key,
        file_size=file_size,
        public_url=None,
        ingestion_method=IngestionMethod.S3,
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
        unprocessed_citations=None,
    )


@FunctionRegistry.register("ingest", "s3")
async def ingest_s3_folder(
    bucket_name: str,
    prefix: Union[str, list[str]] = "",
    ending_with: str = "",
    added_metadata: dict = {},
    write=None,
    batch_size: int = 10,
    **kwargs,
) -> list[Ingestion]:
    all_ingestions = []
    prefixes = prefix if isinstance(prefix, list) else [prefix]

    async def process_batch(s3_client, batch_keys):
        tasks = []
        for key in batch_keys:
            tasks.append(create_ingestion_from_s3(s3_client, bucket_name, key, write))
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            update_ingestion_with_metadata(result, added_metadata)
            for result in batch_results
            if not isinstance(result, Exception)
        ]

    session = aioboto3.Session()
    async with session.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    ) as s3:
        for current_prefix in prefixes:
            current_batch = []
            paginator = s3.get_paginator("list_objects_v2")

            async for page in paginator.paginate(
                Bucket=bucket_name, Prefix=current_prefix
            ):
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    key = obj["Key"]
                    # Skip if it's just a folder marker or doesn't match ending
                    if (key.endswith("/") and obj["Size"] == 0) or (
                        ending_with and not key.lower().endswith(ending_with.lower())
                    ):
                        continue

                    current_batch.append(key)

                    # Process batch when it reaches batch_size
                    if len(current_batch) >= batch_size:
                        try:
                            batch_results = await process_batch(s3, current_batch)
                            all_ingestions.extend(batch_results)
                        except Exception as e:
                            print(f"Error processing batch: {str(e)}")
                        current_batch = []

            # Process remaining items in the last batch
            if current_batch:
                try:
                    batch_results = await process_batch(s3, current_batch)
                    all_ingestions.extend(batch_results)
                except Exception as e:
                    print(f"Error processing final batch: {str(e)}")

    return all_ingestions


if __name__ == "__main__":
    import asyncio
    import sys

    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
        asyncio.run(ingest_s3_folder(directory_path))
    else:
        print("Please provide an absolute directory path as an argument.")
