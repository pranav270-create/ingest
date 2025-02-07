import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

import asyncio
import json
import logging
import os
from typing import Dict, List

import aioboto3

from src.utils.s3_utils import upload_single_file_async

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


async def upload_omnibench_to_s3(base_path: str, bucket_name: str, prefix: str = "omnibench", max_concurrency: int = 10) -> None:
    """
    Upload OmniBench PDFs to S3 organized by document type with a global metadata file
    """
    # Load metadata
    json_path = os.path.join(base_path, "OmniDocBench.json")
    logging.info(f"Loading metadata from {json_path}")
    with open(json_path, encoding="utf-8") as f:
        metadata = json.load(f)
    logging.info(f"Found {len(metadata)} entries in metadata file")

    # Create mapping of files to document types and build global metadata
    file_mapping: Dict[str, List[str]] = {}
    global_metadata: Dict[str, dict] = {}

    for entry in metadata:
        doc_type = entry["page_info"]["page_attribute"]["data_source"]
        image_path = entry["page_info"]["image_path"]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        pdf_path = os.path.join(base_path, "pdfs", f"{base_name}.pdf")

        if doc_type not in file_mapping:
            file_mapping[doc_type] = []

        s3_key = f"{prefix}/{doc_type}/{base_name}.pdf"
        file_mapping[doc_type].append(pdf_path)
        global_metadata[s3_key] = entry

    # Upload files
    session = aioboto3.Session()
    tasks = []
    semaphore = asyncio.Semaphore(max_concurrency)
    stats = {"uploaded": 0, "failed": 0, "missing": 0}

    async def upload_file(pdf_path: str, doc_type: str):
        async with semaphore:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            s3_key = f"{prefix}/{doc_type}/{base_name}.pdf"

            if os.path.exists(pdf_path):
                try:
                    await upload_single_file_async(session, pdf_path, bucket_name, s3_key)
                    stats["uploaded"] += 1
                    logging.info(f"Uploaded {s3_key}")
                except Exception as e:
                    stats["failed"] += 1
                    logging.error(f"Error uploading {pdf_path}: {str(e)}")
            else:
                stats["missing"] += 1
                logging.warning(f"File not found: {pdf_path}")

    # Create upload tasks
    for doc_type, files in file_mapping.items():
        for pdf_path in files:
            task = asyncio.create_task(upload_file(pdf_path, doc_type))
            tasks.append(task)

    # Upload files
    await asyncio.gather(*tasks)

    # Upload global metadata JSON
    metadata_key = f"{prefix}/metadata.json"
    metadata_bytes = json.dumps(global_metadata).encode("utf-8")
    async with session.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_DATA_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("AWS_DATA_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("S3_DATA_REGION", "us-west-1"),
    ) as s3:
        await s3.put_object(Bucket=bucket_name, Key=metadata_key, Body=metadata_bytes, ContentType="application/json")

    # Print statistics
    logging.info("\nUpload Statistics:")
    logging.info(f"PDFs uploaded: {stats['uploaded']}")
    logging.info(f"Failed uploads: {stats['failed']}")
    logging.info(f"Missing files: {stats['missing']}")
    logging.info(f"Metadata file uploaded to {metadata_key}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload OmniBench PDFs to S3")
    parser.add_argument("--base_path", type=str, default="OmniDocBench", help="Path to OmniBench directory")
    parser.add_argument("--bucket", type=str, default="astralis-data-4170a4f6", help="Target S3 bucket")
    parser.add_argument("--prefix", type=str, default="omnibench", help="S3 prefix")
    parser.add_argument("--max_concurrency", type=int, default=10, help="Maximum concurrent uploads")

    args = parser.parse_args()

    asyncio.run(upload_omnibench_to_s3(base_path=args.base_path, bucket_name=args.bucket, prefix=args.prefix, max_concurrency=args.max_concurrency))
