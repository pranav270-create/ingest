import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

import asyncio
import logging
import os

import aioboto3

from src.utils.s3_utils import upload_single_file_async

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DOCBENCH_TYPE_RANGES = {
    "academic_papers": range(0, 49),
    "finance": range(49, 89),
    "government": range(89, 133),
    "laws": range(133, 179),
    "news": range(179, 229),
}


async def upload_docbench_to_s3(base_path: str, bucket_name: str, prefix: str = "docbench", max_concurrency: int = 10) -> None:
    """
    Upload DocBench PDFs and their metadata JSONs to S3, organized by document type
    """
    stats = {"total_folders": 0, "uploaded_pdfs": 0, "uploaded_jsons": 0, "failed_uploads": 0, "missing_files": 0}

    session = aioboto3.Session()
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = []

    def get_doc_type(folder_num: int) -> str:
        for doc_type, range_obj in DOCBENCH_TYPE_RANGES.items():
            if folder_num in range_obj:
                return doc_type
        return "unknown"

    async def upload_folder_content(folder_path: str):
        folder_name = os.path.basename(folder_path)
        folder_num = int(folder_name)
        doc_type = get_doc_type(folder_num)

        all_files = os.listdir(folder_path)
        pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]
        json_files = [f for f in all_files if f.lower().endswith((".json", ".jsonl"))]

        if not pdf_files or not json_files:
            stats["missing_files"] += 1
            logging.warning(f"Missing files in {folder_path} (PDFs: {pdf_files}, JSONs: {json_files})")
            return

        pdf_path = os.path.join(folder_path, pdf_files[0])
        json_path = os.path.join(folder_path, json_files[0])

        async with semaphore:
            try:
                # Upload PDF with document type in path
                pdf_key = f"{prefix}/{doc_type}/{folder_name}/{pdf_files[0]}"
                await upload_single_file_async(session, pdf_path, bucket_name, pdf_key)
                stats["uploaded_pdfs"] += 1
                logging.info(f"Uploaded PDF: {pdf_key}")

                # Upload JSON
                json_key = f"{prefix}/{doc_type}/{folder_name}/{json_files[0]}"
                await upload_single_file_async(session, json_path, bucket_name, json_key)
                stats["uploaded_jsons"] += 1
                logging.info(f"Uploaded JSON: {json_key}")

            except Exception as e:
                stats["failed_uploads"] += 1
                logging.error(f"Error uploading from {folder_path}: {str(e)}")

    # Iterate through all subfolders
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            stats["total_folders"] += 1
            task = asyncio.create_task(upload_folder_content(folder_path))
            tasks.append(task)

    await asyncio.gather(*tasks)

    # Print final statistics
    logging.info("\nUpload Statistics:")
    logging.info(f"Total folders processed: {stats['total_folders']}")
    logging.info(f"PDFs uploaded: {stats['uploaded_pdfs']}")
    logging.info(f"JSONs uploaded: {stats['uploaded_jsons']}")
    logging.info(f"Failed uploads: {stats['failed_uploads']}")
    logging.info(f"Folders with missing files: {stats['missing_files']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload DocBench data to S3")
    parser.add_argument("--base_path", type=str, default="DocBench_Data", help="Path to DocBench directory")
    parser.add_argument("--bucket", type=str, default="astralis-data-4170a4f6", help="Target S3 bucket")
    parser.add_argument("--prefix", type=str, default="docbench", help="S3 prefix")
    parser.add_argument("--max_concurrency", type=int, default=10, help="Maximum concurrent uploads")

    args = parser.parse_args()

    asyncio.run(upload_docbench_to_s3(base_path=args.base_path, bucket_name=args.bucket, prefix=args.prefix, max_concurrency=args.max_concurrency))
