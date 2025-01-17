import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import json
import os
import asyncio
import logging
from typing import Dict, List
from src.utils.s3_utils import upload_folder_async, upload_single_file_async
import aioboto3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def upload_omnibench_to_s3(
    base_path: str,
    bucket_name: str,
    prefix: str = "omnibench",
    max_concurrency: int = 10
) -> None:
    """
    Upload OmniBench PDFs to S3 organized by document type
    """
    # Load metadata
    json_path = os.path.join(base_path, "OmniDocBench.json")
    logging.info(f"Loading metadata from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    logging.info(f"Found {len(metadata)} entries in metadata file")
    
    # Create mapping of files to document types
    file_mapping: Dict[str, List[dict]] = {}
    for entry in metadata:
        doc_type = entry['page_info']['page_attribute']['data_source']
        image_path = entry['page_info']['image_path']
        
        # Extract base name without extension and ensure PDF extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        pdf_path = os.path.join(base_path, 'pdfs', f"{base_name}.pdf")
        
        if doc_type not in file_mapping:
            file_mapping[doc_type] = []
        
        file_mapping[doc_type].append({
            'pdf_path': pdf_path,
            'metadata': entry
        })
    
    logging.info(f"Found {len(file_mapping)} document types:")
    for doc_type, files in file_mapping.items():
        logging.info(f"  - {doc_type}: {len(files)} files")

    # Upload files by document type
    session = aioboto3.Session()
    tasks = []
    semaphore = asyncio.Semaphore(max_concurrency)
    
    # Track upload statistics
    stats = {
        'total_files': sum(len(files) for files in file_mapping.values()),
        'uploaded_pdfs': 0,
        'uploaded_jsons': 0,
        'failed_uploads': 0,
        'missing_files': 0
    }

    async def upload_file_with_metadata(pdf_path: str, doc_type: str, metadata: dict):
        async with semaphore:
            s3_key = f"{prefix}/{doc_type}/{os.path.basename(pdf_path)}"
            metadata_key = f"{s3_key}.json"
            
            if os.path.exists(pdf_path):
                try:
                    # Upload PDF
                    await upload_single_file_async(
                        session,
                        pdf_path,
                        bucket_name,
                        s3_key
                    )
                    stats['uploaded_pdfs'] += 1
                    
                    # Upload metadata JSON
                    metadata_bytes = json.dumps(metadata).encode('utf-8')
                    async with session.client(
                        "s3",
                        aws_access_key_id=os.environ.get("AWS_DATA_ACCESS_KEY"),
                        aws_secret_access_key=os.environ.get("AWS_DATA_SECRET_ACCESS_KEY"),
                        region_name=os.environ.get("S3_DATA_REGION", "us-west-1")
                    ) as s3:
                        await s3.put_object(
                            Bucket=bucket_name,
                            Key=metadata_key,
                            Body=metadata_bytes,
                            ContentType='application/json'
                        )
                        stats['uploaded_jsons'] += 1
                    
                    logging.info(f"Successfully uploaded {os.path.basename(pdf_path)} to {s3_key}")
                except Exception as e:
                    stats['failed_uploads'] += 1
                    logging.error(f"Error uploading {pdf_path}: {str(e)}")
            else:
                stats['missing_files'] += 1
                logging.warning(f"File not found: {pdf_path}")

    # Create upload tasks for each document type
    for doc_type, files in file_mapping.items():
        for file_info in files:
            task = asyncio.create_task(
                upload_file_with_metadata(
                    file_info['pdf_path'],
                    doc_type,
                    file_info['metadata']
                )
            )
            tasks.append(task)
    
    # Wait for all uploads to complete
    await asyncio.gather(*tasks)
    
    # Print final statistics
    logging.info("\nUpload Statistics:")
    logging.info(f"Total files processed: {stats['total_files']}")
    logging.info(f"PDFs uploaded: {stats['uploaded_pdfs']}")
    logging.info(f"JSON metadata files uploaded: {stats['uploaded_jsons']}")
    logging.info(f"Failed uploads: {stats['failed_uploads']}")
    logging.info(f"Missing files: {stats['missing_files']}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Upload OmniBench PDFs to S3')
    parser.add_argument('--base_path', type=str, default="OmniDocBench", help='Path to OmniBench directory')
    parser.add_argument('--bucket', type=str, default="astralis-data-4170a4f6", help='Target S3 bucket')
    parser.add_argument('--prefix', type=str, default='omnibench', help='S3 prefix')
    parser.add_argument('--max_concurrency', type=int, default=10, help='Maximum concurrent uploads')
    
    args = parser.parse_args()
    
    asyncio.run(upload_omnibench_to_s3(
        base_path=args.base_path,
        bucket_name=args.bucket,
        prefix=args.prefix,
        max_concurrency=args.max_concurrency
    ))
