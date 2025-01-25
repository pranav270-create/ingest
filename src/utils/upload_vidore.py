import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

import asyncio
import logging
import os
from typing import List
import shutil

import aioboto3
import pandas as pd
from datasets import load_dataset

from src.utils.s3_utils import upload_single_file_async

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

VIDORE_DATASETS = {
    'arxivqa': {
        'questions': "vidore/arxivqa_test_subsampled",
        'docs_path': "path/to/arxiv/pdfs"
    },
    'docvqa': {
        'questions': "vidore/docvqa_test_subsampled",
        'docs_path': "path/to/docvqa/images"
    },
    'infovqa': {
        'questions': "vidore/infovqa_test_subsampled",
        'docs_path': "path/to/infovqa/images"
    },
    'tatdqa': {
        'questions': "vidore/tatdqa_test",
        'docs_path': "path/to/tatdqa/tables"
    },
    'tabfquad': {
        'questions': "vidore/tabfquad_test_subsampled",
        'docs_path': "path/to/tabfquad/tables"
    }
}

async def upload_vidore_to_s3(
    temp_dir: str,
    bucket_name: str,
    prefix: str = "vidore",
    max_concurrency: int = 10
) -> None:
    """
    Upload ViDoRe datasets to S3
    """
    stats = {
        'total_datasets': 0,
        'uploaded_files': 0,
        'failed_uploads': 0
    }

    session = aioboto3.Session()
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = []

    # Delete existing files in S3 with the same prefix
    async with session.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_DATA_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("AWS_DATA_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("S3_DATA_REGION", "us-west-1")
    ) as s3:
        try:
            # List all objects with the prefix
            paginator = s3.get_paginator('list_objects_v2')
            async for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    objects_to_delete = [{'Key': obj['Key']} for obj in page['Contents']]
                    if objects_to_delete:
                        await s3.delete_objects(
                            Bucket=bucket_name,
                            Delete={'Objects': objects_to_delete}
                        )
            logging.info(f"Cleaned up existing files with prefix: {prefix}")
        except Exception as e:
            logging.error(f"Error cleaning up existing files: {str(e)}")

    # Create temp directory if it doesn't exist, or clean it if it does
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    async def download_dataset(dataset_name: str, dataset_path: str, temp_dir: str) -> str:
        """Download dataset and return local path"""
        local_path = os.path.join(temp_dir, f"{dataset_name}_raw.parquet")
        
        try:
            # Load dataset with streaming to avoid memory issues
            dataset = load_dataset(dataset_path, streaming=True)
            
            # Convert to DataFrame in chunks
            chunks = []
            for chunk in dataset['test'].iter(batch_size=100):
                df_chunk = pd.DataFrame(chunk)
                # Remove image data to avoid "Too many open files" error
                if 'image' in df_chunk.columns:
                    df_chunk = df_chunk.drop('image', axis=1)
                chunks.append(df_chunk)
            
            df = pd.concat(chunks, ignore_index=True)
            df.to_parquet(local_path)
            logging.info(f"Downloaded dataset: {dataset_name}")
            
        except Exception as e:
            logging.error(f"Error downloading {dataset_name}: {str(e)}")
            raise
        
        return local_path

    async def process_dataset(dataset_name: str, dataset_config: dict):
        try:
            # Download dataset first
            local_dataset_path = await download_dataset(
                dataset_name, 
                dataset_config['questions'], 
                temp_dir
            )
            
            # Read the downloaded dataset
            if dataset_name == 'tatdqa':
                df = pd.read_parquet(local_dataset_path)
            else:
                df = pd.read_parquet(local_dataset_path)
            
            # Debug column names
            logging.info(f"Available columns for {dataset_name}: {df.columns.tolist()}")
            
            # Format fields based on dataset type
            if dataset_name == 'docvqa':
                df['document_id'] = df['docId']
                df['page_id'] = df['page']
                df = df[['query', 'document_id', 'page_id', 'answer']]
                
            elif dataset_name == 'arxivqa':
                df['document_id'] = df['image_filename'].apply(lambda x: x.split('.')[0])
                df['page_id'] = df['page']
                df = df[['query', 'document_id', 'page_id', 'answer']]
                
            elif dataset_name == 'infovqa':
                df['document_id'] = df['image_filename'].apply(lambda x: x.split('.')[0])
                df['page_id'] = 0
                df = df[['query', 'document_id', 'page_id', 'answer']]
                
            elif dataset_name == 'tabfquad':
                df['document_id'] = df['image_filename'].apply(lambda x: x.split('.')[0])
                df = df[['query', 'document_id']]
            
            # Add dataset source
            df['dataset'] = dataset_name
            
            # Save processed dataset
            questions_path = os.path.join(temp_dir, f"{dataset_name}_questions.parquet")
            df.to_parquet(questions_path)
            
            s3_key = f"{prefix}/{dataset_name}/{dataset_name}_questions.parquet"
            await upload_single_file_async(session, questions_path, bucket_name, s3_key)
            stats['uploaded_files'] += 1
            
        except Exception as e:
            stats['failed_uploads'] += 1
            logging.error(f"Error processing {dataset_name}: {str(e)}")
            logging.error(f"Full error for {dataset_name}: ", exc_info=True)

    # Create upload tasks
    for dataset_name, dataset_info in VIDORE_DATASETS.items():
        stats['total_datasets'] += 1
        task = asyncio.create_task(process_dataset(dataset_name, dataset_info))
        tasks.append(task)

    await asyncio.gather(*tasks)

    # Print final statistics
    logging.info("\nUpload Statistics:")
    logging.info(f"Total datasets processed: {stats['total_datasets']}")
    logging.info(f"Files uploaded: {stats['uploaded_files']}")
    logging.info(f"Failed uploads: {stats['failed_uploads']}")

    # Clean up temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        logging.info(f"Cleaned up temporary directory: {temp_dir}")

async def upload_documents(
    session: aioboto3.Session,
    dataset_name: str,
    docs_path: str,
    bucket_name: str,
    prefix: str,
    semaphore: asyncio.Semaphore
) -> None:
    """Upload the actual documents (PDFs/images/tables) to S3"""
    if not os.path.exists(docs_path):
        logging.error(f"Documents path not found: {docs_path}")
        return
        
    async with semaphore:
        for doc_file in os.listdir(docs_path):
            doc_path = os.path.join(docs_path, doc_file)
            s3_key = f"{prefix}/{dataset_name}/documents/{doc_file}"
            
            await upload_single_file_async(
                session,
                doc_path,
                bucket_name,
                s3_key
            )
            logging.info(f"Uploaded document: {s3_key}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Upload ViDoRe datasets to S3')
    parser.add_argument('--temp_dir', type=str, default="temp_vidore", help='Temporary directory for processing')
    parser.add_argument('--bucket', type=str, default="astralis-data-4170a4f6", help='Target S3 bucket')
    parser.add_argument('--prefix', type=str, default='vidore', help='S3 prefix')
    parser.add_argument('--max_concurrency', type=int, default=10, help='Maximum concurrent uploads')
    parser.add_argument('--docs_base_path', type=str, required=False, 
                       help='Base path containing document directories for each dataset')

    args = parser.parse_args()

    asyncio.run(upload_vidore_to_s3(
        temp_dir=args.temp_dir,
        bucket_name=args.bucket,
        prefix=args.prefix,
        max_concurrency=args.max_concurrency
    ))
