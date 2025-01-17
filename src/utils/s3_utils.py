import os
import asyncio
import aioboto3
from typing import Optional


aws_access_key_id = os.environ.get("AWS_DATA_ACCESS_KEY")
aws_secret_access_key = os.environ.get("AWS_DATA_SECRET_ACCESS_KEY")
region_name = os.environ.get("S3_DATA_REGION", "us-west-1")


async def upload_single_file_async(
    session,
    file_path: str,
    bucket_name: str,
    s3_key: str
) -> None:
    async with session.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    ) as s3:
        try:
            await s3.upload_file(file_path, bucket_name, s3_key)
        except Exception as e:
            print(f"Error uploading {file_path}: {str(e)}")


async def upload_folder_async(
    folder_path: str,
    bucket_name: str,
    prefix: Optional[str] = None,
    max_concurrency: int = 10
) -> None:
    """
    Upload a folder to S3 asynchronously with optional prefix
    
    Args:
        folder_path: Local folder path to upload
        bucket_name: Target S3 bucket
        prefix: Optional S3 key prefix (e.g., 'data/raw/')
        max_concurrency: Maximum number of concurrent uploads
    """
    session = aioboto3.Session()
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = []

    async def upload_with_semaphore(file_path: str, s3_key: str):
        async with semaphore:
            await upload_single_file_async(session, file_path, bucket_name, s3_key)

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Get relative path and normalize separators
            relative_path = os.path.relpath(file_path, folder_path).replace(os.sep, '/')
            
            # Combine prefix with relative path if prefix is provided
            s3_key = f"{prefix.rstrip('/')}/{relative_path}" if prefix else relative_path
            
            task = asyncio.create_task(
                upload_with_semaphore(file_path, s3_key)
            )
            tasks.append(task)
    
    await asyncio.gather(*tasks)


# Example usage:
if __name__ == "__main__":
    asyncio.run(upload_folder_async(
        folder_path="/Users/pranaviyer/Desktop/AstralisData",
        bucket_name="astralis-data-4170a4f6",
        prefix="unstructured",
        max_concurrency=10
    ))