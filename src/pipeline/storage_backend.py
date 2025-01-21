import asyncio
import io
import os
from abc import ABC, abstractmethod
from typing import Literal, Union

import aiofiles
import boto3


class StorageBackend(ABC):
    @abstractmethod
    async def write(self, file_path: str, content: Union[str, bytes]) -> None:
        pass

    @abstractmethod
    async def read(self, file_path: str) -> Union[str, bytes]:
        pass


class LocalStorageBackend(StorageBackend):
    def __init__(self, base_path: str = "/tmp/s3"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    async def write(self, file_path: str, content: Union[str, bytes]) -> None:
        full_path = os.path.join(self.base_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        mode = "wb" if isinstance(content, bytes) else "w"
        async with aiofiles.open(full_path, mode) as f:
            await f.write(content)

    async def read(self, file_path: str) -> Union[str, bytes]:
        """Read file content, returning bytes for binary files and str for text files"""
        full_path = os.path.join(self.base_path, file_path)
        # Check if file is likely binary based on extension
        binary_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.doc', '.docx']
        text_extensions = ['.json', '.jsonl', '.txt', '.csv']

        is_binary = (
            any(file_path.lower().endswith(ext) for ext in binary_extensions) and
            not any(file_path.lower().endswith(ext) for ext in text_extensions)
        )

        if is_binary:
            async with aiofiles.open(full_path, 'rb') as f:
                return await f.read()
        else:
            # For text files (json, jsonl, txt, etc)
            async with aiofiles.open(full_path, encoding='utf-8') as f:
                return await f.read()


class S3StorageBackend(StorageBackend):
    def __init__(self, bucket_name: str):
        aws_access_key_id = os.environ.get("AWS_DATA_ACCESS_KEY")
        aws_secret_access_key = os.environ.get("AWS_DATA_SECRET_ACCESS_KEY")
        region_name = os.environ.get("S3_DATA_REGION", "us-west-1")
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
        self.bucket_name = bucket_name

    async def write(self, file_path: str, content: Union[str, bytes]) -> None:
        if isinstance(content, str):
            content = content.encode('utf-8')

        await asyncio.get_event_loop().run_in_executor(
            None,
            self.s3.upload_fileobj,
            io.BytesIO(content),
            self.bucket_name,
            file_path
        )

    async def read(self, file_path: str) -> Union[str, bytes]:
        """Read file content, returning bytes for binary files and str for text files"""
        # Check if file is likely binary based on extension BEFORE reading
        binary_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.doc', '.docx']
        text_extensions = ['.json', '.jsonl', '.txt', '.csv']

        is_binary = (
            any(file_path.lower().endswith(ext) for ext in binary_extensions) and
            not any(file_path.lower().endswith(ext) for ext in text_extensions)
        )

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.s3.get_object(Bucket=self.bucket_name, Key=file_path)
        )
        content = await asyncio.get_event_loop().run_in_executor(None, response['Body'].read)
        if is_binary:
            return content
        else:
            # For text files (json, jsonl, txt, etc)
            return content.decode('utf-8')


class StorageFactory:
    @staticmethod
    def create(storage_type: Literal["local", "s3"], **kwargs) -> StorageBackend:
        if storage_type == "local":
            if "base_path" not in kwargs:
                print("creating tmp directory for local storage")
            return LocalStorageBackend(base_path=kwargs.get("base_path", "/tmp/s3"))
        elif storage_type == "s3":
            if "bucket_name" not in kwargs:
                raise ValueError("bucket_name is required for S3 storage")
            return S3StorageBackend(bucket_name=kwargs["bucket_name"])
