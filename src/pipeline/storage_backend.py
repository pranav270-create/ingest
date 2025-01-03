from abc import ABC, abstractmethod
import os
import boto3
import io
from typing import Any, Union
import asyncio
import aiofiles


class StorageBackend(ABC):
    @abstractmethod
    async def write(self, file_path: str, content: str, mode: str) -> None:
        pass

    @abstractmethod
    async def read(self, file_path: str, mode: str = 'rb') -> Union[str, bytes]:
        pass


class LocalStorageBackend(StorageBackend):
    def __init__(self, base_path: str = "/tmp/s3"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    async def write(self, file_path: str, content: str, mode: str = 'w') -> None:
        full_path = os.path.join(self.base_path, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        async with aiofiles.open(full_path, mode) as f:
            await f.write(content)

    async def read(self, file_path: str, mode: str = 'r') -> str:
        full_path = os.path.join(self.base_path, file_path)
        async with aiofiles.open(full_path, mode) as f:
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

    async def write(self, file_path: str, content: Any, mode: str = 'wb') -> None:
        if isinstance(content, str):
            content = content.encode('utf-8')
        elif not isinstance(content, bytes):
            content = io.BytesIO(content)

        await asyncio.get_event_loop().run_in_executor(
            None,
            self.s3.upload_fileobj,
            io.BytesIO(content),
            self.bucket_name,
            file_path
        )

    async def read(self, file_path: str, mode: str = 'rb') -> Union[str, bytes]:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.s3.get_object(Bucket=self.bucket_name, Key=file_path)
        )
        content = await asyncio.get_event_loop().run_in_executor(
            None,
            response['Body'].read
        )

        if 'b' not in mode:
            return content.decode('utf-8')
        return content


class StorageFactory:
    @staticmethod
    def create(type: str, **kwargs) -> StorageBackend:
        if type == "local":
            return LocalStorageBackend(base_path=kwargs.get("base_path", "/tmp/s3"))
        elif type == "s3":
            return S3StorageBackend(bucket_name=kwargs["bucket_name"])
        else:
            raise ValueError(f"Unsupported storage type: {type}")