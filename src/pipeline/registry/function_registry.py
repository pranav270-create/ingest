from typing import Any, Callable, Union
from functools import wraps

from src.pipeline.registry.base import RegistryBase
from src.pipeline.storage_backend import StorageBackend


class FunctionRegistry(RegistryBase):
    _registry: dict[str, dict[str, Callable]] = {
        "ingest": {},
        "parse": {},
        "chunk": {},
        "featurize": {},
        "embed": {},
        "upsert": {},
    }
    _storage_backend: StorageBackend = None

    @classmethod
    def set_storage_backend(cls, storage_backend: StorageBackend):
        cls._storage_backend = storage_backend

    @classmethod
    def register(cls, stage: str, name: str):
        def decorator(func: Callable[..., Any]) -> Callable:
            @wraps(func)
            async def wrapper(*args, simple_mode: bool = True, **kwargs):
                if not simple_mode:
                    if cls._storage_backend is None:
                        raise ValueError("Storage backend not set")

                    async def write(file_path: str, content: str, mode: str = "w") -> None:
                        await cls._storage_backend.write(file_path, content, mode)

                    async def read(file_path: str, mode: str = "r") -> Union[str, None]:
                        return await cls._storage_backend.read(file_path, mode)

                    kwargs["write"] = write
                    kwargs["read"] = read
                return await func(*args, **kwargs)

            cls._registry[stage][name] = wrapper
            return wrapper

        return decorator

    @classmethod
    def get(cls, stage: str, name: str) -> Callable:
        return cls._registry[stage].get(name)

