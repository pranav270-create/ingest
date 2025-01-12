import importlib
import pkgutil
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Union

from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.storage_backend import StorageBackend
from src.prompts.base_prompt import BasePrompt


class RegistryBase:
    _registry: dict = {}

    @classmethod
    def autodiscover(cls, package_name: str) -> None:
        """
        Automatically discover and import all modules in a package.

        Usage:
        FunctionRegistry.autodiscover('src.featurization')
        SchemaRegistry.autodiscover('src.schemas')
        PromptRegistry.autodiscover('src.prompts')
        """
        package = importlib.import_module(package_name)
        package_path = Path(package.__file__).parent

        for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
            full_module_name = f"{package_name}.{module_name}"
            importlib.import_module(full_module_name)


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


class PromptRegistry(RegistryBase):
    _registry: dict[str, type[BasePrompt]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[BasePrompt]], type[BasePrompt]]:
        """
        A decorator that registers a prompt template class.

        Usage:
        @PromptRegistry.register.register("client_qa")
        class ClientQAPrompt:
            ...
        """
        def decorator(prompt_class: type[BasePrompt]) -> type[BasePrompt]:
            if name in cls._registry:
                raise KeyError(f"Prompt template with name '{name}' already exists.")
            cls._registry[name] = prompt_class
            return prompt_class
        return decorator

    @classmethod
    def get(cls, name: str) -> type[BasePrompt]:
        """Retrieve a prompt template class by its name."""
        return cls._registry.get(name)


class SchemaRegistry(RegistryBase):
    _registry: dict[str, type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[BaseModel]], type[BaseModel]]:
        """
        A decorator that registers a schema class.

        Usage:
        @SchemaRegistry.register("my_schema")
        class MySchema(BaseModel):
            ...
        """
        def decorator(schema_class: type[BaseModel]) -> type[BaseModel]:
            if name in cls._registry:
                raise KeyError(f"Schema with name '{name}' already exists.")
            cls._registry[name] = schema_class
            return schema_class
        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseModel]:
        """Retrieve a schema class by its name."""
        return cls._registry.get(name)
