from typing import Callable, TypeVar

from pydantic import BaseModel

from src.pipeline.registry_base import RegistryBase

T = TypeVar('T', bound=BaseModel)

class SchemaRegistry(RegistryBase):
    _registry: dict[str, type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[T]], type[T]]:
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
