from typing import Callable, cast

from pydantic import BaseModel

from src.pipeline.registry.base import RegistryBase


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
        return cast(type[BaseModel], cls._registry.get(name))
