from abc import ABC
from typing import Callable


class SchemaRegistry:
    _registry: dict[str, type[ABC]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[ABC]], type[ABC]]:
        """
        A decorator that registers a schema class.

        Usage:
        @SchemaRegistry.register("my_schema")
        class MySchema(BaseModel):
            ...
        """
        def decorator(schema_class: type[ABC]) -> type[ABC]:
            if name in cls._registry:
                raise KeyError(f"Prompt template with name '{name}' already exists.")
            cls._registry[name] = schema_class
            return schema_class
        return decorator

    @classmethod
    def get(cls, name: str) -> type[ABC]:
        """Retrieve a prompt template class by its name."""
        return cls._registry.get(name)