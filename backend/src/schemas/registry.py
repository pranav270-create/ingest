from typing import Dict, Type, Callable
from abc import ABC

class SchemaRegistry:
    _registry: Dict[str, Type[ABC]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[ABC]], Type[ABC]]:
        """
        A decorator that registers a schema class.
        
        Usage:
        @SchemaRegistry.register("my_schema")
        class MySchema(BaseModel):
            ...
        """
        def decorator(schema_class: Type[ABC]) -> Type[ABC]:
            if name in cls._registry:
                raise KeyError(f"Prompt template with name '{name}' already exists.")
            cls._registry[name] = schema_class
            return schema_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[ABC]:
        """Retrieve a prompt template class by its name."""
        return cls._registry.get(name)