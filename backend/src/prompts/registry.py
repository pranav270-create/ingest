from typing import Dict, Type, Callable
from abc import ABC

class PromptRegistry:
    _registry: Dict[str, Type[ABC]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[ABC]], Type[ABC]]:
        """
        A decorator that registers a prompt template class.
        
        Usage:
        @PromptRegistry.register.register("client_qa")
        class ClientQAPrompt:
            ...
        """
        def decorator(prompt_class: Type[ABC]) -> Type[ABC]:
            if name in cls._registry:
                raise KeyError(f"Prompt template with name '{name}' already exists.")
            cls._registry[name] = prompt_class
            return prompt_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[ABC]:
        """Retrieve a prompt template class by its name."""
        return cls._registry.get(name)