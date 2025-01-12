from abc import ABC
from typing import Callable


class PromptRegistry:
    _registry: dict[str, type[ABC]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[ABC]], type[ABC]]:
        """
        A decorator that registers a prompt template class.

        Usage:
        @PromptRegistry.register.register("client_qa")
        class ClientQAPrompt:
            ...
        """
        def decorator(prompt_class: type[ABC]) -> type[ABC]:
            if name in cls._registry:
                raise KeyError(f"Prompt template with name '{name}' already exists.")
            cls._registry[name] = prompt_class
            return prompt_class
        return decorator

    @classmethod
    def get(cls, name: str) -> type[ABC]:
        """Retrieve a prompt template class by its name."""
        return cls._registry.get(name)