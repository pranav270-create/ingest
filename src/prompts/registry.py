from typing import Callable

from src.prompts.base_prompt import BasePrompt


class PromptRegistry:
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