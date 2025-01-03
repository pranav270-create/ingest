import sys
from pathlib import Path
from typing import Any, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.llm_utils.utils import Provider


def format_embed_prompt(
    model: str, provider: Provider, text: list[str], metadata: list[dict[str, Any]], dimensions: Optional[int] = None
) -> list[dict[str, Any]]:
    """
    Formats the text to be embedded according to the api
    """
    if dimensions and provider != Provider.OPENAI:
        raise ValueError("Dimension is only supported for OpenAI embedding models")

    if isinstance(text, str):
        text = [text]
        metadata = [metadata]
    if provider == Provider.ANYSCALE:
        return [{"model": model, "input": t, "metadata": m} for t, m in zip(text, metadata)]
    elif provider == Provider.OPENAI:
        # only text-embedding-3-small and text-embedding-3-large support dimensions parameter
        if "text-embedding-3" in model:
            return [{"model": model, "input": t, "metadata": m, "dimensions": dimensions} for t, m in zip(text, metadata)]
        else:
            return [{"model": model, "input": t, "metadata": m} for t, m in zip(text, metadata)]
    elif provider == Provider.MISTRAL:
        return [{"model": model, "inputs": t, "metadata": m} for t, m in zip(text, metadata)]
    elif provider in [Provider.COHERE, Provider.VOYAGE]:
        raise ValueError(f"{provider} does not support metadata")
    else:
        raise ValueError(f"Unsupported provider: {provider}")
