import os

from src.llm_utils.utils import Functionality, Provider


def get_request_header(provider: Provider) -> dict:
    api_key = get_api_key(provider)
    if provider == Provider.ANTHROPIC:
        return {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    elif provider in [
        Provider.OPENAI,
        Provider.MISTRAL,
        Provider.COHERE,
        Provider.GROQ,
        Provider.VOYAGE,
    ]:
        return {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }

    else:
        raise ValueError(f"No request header for provider: {provider}")


def get_request_url(provider: Provider, functionality: Functionality) -> str:
    if provider == Provider.OPENAI:
        if functionality == Functionality.CHAT:
            return "https://api.openai.com/v1/chat/completions"
        elif functionality == Functionality.EMBEDDING:
            return "https://api.openai.com/v1/embeddings"
    elif provider == Provider.ANTHROPIC:
        if functionality == Functionality.CHAT:
            return "https://api.anthropic.com/v1/messages"
        elif functionality == Functionality.EMBEDDING:
            raise ValueError("Anthropic does not support embeddings")
    elif provider == Provider.MISTRAL:
        if functionality == Functionality.CHAT:
            return "https://api.mistral.ai/v1/chat/completions"
        elif functionality == Functionality.EMBEDDING:
            return "https://api.mistral.ai/v1/embeddings"
    elif provider == Provider.COHERE:
        if functionality == Functionality.CHAT:
            return "https://api.cohere.com/v1/chat"
        elif functionality == Functionality.EMBEDDING:
            return "https://api.cohere.com/v1/embed"
    elif provider == Provider.GROQ:
        if functionality == Functionality.CHAT:
            return "https://api.groq.com/openai/v1/chat/completions"
        elif functionality == Functionality.EMBEDDING:
            raise ValueError("Groq does not support embeddings")
    elif provider == Provider.VOYAGE:
        if functionality == Functionality.CHAT:
            raise ValueError("Voyage does not support chat")
        if functionality == Functionality.EMBEDDING:
            return "https://api.voyageai.com/v1/embeddings"
    raise ValueError(f"Unsupported provider or functionality: {provider}, {functionality}")


def get_api_key(provider: Provider):
    """
    Fetches the API key for a given model provider
    """
    model_to_api_key = {
        Provider.OPENAI: os.getenv("OPENAI_API_KEY"),
        Provider.ANTHROPIC: os.getenv("ANTHROPIC_API_KEY"),
        Provider.ANYSCALE: os.getenv("ANYSCALE_API_KEY"),
        Provider.COHERE: os.getenv("COHERE_API_KEY"),
        Provider.VOYAGE: os.getenv("VOYAGE_API_KEY"),
        Provider.GROQ: os.getenv("GROQ_API_KEY"),
        Provider.MISTRAL: os.getenv("MISTRAL_API_KEY"),
        Provider.GEMINI: os.getenv("GOOGLE_API_KEY"),
    }
    assert provider in model_to_api_key, f"Provider '{provider}' is not recognized."
    return model_to_api_key[provider]


def get_provider(base_url: str) -> Provider:
    if "anyscale" in str(base_url):
        return Provider.ANYSCALE
    elif "together" in str(base_url):
        return Provider.TOGETHER
    elif "anthropic" in str(base_url):
        return Provider.ANTHROPIC
    elif "groq" in str(base_url):
        return Provider.GROQ
    elif "voyage" in str(base_url):
        return Provider.VOYAGE
    elif "openai" in str(base_url):
        return Provider.OPENAI
    elif "mistral" in str(base_url):
        return Provider.MISTRAL
    elif "cohere" in str(base_url):
        return Provider.COHERE
    elif "gemini" in str(base_url):
        return Provider.GEMINI
    elif "databricks" in str(base_url):
        return Provider.DATABRICKS
    elif "vertexai" in str(base_url):
        return Provider.VERTEXAI
    return Provider.UNKNOWN
