from typing import Optional, Union

# import cohere
# from mistralai.client import MistralClient
# from mistralai.async_client import MistralAsyncClient
# import voyageai
import anthropic
import google.generativeai as gemini
from openai import AsyncOpenAI, OpenAI

from src.llm_utils.api_requests import get_api_key
from src.llm_utils.client import AsyncChat, Chat, from_anthropic, from_gemini, from_openai
from src.llm_utils.utils import Provider


def llm_client(
    provider: Provider,
    model_name: Optional[str] = None,
    async_client: bool = False,
) -> Union[Chat, AsyncChat]:
    """
    Factory function to get the appropriate embedding client for a given provider.

    Args:
        provider (Provider): The provider to use.
        model_name (str): The name of the model to use.
        async_client (bool): Whether to return async clients. Defaults to False.

    Returns:
        Union[AnyEmbedder, AsyncAnyEmbedder]: The embedding client.
    """
    api_key = get_api_key(provider)

    if provider == Provider.OPENAI:
        raw_client = AsyncOpenAI(api_key=api_key) if async_client else OpenAI(api_key=api_key)
        return from_openai(raw_client)

    elif provider == Provider.COHERE:
        # raw_client = (
        #     cohere.AsyncClient(api_key=api_key)
        #     if async_client
        #     else cohere.Client(api_key=api_key)
        # )
        # return infinity_llm.embed_from_cohere(raw_client)
        raise NotImplementedError("Cohere support is not currently available.")

    elif provider == Provider.VOYAGE:
        # raw_client = (
        #     voyageai.AsyncClient(api_key=api_key)
        #     if async_client
        #     else voyageai.Client(api_key=api_key)
        # )
        # return infinity_llm.embed_from_voyage(raw_client)
        raise NotImplementedError("Voyage support is not currently available.")

    elif provider == Provider.MISTRAL:
        # raw_client = (
        #     MistralAsyncClient(api_key=api_key)
        #     if async_client
        #     else MistralClient(api_key=api_key)
        # )
        # return infinity_llm.embed_from_mistral(raw_client)
        raise NotImplementedError("Mistral support is not currently available.")

    elif provider == Provider.ANYSCALE:
        raw_client = (
            AsyncOpenAI(
                base_url="https://api.endpoints.anyscale.com/v1",
                api_key=api_key,
            )
            if async_client
            else OpenAI(
                base_url="https://api.endpoints.anyscale.com/v1",
                api_key=api_key,
            )
        )
        return from_openai(raw_client)

    elif provider == Provider.ANTHROPIC:
        raw_client = anthropic.AsyncAnthropic(api_key=api_key) if async_client else anthropic.Anthropic(api_key=api_key)
        return from_anthropic(raw_client)

    elif provider == Provider.GEMINI:
        assert model_name, "model_name is required for Gemini"
        raw_client = (
            gemini.GenerativeModel(model_name=model_name) if async_client else gemini.GenerativeModel(model_name=model_name)
        )
        client = from_gemini(raw_client, use_async=async_client) if async_client else from_gemini(raw_client)
        return client

    else:
        raise ValueError(f"Cannot create embedding client for unsupported provider: {provider}")
