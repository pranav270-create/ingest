import warnings
from functools import cache

import cohere
import tiktoken
from mistral_common.protocol.embedding.request import EmbeddingRequest
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import AutoTokenizer, GPT2TokenizerFast
from voyageai import Client as voyage_client

from .api_requests import get_api_key
from .utils import Provider


@cache
def get_tokenizer(model: str, provider: Provider):
    """
    Cached version of get_tokenizer to avoid re-instantiating tokenizers.
    """
    if provider == Provider.OPENAI:
        return tiktoken.encoding_for_model(model)

    elif provider in {Provider.ANYSCALE, Provider.GROQ}:
        if model in {"mistralai/Mistral-7B-Instruct-v0.1", "mlabonne/NeuralHermes-2.5-Mistral-7B"}:
            return MistralTokenizer.from_model("open-mistral-7b")
        elif model in {"mixtral-8x7b-32768", "mistralai/Mixtral-8x7B-Instruct-v0.1"}:
            return MistralTokenizer.from_model("open-mixtral-8x7b")
        elif model == "mistralai/Mixtral-8x22B-Instruct-v0.1":
            return MistralTokenizer.from_model("open-mixtral-8x22b")
        elif model in {"gemma-7b-it", "google/gemma-7b-it"}:
            return AutoTokenizer.from_pretrained("google/gemma-7b")
        elif model == "thenlper/gte-large":
            return AutoTokenizer.from_pretrained("thenlper/gte-large")
        elif model == "BAAI/bge-large-en-v1.5":
            return AutoTokenizer.from_pretrained("BAAI/bge-large-zh-v1.5")
        else:
            raise ValueError(f"Unsupported model '{model}' for provider '{provider}'.")
    elif provider == Provider.MISTRAL:
        return MistralTokenizer.from_model(model)
    elif provider == Provider.VOYAGE:
        client = voyage_client(api_key=get_api_key(Provider.VOYAGE))
        return client
    elif provider == Provider.COHERE:
        co = cohere.Client(api_key=get_api_key(Provider.COHERE))
        return co  # Assuming the Cohere client provides tokenizer functionality
    elif provider == Provider.ANTHROPIC:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return GPT2TokenizerFast.from_pretrained("Xenova/claude-tokenizer")
    else:
        raise ValueError(f"Unsupported provider '{provider}'.")


def tokenize_openai(model: str, text: str) -> list[str]:
    tokenizer = get_tokenizer(model, Provider.OPENAI)
    return tokenizer.encode(text)


def tokenize_voyage(model: str, text: str) -> list[str]:
    client = get_tokenizer(model, Provider.VOYAGE)
    tokenized = client.tokenize(text, model)
    return [token for sublist in tokenized for token in sublist]


def tokenize_cohere(model: str, text: str) -> list[str]:
    co = get_tokenizer(model, Provider.COHERE)
    return co.tokenize(text=text, model=model)


def tokenize_anthropic(text) -> list[str]:
    tokenizer = get_tokenizer("Xenova/claude-tokenizer", Provider.ANTHROPIC)
    return tokenizer.encode(text)


def tokenize_mistral(model: str, text: str) -> list[str]:
    tokenizer = get_tokenizer(model, Provider.MISTRAL)

    if model == "mistral-embed":
        request = EmbeddingRequest(inputs=[text], model=model)
        return tokenizer.instruct_tokenizer.encode_embedding(request).tokens
    else:
        request = ChatCompletionRequest(messages=[UserMessage(content=text)])
        return tokenizer.encode_chat_completion(request).tokens


def tokenize_oss(model: str, text: str) -> list[str]:
    if model == "mistralai/Mistral-7B-Instruct-v0.1":
        return tokenize_mistral("open-mistral-7b", text)
    elif model in ["mixtral-8x7b-32768", "mistralai/Mixtral-8x7B-Instruct-v0.1"]:
        return tokenize_mistral("open-mixtral-8x7b", text)
    elif model == "mistralai/Mixtral-8x22B-Instruct-v0.1":
        return tokenize_mistral("open-mixtral-8x22b", text)
    elif model == "mlabonne/NeuralHermes-2.5-Mistral-7B":
        return tokenize_mistral("open-mistral-7b", text)

    elif model in ["gemma-7b-it", "google/gemma-7b-it"]:
        tokenizer = get_tokenizer("google/gemma-7b", Provider.GROQ)
        return tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors="pt")
    elif model == "thenlper/gte-large":
        tokenizer = get_tokenizer("thenlper/gte-large", Provider.GROQ)
        return tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors="pt")
    elif model == "BAAI/bge-large-en-v1.5":
        tokenizer = get_tokenizer("BAAI/bge-large-zh-v1.5", Provider.GROQ)
        return tokenizer(text, padding=True, truncation=True, return_tensors="pt")


def detokenize_openai(model: str, tokens: list[int]) -> str:
    tokenizer = get_tokenizer(model, Provider.OPENAI)
    return tokenizer.decode(tokens)


def detokenize_voyage(model: str, tokens: list[int]) -> str:
    client = get_tokenizer(model, Provider.VOYAGE)
    return client.detokenize(tokens, model)


def detokenize_cohere(model: str, tokens: list[int]) -> str:
    co = get_tokenizer(model, Provider.COHERE)
    return co.detokenize(tokens, model=model)


def detokenize_anthropic(tokens: list[int]) -> str:
    tokenizer = get_tokenizer("Xenova/claude-tokenizer", Provider.ANTHROPIC)
    return tokenizer.decode(tokens)


def detokenize_mistral(model: str, tokens: list[int]) -> str:
    tokenizer = get_tokenizer(model, Provider.MISTRAL)
    if model == "mistral-embed":
        request = EmbeddingRequest(inputs=[], model=model)  # Inputs not needed for detokenization
        return tokenizer.instruct_tokenizer.decode_embedding(request)

    else:
        request = ChatCompletionRequest(messages=[])  # Messages not needed for detokenization
        return tokenizer.decode_chat_completion(request)


def detokenize_oss(model: str, tokens: list[int]) -> str:
    if model == "mistralai/Mistral-7B-Instruct-v0.1":
        return detokenize_mistral("open-mistral-7b", tokens)
    elif model in ["mixtral-8x7b-32768", "mistralai/Mixtral-8x7B-Instruct-v0.1"]:
        return detokenize_mistral("open-mixtral-8x7b", tokens)
    elif model == "mistralai/Mixtral-8x22B-Instruct-v0.1":
        return detokenize_mistral("open-mixtral-8x22b", tokens)
    elif model == "mlabonne/NeuralHermes-2.5-Mistral-7B":
        return detokenize_mistral("open-mistral-7b", tokens)

    elif model in ["gemma-7b-it", "google/gemma-7b-it"]:
        tokenizer = get_tokenizer("google/gemma-7b", Provider.GROQ)
        return tokenizer.decode(tokens)
    elif model == "thenlper/gte-large":
        tokenizer = get_tokenizer("thenlper/gte-large", Provider.GROQ)
        return tokenizer.decode(tokens)
    elif model == "BAAI/bge-large-en-v1.5":
        tokenizer = get_tokenizer("BAAI/bge-large-zh-v1.5", Provider.GROQ)
        return tokenizer.decode(tokens)


def get_embed_input_tokens(request_json, provider: Provider) -> int:
    def get_text(inputs):
        if isinstance(inputs, str):
            return inputs
        elif isinstance(inputs, list):
            if all(isinstance(i, str) for i in inputs):
                return "".join(inputs)
            else:
                raise TypeError('Expected a list of strings for "inputs" field in embedding request')
        else:
            raise TypeError('Expected a list of strings for "inputs" field in embedding request')

    if provider == Provider.OPENAI:
        return tokenize_openai(request_json["model"], get_text(request_json["input"]))
    elif provider == Provider.ANYSCALE:
        return tokenize_oss(request_json["model"], get_text(request_json["input"]))
    elif provider == Provider.MISTRAL:
        return tokenize_mistral(model=request_json["model"], text=get_text(request_json["input"]))
    elif provider == Provider.VOYAGE:
        return tokenize_voyage(model=request_json["model"], text=request_json["texts"])
    elif provider == Provider.COHERE:
        return tokenize_cohere(model=request_json["model"], text=get_text(request_json["texts"]))


def get_chat_prompt_tokens(request_json, provider: Provider) -> int:
    def get_text(messages):
        return "".join(value for message in messages for value in message.values())

    if provider == Provider.OPENAI:
        return tokenize_openai(request_json["model"], get_text(request_json["messages"]))
    elif provider == Provider.ANTHROPIC:
        return tokenize_anthropic(text=get_text(request_json["messages"]))
    elif provider in [Provider.GROQ, Provider.ANYSCALE]:
        return tokenize_oss(model=request_json["model"], text=get_text(request_json["messages"]))
    elif provider == Provider.MISTRAL:
        return tokenize_mistral(model=request_json["model"], text=get_text(request_json["messages"]))
    elif provider == Provider.COHERE:
        return tokenize_cohere(model=request_json["model"], text=get_text(request_json["messages"]))


def tokenize_embed_input(provider: Provider, model: str, texts: str) -> int:
    if provider == Provider.OPENAI:
        return tokenize_openai(model, texts)
    elif provider == Provider.ANYSCALE:
        return tokenize_oss(model, texts)
    elif provider == Provider.MISTRAL:
        return tokenize_mistral(model, texts)
    elif provider == Provider.VOYAGE:
        return tokenize_voyage(model, texts)
    elif provider == Provider.COHERE:
        return tokenize_cohere(model, texts)
    else:
        raise ValueError(f"Unsupported provider '{provider}'.")


def detokenize_embed_input(provider: Provider, model: str, tokens: list[str]) -> int:
    if provider == Provider.OPENAI:
        return detokenize_openai(model, tokens)
    elif provider == Provider.ANYSCALE:
        return detokenize_oss(model, tokens)
    elif provider == Provider.MISTRAL:
        return detokenize_mistral(model, tokens)
    elif provider == Provider.VOYAGE:
        return detokenize_voyage(model, tokens)
    elif provider == Provider.COHERE:
        return detokenize_cohere(model, tokens)
