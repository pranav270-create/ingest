from typing import Any

from pydantic import BaseModel

from src.llm_utils.utils import Provider, get_chat_cost, get_chat_cost_dictionary, model_mapping


def text_cost_parser(completion: Any, model: str) -> tuple[str, float]:
    """
    Given LLM chat completion, return the text and the cost
    """
    if model_mapping[model].provider == Provider.ANTHROPIC:
        return completion.content[0].text, get_chat_cost(completion, model)
    elif model_mapping[model].provider == Provider.GEMINI:
        return completion.text, get_chat_cost(completion, model)
    elif model_mapping[model].provider == Provider.OPENAI or Provider.ANYSCALE:
        return completion.choices[0].message.content, get_chat_cost(completion, model)
    else:
        raise NotImplementedError(f"Provider {model_mapping[model].provider} not supported")


def structured_text_cost_parser(completion: Any, model: str) -> tuple[str, float]:
    """
    Given LLM structuered chat completion, return the text and the cost
    """
    if model_mapping[model].provider == Provider.ANTHROPIC:
        return completion.content[0].text, get_chat_cost(completion, model)
    elif model_mapping[model].provider == Provider.GEMINI:
        return completion.text, get_chat_cost(completion, model)
    elif model_mapping[model].provider == Provider.OPENAI:
        return completion.choices[0].message.parsed, get_chat_cost(completion, model)
    else:
        raise NotImplementedError(f"Provider {model_mapping[model].provider} not supported")


def text_cost_parser_dictionary(completion: dict[str, Any], model: str) -> tuple[str, float]:
    """
    Given LLM chat completion as a dictionary, return the text and the cost
    """
    if model_mapping[model].provider == Provider.ANTHROPIC:
        return completion['content'][0]['text'], get_chat_cost_dictionary(completion, model)
    elif model_mapping[model].provider == Provider.GEMINI:
        return completion['text'], get_chat_cost_dictionary(completion, model)
    elif model_mapping[model].provider == Provider.OPENAI or Provider.ANYSCALE:
        return completion['choices'][0]['message']['content'], get_chat_cost_dictionary(completion, model)
    else:
        raise NotImplementedError(f"Provider {model_mapping[model].provider} not supported")


def structured_text_cost_parser_dictionary(completion: dict[str, Any], model: str) -> tuple[str, float]:
    """
    Given LLM structured chat completion as a dictionary, return the text and the cost
    """
    if model_mapping[model].provider == Provider.ANTHROPIC:
        return completion['content'][0]['text'], get_chat_cost_dictionary(completion, model)
    elif model_mapping[model].provider == Provider.GEMINI:
        return completion['text'], get_chat_cost_dictionary(completion, model)
    elif model_mapping[model].provider == Provider.OPENAI:
        return completion['choices'][0]['message']['parsed'], get_chat_cost_dictionary(completion, model)
    else:
        raise NotImplementedError(f"Provider {model_mapping[model].provider} not supported")


def structured_output_curl_parser(completion: dict[str, Any], model: str) -> tuple[BaseModel, float]:
    """
    Given LLM structured output completion, return the text and the cost
    """

    if model_mapping[model].provider == Provider.OPENAI:
        return completion.choices[0].message.parsed, get_chat_cost(completion, model)
    else:
        raise NotImplementedError(f"Provider {model_mapping[model].provider} not supported")


def text_stream_parser(chunk: Any, model: str) -> str:
    """
    Given a streaming chunk from an LLM, return the text content
    """
    if model_mapping[model].provider == Provider.ANTHROPIC:
        # Handle Anthropic's content_block_delta events
        if hasattr(chunk, 'type') and chunk.type == 'content_block_delta':
            return chunk.delta.text if chunk.delta.text else ""
        return ""
    elif model_mapping[model].provider == Provider.GEMINI:
        return chunk.text if hasattr(chunk, 'text') else ""
    elif model_mapping[model].provider in [Provider.OPENAI, Provider.ANYSCALE]:
        return chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
    else:
        raise NotImplementedError(f"Streaming not supported for provider {model_mapping[model].provider}")
