import os
from typing import Union

chat_model_list = [
    {
        "model_name": "gpt-4o-mini",  # model alias -> loadbalance between models with same `model_name`
        "litellm_params": {
            "model": "openai/gpt-4o-mini",  # actual model name
            "api_key": os.getenv("OPENAI_API_KEY"),
            "rpm": 10000,
            "tpm": 10000000,
        },
    },
    {
        "model_name": "gpt-4o",
        "litellm_params": {
            "model": "openai/gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "rpm": 10000,
            "tpm": 10000000,
        },
    },
    {
        "model_name": "claude-3-5-sonnet-20241022",
        "litellm_params": {
            "model": "anthropic/claude-3-5-sonnet-20241022",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "rpm": 4000,
            "tpm": 40000,
        },
    },
    {
        "model_name": "deepseek-chat",
        "litellm_params": {
            "model": "deepseek/deepseek-chat",
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "rpm": 10000,
            "tpm": 10000000,
        },
    },
    {
        "model_name": "deepseek-reasoner",
        "litellm_params": {
            "model": "deepseek/deepseek-reasoner",
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "rpm": 10000,
            "tpm": 10000000,
        },
    }
]


embedding_model_list: list[dict[str, Union[str, dict[str, Union[str, int]]]]] = [
    {
        "model_name": "openai",  # model alias -> loadbalance between models with same `model_name`
        "litellm_params": {
            "model": "openai/text-embedding-3-large",  # actual model name
            "api_key": os.getenv("OPENAI_API_KEY"),
            "rpm": 10000,
            "tpm": 5000000,
        },
    },
    {
        "model_name": "voyage",
        "litellm_params": {
            "model": "voyage/voyage-3-large",
            "api_key": os.getenv("VOYAGE_API_KEY"),
            "rpm": 2000,
            "tpm": 8000000,
        },
    },
    {
        "model_name": "cohere",
        "litellm_params": {
            "model": "cohere/embed-english-v3.0",
            "api_key": os.getenv("COHERE_API_KEY"),
            "rpm": 2000,
            "tpm": 100000,
        },
    }

]
