import os
from typing import Union

chat_model_list = [
    {
        "model_name": "gpt-4o-mini",  # model alias -> loadbalance between models with same `model_name`
        "litellm_params": {
            "model": "openai/gpt-4o-mini",  # actual model name
            "api_key": os.getenv("OPENAI_API_KEY"),
            "rpm": 5000,
            "tpm": 600000,
        },
    }
]


embedding_model_list: list[dict[str, Union[str, dict[str, Union[str, int]]]]] = [
    {
        "model_name": "openai",  # model alias -> loadbalance between models with same `model_name`
        "litellm_params": {
            "model": "openai/text-embedding-3-large",  # actual model name
            "api_key": os.getenv("OPENAI_API_KEY"),
            "rpm": 5000,
            "tpm": 5000000,
        },
    },
    {
        "model_name": "voyageai",
        "litellm_params": {
            "model": "voyageai/voyage-1-mini",
            "api_key": os.getenv("VOYAGE_API_KEY"),
            "rpm": 5000,
            "tpm": 5000000,
        },
    },
    {
        "model_name": "cohere",
        "litellm_params": {
            "model": "cohere/embed-english-v3.0",
            "api_key": os.getenv("COHERE_API_KEY"),
            "rpm": 100,
            "tpm": 100000,
        },
    }

]
