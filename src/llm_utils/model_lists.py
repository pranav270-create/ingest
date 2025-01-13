import os

chat_model_list = [
    {
        "model_name": "gpt-4o-mini",  # model alias -> loadbalance between models with same `model_name`
        "litellm_params": {
            "model": "gpt-4o-mini",  # actual model name
            "api_key": os.getenv("OPENAI_API_KEY"),
            "rpm": 5000,
            "tpm": 600000,
        },
    }
]


embedding_model_list = [
    {
        "model_name": "openai",  # model alias -> loadbalance between models with same `model_name`
        "litellm_params": {
            "model": "text-embedding-3-large",  # actual model name
            "api_key": os.getenv("OPENAI_API_KEY"),
            "rpm": 5000,
            "tpm": 5000000,
        },
    }
]
