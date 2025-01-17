import base64
import os
import sys
from pathlib import Path
from typing import Any, Literal

import httpx
from litellm import Router

sys.path.append(str(Path(__file__).parents[2]))

from src.llm_utils.model_lists import embedding_model_list
from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import EmbeddedFeatureType, Embedding, Entry
from src.utils.datetime_utils import get_current_utc_datetime

router = Router(
    model_list=embedding_model_list,
    allowed_fails=2,
    cooldown_time=20,
    num_retries=3,
)


def is_base64_image(s: str) -> bool:
    """Check if a string is a base64 encoded image"""
    try:
        base64.b64decode(s)
        return s.startswith(("data:image/", "/9j/", "iVBORw0KGgo"))
    except Exception:
        return False


def supports_image_embedding(model: str) -> bool:
    """Check if model supports image embedding based on prefix"""
    return model.startswith(("cohere/", "voyage/", "jina-"))


async def get_jina_embeddings(
    inputs: list[str], model_name: str, dimensions: int, embedding_type: Literal["ubinary", "float", "binary"] = "float"
) -> dict[str, Any]:
    """Handle embedding requests specifically for Jina AI models"""
    jina_api_key = os.environ.get("JINA_API_KEY")
    if not jina_api_key:
        raise ValueError("JINA_API_KEY is not set in the environment variables")

    # Prepare the input data
    input_data = []
    for text in inputs:
        if is_base64_image(text):
            input_data.append({"image": text})
        else:
            input_data.append({"text": text})

    payload = {"model": model_name, "dimensions": dimensions, "normalized": True, "embedding_type": embedding_type, "input": input_data}

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.jina.ai/v1/embeddings", headers={"Content-Type": "application/json", "Authorization": f"Bearer {jina_api_key}"}, json=payload
        )
        response.raise_for_status()
        return response.json()


@FunctionRegistry.register("embed", "embed")
async def get_embeddings(
    basemodels: list[Entry],
    model_name: str,
    dimensions: int,
    write=None,  # noqa
    read=None,  # noqa
) -> list[Embedding]:
    """
    Get embeddings for a list of entries
    """
    inputs = []
    flat_entries = []
    using_image_embedding = False
    for i, entry in enumerate(basemodels):
        if i == 0:
            if is_base64_image(entry.string):
                using_image_embedding = True

        # update metadata
        if not entry.embedded_feature_type:
            if using_image_embedding:
                entry.embedded_feature_type = EmbeddedFeatureType.IMAGE
            else:
                entry.embedded_feature_type = EmbeddedFeatureType.TEXT
        entry.embedding_date = get_current_utc_datetime()
        entry.embedding_model = model_name
        entry.embedding_dimensions = dimensions

        inputs.append(entry.string)

        # pop schema so we can create Embedding(Entry) later
        metadata = entry.model_dump(exclude_none=True)
        metadata.pop("schema__")
        flat_entries.append(metadata)

    # Check if the model supports image embedding when embedding images
    if using_image_embedding:
        model_params = next((mp for mp in embedding_model_list if mp["model_name"] == model_name), None)
        if model_params and not supports_image_embedding(model_params["litellm_params"]["model"]):
            raise ValueError(f"Model '{model_name}' does not support image embedding")

    # Check if using Jina AI model
    if model_name.startswith("jina-"):
        response_data = await get_jina_embeddings(inputs, model_name, dimensions)
        embeddings = response_data["data"]
        tokens = response_data.get("usage", {}).get("total_tokens", 0)
    else:
        embedding_response = await router.aembedding(model=model_name, input=inputs, dimensions=dimensions)
        embeddings = embedding_response.data
        tokens = embedding_response.usage.total_tokens

    return [
        Embedding(**{**entry, "embedding": embed_dict["embedding"], "string": string, "tokens": tokens})
        for entry, embed_dict, string in zip(flat_entries, embeddings, inputs)
    ]


if __name__ == "__main__":
    import asyncio

    from src.schemas.schemas import Embedding, Entry, Ingestion, IngestionMethod, Scope

    async def test():
        def convert_image_to_base64(image_path: str) -> str:
            """Convert an image file to a base64 encoded string."""
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded_string}"

        base64_image = convert_image_to_base64(r"C:\Users\marka\fun\ingest\src\test_image.jpg")

        # Test creating an Ingestion object
        ingestion = Ingestion(
            document_hash="123abc",
            scope=Scope.INTERNAL,
            creator_name="Test User",
            ingestion_method=IngestionMethod.LOCAL_FILE,
            ingestion_date="2024-03-20T12:00:00Z",
        )

        # Test creating an Entry object
        entry = Entry(
            uuid="test-uuid-123",
            ingestion=ingestion,
            string=base64_image,
            keywords=["test", "example"],
            embedded_feature_type=EmbeddedFeatureType.TEXT,
        )

        embeddings = await get_embeddings([entry], "voyage", 1024)

        print("\nEmbedding created successfully:", embeddings[0].model_dump_json(indent=2))

    asyncio.run(test())
