import base64
import sys
from pathlib import Path
from typing import Union

from litellm import Router

sys.path.append(str(Path(__file__).parents[2]))

from src.llm_utils.model_lists import embedding_model_list
from src.pipeline.registry import FunctionRegistry
from src.schemas.schemas import EmbeddedFeatureType, Embedding, Entry
from src.utils.datetime_utils import get_current_utc_datetime

router = Router(
    model_list=embedding_model_list,
    default_max_parallel_requests=50,
    allowed_fails=2,
    cooldown_time=20,
    num_retries=3,
)


def is_base64_image(s: str) -> bool:
    """Check if a string is a base64 encoded image"""
    try:
        base64.b64decode(s)
        return s.startswith(('data:image/', '/9j/', 'iVBORw0KGgo'))
    except Exception:
        return False


def supports_image_embedding(model: str) -> bool:
    """Check if model supports image embedding based on prefix"""
    return model.startswith(("cohere/", "voyage/"))


@FunctionRegistry.register("embed", "embed")
async def get_embeddings(
    basemodels: Union[list[Entry]],
    model_name: str,
    dimensions: int,
    write=None,  # noqa
    read=None,  # noqa
    **kwargs,  # noqa
) -> list[Embedding]:

    inputs = []
    flat_entries = []
    using_image_embedding = False

    def iterate_over_entries(entries: list[Entry]):
        """
        for each entry, update the metadata and append the text to the all_prompts list
        """
        nonlocal using_image_embedding

        for i, entry in enumerate(entries):
            # update metadata
            entry.ingestion.embedded_feature_type = (
                EmbeddedFeatureType.TEXT
                if entry.ingestion.embedded_feature_type is None
                else entry.ingestion.embedded_feature_type
            )
            entry.ingestion.embedding_date = get_current_utc_datetime()
            entry.ingestion.embedding_model = model_name
            entry.ingestion.embedding_dimensions = dimensions

            # get the text to embed
            text = entry.string

            if i == 0:
                if is_base64_image(text):
                    using_image_embedding = True

            text += entry.context_summary_string or ""
            inputs.append(text)

            # pop schema so we can create Embedding(Entry) later
            metadata = entry.model_dump(exclude_none=True)
            metadata.pop("schema__")
            flat_entries.append(metadata)

    iterate_over_entries(basemodels)

    # if embedding images, check if model supports image embedding
    if using_image_embedding:
        if not supports_image_embedding(model_name):
            raise ValueError(f"Model {model_name} does not support image embedding")

    embedding_response = await router.aembedding(model=model_name, input=inputs, dimensions=dimensions)
    embeddings = embedding_response.data
    tokens = embedding_response.usage.total_tokens
    return [
        Embedding(
            **{**entry, "embedding": embed_dict["embedding"], "string": string, "tokens": tokens}
        )
        for entry, embed_dict, string in zip(flat_entries, embeddings, inputs)
    ]
