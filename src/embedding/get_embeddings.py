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

@FunctionRegistry.register("embed", "embed")
async def get_embeddings(
    basemodels: Union[list[Entry]],
    model: str,
    provider: str, # noqa
    dimensions: int,
    write=None,  # noqa
    read=None,  # noqa
    **kwargs,  # noqa
) -> list[Embedding]:
    all_prompts = []
    flat_entries = []

    def iterate_over_entries(entries: list[Entry]):
        """
        for each entry, update the metadata and append the text to the all_prompts list
        """
        for entry in entries:
            # update metadata
            entry.ingestion.embedded_feature_type = (
                EmbeddedFeatureType.TEXT
                if entry.ingestion.embedded_feature_type is None
                else entry.ingestion.embedded_feature_type
            )
            entry.ingestion.embedding_date = get_current_utc_datetime()
            entry.ingestion.embedding_model = model
            entry.ingestion.embedding_dimensions = dimensions

            # get the text to embed
            text = entry.string
            text += entry.context_summary_string or ""
            all_prompts.append(text)

            # pop schema so we can create Embedding(Entry) later
            metadata = entry.model_dump(exclude_none=True)
            metadata.pop("schema__")
            flat_entries.append(metadata)

    iterate_over_entries(basemodels)
    embedding_response = await router.aembedding(model="openai", input=all_prompts, dimensions=dimensions)
    embeddings = embedding_response.data
    tokens = embedding_response.usage.total_tokens
    return [
        Embedding(
            **{**entry, "embedding": embed_dict["embedding"], "string": prompt, "tokens": tokens}
        )
        for entry, embed_dict, prompt in zip(flat_entries, embeddings, all_prompts)
    ]
