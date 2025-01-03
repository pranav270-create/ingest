import json
import os
import sys
from pathlib import Path
from typing import Union

from pydantic import BaseModel

sys.path.append(str(Path(__file__).parents[2]))

from src.llm_utils.format_embed_prompt import format_embed_prompt
from src.llm_utils.process_api_requests import process_api_requests_from_file
from src.llm_utils.utils import Functionality, Provider, save_prompts_to_jsonl
from src.pipeline.registry import FunctionRegistry
from src.schemas.schemas import Document, EmbeddedFeatureType, Embedding, Entry
from src.utils.datetime_utils import get_current_utc_datetime


@FunctionRegistry.register("embed", "embed")
async def get_embeddings(
    basemodels: Union[list[Document], list[Entry]],
    model: str,
    provider: str,
    dimensions: int,
    write=None,  # noqa
    read=None,  # noqa
    **kwargs,  # noqa
) -> list[BaseModel]:
    provider = Provider(provider)
    all_prompts = []

    def iterate_over_entries(entries: list[Entry]):
        for entry in entries:
            entry.ingestion.embedded_feature_type = (
                EmbeddedFeatureType.TEXT
                if entry.ingestion.embedded_feature_type is None
                else entry.ingestion.embedded_feature_type
            )
            entry.ingestion.embedding_date = get_current_utc_datetime()
            entry.ingestion.embedding_model = model
            entry.ingestion.embedding_dimensions = dimensions
            metadata = entry.model_dump(exclude_none=True)
            if "context_summary_string" in metadata and metadata["context_summary_string"]:
                context_summary = metadata["context_summary_string"]
            else:
                context_summary = ""
            text = metadata.pop("string")
            text += context_summary
            prompts = format_embed_prompt(model, provider, text, metadata, dimensions=dimensions)
            all_prompts.extend(prompts)

    if isinstance(basemodels[0], Document):
        for basemodel in basemodels:
            iterate_over_entries(basemodel.entries)

    elif isinstance(basemodels[0], Entry):
        iterate_over_entries(basemodels)

    else:
        raise ValueError("Invalid input type")

    # format prompts
    input_file = "/tmp/embed_prompts.jsonl"
    open(input_file, "w").close()  # make sure to clear the file
    save_prompts_to_jsonl(all_prompts, input_file)

    # Set up parameters for API call
    output_filepath = "/tmp/embed_responses"
    if os.path.exists(output_filepath):
        for file in Path(output_filepath).iterdir():
            os.remove(file)
    else:
        os.makedirs(output_filepath, exist_ok=True)

    await process_api_requests_from_file(
        requests_filepath=input_file,
        save_filepath=output_filepath,
        provider=provider,
        functionality=Functionality.EMBEDDING,
        model=model,
        max_attempts=3,
    )

    # iterate over the output directory
    all_embeddings = []
    for file in Path(output_filepath).iterdir():
        with open(file) as f:
            for line in f:
                try:
                    raw_data = json.loads(line)
                    string = raw_data[0].get("input")
                    embedding_data = raw_data[1]
                    embedding = embedding_data.get("data")[0].get("embedding")
                    tokens = embedding_data.get("usage", {}).get("total_tokens", 0)
                    entry = raw_data[2]
                    entry.pop("schema__")
                    new_embedding = Embedding(**raw_data[2], embedding=embedding, string=string, tokens=tokens)
                    all_embeddings.append(new_embedding)
                except Exception as e:
                    print(e)
                    continue
    return all_embeddings
