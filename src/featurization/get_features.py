import asyncio
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.append(str(Path(__file__).parents[2]))

from litellm import Router

from src.llm_utils.model_lists import chat_model_list
from src.pipeline.registry.function_registry import FunctionRegistry
from src.pipeline.registry.prompt_registry import PromptRegistry
from src.schemas.schemas import BaseModelListType, Ingestion
from src.utils.datetime_utils import get_current_utc_datetime
from src.utils.filter_utils import filter_basemodels

router = Router(
    model_list=chat_model_list,
    allowed_fails=3,
    cooldown_time=10,
    num_retries=3,
)


def update_ingestion_metadata(obj: Ingestion, model: str, prompt_name: str) -> None:
    """
    Update feature metadata for an ingestion object.
    """
    obj.feature_models = (obj.feature_models or []) + [model]
    obj.feature_dates = (obj.feature_dates or []) + [get_current_utc_datetime()]
    obj.feature_types = (obj.feature_types or []) + [prompt_name]


@FunctionRegistry.register("featurize", "featurize_model")
async def featurize(
    basemodels: BaseModelListType,
    prompt_name: str,
    model_name: str = "gpt-4o-mini",
    write=None,  # noqa
    read=None,
    update_metadata=True,
    model_params: dict[str, Any] = {},
    filter_params: Optional[dict[str, Any]] = None,
) -> BaseModelListType:
    """
    Use LLMs to featurize the data contained in an Ingestion or Entry

    args:
        basemodels: a list of objects to featurize
        prompt_name: the name of the prompt from the PromptRegistry
        model_name: the name of the group of models for litellm Router
        filter_params: dictionary of field conditions to filter basemodels
            Example: {'consolidated_feature_type': ExtractedFeatureType.image}
    """
    # If filter_params provided, only process matching models
    models_to_process = basemodels
    print(f"Number of models to process: {len(models_to_process)}")
    unfiltered_models = []
    if filter_params:
        models_to_process, unfiltered_models = filter_basemodels(basemodels, filter_params)
        if not models_to_process:
            return basemodels  # Return original list if no models match filter
    print(f"Number of models to process after filtering: {len(models_to_process)}")

    prompt = PromptRegistry.get(prompt_name)

    if prompt.has_data_model():
        model_params["response_format"] = prompt.DataModel

    if not update_metadata:
        for base_model in models_to_process:
            if base_model.schema__ == "Ingestion":
                update_ingestion_metadata(base_model, model_name, prompt_name)
            elif base_model.schema__ == "Entry":
                update_ingestion_metadata(base_model.ingestion, model_name, prompt_name)

    # format prompts
    messages_list = await asyncio.gather(*(prompt.format_prompt(basemodel, read=read) for basemodel in models_to_process))

    # Run litellm Router, add results to basemodels
    responses = await asyncio.gather(*(router.acompletion(model=model_name, messages=messages, **model_params) for messages in messages_list))
    print("Done running litellm router")

    # create new entries
    parsed_responses = [prompt.parse_response(basemodel, response) for basemodel, response in zip(models_to_process, responses)]

    # flatten the results if they are lists
    new_entries = []
    for result in parsed_responses:
        if isinstance(result, list):
            new_entries.extend(result)
        else:
            new_entries.append(result)

    # Combine new entries with unfiltered models
    return new_entries + unfiltered_models
