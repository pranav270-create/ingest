import asyncio
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.append(str(Path(__file__).parents[2]))

from litellm import Router

from src.llm_utils.model_lists import chat_model_list
from src.pipeline.registry.function_registry import FunctionRegistry
from src.pipeline.registry.prompt_registry import PromptRegistry
from src.schemas.schemas import BaseModelListType, Ingestion, ExtractedFeatureType, EmbeddedFeatureType
from src.utils.datetime_utils import get_current_utc_datetime

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


def filter_basemodels(basemodels: BaseModelListType, filter_params: dict[str, Any]) -> tuple[BaseModelListType, BaseModelListType]:
    """
    Filter basemodels based on provided field conditions.
    Returns tuple of (filtered_models, unfiltered_models)

    Example usage in yaml:
        filter_params:
            consolidated_feature_type: "image"
            chunk_index: 1
            embedded_feature_type: ["raw", "synthetic"]
    """
    # Map of field names to their enum classes
    enum_fields = {
        'consolidated_feature_type': ExtractedFeatureType,
        'embedded_feature_type': EmbeddedFeatureType,
    }

    # Preprocess filter_params to convert string values to enums where needed
    processed_params = {}
    for key, value in filter_params.items():
        if key in enum_fields:
            enum_class = enum_fields[key]
            if isinstance(value, list):
                # Handle list of values
                processed_values = []
                for v in value:
                    if isinstance(v, str):
                        try:
                            matching_member = next(
                                member for member in enum_class
                                if member.value == v.lower()
                            )
                            processed_values.append(matching_member)
                        except StopIteration:
                            valid_values = [member.value for member in enum_class]
                            raise ValueError(f"Invalid value '{v}' for {key}. Must be one of {valid_values}")
                    else:
                        processed_values.append(v)
                processed_params[key] = processed_values
            elif isinstance(value, str):
                # Handle single string value
                try:
                    matching_member = next(
                        member for member in enum_class 
                        if member.value == value.lower()
                    )
                    processed_params[key] = matching_member
                except StopIteration:
                    valid_values = [member.value for member in enum_class]
                    raise ValueError(f"Invalid value '{value}' for {key}. Must be one of {valid_values}")
            else:
                processed_params[key] = value
        else:
            processed_params[key] = value

    filtered = []
    unfiltered = []

    for model in basemodels:
        matches_all_conditions = True
        for field, expected_value in processed_params.items():
            actual_value = getattr(model, field, None)

            # Handle list of allowed values
            if isinstance(expected_value, list):
                if actual_value not in expected_value:
                    matches_all_conditions = False
                    break
            # Handle single value match
            elif actual_value != expected_value:
                matches_all_conditions = False
                break

        if matches_all_conditions:
            filtered.append(model)
        else:
            unfiltered.append(model)

    return filtered, unfiltered


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
