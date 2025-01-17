import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parents[2]))

from litellm import Router

from src.llm_utils.model_lists import chat_model_list
from src.pipeline.registry.function_registry import FunctionRegistry
from src.pipeline.registry.prompt_registry import PromptRegistry
from src.schemas.schemas import BaseModelListType, Ingestion
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


@FunctionRegistry.register("featurize", "featurize_model")
async def featurize(
    basemodels: BaseModelListType,
    prompt_name: str,
    model_name: str = "gpt-4o-mini", # model_name from litellm router
    write=None,  # noqa
    read=None,
    update_metadata=True,
    model_params: dict[str, Any] = {},
) -> BaseModelListType:
    """
    Use LLMs to featurize the data contained in an Ingestion or Entry

    args:
        basemodels: a list of objects to featurize
        prompt_name: the name of the prompt from the PromptRegistry
        model_name: the name of the group of models for litellm Router
    """

    prompt = PromptRegistry.get(prompt_name)

    if prompt.has_data_model():
        model_params["response_format"] = prompt.DataModel


    if not update_metadata:
        for base_model in basemodels:
            if base_model.schema__ == "Ingestion":
                update_ingestion_metadata(base_model, model_name, prompt_name)
            elif base_model.schema__ == "Entry":
                update_ingestion_metadata(base_model.ingestion, model_name, prompt_name)

    # format prompts
    messages_list = await asyncio.gather(*(prompt.format_prompt(basemodel, read=read) for basemodel in basemodels))

    # Run litellm Router, add results to basemodels
    responses = await asyncio.gather(*(router.acompletion(model=model_name, messages=messages, **model_params) for messages in messages_list))
    print("Done running litellm router")

    # create new entries
    new_entries = [prompt.parse_response(basemodel, response) for basemodel, response in zip(basemodels, responses)]

    return new_entries