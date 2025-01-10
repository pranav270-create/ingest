import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Literal, Optional, Union

sys.path.append(str(Path(__file__).parents[2]))

from litellm import Router

from src.llm_utils.utils import text_cost_parser
from src.pipeline.registry import FunctionRegistry
from src.prompts.registry import PromptRegistry
from src.schemas.schemas import Document, Entry, Ingestion
from src.utils.datetime_utils import get_current_utc_datetime

model_list = [
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

router = Router(
    model_list=model_list,
    default_max_parallel_requests=50,
    allowed_fails=2,
    cooldown_time=20,
    num_retries=3,
)


def update_metadata(obj: Ingestion, model: str, feature_class_name: str) -> None:
    """Update feature metadata for an ingestion object.

    Args:
        obj: The Entry object to update
        model: LLM model name
        feature_class_name: The feature class name to add
    """
    if obj.feature_models is None:
        obj.feature_models = [model]
    else:
        obj.feature_models.append(model)

    if obj.feature_dates is None:
        obj.feature_dates = [get_current_utc_datetime()]
    else:
        obj.feature_dates.append(get_current_utc_datetime())

    if obj.feature_types is None:
        obj.feature_types = [feature_class_name]
    else:
        obj.feature_types.append(feature_class_name)


@FunctionRegistry.register("featurize", "featurize_model")
async def featurize(
    basemodels: Union[list[Ingestion], list[Document], list[Entry]],
    feature_class_name: str,
    basemodel_name: Optional[Literal["Ingestion", "Document", "Entry"]] = None,
    write=None,  # noqa
    read=None,
    sql_only=False,
    **kwargs,
) -> Union[list[Ingestion], list[Document], list[Entry]]:
    """
    Use LiteLLM to featurize the data contained in an Ingestion, Document, or Entry

    args:
        basemodels: a list of objects to featurize
        feature_class_name: the name of the prompt
        basemodel_name: Ingestion, Document, or Entry
    """

    feature_class = PromptRegistry.get(feature_class_name)

    # grab kwargs from yaml
    model = kwargs.get("model", "gpt-4o-mini")
    max_tokens = kwargs.get("max_tokens", 512)

    if not sql_only:
        for base_model in basemodels:
            if base_model.schema__ == "Ingestion":
                update_metadata(base_model, model, feature_class_name)
            elif base_model.schema__ == "Document":
                for entry in base_model.entries:
                    update_metadata(entry.ingestion, model, feature_class_name)
            elif base_model.schema__ == "Entry":
                update_metadata(base_model.ingestion, model, feature_class_name)

        # if we are featurizing a Document, we need to flatten the entries
        if basemodel_name and basemodel_name != basemodels[0].schema__:
            assert basemodels[0].schema__ == "Document" and basemodel_name == "Entry"
            documents = basemodels
            basemodels = [entry for document in documents for entry in document.entries]

    # format prompts
    system_prompts, user_prompts = zip(*[await feature_class.format_prompt(entry, read=read) for entry in basemodels])

    kwargs = {
        "max_tokens": max_tokens,
    }
    if feature_class.DataModel:
        kwargs["response_format"] = feature_class.DataModel

    # Run LLM Router
    tasks = [
        router.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sp}, {"role": "user", "content": up}],
            **kwargs,
        )
        for sp, up in zip(system_prompts, user_prompts)
    ]
    responses = await asyncio.gather(*tasks)

    # add the response to the entries/ingestions/documents
    basemodel_responses = {i: json.loads(text_cost_parser(response)[0]) for i, response in enumerate(responses)}
    return feature_class.parse_response(basemodels, basemodel_responses)