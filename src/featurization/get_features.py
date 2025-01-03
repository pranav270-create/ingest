import json
import os
import sys
from pathlib import Path
from typing import Optional
import uuid
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parents[2]))

from src.llm_utils.chat import format_chat_prompt
from src.llm_utils.process_api_requests import process_api_requests_from_file
from src.llm_utils.utils import Functionality, Provider, save_prompts_to_jsonl
from src.pipeline.registry import FunctionRegistry
from src.prompts.registry import PromptRegistry
from src.utils.datetime_utils import get_current_utc_datetime


@FunctionRegistry.register("featurize", "featurize_model")
async def featurize(basemodels: list[BaseModel], feature_class_name: str, basemodel_name: Optional[str] = None, write=None, read=None, sql_only=False, **kwargs) -> list[BaseModel]:
    feature_class = PromptRegistry.get(feature_class_name)
    # Create prompts to embed
    model = kwargs.get("model", "gpt-4o")
    provider = Provider(kwargs.get("provider", "openai"))
    functionality = Functionality(kwargs.get("functionality", "chat"))
    max_tokens = kwargs.get("max_tokens", 512)

    if not sql_only:  # We only do this when we are not just using the SQL
        # Update the entries with the feature model
        for base_model in basemodels:
            if base_model.schema__ == "Entry":
                entry = base_model
                entry.ingestion.feature_models = [model] if entry.ingestion.feature_models is None else entry.ingestion.feature_models.append(model)
                entry.ingestion.feature_dates = [get_current_utc_datetime()] if entry.ingestion.feature_dates is None else entry.ingestion.feature_dates.append(get_current_utc_datetime())
                entry.ingestion.feature_types = [feature_class_name] if entry.ingestion.feature_types is None else entry.ingestion.feature_types.append(feature_class_name)
            elif base_model.schema__ == "Ingestion":
                ingestion = base_model
                ingestion.feature_models = [model] if ingestion.feature_models is None else ingestion.feature_models.append(model)
                ingestion.feature_dates = [get_current_utc_datetime()] if ingestion.feature_dates is None else ingestion.feature_dates.append(get_current_utc_datetime())
                ingestion.feature_types = [feature_class_name] if ingestion.feature_types is None else ingestion.feature_types.append(feature_class_name)
            elif base_model.schema__ == "Document":
                document = base_model
                for entry in document.entries:
                    entry.ingestion.feature_models = [model] if entry.ingestion.feature_models is None else entry.ingestion.feature_models.append(model)
                    entry.ingestion.feature_dates = [get_current_utc_datetime()] if entry.ingestion.feature_dates is None else entry.ingestion.feature_dates.append(get_current_utc_datetime())
                    entry.ingestion.feature_types = [feature_class_name] if entry.ingestion.feature_types is None else entry.ingestion.feature_types.append(feature_class_name)

    # Format prompts, save to jsonl
    tmp_folder = os.path.join(os.getcwd(), f"tmp_{uuid.uuid4()}")
    os.makedirs(tmp_folder, exist_ok=True)
    input_file = os.path.join(os.getcwd(), tmp_folder, "featurize_prompts.jsonl")
    open(input_file, "w").close()  # Make sure to clear the file

    if not sql_only:  # We only do this when we are not just using the SQL
        if basemodel_name and basemodel_name != basemodels[0].schema__:
            assert basemodels[0].schema__ == "Document" and basemodel_name == "Entry"
            documents = basemodels
            basemodels = [entry for document in documents for entry in document.entries]        
    metadata = [{"id": i} for i in range(len(basemodels))]
    system_prompts, user_prompts = zip(*[await feature_class.format_prompt(entry, read=read) for entry in basemodels])

    prompts = format_chat_prompt(
        model,
        provider,
        system_prompts=system_prompts,
        user_prompts=user_prompts,
        metadata=metadata,
        response_model=feature_class.DataModel,
        max_tokens=max_tokens,
    )
    save_prompts_to_jsonl(prompts, input_file)

    # Set up parameters for API call
    output_filepath = os.path.join(os.getcwd(), tmp_folder, "featurize_responses")
    if os.path.exists(output_filepath):
        for file in Path(output_filepath).iterdir():
            os.remove(file)
    else:
        os.makedirs(output_filepath, exist_ok=True)

    num_entries = len(metadata)
    print(f"Processing {num_entries} entries")
    await process_api_requests_from_file(
        requests_filepath=input_file,
        save_filepath=output_filepath,
        provider=provider,
        functionality=functionality,
        model=model,
        max_attempts=3,
    )

    # Iterate over the output directory
    basemodel_responses = {}
    for file in Path(output_filepath).iterdir():
        with open(file) as f:
            for line in f:
                try:
                    raw_data = json.loads(line)
                    response = raw_data[1].get("choices")
                    entry_id = raw_data[2].get("id")
                    if feature_class.DataModel:
                        try:
                            response_dict = response[0].get("message").get("tool_calls")[0].get("function").get("arguments")
                            basemodel_responses[entry_id] = json.loads(response_dict)
                        except Exception:
                            basemodel_responses[entry_id] = feature_class.DataModel().model_dump()  # Use the default model
                    else:
                        basemodel_responses[entry_id] = {"response": response[0].get("message").get("content")}
                except Exception as e:
                    basemodel_responses[entry_id] = {"error": str(e)}
                    continue
    return feature_class.parse_response(basemodels, basemodel_responses)
