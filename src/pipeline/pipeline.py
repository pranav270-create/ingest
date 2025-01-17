import inspect
import sys
from pathlib import Path
from typing import Union, get_args, get_origin
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry.function_registry import FunctionRegistry
from src.pipeline.registry.prompt_registry import PromptRegistry
from src.pipeline.registry.schema_registry import SchemaRegistry
from src.pipeline.storage_backend import StorageFactory
from src.schemas.schemas import BaseModelListType, Embedding, Entry, Ingestion, RegisteredSchemaListType


class PipelineOrchestrator:
    def __init__(self, config_path: str):
        FunctionRegistry.autodiscover("src.ingestion.files")
        FunctionRegistry.autodiscover("src.ingestion.web")
        FunctionRegistry.autodiscover("src.extraction")
        FunctionRegistry.autodiscover("src.chunking")
        FunctionRegistry.autodiscover("src.featurization")
        FunctionRegistry.autodiscover("src.embedding")
        FunctionRegistry.autodiscover("src.upsert")

        PromptRegistry.autodiscover("src.prompts")
        SchemaRegistry.autodiscover("src.schemas")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.storage = self.setup_storage()

        self.register_functions()
        # Run parameter verification and raise exception if there are errors
        success, errors = self.verify_config_stage_parameters()
        if not success:
            error_msg = "\nParameter Validation Errors:\n"
            error_msg += "\n".join(f"- {error}" for error in errors)
            raise ValueError(error_msg)

    def setup_storage(self):
        storage_config = self.config.get("storage", {})
        return StorageFactory.create(**storage_config)


    def register_functions(self):
        for stage in self.config.get("stages", []):
            for function in stage.get("functions", []):
                FunctionRegistry.register(stage["name"], function["name"])


    def get_registered_functions(self):
        """Returns a dictionary of all registered functions by stage."""
        return {stage["name"]: [func["name"] for func in stage.get("functions", [])] for stage in self.config.get("stages", [])}


    def verify_config_stage_parameters(self):
        """Verifies that all function parameters specified in config match their function signatures.
        Returns (bool, list): Success status and list of any parameter mismatches."""

        def should_skip_param(param):
            """Helper to determine if parameter should be skipped from validation based on its type."""
            annotation = param.annotation
            if annotation == inspect.Parameter.empty:
                return False

            # Get the base type if it's a Union
            if get_origin(annotation) is Union:
                types = get_args(annotation)
                # Check each type in the Union
                return any(
                    (get_origin(t) is list and get_args(t)[0] in (Entry, Ingestion, Embedding)) or
                    t in (BaseModelListType, RegisteredSchemaListType)
                    for t in types
                )

            # Check if it's a List[Entry], List[Ingestion]
            return ((get_origin(annotation) is list and
                    get_args(annotation)[0] in (Entry, Ingestion, Embedding)) or
                    annotation in (BaseModelListType, RegisteredSchemaListType))

        validation_errors = []
        stages = self.config.get("stages", [])

        for stage in stages:
            stage_name = stage["name"]
            for function_config in stage.get("functions", []):
                func_name = function_config["name"]

                # Get the actual function from registry
                func = FunctionRegistry.get(stage_name, func_name)
                if not func:
                    validation_errors.append(f"Function {stage_name}.{func_name} not found in registry")
                    continue

                # Get the function's parameters
                sig = inspect.signature(func)
                required_params = {
                    name: param
                    for name, param in sig.parameters.items()
                    if param.default == inspect.Parameter.empty
                    and param.kind != inspect.Parameter.VAR_POSITIONAL
                    and param.kind != inspect.Parameter.VAR_KEYWORD
                    and not should_skip_param(param)  # Skip parameters of specified types
                }

                # Get configured parameters (if any)
                configured_params = function_config.get("params", {})

                # Check for missing required parameters
                for param_name in required_params:
                    if param_name not in configured_params:
                        validation_errors.append(f"Required parameter '{param_name}' missing for {stage_name}.{func_name}")

                # Check for extra parameters that don't exist in function signature
                for param_name in configured_params:
                    if param_name not in sig.parameters:
                        validation_errors.append(f"Unknown parameter '{param_name}' specified for {stage_name}.{func_name}")

                # Validate added_metadata keys if return type is Ingestion
                if function_config.get("return") == "Ingestion" and "added_metadata" in configured_params:
                    metadata = configured_params["added_metadata"]
                    ingestion_fields = Ingestion.model_fields.keys()

                    for key in metadata:
                        if key not in ingestion_fields:
                            validation_errors.append(f"Metadata key '{key}' in {stage_name}.{func_name} is not a valid field in Ingestion model")

        return len(validation_errors) == 0, validation_errors