import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry.function_registry import FunctionRegistry
from src.pipeline.registry.prompt_registry import PromptRegistry
from src.pipeline.registry.schema_registry import SchemaRegistry
from src.pipeline.storage_backend import StorageFactory

# # import all prompts
# import src.prompts

# # import all schemas
# import src.schemas

# # Ingestion
# from src.ingestion.web.webcrawl import scrape_ingestion, scrape_urls
# from src.ingestion.files.local import ingest_local_files
# from src.ingestion.web.basic import manual_ingest

# # Extraction
# from src.extraction.google_html import parse_html
# from src.extraction.dummy import parse_dummy
# from src.extraction.ocr_service import main_ocr
# from src.extraction.simple import main_simple
# from src.extraction.datalab_service import main_datalab

# # Chunking
# from src.chunking.distance_chunking import distance_chunks
# from src.chunking.embedding_chunking import embedding_chunks
# from src.chunking.fixed_chunking import fixed_chunks
# from src.chunking.regex_chunking import regex_chunks
# from src.chunking.nlp_sentence_chunking import sentence_chunks
# from src.chunking.sliding_chunking import sliding_chunks
# from src.chunking.topic_chunking import topic_chunks

# # Featurization
# from src.featurization.get_clusters import get_clusters
# from src.featurization.get_features import featurize

# # Embedding
# from src.embedding.get_embeddings import get_embeddings

# # Upsert
# from src.vector_db.etl_upsert import upsert_embeddings


class PipelineOrchestrator:
    def __init__(self, config_path: str):

        FunctionRegistry.autodiscover('src.ingestion.files')
        FunctionRegistry.autodiscover('src.ingestion.web')
        FunctionRegistry.autodiscover('src.extraction')
        FunctionRegistry.autodiscover('src.chunking')
        FunctionRegistry.autodiscover('src.featurization')
        FunctionRegistry.autodiscover('src.embedding')
        FunctionRegistry.autodiscover('src.vector_db')

        PromptRegistry.autodiscover('src.prompts')
        SchemaRegistry.autodiscover('src.schemas')


        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.storage = self.setup_storage()

        self.register_functions()

    def setup_storage(self):
        storage_config = self.config.get('storage', {})
        return StorageFactory.create(**storage_config)

    def register_functions(self):
        for stage in self.config.get('stages', []):
            for function in stage.get('functions', []):
                FunctionRegistry.register(stage['name'], function['name'])

    def get_registered_functions(self):
        """Returns a dictionary of all registered functions by stage."""
        return {
            stage['name']: [
                func['name'] for func in stage.get('functions', [])
            ]
            for stage in self.config.get('stages', [])
        }

    def verify_registration(self):
        """Verifies that all functions specified in config are properly registered.
        Returns (bool, list): Success status and list of any missing functions."""
        configured_functions = self.get_registered_functions()
        missing_functions = []

        print("\nDebug Registration Info:")
        print("------------------------")
        print(f"Configured functions: {configured_functions}")
        print(f"Registry contents: {FunctionRegistry._registry}")

        for stage, functions in configured_functions.items():
            for func_name in functions:
                registered_func = FunctionRegistry.get(stage, func_name)
                print(f"Checking {stage}.{func_name}: {'Found' if registered_func else 'Not found'}")
                if not registered_func:
                    missing_functions.append(f"{stage}.{func_name}")

        return len(missing_functions) == 0, missing_functions