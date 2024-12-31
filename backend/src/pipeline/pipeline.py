import asyncio
import yaml
import sys
from pathlib import Path
from typing import Any
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry import FunctionRegistry
from src.pipeline.storage_backend import StorageFactory
import src.prompts  # This import will execute all the class definitions and register the prompts
import src.schemas  # This import will execute all the class definitions and register the schemas
# Ingestion
from src.ingestion.web.webcrawl import scrape_ingestion, scrape_urls
from src.ingestion.files.local import ingest_local_files
from src.ingestion.web.basic import manual_ingest
# Extraction
from src.extraction.google_html import parse_html
from src.extraction.dummy import parse_dummy
from src.extraction.ocr_service import main_ocr
from src.extraction.simple import main_simple
from src.extraction.datalab_service import main_datalab
# Chunking
from src.chunking.distance_chunking import distance_chunks
from src.chunking.embedding_chunking import embedding_chunks
from src.chunking.fixed_chunking import fixed_chunks
from src.chunking.regex_chunking import regex_chunks
from src.chunking.nlp_sentence_chunking import sentence_chunks
from src.chunking.sliding_chunking import sliding_chunks
from src.chunking.topic_chunking import topic_chunks
# Featurization
from src.featurization.get_clusters import get_clusters
from src.featurization.get_features import featurize
# Embedding
from src.embedding.get_embeddings import get_embeddings
# Upsert
from src.vector_db.etl_upsert import upsert_embeddings


class PipelineOrchestrator:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.storage = self.setup_storage()
        self.register_functions()

    def register_functions(self):
        for stage in self.config.get('stages', []):
            for function in stage.get('functions', []):
                FunctionRegistry.register(stage['name'], function['name'])
        print(f"Registered functions: {FunctionRegistry._registry}")

    def setup_storage(self):
        storage_config = self.config.get('storage', {})
        return StorageFactory.create(**storage_config)
