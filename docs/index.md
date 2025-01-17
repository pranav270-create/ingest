# Ingest

## Architecture Overview

The pipeline is built around a flexible, modular architecture that ingests documents, transforms the data, and upserts vectors into a vector database, all while saving important metadata. This makes adding new data and experimenting with new retrieval techniques easy.

### Make any data retrievable in just two steps:

1) Create a new YAML config file `src/config/[file_name].yaml`

2) Run the pipeline with `python -m src.pipeline.run_pipeline --config [file_name]`


### Core Components

**[Pipeline Orchestrator](pipeline_orchestrator.md)**

   Manages the overall pipeline configuration and execution

   - Loads YAML configuration from `config/`
   - Sets up storage backend `StorageBackend`
   - Imports and registers all functions, schemas, and prompts

**[Registries](registry.md)**

   Registries enable real functions, schemas, and prompts to be invoked with the yaml file.

   - `FunctionRegistry`: Manages data processing functions for each pipeline stage
   - `SchemaRegistry`: Manages data [schemas](schemas.md) for different units of data
   - `PromptRegistry`: Manages prompt templates for LLM interactions

**[Storage Backend](storage.md)**

   - Abstracted read/write interface to save intermediate results
   - Supports local filesystem `LocalStorageBackend` and S3 `S3StorageBackend`

### Configuration

The pipeline is configured [through YAML files](configuration.md) that specify:

- Pipeline stages and their order
- Function parameters for each stage
- Storage backend configuration for saving intermediates
- Qdrant collection names for saving vectors
- Pipeline execution instructions

### Pipeline Stages

There are many types of pipeline stages. Each stage is just a function in the `FunctionRegistry`.

**[Ingestion](ingestion.md)** (`/ingestion`)

   - Handles document input from various sources
   - Creates initial document metadata

**Extraction** (`/extraction`)

   - Extracts text and structure from documents
   - Supports multiple parsing methods (OCR, HTML parsing, etc.)

**Chunking** (`/chunking`)

   - Breaks documents into smaller, processable pieces

**[Featurization](featurization.md)** (`/featurization`)

   - Uses LLMs to create synthetic representations of chunks

**Embedding** (`/embedding`)

   - Converts text/image into vector representations

**Upsert** (`upsert`)

   - Stores processed documents in [Qdrant](https://qdrant.tech/) vector database
   - Manages document metadata and relationships