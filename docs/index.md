# Ingest

## Architecture Overview

The pipeline is built around a flexible, modular architecture that ingests documents, executes multiple layers of computation, and upserts vectors into a vector database, all while saving important metadata. 

### Core Components

**[Pipeline Orchestrator](pipeline_orchestrator.md)**

   Manages the overall pipeline configuration and execution

   - Loads configuration from YAML files
   - Sets up storage backend
   - Registers processing functions

**[Registry System](registry.md)**

   Three main registries handle different aspects of the pipeline:

   - `FunctionRegistry`: Manages data processing functions for each pipeline stage
   - `SchemaRegistry`: Handles data schemas for different units of data
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

The pipeline processes documents through the following stages:

**Ingestion** (`/ingestion`)

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