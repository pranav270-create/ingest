# Ingest

## Architecture Overview

The pipeline is built around a flexible, modular architecture that processes documents through multiple stages. It uses a registry pattern for function management and supports various storage backends.

### Core Components

**[Pipeline Orchestrator](pipeline_orchestrator.md)**

   Manages the overall pipeline configuration and execution

   - Loads configuration from YAML files
   - Sets up storage backend
   - Registers processing functions

**[Registry System](registry.md)**

   Three main registries handle different aspects of the pipeline:

   - `FunctionRegistry`: Manages processing functions for each pipeline stage
   - `SchemaRegistry`: Handles data schemas for different units of data
   - `PromptRegistry`: Manages prompt templates for LLM interactions

**[Storage Backend](storage.md)**

   - Abstracted read/write interface to save intermediate results
   - Supports local filesystem `LocalStorageBackend` and S3 `S3StorageBackend`

### Pipeline Stages

The pipeline processes documents through the following stages:

**Ingestion** (`/ingestion`)

   - Handles document input from various sources
   - Supports multiple ingestion methods (web scraping, local files, APIs)
   - Creates initial document metadata

**Extraction** (`/extraction`)

   - Extracts text and structure from documents
   - Supports multiple parsing methods (OCR, HTML parsing, etc.)
   - Creates structured document representations

**Chunking** (`/chunking`)

   - Breaks documents into smaller, processable pieces
   - Multiple chunking strategies available:
     - Fixed length
     - Sliding window
     - Embedding-based
     - Distance-based
     - NLP sentence
     - Topic-based
     - Regex-based

**Featurization** (`/featurization`)

   - Extracts features from document chunks
   - Supports multiple feature types and models

**Embedding** (`/embedding`)

   - Converts text into vector representations
   - Supports different embedding models and dimensions

**Upsert** (`upsert`)

   - Stores processed documents in vector database
   - Manages document metadata and relationships

### Data Schemas

The pipeline uses several core data schemas:

1. **Ingestion**
   - Tracks document metadata and processing status
   - Includes source information and timestamps

2. **Entry**
   - Represents individual document chunks
   - Contains text content and contextual information
   - Supports parent-child relationships

3. **Document**
   - Contains collections of entries
   - Maintains document-level metadata

4. **Embedding**
   - Extends Entry with vector representations
   - Includes embedding metadata

5. **Upsert**
   - Combines all necessary information for database storage
   - Includes both dense and sparse vectors

### Configuration

The pipeline is configured through YAML files that specify:

- Pipeline stages and their order
- Function parameters for each stage
- Storage backend configuration
- Collection names and metadata
- Processing options