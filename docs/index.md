# Document Processing Pipeline

## Architecture Overview

The pipeline is built around a flexible, modular architecture that processes documents through multiple stages. It uses a registry pattern for function management and supports various storage backends.

### Core Components

1. **Pipeline Orchestrator**
   - Manages the overall pipeline configuration and execution
   - Loads configuration from YAML files
   - Sets up storage backend
   - Registers processing functions

2. **Registry System**
   Three main registries handle different aspects of the pipeline:

   - `FunctionRegistry`: Manages processing functions for each pipeline stage
   - `SchemaRegistry`: Handles data schemas for different document types
   - `PromptRegistry`: Manages prompt templates for LLM interactions

3. **Storage Backend**
   - Abstracted storage interface supporting different backends
   - Handles reading and writing of intermediate results
   - Configurable through pipeline configuration

### Pipeline Stages

The pipeline processes documents through the following stages:

1. **Ingestion** (`ingest`)
   - Handles document input from various sources
   - Supports multiple ingestion methods (web scraping, local files, APIs)
   - Creates initial document metadata

2. **Parsing** (`parse`)
   - Extracts text and structure from documents
   - Supports multiple parsing methods (OCR, HTML parsing, etc.)
   - Creates structured document representations

3. **Chunking** (`chunk`)
   - Breaks documents into smaller, processable pieces
   - Multiple chunking strategies available:
     - Fixed length
     - Sliding window
     - Embedding-based
     - Distance-based
     - NLP sentence
     - Topic-based
     - Regex-based

4. **Featurization** (`featurize`)
   - Extracts features from document chunks
   - Supports multiple feature types and models

5. **Embedding** (`embed`)
   - Converts text into vector representations
   - Supports different embedding models and dimensions

6. **Upsert** (`upsert`)
   - Stores processed documents in vector database
   - Manages document metadata and relationships

### Data Models

The pipeline uses several core data models:

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