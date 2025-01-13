# Schema Documentation

## Core Models Overview

The schema system provides a structured way to handle document processing and storage in the pipeline, with two main models: `Ingestion` and `Entry`.

### Ingestion Model

Ingestion represents metadata about a document and its processing status. It tracks the entire lifecycle of a document from ingestion through processing.

### Entry Model

Entry represents a processed chunk or segment of a document, along with its extracted features and metadata.


### Relationship Between Models

- An `Ingestion` represents a complete document
- Multiple `Entry` objects are created from a single `Ingestion`
- Each `Entry` maintains a reference to its parent `Ingestion`
- `Entry` objects can be used independently for search and retrieval
