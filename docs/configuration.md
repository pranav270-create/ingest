# Pipeline Configuration Guide

## Overview
Pipelines are configured using YAML files that define the processing stages, their functions, and pipeline-level settings. This guide explains the configuration structure and advanced features like resuming and forking pipelines.

## Basic Configuration Structure

A pipeline configuration consists of two main sections:

- `stages`: Defines the processing steps
- `pipeline`: Contains pipeline execution settings
- `storage`: Specifies where intermediate results are stored

### Example Configuration

```yaml
stages:
 - name: ingest
    functions:
        name: local
        input: null
        return: Ingestion
        params:
            directory_path: "/path/to/data"
 - name: parse
    functions:
        name: textract
        input: Ingestion
        return: Entry

pipeline:
    version: "1.0"
    description: "my_pipeline"
    collection_name: "my_collection"
    pipeline_id: 123

storage:
    storage_type: "s3"
    bucket_name: "my-bucket"
```


## Stage Configuration

Each stage requires:

- `name`: Stage identifier
- `functions`: List of functions to execute in this stage

Review individual functions for more details on their parameters.

## Advanced Pipeline Execution Settings

It might not always be wise to run the pipeline from the start. Each step takes time, and often costs money, so we need to be able to pickup where we left off.

### Resuming from a Step

You can resume pipeline execution from a specific step by adding `resume_from_step` to the pipeline configuration:

```yaml
pipeline:
    resume_from_step: 3
    pipeline_id: 123
```
Note: The pipeline is **0-indexed**.

#### How Resume Works
1. Loads the results from the specified step (in the above example, step 2)
2. Continues processing + overwrites results from that step onwards (in the above example, step 3)
3. Useful for:
   - Recovering from failures
   - Rerunning specific stages with different parameters
   - Debugging pipeline stages

### Pipeline Forking

What if you want to run the pipeline from a specific step without overwriting the results from the previous execution? For example, you could be experimenting with different extraction or chunking methods or even hyperparameters on the same method.

Forking creates a new pipeline branch starting from a specific step. Enable forking by combining `resume_from_step` with `fork_pipeline`:

```yaml
pipeline:
    resume_from_step: 3
    fork_pipeline: true
    pipeline_id: 124
```

#### How Forking Works

1. Creates a new pipeline with a new ID
2. Copies results from the original pipeline up to the specified step
3. Continues processing in the new pipeline
4. Original pipeline remains unchanged