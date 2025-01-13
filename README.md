# ingest
ingest it all.


To get started, use `setup.py` to install the packages.

```bash
pip install . 
```

Create a YAML config file in the `config` directory.

```yaml
# example_pipeline.yaml

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

To run this, use:

```bash
python -m src.pipeline.run_pipeline --config example_pipeline
```


