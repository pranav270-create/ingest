

This example ingests all files in a folder and adds metadata to each file.
The only required field is `directory_path`.
The keys in `added_metadata` are fields in the `Ingestion` model.

```yaml
  - name: ingest
    functions:
      - name: local
        input: null
        return: Ingestion
        params:
            directory_path: "/path/to/folder/with/data"
            added_metadata:
                content_type: "other"
                ingestion_method: "local_file"
                file_type: "pdf"
```

```python
@FunctionRegistry.register("ingest", "local")
async def ingest_local_files(
    directory_path: str,
    added_metadata: dict = None,
    write=None, **kwargs
) -> list[Ingestion]: # noqa
```