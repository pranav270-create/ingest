# Storage Factory

## Purpose
The StorageFactory is a simple factory class that creates storage backends for the pipeline. It abstracts away the details of which storage system to use (local filesystem or S3), and provides a read/write interface.

## Usage

```python
# Create a local storage backend
storage = StorageFactory.create(storage_type="local", base_path="/tmp/s3")

# Create an S3 storage backend
storage = StorageFactory.create(storage_type="s3", bucket_name="my-bucket")
```

## How It Works
1. Takes a `type` parameter ("local" or "s3")
2. Takes additional configuration as keyword arguments
    - optional `base_path` for local storage, otherwise a tmp directory is used
    - required `bucket_name` for S3 storage
3. Returns the appropriate StorageBackend instance:
   - `LocalStorageBackend` for local file storage
   - `S3StorageBackend` for AWS S3 storage

The returned storage backend provides a consistent interface for reading and writing data, regardless of the underlying storage system.