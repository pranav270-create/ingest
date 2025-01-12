# Pipeline Orchestrator

## Overview
The Pipeline Orchestrator is a class that initializes and sets up the pipeline environment. It has three main responsibilities:

1. **Load Configuration**

- Loads a YAML configuration file that defines pipeline stages and settings
- Configuration includes which functions to run and their parameters

2. **Setup Storage**

```python
def setup_storage(self):
    storage_config = self.config.get('storage', {})
    return StorageFactory.create(storage_config)
```

- Creates a storage backend based on configuration
- Storage is used to save intermediate results between pipeline stages

3. **Register Functions**

```python
def register_functions(self):
    for stage in self.config.get('stages', []):
        for function in stage.get('functions', []):
            FunctionRegistry.register(stage['name'], function['name'])
```

- Registers functions defined in the configuration
- Each function is wrapped with storage access capabilities
- Functions can then be looked up by stage and name during pipeline execution

## Key Point
The Orchestrator itself doesn't run the pipeline - it just sets up the environment. The actual pipeline execution happens in `run_pipeline.py`, which uses the environment the Orchestrator has prepared.