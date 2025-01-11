# Registration System

## Overview
The pipeline uses a registration system to manage three types of components:
1. Functions (processing steps)
2. Schemas (data models)
3. Prompts (LLM templates)

Each component type has its own registry - a global dictionary that maps names to implementations.

## The Registry Classes

### 1. Function Registry

```python
class FunctionRegistry:
    registry: dict[str, dict[str, Callable]] = {
        "ingest": {},
        "parse": {},
        "chunk": {},
        "featurize": {},
        "embed": {},
        "upsert": {},
    }
```

- Organizes functions by pipeline stage
- Each stage (ingest, parse, etc.) has its own dictionary of functions
- Functions are registered using the `@FunctionRegistry.register(stage, name)` decorator

### 2. Schema Registry

```pythonpython
class SchemaRegistry:
registry: dict[str, type[ABC]] = {}
```

- Maps schema names to Pydantic model classes
- Used to validate and serialize data between pipeline stages
- Schemas are registered using the `@SchemaRegistry.register(name)` decorator

### 3. Prompt Registry

```python
class PromptRegistry:
    registry: dict[str, type[ABC]] = {}
```

- Maps prompt names to prompt template classes
- Used for LLM interactions
- Prompts are registered using the `@PromptRegistry.register(name)` decorator

## Important Note About Imports

For the registration system to work, all components must be imported when the pipeline starts. Currently, this is handled in `pipeline.py`:

```python
# Import all prompts
import src.prompts

# Import all schemas
import src.schemas

# Import all functions by stage
from src.ingestion.web.webcrawl import scrape_ingestion, scrape_urls
from src.extraction.google_html import parse_html
```

**Why This Matters:**
- Python only executes decorators when the module is imported
- If a module isn't imported, its components won't be registered
- Missing imports = missing functionality in the pipeline

**Developer Responsibility:**
When adding new components:
1. Add the appropriate decorator to your component
2. Import the module in pipeline.py
3. Use the registered name in your pipeline configuration

This manual import system is one area that could be improved through auto-discovery or a plugin system.