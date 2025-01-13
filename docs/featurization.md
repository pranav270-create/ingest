# Featurization Stage Guide

## Overview

The featurization stage is used to extract features from the data. This is useful for creating a larger surface area of given piece of data to a query it answers. Examples include summaries, keywords, claims, facts, questions (HyDE), and more.

## Configuration

The featurization stage is configured using the `featurization` stage in the pipeline configuration file.

```yaml
stages:
  - name: featurize
    functions:
      - name: featurize_model # name of function from FunctionRegistry
        input: Entry
        return: Entry
        params:
          prompt_name: "summarize_entry" # name of prompt from PromptRegistry
          basemodel_name: "Entry"
          model_name: "openai" # name of model family from litellm router
          max_tokens: 1000
```

## Prompt Configuration

Prompts all inherit from the `BasePrompt` class.


The BasePrompt class containnsi the following:
`system`: the system prompt
`user`: the user prompt
`DataModel`: an Optional pydantic model used for structured output
`format_prompt`: a method that formats the prompts into the OpenAI messages format
`parse_response`: a method that parses the response from the LLM and returns a list of entries

Structurd Output Example:
```python
from pydantic import BaseModel
from litellm import ModelResponse

from src.schemas.schemas import BaseModelListType
from src.llm_utils.utils import text_cost_parser

class SummarizeEntry(BasePrompt):
    system = "Summarize the following entry"
    user = "Entry: {entry}"
    
    class DataModel(BaseModel):
        summary: str

    @classmethod
    def format_prompt(cls, entry: Entry) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": cls.system},
            {"role": "user", "content": cls.user.format(entry=entry)}
        ]

    @staticmethod
    def parse_response(
        entries: BaseModelListType, responses: list[ModelResponse]
    ) -> BaseModelListType:
        for entry, response in zip(entries, responses):
            entry.summary = text_cost_parser(response).summary
        return entries
```