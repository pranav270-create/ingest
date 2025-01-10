# LLM Agent

A flexible wrapper for making LLM API calls using litellm. 
The goal is fast experimentation.
Define a prompt without any concern for the model. Try standard chat completion, structured output, etc.

Select a model and execution type without any concern for the prompt.

## Features

- **Versatility**: Using litellm, we get lots of models, sync/async, structured output, image support, etc. for very little code.
- **Separation of Concerns**: Prompts and Models are completely separate abstractions. Prompts can be structured / unstructured. Models can be any model, any provider, async/sync/streaming etc.


## Quick Start

### 1. Define Your Prompt

```python
from src.prompts.base_prompt import BasePrompt

class MyPrompt(BasePrompt):
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is the capital of {country}?"

@classmethod
def format_prompt(self, country: str) -> dict[str, str]:
    return cls.system_prompt, cls.user_prompt.format(country=country)

@classmethod
def parse_response(self, response) -> tuple[str, float]:
    content = response.choices[0].message.content
    cost = response.usage.total_cost
    return content, cost
```

### 2. Initialize the Agent

```python
from src.llm_utils.agent import Agent
from src.prompts.example import MyPrompt

agent = Agent(prompt=MyPrompt)
```

### 3. Make an API Call

```python
response, cost = agent.call(provider="openai", model="gpt-4o", country="France")
print(response)
print(cost)
```

```bash
# Output
The capital of France is Paris.
0.0001
```

