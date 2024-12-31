import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.llm_utils.tokenize_utils import detokenize_embed_input, tokenize_embed_input
from src.llm_utils.utils import Provider, get_embed_cost, model_mapping


def sliding_chunking(input_array: np.ndarray, chunk_size: int, overlap_size: int, padding_value=0) -> np.ndarray:
    """crop sequences to chunk_size length, each sequence overlaps by overlap_size"""
    pad_dim = overlap_size - (input_array.shape[1] % overlap_size)
    if pad_dim > 0:
        input_array = np.pad(input_array, ((0, 0), (0, pad_dim)), constant_values=padding_value)

    output = []
    for i in range(0, input_array.shape[1] - chunk_size + 1, chunk_size - overlap_size):
        output.append(input_array[:, i : i + chunk_size])
    return np.concatenate(output, axis=0)


def batched_chunking(input_array: np.ndarray, chunk_size: int, padding_value=0) -> np.ndarray:
    """naively crop sequences to the chunk_size length, batch resulting chunks"""
    pad_dim = chunk_size - (input_array.shape[1] % chunk_size)
    if pad_dim > 0:
        input_array = np.pad(input_array, ((0, 0), (0, pad_dim)), constant_values=padding_value)

    return input_array.reshape(-1, chunk_size)


def chunk_tensors(tokens: list[int], chunk_size: int, overlap: int = None) -> tuple[np.ndarray, int]:
    """route chunking to sliding window or batched"""
    tokens_array = np.array(tokens)
    if chunk_size > tokens_array.shape[1]:
        return tokens_array
    if overlap:
        return sliding_chunking(tokens_array, chunk_size, overlap)
    else:
        return batched_chunking(tokens_array, chunk_size)


async def async_embed_text(
    client: str, model: str, provider: Provider, embed_str: list[str], overlap_window: int = None, **kwargs
) -> tuple[list[float], float]:
    """
    Accepts multidimensional array of strings, batch processes with asyncio
    """
    max_tokens = model_mapping[model].context_length
    tokenized_inputs = [tokenize_embed_input(provider, model, string) for string in embed_str]
    max_len = max(len(tokenized_input) for tokenized_input in tokenized_inputs)
    encoded_input = np.array([tokenized_input + [0] * (max_len - len(tokenized_input)) for tokenized_input in tokenized_inputs])
    chunk_input_ids = chunk_tensors(encoded_input, max_tokens, overlap_window)
    chunk_input_strings = [detokenize_embed_input(provider, model, idx) for idx in chunk_input_ids.tolist()]

    embeddings, total_cost = [], 0.0
    step_size = 128
    for i in range(0, len(chunk_input_strings), step_size):
        batch_strings = chunk_input_strings[i : i + step_size]
        if provider in [Provider.OPENAI, Provider.ANYSCALE]:
            if "dimensions" in kwargs.keys() and provider == Provider.OPENAI:
                response = await client.embeddings.create(input=batch_strings, model=model, dimensions=kwargs["dimensions"])
            else:
                response = await client.embeddings.create(input=batch_strings, model=model)
            for r in response.data:
                embeddings.extend(r.embedding)
        elif provider == Provider.VOYAGE:
            response = await client.embed(batch_strings, input_type="query", model=model)
            embeddings.extend(response.embeddings)
        elif provider == Provider.MISTRAL:
            response = await client.embeddings.create(input=batch_strings, model=model)
            embed_dict = dict(sorted({d.index: d.embedding for d in response.data}.items()))
            embeddings.extend(list(embed_dict.values()))
        elif provider == Provider.COHERE:
            response = await client.embed(batch_strings, input_type="search_query", model=model)
            embeddings.extend(response.embeddings)

        total_cost += get_embed_cost(response, model_name=model)

    # return mean across chunks
    embeddings = np.array(embeddings)
    num_chunks = int(chunk_input_ids.shape[0] / encoded_input.shape[0])
    mean_embeddings = np.mean(embeddings.reshape(num_chunks, encoded_input.shape[0], -1), axis=0)
    embeddings_list = [t.tolist() for t in mean_embeddings]
    return embeddings_list, total_cost


if __name__ == "__main__":
    import asyncio
    import os

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = "text-embedding-3-large"
    provider = Provider.OPENAI
    embed_str = ["Hello world", "This is a test"]
    embeddings, total_cost = asyncio.run(async_embed_text(client, model, provider, embed_str, dimensions=512))
    print(type(embeddings))
    print(total_cost)
