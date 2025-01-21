"""
Module for LLM-based evaluation of document chunks.
Implements text-only evaluation methods using Language Models.
Provides structured prompts and scoring mechanisms for chunk quality assessment.
"""

import asyncio
from typing import Dict, List

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from src.evaluation.chunking_eval.elo_system import ELOSystem, run_elo_analysis
from src.evaluation.experimental.chunking_evaluation import ExtractionMethod, evaluate_extraction_chunking
from src.featurization.get_features import featurize
from src.prompts.evaluation_prompts.llm_relative_prompt import LLMRelativeEvaluationPrompt
from src.schemas.schemas import ChunkComparison, ChunkingMethod, Entry

client = AsyncOpenAI()


async def get_completion(messages: list[dict[str, str]], model: str = "gpt-4-turbo-preview") -> dict:
    """Use OpenAI API to get completion."""
    try:
        completion = await client.chat.completions.create(model=model, messages=messages, temperature=0.0)

        # Get the raw text response
        text = completion.choices[0].message.content

        # Clean up markdown if present
        if text.startswith("```") and text.endswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
            if text.startswith("json"):
                text = "\n".join(text.split("\n")[1:])

        completion.choices[0].message.content = text.strip()
        return completion

    except Exception as e:
        print(f"Error in get_completion: {str(e)}")
        raise


async def evaluate_chunk_quality(chunk: Entry) -> Dict:
    """Evaluate a single chunk's quality using the rubric prompt."""
    result = await featurize([chunk], "LLM_chunk_rubric")
    # Return the evaluation scores from the first (and only) result
    return result[0].evaluation_scores


async def compare_chunk_sets(chunks_a: List[Entry], chunks_b: List[Entry], pipeline_a: str = None, pipeline_b: str = None) -> Dict:
    """Compare two sets of chunks using relative evaluation prompt."""
    comparison = ChunkComparison(chunks_a=chunks_a, chunks_b=chunks_b, winner=None, reasoning="")

    system_prompt, user_prompt = await LLMRelativeEvaluationPrompt.format_prompt(comparison)
    response = await get_completion([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
    comparison = LLMRelativeEvaluationPrompt.parse_response([comparison], {0: response})[0]

    result = {"winner": comparison.winner, "reasoning": comparison.reasoning}

    # Handle ELO updates if pipeline IDs provided
    if pipeline_a and pipeline_b:
        elo_system = ELOSystem()
        elo_score = 1.0 if result["winner"] == "A" else 0.0
        elo_system.update_ratings(pipeline_a, pipeline_b, elo_score, num_comparisons=len(chunks_a) + len(chunks_b))
        analysis = run_elo_analysis([pipeline_a, pipeline_b])
        result["elo_ratings"] = analysis["current_ratings"]

    return result


async def run_single_evaluation(pdf_path: str, extraction: ExtractionMethod, chunking: ChunkingMethod, **kwargs):
    """Run evaluation for a single extraction + chunking combination."""
    chunks, metrics = await evaluate_extraction_chunking(pdf_path=pdf_path, extraction_method=extraction, chunking_method=chunking, **kwargs)

    results = await featurize(chunks, "LLM_chunk_rubric", "Entry")

    print(results)

    exit()

    # Evaluate chunks in parallel batches
    batch_size = 5  # Adjust based on rate limits
    quality_scores = []

    chunks_with_progress = tqdm_asyncio(range(0, len(chunks), batch_size), desc="Evaluating chunks")
    async for i in chunks_with_progress:
        batch = chunks[i : i + batch_size]
        tasks = [evaluate_chunk_quality(chunk) for chunk in batch]
        batch_scores = await asyncio.gather(*tasks)
        quality_scores.extend(batch_scores)

        if i + batch_size < len(chunks):
            await asyncio.sleep(1)

    return {
        "extraction": extraction.value,
        "chunking": chunking.value,
        "metrics": metrics,
        "quality_scores": quality_scores,
        "num_chunks": len(chunks),
    }


if __name__ == "__main__":
    asyncio.run(run_single_evaluation("C:/Users/marka/fun/ingest/2407.10701v1.pdf", ExtractionMethod.TEXTRACT, ChunkingMethod.SLIDING_WINDOW))
