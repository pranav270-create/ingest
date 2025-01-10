import asyncio

from tqdm.asyncio import tqdm_asyncio

from src.evaluation.chunking_evaluation import ExtractionMethod, evaluate_extraction_chunking
from src.llm_utils.agent import Agent
from src.llm_utils.utils import Provider
from src.prompts.evaluation_prompts import ChunkEvaluationPrompt
from src.schemas.schemas import ChunkingMethod, Entry


async def evaluate_chunk_quality(chunk: Entry) -> dict:

    # instantiate agent
    agent = Agent(prompt=ChunkEvaluationPrompt)

    # call agent
    response, _ = await agent.call(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        chunk=chunk.string
    )
    return response


async def run_single_evaluation(
        pdf_path: str,
        extraction: ExtractionMethod,
        chunking: ChunkingMethod,
        batch_size: int = 5,
        delay_seconds: float = 1.0,
        **kwargs):
    """Run evaluation for a single extraction + chunking combination."""
    chunks, metrics = await evaluate_extraction_chunking(
        pdf_path=pdf_path,
        extraction_method=extraction,
        chunking_method=chunking,
        **kwargs)

    # Evaluate chunks in parallel batches
    quality_scores = []
    total_chunks = len(chunks)

    async for batch_start in tqdm_asyncio(range(0, total_chunks, batch_size), desc="Evaluating chunks", total=(total_chunks + batch_size - 1) // batch_size): # noqa
        # Get current batch of chunks
        batch_end = min(batch_start + batch_size, total_chunks)
        current_batch = chunks[batch_start:batch_end]

        # Evaluate batch in parallel
        tasks = [evaluate_chunk_quality(chunk) for chunk in current_batch]
        batch_scores = await asyncio.gather(*tasks)
        quality_scores.extend(batch_scores)

        # Add delay between batches if not the last batch
        if batch_end < total_chunks:
            await asyncio.sleep(delay_seconds)

    return {
        "extraction": extraction.value,
        "chunking": chunking.value,
        "metrics": metrics,
        "quality_scores": quality_scores,
        "num_chunks": len(chunks),
    }
