from typing import List
from base64 import b64encode
import asyncio
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

from src.schemas.schemas import Entry, ChunkingMethod
from src.llm_utils.utils import Provider
from src.prompts.parser import text_cost_parser
from src.llm_utils.api_requests import get_api_key
from src.evaluation.chunking_evaluation import ExtractionMethod, evaluate_extraction_chunking

client = AsyncOpenAI(api_key=get_api_key(Provider.OPENAI))

async def get_completion(prompt: str, model: str = "gpt-4-turbo-preview") -> str:
    """Use existing OpenAI infrastructure."""
    completion = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    text, _ = text_cost_parser(completion, model)
    return text

CHUNK_RUBRIC = """
Rate this text chunk from 1-5 on the following criteria:
1. Text Clarity (1-5): Is the text well-formed and readable?
2. Coherence (1-5): Does the chunk represent a complete, coherent unit of information?
3. Organization (1-5): Is the content well-structured within the chunk?

Provide ratings and brief explanations in JSON format.
"""

async def evaluate_chunk_quality(chunk: Entry) -> dict:
    prompt = f"{CHUNK_RUBRIC}\n\nText chunk:\n{chunk.string}"
    
    response = await get_completion(prompt)
    try:
        scores = eval(response)  # Convert string JSON to dict
        return scores
    except:
        return {
            "text_clarity": 0,
            "coherence": 0, 
            "organization": 0,
            "error": "Failed to parse LLM response"
        }

async def compare_chunk_sets(
    chunks_a: List[Entry],
    chunks_b: List[Entry],
    page_image: bytes = None
) -> dict:
    """Compare two sets of chunks for the same page."""
    if page_image:
        # Use VLM comparison when image is available
        all_chunks = [c.string for c in chunks_a + chunks_b]
        vlm_result = await compare_with_vlm(all_chunks, page_image)
        
        # Add A/B comparison to VLM result
        vlm_result["winner"] = "A" if vlm_result.get("score", 0) >= 3 else "B"
        return vlm_result
    
    # Use text-only LLM comparison when no image
    prompt = f"""Compare these two ways of chunking the same page content.
    Which better preserves the meaning and structure of the original text?
    
    Chunking A:
    {[c.string for c in chunks_a]}
    
    Chunking B:
    {[c.string for c in chunks_b]}
    
    Respond with a JSON containing:
    1. winner: "A" or "B"
    2. reasoning: Brief explanation
    """
    
    response = await get_completion(prompt)
    try:
        result = eval(response)
        return result
    except:
        return {"error": "Failed to parse LLM response"}

async def compare_with_vlm(chunks: List[str], page_image: bytes) -> dict:
    """Compare chunks against original page image using GPT-4V."""
    client = AsyncOpenAI(api_key=get_api_key(Provider.OPENAI))
    
    # Convert image to base64
    image_b64 = b64encode(page_image).decode('utf-8')
    
    prompt = f"""Here is a page from a document and its extracted text chunks.
    Evaluate how well the chunks preserve the original layout and content structure.
    
    Consider:
    1. Do the chunks respect natural document boundaries (paragraphs, sections)?
    2. Is the reading order preserved?
    3. Are visual elements (tables, figures) kept intact?
    
    Chunks:
    {chunks}
    
    Provide a score from 1-5 and explanation in JSON format.
    """
    
    response = await client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ],
        max_tokens=1000
    )
    
    text, _ = text_cost_parser(response, "gpt-4-vision-preview")
    try:
        return eval(text)
    except:
        return {"error": "Failed to parse VLM response"} 

async def run_single_evaluation(pdf_path: str, extraction: ExtractionMethod, chunking: ChunkingMethod, **kwargs):
    """Run evaluation for a single extraction + chunking combination."""
    chunks, metrics = await evaluate_extraction_chunking(
        pdf_path=pdf_path,
        extraction_method=extraction,
        chunking_method=chunking,
        **kwargs
    )
    
    # Evaluate chunks in parallel batches
    batch_size = 5  # Adjust based on rate limits
    quality_scores = []
    
    chunks_with_progress = tqdm_asyncio(
        range(0, len(chunks), batch_size),
        desc="Evaluating chunks"
    )
    async for i in chunks_with_progress:
        batch = chunks[i:i + batch_size]
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
        "num_chunks": len(chunks)
    } 