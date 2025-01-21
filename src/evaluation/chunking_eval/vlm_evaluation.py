"""
Module for VLM-based evaluation using ELO ratings and structured prompts.
This module handles the evaluation of document chunks using Vision Language Models (VLMs).
It compares two sets of chunks using visual context and maintains ELO ratings for different chunking pipelines.
Key features:
- VLM-based chunk comparison
- ELO rating system integration
- Visual context validation
"""

from typing import Optional, Dict
from src.schemas.schemas import Entry, ChunkEvaluation
from src.evaluation.chunking_eval.elo_system import ELOSystem, run_elo_analysis
from src.featurization.get_features import featurize
import src.prompts.evaluation_prompts.vlm_prompt
from pathlib import Path

DEFAULT_VLM_CONFIG = {
    "model": "gpt-4-vision-preview",
    "max_tokens": 1000,
    "temperature": 0.0,
    "image_detail": "auto"
}

async def evaluate_chunks(
    chunks_a: list[Entry],
    chunks_b: list[Entry],
    page_image: Optional[bytes] = None,
    pipeline_ids: Optional[tuple[str, str]] = None,
    vlm_config: Optional[dict] = None
) -> Dict:
    """Unified chunk evaluation function"""
    
    # Merge provided config with defaults
    config = {**DEFAULT_VLM_CONFIG, **(vlm_config or {})}
    
    evaluation = ChunkEvaluation(chunks_a=chunks_a, chunks_b=chunks_b)
    result = await featurize(evaluation, "vlm_elo", vlm_config=config)

    # Handle ELO updates if pipeline IDs provided
    if pipeline_ids:
        pipeline_a, pipeline_b = pipeline_ids
        elo_system = ELOSystem()
        elo_score = 1.0 if result["winner"] == "A" else 0.0
        elo_system.update_ratings(
            pipeline_a,
            pipeline_b,
            elo_score,
            num_comparisons=len(chunks_a) + len(chunks_b)
        )
        # Add ELO analysis to result
        analysis = run_elo_analysis([pipeline_a, pipeline_b])
        result["elo_ratings"] = analysis["current_ratings"]
    
    return result 

def can_use_vlm(chunks_a: list[Entry], chunks_b: list[Entry]) -> bool:
    """Check if VLM evaluation is possible for these chunks."""
    # Check if chunks have locations
    has_locations_a = all(hasattr(chunk, 'chunk_locations') and chunk.chunk_locations for chunk in chunks_a)
    has_locations_b = all(hasattr(chunk, 'chunk_locations') and chunk.chunk_locations for chunk in chunks_b)
    
    if not (has_locations_a and has_locations_b):
        return False
        
    # Check if all referenced pages exist
    page_paths_a = [loc.page_file_path for chunk in chunks_a for loc in chunk.chunk_locations]
    page_paths_b = [loc.page_file_path for chunk in chunks_b for loc in chunk.chunk_locations]
    
    all_paths = set(page_paths_a + page_paths_b)
    return all(Path(path).exists() for path in all_paths) 