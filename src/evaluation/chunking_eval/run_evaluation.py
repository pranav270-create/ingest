"""
Main entry point for running comparative evaluations between different chunking pipelines.
This module orchestrates the evaluation process by:
- Loading and comparing chunks from different pipelines
- Running both LLM and VLM evaluations
- Aggregating results and updating ELO ratings
- Generating evaluation reports
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import List, Dict, Optional
from datetime import datetime
import os

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import Entry
from src.evaluation.chunking_eval.chunk_retrieval import compare_pipeline_chunks, get_single_pipeline_entries
from src.evaluation.chunking_eval.llm_evaluation import evaluate_chunk_quality, compare_chunk_sets
from src.evaluation.chunking_eval.vlm_evaluation import evaluate_chunks as evaluate_chunks_vlm, can_use_vlm
from src.evaluation.chunking_eval.elo_system import ELOSystem, run_elo_analysis
import yaml

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]

async def run_evaluation_pipeline(config_path: str, eval_type: str = "comparative"):
    """Run evaluation pipeline based on config file."""
    # Resolve config path relative to project root
    config_path = PROJECT_ROOT / "src" / "config" / config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if eval_type == "comparative":
        results = await run_comparative_evaluation(config)
    else:  # individual
        results = await run_individual_evaluation(config)
    
    # Save results if enabled
    output_config = config.get('output', {})
    if output_config.get('save_results'):
        save_evaluation_results(results, output_config)

async def get_pipeline_chunks(pipeline_id: str) -> List[Entry]:
    """Get chunks from a single pipeline."""
    return await get_single_pipeline_entries(pipeline_id)

async def run_individual_evaluation(config: dict) -> List[Dict]:
    """Run individual chunk quality evaluation."""
    pipeline_config = config.get('pipeline_evaluation', {})
    if not pipeline_config.get('enabled'):
        print("Pipeline evaluation not enabled in config")
        return []
    
    pipeline_id = pipeline_config.get('pipeline_id')
    delay_seconds = pipeline_config.get('delay_seconds', 1.0)
    
    # Get chunks from pipeline
    chunks = await get_pipeline_chunks(str(pipeline_id))
    results = []
    
    print(f"\nEvaluating {len(chunks)} chunks from pipeline {pipeline_id}")
    
    for i, chunk in enumerate(chunks, 1):
        # Evaluate individual chunk quality
        quality_scores = await evaluate_chunk_quality(chunk)
        
        result = {
            'pipeline_id': pipeline_id,
            'chunk_uuid': chunk.uuid,  # Using uuid instead of id
            'chunk_text': chunk.string,  # Adding text for reference
            'quality_scores': quality_scores,
            'document_title': chunk.ingestion.document_title if chunk.ingestion else "Unknown",
            'page_range': (chunk.min_primary_index, chunk.max_primary_index)
        }
        results.append(result)
        
        print(f"Evaluated chunk {i}/{len(chunks)}")
        await asyncio.sleep(delay_seconds)
    
    # Print summary statistics
    if results:
        print("\nEvaluation Summary:")
        avg_clarity = sum(r['quality_scores']['text_clarity'] for r in results) / len(results)
        avg_coherence = sum(r['quality_scores']['coherence'] for r in results) / len(results)
        avg_organization = sum(r['quality_scores']['organization'] for r in results) / len(results)
        total_score = avg_clarity + avg_coherence + avg_organization
        
        print(f"Average Text Clarity: {avg_clarity:.2f}/5.00")
        print(f"Average Coherence: {avg_coherence:.2f}/5.00")
        print(f"Average Organization: {avg_organization:.2f}/5.00")
        print(f"Total Score: {total_score:.2f}/15.00")
    
    return results

async def run_comparative_evaluation(config: dict):
    """Run comparative evaluation between pipelines."""
    pipeline_config = config.get('pipeline_comparison', {})
    if not pipeline_config.get('enabled'):
        print("Pipeline comparison not enabled in config")
        return []
    
    pipeline_configs = pipeline_config.get('pipeline_ids', [])
    evaluation_type = pipeline_config.get('evaluation_type', 'LLM')
    delay_seconds = pipeline_config.get('delay_seconds', 1.0)
    results = []
    
    for i, pipeline_a in enumerate(pipeline_configs):
        for pipeline_b in pipeline_configs[i+1:]:
            print(f"\nComparing Pipeline {pipeline_a['id']} vs Pipeline {pipeline_b['id']}")
            
            # Get chunks from both pipelines
            comparisons = await compare_pipeline_chunks(str(pipeline_a['id']), str(pipeline_b['id']))
            
            for content_hash, comps in comparisons.items():
                for comp in comps:
                    # Use VLM evaluation if specified and possible
                    if evaluation_type == "VLM":
                        can_use_vlm_eval = can_use_vlm(comp.pipeline_a_chunks, comp.pipeline_b_chunks)
                        if can_use_vlm_eval:
                            try:
                                evaluation_result = await evaluate_chunks_vlm(
                                    comp.pipeline_a_chunks,
                                    comp.pipeline_b_chunks,
                                    pipeline_ids=(str(pipeline_a['id']), str(pipeline_b['id']))
                                )
                            except Exception as e:
                                print(f"\nVLM evaluation failed with error: {e}")
                                print(f"Falling back to LLM evaluation for page range: {comp.page_range}")
                                evaluation_result = await compare_chunk_sets(
                                    comp.pipeline_a_chunks,
                                    comp.pipeline_b_chunks,
                                    str(pipeline_a['id']),
                                    str(pipeline_b['id'])
                                )
                        else:
                            print(f"\nVLM evaluation not possible for page range: {comp.page_range} (missing locations or page files)")
                            evaluation_result = await compare_chunk_sets(
                                comp.pipeline_a_chunks,
                                comp.pipeline_b_chunks,
                                str(pipeline_a['id']),
                                str(pipeline_b['id'])
                            )
                    else:
                        evaluation_result = await compare_chunk_sets(
                            comp.pipeline_a_chunks,
                            comp.pipeline_b_chunks,
                            str(pipeline_a['id']),
                            str(pipeline_b['id'])
                        )
                    
                    if 'error' in evaluation_result:
                        print(f"Error in evaluation: {evaluation_result['error']}")
                        continue
                    
                    # Run LLM quality evaluation on individual chunks
                    quality_scores_a = []
                    quality_scores_b = []
                    
                    for chunk in comp.pipeline_a_chunks:
                        score = await evaluate_chunk_quality(chunk)
                        quality_scores_a.append(score)
                        
                    for chunk in comp.pipeline_b_chunks:
                        score = await evaluate_chunk_quality(chunk)
                        quality_scores_b.append(score)
                    
                    result = {
                        'content_hash': content_hash,
                        'page_range': comp.page_range,
                        'pipeline_a': pipeline_a['id'],
                        'pipeline_b': pipeline_b['id'],
                        'evaluation_type': evaluation_type,
                        'evaluation_result': evaluation_result,
                        'quality_scores_a': quality_scores_a,
                        'quality_scores_b': quality_scores_b
                    }
                    results.append(result)
                    
                    # Add delay between evaluations
                    await asyncio.sleep(delay_seconds)
    
    # Run final ELO analysis
    if config.get('elo', {}).get('enabled'):
        elo_analysis = run_elo_analysis([str(p['id']) for p in pipeline_configs])
        print("\nFinal ELO Ratings:")
        for pipeline_id, rating in elo_analysis['current_ratings'].items():
            print(f"Pipeline {pipeline_id}: {rating}")
    
    return results

def save_evaluation_results(results: List[Dict], output_config: dict):
    """Save evaluation results to file."""
    output_dir = Path(output_config.get('output_dir', 'evaluation_results'))
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'evaluation_results_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to evaluation config file")
    parser.add_argument(
        "--eval_type", 
        choices=["comparative", "individual"],
        default="comparative",
        help="Type of evaluation to run"
    )
    args = parser.parse_args()
    
    asyncio.run(run_evaluation_pipeline(args.config, args.eval_type))
