import asyncio
import json
from pathlib import Path
import sys
from typing import List, Dict, Optional
from datetime import datetime
import os

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.get_chunks import compare_pipeline_chunks
from src.evaluation.llm_evaluation import evaluate_chunk_quality, compare_chunk_sets
from src.evaluation.vlm_evaluation import evaluate_chunks as evaluate_chunks_vlm
from src.evaluation.evaluation_utils import ELOSystem, run_elo_analysis
import yaml

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

async def run_evaluation_pipeline(config_path: str):
    """Run evaluation pipeline based on config file."""
    # Resolve config path relative to project root if not absolute
    if not os.path.isabs(config_path):
        config_path = os.path.join(PROJECT_ROOT, "src", "config", config_path)
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Extract configuration
    pipeline_comparison = config.get('pipeline_comparison', {})
    if not pipeline_comparison.get('enabled'):
        print("Pipeline comparison not enabled in config")
        return
    
    evaluation_type = pipeline_comparison.get('evaluation_type', 'LLM')  # Default to LLM
    pipeline_configs = pipeline_comparison.get('pipeline_ids', [])
    batch_size = pipeline_comparison.get('batch_size', 5)
    delay_seconds = pipeline_comparison.get('delay_seconds', 1.0)
    
    # Get model settings
    model_config = config.get('model', {})
    provider = model_config.get('provider', 'openai')
    model_name = model_config.get('name', 'gpt-4-vision-preview')
    
    results = []
    
    # Compare each pair of pipelines
    for i, pipeline_a in enumerate(pipeline_configs):
        for j, pipeline_b in enumerate(pipeline_configs[i+1:], i+1):
            print(f"\nComparing Pipeline {pipeline_a['id']} vs Pipeline {pipeline_b['id']}")
            
            # Get chunks from both pipelines
            comparisons = await compare_pipeline_chunks(
                str(pipeline_a['id']), 
                str(pipeline_b['id'])
            )
            
            # Evaluate each comparison
            for content_hash, chunk_comparisons in comparisons.items():
                for comp in chunk_comparisons:
                    # Choose evaluation method based on config
                    if evaluation_type == "VLM":
                        evaluation_result = await evaluate_chunks_vlm(
                            comp.pipeline_a_chunks,
                            comp.pipeline_b_chunks,
                            pipeline_ids=(str(pipeline_a['id']), str(pipeline_b['id']))
                        )
                    else:  # LLM evaluation
                        print(f"\nEvaluating chunks for page range: {comp.page_range}")
                        evaluation_result = await compare_chunk_sets(
                            comp.pipeline_a_chunks,
                            comp.pipeline_b_chunks,
                            pipeline_a=str(pipeline_a['id']),
                            pipeline_b=str(pipeline_b['id'])
                        )
                        print(f"Evaluation result: {evaluation_result}")
                    
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
    
    # Save results if enabled
    output_config = config.get('output', {})
    if output_config.get('save_results'):
        output_dir = Path(output_config.get('output_dir', 'evaluation_results'))
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'evaluation_results_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Run final ELO analysis
    if config.get('elo', {}).get('enabled'):
        elo_analysis = run_elo_analysis([str(p['id']) for p in pipeline_configs])
        print("\nFinal ELO Ratings:")
        for pipeline_id, rating in elo_analysis['current_ratings'].items():
            print(f"Pipeline {pipeline_id}: {rating}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run evaluation pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to evaluation config file')
    
    args = parser.parse_args()
    asyncio.run(run_evaluation_pipeline(args.config))
