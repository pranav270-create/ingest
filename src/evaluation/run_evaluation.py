import asyncio
import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import List, Dict

# Import chunking modules to ensure registration
import src.chunking.embedding_chunking
import src.chunking.textract_chunking
import src.chunking.sliding_chunking

from src.evaluation.chunking_evaluation import ExtractionMethod, evaluate_extraction_chunking
from src.evaluation.llm_evaluation import evaluate_chunk_quality, compare_chunk_sets
from src.schemas.schemas import ChunkingMethod

async def run_single_evaluation(pdf_path: str, extraction: ExtractionMethod, chunking: ChunkingMethod, **kwargs):
    """Run evaluation for a single extraction + chunking combination."""
    chunks, metrics = await evaluate_extraction_chunking(
        pdf_path=pdf_path,
        extraction_method=extraction,
        chunking_method=chunking,
        **kwargs
    )
    
    # Get LLM quality scores for each chunk
    quality_scores = []
    for chunk in chunks:
        score = await evaluate_chunk_quality(chunk)
        quality_scores.append(score)
    
    return {
        "extraction": extraction.value,
        "chunking": chunking.value,
        "metrics": metrics,
        "quality_scores": quality_scores,
        "num_chunks": len(chunks)
    }

def plot_comparison(results: List[Dict]):
    """Plot comparison of different methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot diversity metrics
    methods = [f"{r['extraction']}-{r['chunking']}" for r in results]
    diversities = [r['metrics']['embedding_diversity'] for r in results]
    ax1.bar(methods, diversities)
    ax1.set_title('Embedding Diversity')
    ax1.set_xticklabels(methods, rotation=45)
    
    # Plot average quality scores
    avg_qualities = [
        sum(s['text_clarity'] + s['coherence'] + s['organization'])/3 
        for r in results 
        for s in r['quality_scores']
    ]
    ax2.bar(methods, avg_qualities)
    ax2.set_title('Average Quality Score')
    ax2.set_xticklabels(methods, rotation=45)
    
    plt.tight_layout()
    plt.show()

async def main():
    parser = argparse.ArgumentParser(description='Evaluate different extraction and chunking methods')
    parser.add_argument('--pdf_path', type=str, required=True, help='Path to PDF file')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Chunk size for methods that use it')
    parser.add_argument('--output', type=str, help='Path to save results JSON')
    parser.add_argument('--plot', action='store_true', help='Plot comparison results')
    
    args = parser.parse_args()
    
    # Run all combinations
    combinations = [
        (ExtractionMethod.TEXTRACT, ChunkingMethod.EMBEDDING),
        (ExtractionMethod.TEXTRACT, ChunkingMethod.TEXTRACT),
        (ExtractionMethod.OCR, ChunkingMethod.EMBEDDING)
    ]
    
    results = []
    for extraction, chunking in combinations:
        print(f"\nEvaluating {extraction.value} + {chunking.value}...")
        result = await run_single_evaluation(
            args.pdf_path,
            extraction,
            chunking,
            chunk_size=args.chunk_size
        )
        results.append(result)
        print(f"Done. Found {result['num_chunks']} chunks.")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")
    
    if args.plot:
        plot_comparison(results)

if __name__ == "__main__":
    asyncio.run(main()) 