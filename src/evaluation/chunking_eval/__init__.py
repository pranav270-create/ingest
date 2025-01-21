"""
Chunking Evaluation Package
--------------------------

A comprehensive suite for evaluating and comparing document chunking methods.

Key Components:
- evaluation_utils.py: Core evaluation metrics and utilities
- elo_system.py: ELO rating system for comparing chunking methods
- llm_evaluation.py: LLM-based chunk quality assessment
- vlm_evaluation.py: Vision Language Model evaluation using document images
- run_comparative_evaluation.py: Pipeline for running comparative evaluations
- chunk_retrieval.py: Functions for retrieving and comparing chunks from different pipelines

Configuration:
- Uses chunk_evaluation.yaml for pipeline and evaluation settings
- Supports both LLM and VLM evaluation methods
- Implements ELO rating system for tracking method performance

Usage:
1. Single pipeline evaluation: evaluation_utils.evaluate_single_pipeline()
2. Comparative evaluation: run_comparative_evaluation.run_evaluation_pipeline()
3. Direct chunk comparison: llm_evaluation.compare_chunk_sets()
"""
