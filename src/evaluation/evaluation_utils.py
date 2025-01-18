from dataclasses import dataclass
from typing import Dict, Optional
import json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime


@dataclass
class ELOSystem:
    """ELO rating system for chunking methods."""
    k_factor: float = 32.0
    default_rating: float = 1500.0
    ratings_file: str = "chunking_elo_ratings.json"
    history_file: str = "chunking_elo_history.csv"
    
    def __post_init__(self):
        self.ratings = self._load_ratings()
        self.history = self._load_history()
    
    def _load_ratings(self) -> Dict[str, float]:
        """Load existing ratings from file or create new."""
        try:
            with open(self.ratings_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _load_history(self) -> pd.DataFrame:
        """Load rating history from CSV or create new."""
        try:
            return pd.read_csv(self.history_file)
        except FileNotFoundError:
            return pd.DataFrame(columns=[
                'timestamp', 'pipeline_a', 'pipeline_b', 
                'rating_a_before', 'rating_b_before',
                'rating_a_after', 'rating_b_after',
                'score', 'num_comparisons'
            ])
    
    def _save_ratings(self):
        """Save current ratings to file."""
        with open(self.ratings_file, 'w') as f:
            json.dump(self.ratings, f, indent=2)
    
    def _save_history(self):
        """Save rating history to CSV."""
        self.history.to_csv(self.history_file, index=False)
    
    def get_rating(self, pipeline_id: str) -> float:
        """Get rating for a pipeline, initializing if needed."""
        return self.ratings.get(str(pipeline_id), self.default_rating)
    
    def get_rating_history(self, pipeline_id: str) -> pd.DataFrame:
        """Get historical ratings for a specific pipeline."""
        pipeline_id = str(pipeline_id)
        # Combine ratings where pipeline was either A or B
        history_a = self.history[self.history['pipeline_a'] == pipeline_id][['timestamp', 'rating_a_after']].rename(columns={'rating_a_after': 'rating'})
        history_b = self.history[self.history['pipeline_b'] == pipeline_id][['timestamp', 'rating_b_after']].rename(columns={'rating_b_after': 'rating'})
        combined = pd.concat([history_a, history_b]).sort_values('timestamp')
        return combined
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score using ELO formula."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    
    def update_ratings(self, pipeline_a: str, pipeline_b: str, score: float, num_comparisons: int = 1):
        """Update ratings based on comparison outcome.
        
        Args:
            pipeline_a: ID of first pipeline
            pipeline_b: ID of second pipeline
            score: 1.0 for A win, 0.0 for B win, 0.5 for draw
            num_comparisons: Number of individual comparisons that contributed to the score
        """
        # Initialize ratings if needed
        rating_a_before = self.get_rating(pipeline_a)
        rating_b_before = self.get_rating(pipeline_b)
        
        # Calculate expected scores
        expected_a = self.expected_score(rating_a_before, rating_b_before)
        
        # Update ratings
        rating_a_after = rating_a_before + self.k_factor * (score - expected_a)
        rating_b_after = rating_b_before + self.k_factor * ((1.0 - score) - (1.0 - expected_a))
        
        # Store new ratings
        self.ratings[str(pipeline_a)] = rating_a_after
        self.ratings[str(pipeline_b)] = rating_b_after
        
        # Record history
        new_record = pd.DataFrame([{
            'timestamp': datetime.now().isoformat(),
            'pipeline_a': str(pipeline_a),
            'pipeline_b': str(pipeline_b),
            'rating_a_before': rating_a_before,
            'rating_b_before': rating_b_before,
            'rating_a_after': rating_a_after,
            'rating_b_after': rating_b_after,
            'score': score,
            'num_comparisons': num_comparisons
        }])
        self.history = pd.concat([self.history, new_record], ignore_index=True)
        
        # Save both files
        self._save_ratings()
        self._save_history()
        
        return rating_a_after, rating_b_after
    
    def plot_rating_history(self, pipeline_ids: list[str] = None):
        """Plot rating history for specified pipelines (or all if none specified)."""
        import matplotlib.pyplot as plt
        
        if pipeline_ids is None:
            # Get all unique pipeline IDs from history
            pipeline_ids = set(self.history['pipeline_a'].unique()) | set(self.history['pipeline_b'].unique())
        
        plt.figure(figsize=(12, 6))
        for pipeline_id in pipeline_ids:
            history = self.get_rating_history(pipeline_id)
            plt.plot(pd.to_datetime(history['timestamp']), history['rating'], label=f'Pipeline {pipeline_id}')
        
        plt.title('ELO Rating History')
        plt.xlabel('Time')
        plt.ylabel('Rating')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def calculate_chunk_comparison_score(comp_a: int, comp_b: int) -> float:
    """Calculate comparison score between two chunk sets.
    Returns score between 0 and 1, where higher favors set A.
    """
    if comp_a == comp_b:
        return 0.5
    elif comp_a == 0 and comp_b == 0:
        return 0.5
    else:
        # Normalize to 0-1 range
        total = comp_a + comp_b
        if total == 0:
            return 0.5
        return comp_a / total


def get_chunk_metrics(chunks: list) -> dict:
    """Calculate metrics for a chunk set."""
    if not chunks:
        return {"count": 0, "avg_length": 0}
    
    lengths = [len(chunk.string) for chunk in chunks]
    return {
        "count": len(chunks),
        "avg_length": np.mean(lengths),
        "std_length": np.std(lengths),
    } 

def run_elo_analysis(pipeline_ids: list[str] = None) -> dict:
    """Analyze ELO history and current standings for pipelines."""
    elo_system = ELOSystem()
    
    # If no pipeline IDs specified, get all from history
    if pipeline_ids is None:
        history_df = elo_system._load_history()
        pipeline_ids = list(set(history_df['pipeline_a'].unique()) | set(history_df['pipeline_b'].unique()))
    
    # Get current ratings and stats
    results = {
        "current_ratings": {},
        "total_comparisons": {},
        "win_rates": {}
    }
    
    for pipeline_id in pipeline_ids:
        # Current rating
        results["current_ratings"][pipeline_id] = elo_system.get_rating(pipeline_id)
        
        # Get pipeline history
        history = elo_system.history
        pipeline_matches = history[
            (history['pipeline_a'] == str(pipeline_id)) | 
            (history['pipeline_b'] == str(pipeline_id))
        ]
        
        # Calculate total comparisons
        total_comparisons = pipeline_matches['num_comparisons'].sum()
        results["total_comparisons"][pipeline_id] = int(total_comparisons)
        
        # Calculate win rate
        wins = len(pipeline_matches[
            ((history['pipeline_a'] == str(pipeline_id)) & (history['score'] > 0.5)) |
            ((history['pipeline_b'] == str(pipeline_id)) & (history['score'] < 0.5))
        ])
        if len(pipeline_matches) > 0:
            win_rate = wins / len(pipeline_matches)
            results["win_rates"][pipeline_id] = round(win_rate, 3)
        else:
            results["win_rates"][pipeline_id] = 0.0
            
    return results

def print_elo_report(analysis_results: dict):
    """Print formatted ELO analysis results."""
    print("\nELO Rating Report")
    print("=" * 50)
    
    # Sort pipelines by current rating
    sorted_pipelines = sorted(
        analysis_results["current_ratings"].keys(),
        key=lambda x: analysis_results["current_ratings"][x],
        reverse=True
    )
    
    for pipeline_id in sorted_pipelines:
        print(f"\nPipeline {pipeline_id}")
        print(f"Current Rating: {analysis_results['current_ratings'][pipeline_id]:.1f}")
        print(f"Total Comparisons: {analysis_results['total_comparisons'][pipeline_id]}")
        print(f"Win Rate: {analysis_results['win_rates'][pipeline_id]:.1%}") 