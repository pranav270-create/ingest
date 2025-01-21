import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import pandas as pd

"""
Implementation of the ELO rating system for chunking pipelines.
Manages and updates ELO ratings based on head-to-head comparisons.
Provides methods for rating calculation and historical tracking.
"""


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
            with open(self.ratings_file) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _load_history(self) -> pd.DataFrame:
        """Load rating history from CSV or create new with proper dtypes."""
        try:
            dtypes = {
                "timestamp": "str",
                "pipeline_a": "str",
                "pipeline_b": "str",
                "rating_a_before": "float64",
                "rating_b_before": "float64",
                "rating_a_after": "float64",
                "rating_b_after": "float64",
                "score": "float64",
                "num_comparisons": "int64",
            }
            return pd.read_csv(self.history_file, dtype=dtypes)
        except FileNotFoundError:
            return pd.DataFrame(
                {
                    "timestamp": pd.Series(dtype="str"),
                    "pipeline_a": pd.Series(dtype="str"),
                    "pipeline_b": pd.Series(dtype="str"),
                    "rating_a_before": pd.Series(dtype="float64"),
                    "rating_b_before": pd.Series(dtype="float64"),
                    "rating_a_after": pd.Series(dtype="float64"),
                    "rating_b_after": pd.Series(dtype="float64"),
                    "score": pd.Series(dtype="float64"),
                    "num_comparisons": pd.Series(dtype="int64"),
                }
            )

    def _save_ratings(self):
        """Save current ratings to file."""
        with open(self.ratings_file, "w") as f:
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
        history_a = self.history[self.history["pipeline_a"] == pipeline_id][["timestamp", "rating_a_after"]].rename(
            columns={"rating_a_after": "rating"}
        )
        history_b = self.history[self.history["pipeline_b"] == pipeline_id][["timestamp", "rating_b_after"]].rename(
            columns={"rating_b_after": "rating"}
        )
        combined = pd.concat([history_a, history_b]).sort_values("timestamp")
        return combined

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score using ELO formula."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(self, pipeline_a: str, pipeline_b: str, score: float, num_comparisons: int = 1):
        """Update ratings based on comparison outcome."""
        # Debug logging
        print("\nELO Update:")
        print(f"Pipeline A ({pipeline_a}) vs Pipeline B ({pipeline_b})")
        print(f"Score: {score}")

        # Initialize ratings if needed
        rating_a_before = self.get_rating(pipeline_a)
        rating_b_before = self.get_rating(pipeline_b)

        print(f"Ratings before - A: {rating_a_before:.1f}, B: {rating_b_before:.1f}")

        # Calculate expected scores
        expected_a = self.expected_score(rating_a_before, rating_b_before)
        print(f"Expected score for A: {expected_a:.3f}")

        # Update ratings
        rating_a_after = rating_a_before + self.k_factor * (score - expected_a)
        rating_b_after = rating_b_before + self.k_factor * ((1.0 - score) - (1.0 - expected_a))

        print(f"Ratings after - A: {rating_a_after:.1f}, B: {rating_b_after:.1f}")

        # Store new ratings
        self.ratings[str(pipeline_a)] = rating_a_after
        self.ratings[str(pipeline_b)] = rating_b_after

        # Record history with proper types
        new_record = pd.DataFrame(
            {
                "timestamp": [datetime.now().isoformat()],
                "pipeline_a": [str(pipeline_a)],
                "pipeline_b": [str(pipeline_b)],
                "rating_a_before": [float(rating_a_before)],
                "rating_b_before": [float(rating_b_before)],
                "rating_a_after": [float(rating_a_after)],
                "rating_b_after": [float(rating_b_after)],
                "score": [float(score)],
                "num_comparisons": [int(num_comparisons)],
            }
        )

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
            pipeline_ids = set(self.history["pipeline_a"].unique()) | set(self.history["pipeline_b"].unique())

        plt.figure(figsize=(12, 6))
        for pipeline_id in pipeline_ids:
            history = self.get_rating_history(pipeline_id)
            plt.plot(pd.to_datetime(history["timestamp"]), history["rating"], label=f"Pipeline {pipeline_id}")

        plt.title("ELO Rating History")
        plt.xlabel("Time")
        plt.ylabel("Rating")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def get_rating_confidence(self, pipeline_id: str) -> float:
        """Calculate confidence interval for a pipeline's rating."""
        history = self.get_rating_history(pipeline_id)
        n_matches = len(history)
        if n_matches < 2:
            return float("inf")

        # Standard error of the mean
        std_dev = history["rating"].std()
        return 1.96 * std_dev / np.sqrt(n_matches)

    def run_elo_analysis(self, pipeline_ids: List[str]) -> Dict:
        """Run ELO analysis for a set of pipelines."""
        # Get current ratings and stats
        results = {"current_ratings": {}, "total_comparisons": {}, "win_rates": {}}

        for pipeline_id in pipeline_ids:
            # Current rating
            results["current_ratings"][pipeline_id] = self.get_rating(pipeline_id)

            # Get pipeline history
            history = self.history
            pipeline_matches = history[(history["pipeline_a"] == str(pipeline_id)) | (history["pipeline_b"] == str(pipeline_id))]

            # Calculate total comparisons
            total_comparisons = pipeline_matches["num_comparisons"].sum()
            results["total_comparisons"][pipeline_id] = int(total_comparisons)

            # Calculate win rate
            wins = len(
                pipeline_matches[
                    ((history["pipeline_a"] == str(pipeline_id)) & (history["score"] > 0.5))
                    | ((history["pipeline_b"] == str(pipeline_id)) & (history["score"] < 0.5))
                ]
            )

            if len(pipeline_matches) > 0:
                win_rate = wins / len(pipeline_matches)
                results["win_rates"][pipeline_id] = round(win_rate, 3)
            else:
                results["win_rates"][pipeline_id] = 0.0

        return results


def run_elo_analysis(pipeline_ids: List[str]) -> Dict:
    """Run ELO analysis for a set of pipelines."""
    elo_system = ELOSystem()
    return elo_system.run_elo_analysis(pipeline_ids)
