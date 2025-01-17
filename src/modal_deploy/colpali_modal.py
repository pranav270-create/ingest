import modal
from modal import App, build, enter, method
import subprocess
import requests
import time
import os
import json
app = App("colpali-modal")
MINUTES = 60  # seconds
HOURS = 60 * MINUTES

dependencies = [
    "torch",
    "transformers",
    "numpy",
    "pandas",
    "scikit-learn",
    "datasets",
    "evaluate",
    "accelerate",
]

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "build-essential",
        "curl",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "git",
    )
    .pip_install(*dependencies)
    .run_commands(
        "git clone https://github.com/illuin-tech/vidore-benchmark.git /vidore",
        "cd /vidore && pip install -e .",
    )
)

@app.cls(
    gpu="A100",
    image=image,
    concurrency_limit=5,
    keep_warm=0,
    allow_concurrent_inputs=1,
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
)
class Model:
    @build()
    @enter()
    def load_model(self):
        print("Initializing Vidore benchmark environment...")
        # The environment is already set up through the image setup
        print("Vidore benchmark initialization complete!")

    @method()
    def warmup(self):
        return {"status": "success", "message": "Vidore benchmark is ready"}

    @method()
    def run_benchmark(self, model_name: str, dataset_name: str) -> dict:
        """
        Run the Vidore benchmark for a specific model and dataset.
        
        Args:
            model_name: Name of the model to benchmark
            dataset_name: Name of the dataset to use
            
        Returns:
            Dictionary containing benchmark results
        """
        try:
            # Change to the Vidore directory
            os.chdir("/vidore")
            
            # Run the benchmark command
            cmd = [
                "python", 
                "-m", 
                "vidore.run_benchmark",
                "--model", model_name,
                "--dataset", dataset_name,
                "--output_dir", "/tmp/results"
            ]
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Read the results
            results_file = f"/tmp/results/{model_name}_{dataset_name}_results.json"
            with open(results_file, 'r') as f:
                results = json.load(f)
                
            return {
                "status": "success",
                "results": results,
                "stdout": process.stdout,
                "stderr": process.stderr
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "message": "Benchmark execution failed",
                "stdout": e.stdout,
                "stderr": e.stderr
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
