import subprocess
import time

import modal
import requests
from modal import App, build, enter, method

app = App("grobid-modal")
MINUTES = 60  # seconds
HOURS = 60 * MINUTES

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
        "default-jdk",
        "maven",
        "unzip",
        "wget",
    )
    .pip_install("requests")
    .run_commands(
        "git clone https://github.com/kermitt2/grobid.git /grobid",
        "cd /grobid && ./gradlew clean install",
        "cd /grobid && ./gradlew clean assemble",
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
        print("Starting GROBID build and service...")

        # Change directory to grobid and start service
        process = subprocess.Popen(
            ["./gradlew", "run"],
            cwd="/grobid",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Monitor the build process
        max_retries = 60
        for i in range(max_retries):
            try:
                # Check if process is still running
                if process.poll() is not None:
                    out, err = process.communicate()
                    print("GROBID process output:", out)
                    print("GROBID process error:", err)
                    raise Exception("GROBID process terminated unexpectedly")

                # Try to connect to the service
                response = requests.get("http://localhost:8070/api/isalive")
                if response.status_code == 200:
                    print("GROBID service is ready!")
                    break
            except requests.exceptions.ConnectionError:
                print(f"Waiting for GROBID service to start... ({i+1}/{max_retries})")
                # Read any available output
                out = process.stdout.readline()
                if out:
                    print("GROBID:", out.strip())
                time.sleep(5)
        else:
            raise Exception("GROBID service failed to start")

        print("GROBID initialization complete!")

    @method()
    def warmup(self):
        return {"status": "success", "message": "GROBID service is ready"}

    @method()
    def parse_document(self, pdf_bytes: bytes) -> dict:
        # First verify GROBID is running
        health_check = requests.get("http://localhost:8070/api/isalive")
        if health_check.status_code != 200:
            raise Exception("GROBID service is not responding")

        # Prepare the PDF file for processing
        files = {"input": ("input.pdf", pdf_bytes, "application/pdf")}

        # Parameters for full processing
        params = {
            "consolidateHeader": "1",
            "consolidateCitations": "1",
            "includeRawCitations": "1",
            "segmentSentences": "1",
            "teiCoordinates": ["ref", "biblStruct", "formula", "figure"],
        }

        # Process the document
        try:
            response = requests.post(
                "http://localhost:8070/api/processFulltextDocument",
                files=files,
                params=params,
                timeout=300,  # 5 minute timeout
            )

            if response.status_code == 200:
                return {"status": "success", "tei": response.text}
            elif response.status_code == 204:
                return {
                    "status": "no_content",
                    "message": "No content could be extracted from the document",
                }
            else:
                return {
                    "status": "error",
                    "message": f"GROBID returned status code {response.status_code}",
                    "details": response.text,
                }

        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": "Failed to process document",
                "details": str(e),
            }
