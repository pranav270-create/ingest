import modal
from modal import App, enter, method
import requests
import subprocess
from typing import Optional, Union, List
import time

app = App("grobid-modal")
MINUTES = 60
HOURS = 60 * MINUTES

# Use GROBID's base image and add Python 3.9
image = (
    modal.Image.from_registry("grobid/grobid:0.8.1")
    .apt_install(
        "software-properties-common",
        "curl",
        "git",  # Add git for cloning
    )
    .run_commands(
        # Add deadsnakes PPA for Python 3.9
        "add-apt-repository ppa:deadsnakes/ppa",
        "apt-get update",
        "apt-get install -y python3.9 python3.9-distutils",
        # Install pip for Python 3.9
        "curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9",
        # Make Python 3.9 the default
        "update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1",
        "update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1",
        # Clone GROBID repo and set up directories
        "git clone --depth 1 https://github.com/kermitt2/grobid.git /tmp/grobid",
        "mkdir -p /root/grobid-home",
        "cp -r /tmp/grobid/grobid-home/* /root/grobid-home/",
        "mkdir -p /opt/grobid/grobid-service/config/",
        # Copy and modify config
        "sed 's|grobidHome: \"grobid-home\"|grobidHome: \"/root/grobid-home\"|' /tmp/grobid/grobid-home/config/grobid.yaml > /opt/grobid/grobid-service/config/config.yaml",
        # Cleanup
        "rm -rf /tmp/grobid",
        # Verify Python version
        "python3 --version",
    )
    .pip_install(
        "requests",
        "modal",
        "typing-extensions",
        "synchronicity",
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
class GrobidService:
    def __init__(self):
        self.grobid_url = "http://localhost:8070"

    @enter()
    def start_service(self):
        # GROBID should already be configured in the base image
        # Just need to start the service
        subprocess.run([
            "/opt/grobid/grobid-service/bin/grobid-service", 
            "server", 
            "/opt/grobid/grobid-service/config/config.yaml"
        ], check=True)

        time.sleep(30)  # Give GROBID time to start

    @method()
    def process_pdf(
        self, 
        pdf_bytes: bytes,
        consolidate_header: Optional[int] = 1,
        consolidate_citations: Optional[int] = 0,
        consolidate_funders: Optional[int] = 0,
        include_raw_citations: Optional[bool] = False,
        include_raw_affiliations: Optional[bool] = False,
        include_raw_copyrights: Optional[bool] = False,
        tei_coordinates: Optional[List[str]] = None,
        segment_sentences: Optional[bool] = False,
        generate_ids: Optional[bool] = False,
        start_page: Optional[int] = -1,
        end_page: Optional[int] = -1,
        flavor: Optional[str] = None,
    ) -> dict:
        """
        Process a PDF document using GROBID with full parameter support.
        """
        # Prepare the base files dict with PDF
        files = {
            'input': ('input.pdf', pdf_bytes, 'application/pdf')
        }
        
        # Add optional parameters
        data = {}
        if consolidate_header is not None:
            data['consolidateHeader'] = str(consolidate_header)
        if consolidate_citations is not None:
            data['consolidateCitations'] = str(consolidate_citations)
        if consolidate_funders is not None:
            data['consolidateFunders'] = str(consolidate_funders)
        if include_raw_citations:
            data['includeRawCitations'] = '1'
        if include_raw_affiliations:
            data['includeRawAffiliations'] = '1'
        if include_raw_copyrights:
            data['includeRawCopyrights'] = '1'
        if segment_sentences:
            data['segmentSentences'] = '1'
        if generate_ids:
            data['generateIDs'] = '1'
        if start_page != -1:
            data['start'] = str(start_page)
        if end_page != -1:
            data['end'] = str(end_page)
        if flavor:
            data['flavor'] = flavor
            
        # Handle TEI coordinates (multiple values allowed)
        if tei_coordinates:
            for coord in tei_coordinates:
                if 'teiCoordinates' not in files:
                    files['teiCoordinates'] = []
                files['teiCoordinates'].append(('teiCoordinates', coord))

        try:
            response = requests.post(
                f"{self.grobid_url}/api/processFulltextDocument",
                files=files,
                data=data,
                timeout=300
            )
            
            if response.status_code == 200:
                return {"text": response.text}
            elif response.status_code == 204:
                return {"error": "No content could be extracted"}
            elif response.status_code == 400:
                return {"error": "Bad request - check parameters"}
            elif response.status_code == 503:
                return {"error": "Service unavailable - server busy"}
            else:
                return {"error": f"GROBID error: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Processing error: {str(e)}"}