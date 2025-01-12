import modal
import os
from modal import App, build, enter, method
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

app = App("document-parsing-modal")
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
    )
    .pip_install("marker-pdf==1.2.3")
    .run_commands(
        "echo 'Starting sed commands...'",
        "sed -i 's/for block in page.contained_blocks(document, self.block_types)/for block in page.children/' /usr/local/lib/python3.10/site-packages/marker/processors/sectionheader.py",
        "sed -i 's/block_height = line_heights\\[block.id\\]/block_height = line_heights.get(block.id, 0)/' /usr/local/lib/python3.10/site-packages/marker/processors/sectionheader.py",
        "echo 'Sed command completed'",
        "cat /usr/local/lib/python3.10/site-packages/marker/processors/sectionheader.py",
    )
)

google_secret = modal.Secret.from_dict({"GOOGLE_API_KEY": os.environ["GOOGLE_API_KEY"]})


@app.cls(
    gpu="A100",
    image=image,
    concurrency_limit=5,
    keep_warm=0,
    allow_concurrent_inputs=1,
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    secrets=[google_secret],
)
class Model:
    @build()
    @enter()
    def load_model(self):
        config = {
            "use_llm": True,
            "force_ocr": True,
            "disable_image_extraction": True,
            "output_format": "json",
            "use_fast": True,
        }
        config_parser = ConfigParser(config)
        self.converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

    @method()
    def parse_document(self, fname: bytes) -> dict:
        # Print at start
        print("=== BEFORE CONVERSION ===")
        with open(
            "/usr/local/lib/python3.10/site-packages/marker/processors/sectionheader.py",
            "r",
        ) as f:
            print(f.read())

        # Print the module's location
        import marker.processors.sectionheader as sh

        print("\n=== MODULE LOCATION ===")
        print(f"Module path: {sh.__file__}")

        # Print file contents at that location
        with open(sh.__file__, "r") as f:
            print("\n=== MODULE FILE CONTENTS ===")
            print(f.read())

        with open("temp.pdf", "wb") as f:
            f.write(fname)
        rendered = self.converter("temp.pdf")
        os.remove("temp.pdf")

        # Print after conversion
        print("\n=== AFTER CONVERSION ===")
        with open(
            "/usr/local/lib/python3.10/site-packages/marker/processors/sectionheader.py",
            "r",
        ) as f:
            print(f.read())

        text = rendered.model_dump_json(exclude=["metadata"], indent=2)
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        return text
