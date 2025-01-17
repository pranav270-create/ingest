import modal
import os
from modal import App, build, enter, method
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

app = App("marker-modal")
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
    .pip_install("git+https://github.com/pranav270-create/marker.git")
)

google_secret = modal.Secret.from_dict({"GOOGLE_API_KEY": os.environ["GOOGLE_API_KEY"]})


@app.cls(
    gpu="A100",
    image=image,
    concurrency_limit=5,
    keep_warm=1,
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
    def warmup(self):
        return

    @method()
    def parse_document(self, fname: bytes) -> dict:
        with open("temp.pdf", "wb") as f:
            f.write(fname)
        rendered = self.converter("temp.pdf")
        os.remove("temp.pdf")
        text = rendered.model_dump_json(exclude=["metadata"], indent=2)
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        return text
