import modal
from modal import App, build, enter, method
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

app = App("document-parsing-modal")
MINUTES = 60  # seconds
HOURS = 60 * MINUTES

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("build-essential", "curl", "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "git")
    .pip_install(
        "scikit-learn>=1.3.2,<=1.4.2",
        "Pillow>=10.1.0",
        "pydantic>=2.4.2",
        "pydantic-settings>=2.0.3",
        "transformers>=4.36.2",
        "numpy>=1.26.1",
        "python-dotenv>=1.0.0",
        "torch>=2.2.2",
        "tqdm>=4.66.1",
        "tabulate>=0.9.0",
        "ftfy>=6.1.1",
        "texify>=0.1.10",
        "rapidfuzz>=3.8.1",
        "surya-ocr>=0.4.15",
        "filetype>=1.2.0",
        "regex>=2024.4.28",
        "pdftext>=0.3.10",
        "grpcio>=1.63.0",
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
    config = {
        "use_llm": True,
        "output_format": "json",
    }
    config_parser = ConfigParser(config)

    @build()
    @enter()
    def load_model(self):
        self.converter = PdfConverter(
            config=self.config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=self.config_parser.get_processors(),
            renderer=self.config_parser.get_renderer(),
        )

    @method()
    def parse_document(self, fname: bytes) -> dict:
        # save the file to a temp folder
        with open("temp.pdf", "wb") as f:
            f.write(fname)
        rendered = self.converter("temp.pdf")
        return rendered.model_dump_json(exclude=["metadata"], indent=2)
