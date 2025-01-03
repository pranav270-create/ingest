import base64
import re
from io import BytesIO
import modal
from marker.convert import convert_single_pdf
from marker.models import load_all_models
from modal import App, build, enter, method

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


# TODO: Fix for cuDNN error on the backend
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
    model_list: list = None

    @build()
    @enter()
    def load_model(self):
        model_lst = load_all_models()
        self.model_list = model_lst

    @method()
    def parse_document(self, fname: bytes) -> dict:
        full_text, images, out_meta = convert_single_pdf(
            fname, self.model_list, max_pages=None, langs=["English"], batch_multiplier=1, start_page=1
        )

        # Find all page numbers and their corresponding text
        page_pattern = r"\[ Page Number: (\d+) \]\n-{17}\n\n([\s\S]*?)(?=\[ Page Number: \d+ \]|\Z)"
        matches = re.findall(page_pattern, full_text, re.DOTALL)

        # Create a dictionary of page numbers and their corresponding text
        text_by_page = {int(page_num) + 1: text.strip() for page_num, text in matches}

        # Convert images to base64 encoded strings
        image_strings = {}
        for filename, image in images.items():
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_strings[filename] = img_str

        return {"result": text_by_page, "images": image_strings, "metadata": out_meta}
