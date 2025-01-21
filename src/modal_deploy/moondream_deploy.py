from io import BytesIO

import modal
from modal import App, build, enter, method
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

app = App("moondream-modal")
MINUTES = 60  # seconds
HOURS = 60 * MINUTES

dependencies = [
    "Pillow",
    "torch",
    "transformers",
    "numpy",
    "einops",
    "accelerate",
    "torchvision",
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
    model = None
    tokenizer = None

    @build()
    @enter()
    def load_model(self):

        model_id = "vikhyatk/moondream2"
        revision = "2024-08-26"  # Pin to specific version
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision
        ).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    @method()
    def warmup(self): return

    @method()
    def moondream_ocr(self, image_data: bytes, question: str) -> dict:
        image = Image.open(BytesIO(image_data))
        enc_image = self.model.encode_image(image).to("cuda")
        return self.model.answer_question(enc_image, question, self.tokenizer)


if __name__ == "__main__":
    cls = modal.Cls.lookup("moondream-modal", "Model")
    obj = cls()
    image_data = open("/Users/pranaviyer/Desktop/Screenshot 2025-01-06 at 7.12.32 PM.png", "rb").read()
    print(obj.moondream_ocr.remote(image_data, "Please describe this image in detail"))
