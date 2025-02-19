import os
import uuid
from io import BytesIO

import modal
from modal import App, build, enter, method
from PIL import Image
from transformers import AutoModel, AutoTokenizer

app = App("ocr-modal")
MINUTES = 60  # seconds
HOURS = 60 * MINUTES

dependencies = [
    "markdown2[all]", "numpy", "verovio",
    "requests", "sentencepiece", "tokenizers>=0.15.2",
    "torch", "torchvision", "wandb",
    "shortuuid", "httpx==0.24.0",
    "deepspeed==0.12.3",
    "peft==0.4.0",
    "albumentations",
    "opencv-python",
    "tiktoken==0.6.0",
    "accelerate==0.28.0",
    "transformers==4.37.2",
    "bitsandbytes==0.41.0",
    "scikit-learn==1.2.2",
    "sentencepiece==0.1.99",
    "einops==0.6.1", "einops-exts==0.0.4", "timm==0.6.13",
    "ninja",
]

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("build-essential", "curl", "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "git")
    .pip_install(  # required to build flash-attn
        "ninja",
        "packaging",
        "wheel",
        "torch",
    )
    .run_commands(  # add flash-attn
        "CXX=g++ pip install flash-attn==2.5.8 --no-build-isolation"
    )
    .pip_install(*dependencies)
)


@app.cls(
    gpu="a10g",
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
        tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
        model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
        model = model.eval().cuda()
        self.model = model
        self.tokenizer = tokenizer

    @method()
    def warmup(self): return

    @method()
    def ocr_document(self, image_data) -> dict:
        image_bytes, mode = image_data
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        # Save it locally with a uuid
        # Create a local tmp folder
        if not os.path.exists('./tmp'):
            os.makedirs('./tmp')

        filename = f"./tmp/{uuid.uuid4()}.png"
        image.save(filename)

        try:
            # Plain texts OCR
            if mode == 'plain':
                res = self.model.chat(self.tokenizer, filename, ocr_type='ocr')
            # Format texts OCR
            elif mode == 'format':
                res = self.model.chat(self.tokenizer, filename, ocr_type='format', ocr_box='')
        finally:
            # Clean up the temporary file
            os.remove(filename)
        return res


if __name__ == "__main__":
    cls = modal.Cls.lookup("ocr-modal", "Model")
    obj = cls()
    image_data = open("/Users/pranaviyer/Desktop/Screenshot 2025-01-06 at 7.12.32 PM.png", "rb").read()
    print(obj.moondream_ocr.remote(image_data, "Please describe this image in detail"))
