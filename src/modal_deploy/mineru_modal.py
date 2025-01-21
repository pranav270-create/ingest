import json
import os

import modal
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from modal import App, method

app = App("mineru-modal")

MINUTES = 60
HOURS = 60 * MINUTES

# CUDA setup
cuda_version = "12.2.2"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    # Set noninteractive frontend for apt
    .env(
        {
            "DEBIAN_FRONTEND": "noninteractive",
        }
    )
    .apt_install(
        "build-essential",
        "wget",
        "git",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        # "libreoffice",
    )
    # Install Python packages
    .pip_install("magic-pdf[full]", extra_index_url="https://wheels.myhloli.com")
    # Install paddlepaddle for OCR acceleration
    .pip_install(
        "paddlepaddle-gpu==3.0.0b1",
        index_url="https://www.paddlepaddle.org.cn/packages/stable/cu118/",
    )
    # Install huggingface_hub for model downloads
    .pip_install("huggingface_hub")
    .run_commands(
        # Download model download script
        "wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py -O download_models_hf.py",
        # Run the model download script
        "python download_models_hf.py",
    )
    .apt_install("jq")
    .run_commands(
        """jq '. + {"device-mode": "cuda", "table-config": (.["table-config"] + {"enable": true}), "llm-aided-config": {"formula_aided": {"enable": false}, "text_aided": {"enable": false}, "title_aided": {"enable": false, "api_key": "", "base_url": "https://api.openai.com/v1/chat/completions"}}}' /root/magic-pdf.json > tmp.json && mv tmp.json /root/magic-pdf.json"""
    )
)


@app.cls(
    gpu="A100",  # Requires GPU with at least 8GB VRAM
    image=image,
    concurrency_limit=5,
    keep_warm=1,
    allow_concurrent_inputs=1,
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
)
class MinerU:
    @method()
    def process_pdf_full(self, pdf_data: bytes) -> dict:
        # print if there is the config file
        print(f"Config file contents: {os.path.exists('/root/magic-pdf.json')}")
        print(f"Config file contents: {open('/root/magic-pdf.json').read()}")
        # args
        pdf_file_name = "/tmp/abc.pdf"  # replace with the real pdf path
        with open(pdf_file_name, "wb") as f:
            f.write(pdf_data)
        name_without_suff = pdf_file_name.split(".")[0]

        # prepare env
        local_image_dir, local_md_dir = "output/images", "output"
        image_dir = str(os.path.basename(local_image_dir))

        os.makedirs(local_image_dir, exist_ok=True)

        image_writer, md_writer = FileBasedDataWriter(
            local_image_dir
        ), FileBasedDataWriter(local_md_dir)

        # read bytes
        reader1 = FileBasedDataReader("")
        pdf_bytes = reader1.read(pdf_file_name)  # read the pdf content

        # proc
        ## Create Dataset Instance
        ds = PymuDocDataset(pdf_bytes)

        ## inference
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)

            ## pipeline
            pipe_result = infer_result.pipe_ocr_mode(image_writer)

        else:
            infer_result = ds.apply(doc_analyze, ocr=False)

            ## pipeline
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        ### draw model result on each page
        model_pdf_path = os.path.join(local_md_dir, f"{name_without_suff}_model.pdf")
        infer_result.draw_model(model_pdf_path)

        ### get model inference result
        model_inference_result = infer_result.get_infer_res()

        ### draw layout result on each page
        layout_pdf_path = os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf")
        pipe_result.draw_layout(layout_pdf_path)

        ### draw spans result on each page
        spans_pdf_path = os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf")
        pipe_result.draw_span(spans_pdf_path)

        ### get markdown content
        md_content = pipe_result.get_markdown(image_dir)

        ### dump markdown
        pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

        ### get content list content
        content_list_content = pipe_result.get_content_list(image_dir)

        ### dump content list
        pipe_result.dump_content_list(
            md_writer, f"{name_without_suff}_content_list.json", image_dir
        )

        ### get middle json
        middle_json_content = pipe_result.get_middle_json()

        ### dump middle json
        pipe_result.dump_middle_json(md_writer, f"{name_without_suff}_middle.json")

        # Collect all files from output directory AND the annotated PDFs
        output_files = {}
        for root, _, files in os.walk(local_md_dir):
            for file in files:
                print(f"Processing file: {file}")
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as f:
                    output_files[file] = f.read()

        return json.loads(middle_json_content)

    @method()
    def process_pdf(self, pdf_data: bytes) -> dict:
        pdf_file_name = "/tmp/abc.pdf"  # replace with the real pdf path
        with open(pdf_file_name, "wb") as f:
            f.write(pdf_data)
        local_image_dir, local_md_dir = "output/images", "output"
        os.makedirs(local_image_dir, exist_ok=True)
        image_writer, md_writer = FileBasedDataWriter(
            local_image_dir
        ), FileBasedDataWriter(local_md_dir)
        reader1 = FileBasedDataReader("")
        pdf_bytes = reader1.read(pdf_file_name)  # read the pdf content
        ds = PymuDocDataset(pdf_bytes)
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)
        middle_json_content = pipe_result.get_middle_json()
        return json.loads(middle_json_content)


if __name__ == "__main__":
    cls = modal.Cls.lookup("mineru-modal", "MinerU")
    obj = cls()
    pdf_data = open("/Users/pranaviyer/Desktop/AstralisData/E5_Paper.pdf", "rb").read()

    # Process the PDF and get results
    result = obj.process_pdf.remote(pdf_data)

    # Create output directory
    output_dir = "modal_output"
    os.makedirs(output_dir, exist_ok=True)

    # Save the JSON data
    with open(os.path.join(output_dir, "data.json"), "w") as f:
        import json
        # Parse the string into a Python object first, then dump with indentation
        json_data = json.loads(result["middle_json"]) if isinstance(result["middle_json"], str) else result["middle_json"]
        json.dump(json_data, f, indent=2)

    # Save the inference result
    with open(os.path.join(output_dir, "inference_result.json"), "w") as f:
        json.dump(result["inference_result"], f, indent=2)

    # Save the content list
    with open(os.path.join(output_dir, "content_list.json"), "w") as f:
        json.dump(result["content_list"], f, indent=2)

    # Save all output files
    for filename, file_content in result["output_files"].items():
        output_path = os.path.join(output_dir, filename)
        # Create subdirectories if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Write the file
        with open(output_path, "wb") as f:
            f.write(file_content)
