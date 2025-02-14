import sys
from base64 import b64encode
from pathlib import Path
import fitz
from PIL import Image
import io
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from dotenv import load_dotenv
import uuid

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.llm_utils.utils import text_cost_parser
from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import (
    Entry, ExtractionMethod, Ingestion, 
    ChunkLocation, Index, ExtractedFeatureType,
    EmbeddedFeatureType
)
from src.utils.datetime_utils import get_current_utc_datetime

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

# Define configurations first
generation_config = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# Define the chunking prompt
CHUNKING_PROMPT = """\
OCR the following page into Markdown. Tables should be formatted as HTML. 
Do not surround your output with triple backticks.

Chunk the document into sections of roughly 250 - 1000 words. Our goal is 
to identify parts of the page with same semantic theme. These chunks will 
be embedded and used in a RAG pipeline. 

Surround the chunks with <chunk> </chunk> html tags.
"""

async def extract_with_gemini(page_image: Image.Image) -> dict:
    """Extract text and structure from an image using Gemini."""
    response = model.generate_content(
        contents=[CHUNKING_PROMPT, page_image],
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    return response.text

def pdf_to_images(pdf_content):
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images

@FunctionRegistry.register("extract", "gemini_vlm")
async def gemini_parse(ingestions: list[Ingestion], read=None, write=None, **kwargs):
    all_entries = []
    for ingestion in ingestions:
        if read:
            document = await read(ingestion.file_path)
        else:
            with open(ingestion.file_path, "rb") as f:
                document = f.read()

        # Create output filename from input filename
        input_filename = Path(ingestion.file_path).name
        output_filename = input_filename.replace(".pdf", "_gemini.md")
        parsed_file_path = str(Path("data/output") / output_filename)
        
        # Ensure output directory exists
        os.makedirs("data/output", exist_ok=True)

        images = pdf_to_images(document)
        
        print(f"\nProcessing {len(images)} pages from {ingestion.file_path}")
        all_text = {}
        for page_num, image in enumerate(images, 1):
            print(f"Processing page {page_num}/{len(images)}...")
            extracted_text = await extract_with_gemini(image)
            all_text[str(page_num)] = extracted_text
            
            # Process chunks
            chunks = [chunk.strip() for chunk in extracted_text.split("<chunk>") 
                     if "</chunk>" in chunk]
            print(f"Found {len(chunks)} chunks in page {page_num}")
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_text = chunk.split("</chunk>")[0].strip()
                entry = Entry(
                    uuid=str(uuid.uuid4()),
                    ingestion=ingestion,
                    string=chunk_text,
                    chunk_index=chunk_idx + 1,
                    min_primary_index=page_num,
                    max_primary_index=page_num,
                    chunk_locations=[ChunkLocation(
                        index=Index(primary=page_num),
                        extracted_feature_type=ExtractedFeatureType.text,
                        page_file_path=ingestion.file_path
                    )],
                    embedded_feature_type=EmbeddedFeatureType.TEXT,
                    consolidated_feature_type=ExtractedFeatureType.text,
                    citations=[]
                )
                all_entries.append(entry)

        if write:
            await write(parsed_file_path, "\n\n".join(all_text.values()))
        else:
            with open(parsed_file_path, "w") as f:
                f.write("\n\n".join(all_text.values()))
                
        ingestion.extraction_method = ExtractionMethod.GEMINI_VLM
        ingestion.extraction_date = get_current_utc_datetime()
        ingestion.extracted_document_file_path = parsed_file_path
        
    return all_entries 

if __name__ == "__main__":
    import asyncio
    import argparse
    import yaml
    from pathlib import Path
    from src.ingestion.files.local import create_ingestion

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test_gemini_vlm.yaml")
    parser.add_argument("--pdf", help="Direct path to PDF file (overrides config)")
    args = parser.parse_args()

    async def main():
        # Load config
        config_path = Path(args.config)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Setup input/output dirs from config
        if args.pdf:
            input_files = [Path(args.pdf)]
        else:
            input_pattern = config["pipeline"]["input"]["config"]["file_patterns"][0]
            input_files = list(Path().glob(input_pattern))
            
        if not input_files:
            print(f"No files found matching pattern: {input_pattern}")
            return

        # Create output directory
        output_dir = Path(config["pipeline"]["output"]["config"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process files
        ingestions = []
        for file_path in input_files:
            print(f"Processing: {file_path}")
            ingestion = await create_ingestion(str(file_path))
            ingestions.append(ingestion)

        # Run extraction
        entries = await gemini_parse(ingestions)

        # Print results
        for entry in entries:
            print(f"\nProcessed {entry.ingestion.file_path}:")
            print(f"Extraction method: {entry.ingestion.extraction_method}")
            print(f"Number of chunks: {len(entry.chunk_locations) if entry.chunk_locations else 1}")
            print(f"Output saved to: {entry.ingestion.extracted_document_file_path}")

    asyncio.run(main()) 