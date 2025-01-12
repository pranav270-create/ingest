import sys
from pathlib import Path
from openai import AsyncOpenAI
from base64 import b64encode
import fitz
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import Document, Entry, Ingestion, ParsedFeatureType, ParsingMethod
from src.pipeline.registry import FunctionRegistry
from src.utils.datetime_utils import get_current_utc_datetime
from src.llm_utils.utils import Provider
from src.prompts.parser import text_cost_parser
from src.llm_utils.api_requests import get_api_key


client = AsyncOpenAI(api_key=get_api_key(Provider.OPENAI))


async def compare_with_vlm(page_image: bytes) -> dict:
    # Convert image to base64
    image_b64 = b64encode(page_image).decode('utf-8')
    
    prompt = """Here is a page from a document
    Please extract ALL the text from the document, as it will be used later on in an retrievel augmented generation pipeline.
    If there is an image, please describe the image in detail. If there is a graph, please describe the graph to detail and attempt to convert the data to a text if possible.
    If there is a table, please convert the table to a markdown string.
    Structure your response as a JSON with the following.
    """
    
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }
        ],
        max_tokens=2000
    )
    
    text, _ = text_cost_parser(response, "gpt-4-vision-preview")
    try:
        return eval(text)
    except:
        return {"error": "Failed to parse VLM response"} 


def pdf_to_images(pdf_content):
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


@FunctionRegistry.register("parse", "vlm")
async def vlm_parse(ingestions: list[Ingestion], read=None, write=None, **kwargs):
    documents = []
    for ingestion in ingestions:
        if read:
            document = await read(ingestion.file_path)
        else:
            with open(ingestion.file_path) as f:
                document = f.read()

        parsed_file_path = ingestion.file_path.replace(".pdf", ".txt")
        images = pdf_to_images(document)

        entries = []
        for image in images:
            all_text = await compare_with_vlm(image)
            if write:
                await write(parsed_file_path, all_text)
            else:
                with open(parsed_file_path, "w") as f:
                    f.write(all_text)
            entry = Entry(ingestion=ingestion, string=all_text, index_numbers=None, citations=None)
            entries.append(entry)
        document = Document(entries=entries)
        ingestion.parsing_method = ParsingMethod.GOOGLE_LABS_HTML_CHUNKER
        ingestion.parsing_date = get_current_utc_datetime()
        ingestion.parsed_feature_type = [ParsedFeatureType.TEXT]
        ingestion.parsed_file_path = parsed_file_path
        documents.append(document)
    return documents


if __name__ == "__main__":
    pass
