import json
from uuid import uuid4
from litellm import ModelResponse
from pydantic import BaseModel, Field
import base64
import copy

from src.llm_utils.utils import structure_image_prompt, text_cost_parser
from src.pipeline.registry.prompt_registry import PromptRegistry
from src.prompts.base_prompt import BasePrompt
from src.schemas.schemas import Citation, EmbeddedFeatureType, Entry, ExtractedFeatureType, RelationshipType


@PromptRegistry.register("describe_image")
class DescribeImagePrompt(BasePrompt):
    """
    A prompt class for describing image contents and generating appropriate captions.
    """

    system_prompt = """
    You are an advanced vision model designed to analyze and describe images in detail. Your task is to:
    1. Provide a detailed description of the image contents, including:
       - Main subjects/objects
       - Actions or activities
       - Setting and background
       - Notable visual elements (colors, composition, lighting)
    2. Generate a concise, informative caption that summarizes the key elements of the image.

    Focus on being accurate, clear, and comprehensive in your descriptions while keeping captions concise and engaging.
    """

    user_prompt = """
    Please analyze the following image and provide:
    1. A detailed description of its contents
    2. A concise caption that captures the essence of the image

    Image to Analyze:
    """

    class DataModel(BaseModel):
        description: str = Field(description="A detailed description of the image contents")
        caption: str = Field(description="A concise caption summarizing the image")

    @classmethod
    async def format_prompt(cls, entry: Entry, read=None):
        filepath = entry.chunk_locations[0].extracted_file_path
        if read is not None:
            image = await read(filepath)
        else:
            with open(filepath) as f:
                image = f.read()
        base64_image = base64.b64encode(image).decode('utf-8')
        return structure_image_prompt(cls.system_prompt, cls.user_prompt, base64_image)

    @staticmethod
    def parse_response(entry: Entry, response: ModelResponse) -> list[Entry]:
        original_entry = copy.deepcopy(entry)

        # extract synthetic data
        text, _ = text_cost_parser(response)
        description = json.loads(text).get('description', '')
        caption = json.loads(text).get('caption', '')

        # generate a uuid
        caption_uuid = str(uuid4())
        description_uuid = str(uuid4())

        # create citation
        caption_citation = Citation(
            relationship_type=RelationshipType.SYNTHETIC,
            source_uuid=caption_uuid,  # current
            target_uuid=entry.uuid,  # parent
        )
        description_citation = Citation(
            relationship_type=RelationshipType.SYNTHETIC,
            source_uuid=description_uuid,  # current
            target_uuid=entry.uuid,  # parent
        )

        # create two new Entries, one for the description and one for the caption
        caption_entry = Entry(
            uuid=caption_uuid,
            string=caption,
            ingestion=entry.ingestion,
            citations=[caption_citation],
            embedded_feature_type=EmbeddedFeatureType.SYNTHETIC_FEATURE_DESCRIPTION,
            consolidated_feature_type=ExtractedFeatureType.text
        )

        description_entry = Entry(
            uuid=description_uuid,
            string=description,
            ingestion=entry.ingestion,
            citations=[description_citation],
            embedded_feature_type=EmbeddedFeatureType.SYNTHETIC_FEATURE_DESCRIPTION,
            consolidated_feature_type=ExtractedFeatureType.text
        )

        return [original_entry, caption_entry, description_entry]
