import base64
import json
from typing import Union
from uuid import uuid4

from litellm import ModelResponse
from pydantic import BaseModel, Field

from src.llm_utils.utils import structure_image_prompt, text_cost_parser
from src.pipeline.registry.prompt_registry import PromptRegistry
from src.prompts.base_prompt import BasePrompt
from src.schemas.schemas import Citation, EmbeddedFeatureType, Entry, ExtractedFeatureType, Ingestion, RelationshipType


async def read_content(filepath, read) -> Union[str, bytes]:
    if read is not None:
        return await read(filepath)
    with open(filepath) as f:
        return f.read()


async def base_model_to_encoded_image(base_model: BaseModel, read=None):
    if isinstance(base_model, Ingestion):
        content = await read_content(base_model.file_path, read)

    elif isinstance(base_model, Entry):
        content = base_model.string

    # If content is already bytes (from S3), use it directly
    if isinstance(content, bytes):
        return base64.b64encode(content).decode('utf-8')

    # If it's a string path, read and encode the local file
    return base64.b64encode(content.encode('utf-8')).decode('utf-8')


@PromptRegistry.register("filter_indexing")
class FilterIndexingPrompt(BasePrompt):
    """
    A prompt class for evaluating whether a given text chunk is pertinent for indexing in the company's vector database.
    """

    system_prompt = (
        "You are an advanced AI assistant specialized in evaluating the relevance and quality of content for indexing in a health and wellness vector database."  # noqa
        " Your role is to analyze provided text data and determine its suitability for inclusion based on the company's focus areas and quality standards."  # noqa
    )

    user_prompt = (
        "You are provided with a text chunk that needs to be evaluated for indexing in our company's vector database."
        " Our company is dedicated to health and wellness, offering exceptional services in the following areas: nutrition, wellness, longevity, functional health, blood testing, exercise, and sleep."  # noqa
        "\n\n"
        "Content is considered pertinent if it aligns with our core areas, such as AI in healthcare, proteomic markers for longevity, systems health, and related scientific and actionable health information."  # noqa
        " It is generally preferable to include content rather than exclude it, but we aim to minimize the inclusion of irrelevant or low-quality content."  # noqa
        " Examples of content to avoid include AI applications solely focused on mental health triaging without scientific grounding, highly promotional or clickbait-esque articles, and content that does not provide substantial value to our services."  # noqa
        "\n\n"
        "Please analyze the following text and decide whether it should be indexed in our vector database."
        "\n\n"
        "Text to Evaluate:\n{entry}\n\n"
        "Provide a clear decision: `True` to index, `False` to exclude."
        " Additionally, include a brief rationale (1-2 sentences) explaining your decision."
    )

    class DataModel(BaseModel):
        rationale: str = Field(description="Reason for the decision")
        should_index: bool = Field(description="Whether the text should be indexed")

    @classmethod
    async def format_prompt(cls, base_model: BaseModel, read=None):
        if isinstance(base_model, Ingestion):
            document = await read_content(base_model.file_path, read)  # Call the class method using cls
        elif isinstance(base_model, Entry):
            document = base_model.string
        return cls.system_prompt, cls.user_prompt.format(entry=document)

    @staticmethod
    def parse_response(base_models: list[BaseModel], parsed_items: dict[str, BaseModel]) -> list[BaseModel]:
        # add the "summary" field to the entry.string in the front
        new_base_models = []
        for i, basemodel in enumerate(base_models):
            model_response = parsed_items.get(i)
            should_index = model_response.get("should_index", False)
            rationale = model_response.get("rationale", "")
            if should_index:
                if isinstance(basemodel, Ingestion):
                    basemodel.metadata["should_index"] = True
                    basemodel.metadata["rationale"] = rationale
                elif isinstance(basemodel, Entry):
                    basemodel.ingestion.document_metadata["should_index"] = True
                    basemodel.ingestion.document_metadata["rationale"] = rationale
                new_base_models.append(basemodel)
        return new_base_models


@PromptRegistry.register("summarize_ingestion")
class SummarizeIngestionPrompt(BasePrompt):
    """
    A prompt class for summarizing text ingestions
    """

    system_prompt = """
    You are an advanced language model designed to assist in summarizing text data with contextual awareness.
    """

    user_prompt = """
    Please analyze the following chunk of text and provide a concise and informative summary.
    Text Chunk to Summarize:
    {entry}
    """

    class DataModel(BaseModel):
        summary: str = Field(description="A summary of the text")

    @classmethod
    async def format_prompt(cls, ingestion: Ingestion, read=None):
        if read is not None:
            document = await read(ingestion.file_path)
        else:
            with open(ingestion.file_path) as f:
                document = f.read()
        return cls.system_prompt, cls.user_prompt.format(entry=document)

    @staticmethod
    def parse_response(ingestions: list[Ingestion], parsed_ingestions: dict[str, DataModel]) -> bool:
        # add the "summary" field to the entry.string in the front
        for i, ingestion in enumerate(ingestions):
            ingestion.document_metadata["summary"] = parsed_ingestions.get(i).get("summary", "")
        return ingestions


@PromptRegistry.register("summarize_entry")
class SummarizeEntryPrompt(BasePrompt[Entry]):
    """
    A prompt class for summarizing text entries while considering their context within the full document.
    """

    system_prompt = """
    You are an advanced language model designed to assist in summarizing text data with contextual awareness. Your task is to
    analyze chunks of text extracted from websites and PDFs and provide a concise and informative summary. The summary should
    capture the main points and key information from the text chunk, while considering its place within the larger document.

    Your goals are to:
    1. Provide a clear and coherent summary of the specific text chunk.
    2. Contextualize the summary within the broader document, if relevant.
    3. Maintain the original meaning and context of the text.
    4. Omit unnecessary details or redundant information.

    Ensure that the summary is written in complete sentences and accurately reflects both the content of the provided text chunk
    and its relationship to the overall document.
    """

    user_prompt = """
    Please analyze the following chunk of text and provide a concise and informative summary. The summary should capture the main
    points and key information from the text, while also considering its context within the larger document. If the chunk's
    content is significantly related to or dependent on other parts of the document, please include that context in your summary.

    Full Document (for context):
    {document}
    ------------------------------------------
    Text Chunk to Summarize:
    {entry}
    """

    class DataModel(BaseModel):
        summary: str = Field(description="A summary of the text")

    @classmethod
    async def format_prompt(cls, entry: Entry, read=None):
        if read is not None:
            document = await read(entry.ingestion.extracted_document_file_path)
        else:
            with open(entry.ingestion.extracted_document_file_path) as f:
                document = f.read()

        return [{"role": "system", "content": cls.system_prompt},
                {"role": "user", "content": cls.user_prompt.format(entry=entry.string, document=document)}]

    @staticmethod
    def parse_response(entry: Entry, response: ModelResponse) -> Entry:

        # extract synthetic data
        text, _ = text_cost_parser(response)
        summary = json.loads(text).get('summary', '')

        # generate a uuid
        uuid = str(uuid4())

        # create citation
        citation = Citation(
            relationship_type=RelationshipType.SYNTHETIC,
            source_uuid=uuid, # current
            target_uuid=entry.uuid, # parent
        )

        # create new Entry
        new_entry = Entry(
            uuid=uuid,
            string=summary,
            ingestion=entry.ingestion,
            citations=[citation],
            embedded_feature_type=EmbeddedFeatureType.SYNTHETIC_SUMMARY,
            consolidated_feature_type=ExtractedFeatureType.text
        )
        return new_entry


@PromptRegistry.register("clean_entry")
class CleanEntryPrompt(BasePrompt):
    """
    A prompt class for cleaning and filtering text data, focusing on removing only pure citation pages or completely
    non-informative content.
    """

    system_prompt = """
    You are an advanced language model designed to assist in cleaning and filtering text data. Your task is to analyze chunks
    of text extracted from RSS feeds, Slack channels, Websites, PDFs, etc. and determine whether they should be retained or discarded. The goal is to keep all
    informative content while only removing completely non-informative text.

    Your primary objective is to return a boolean value indicating whether the text should be retained (True) or discarded (False)

    Guidelines for discarding text (return False):
    1. If it is a chunk of 90 percent or more citations or references (e.g., the last page of an academic paper that contains only
    a list of references).
    2. If it contains only random characters or completely nonsensical text that provides absolutely no information.
    3. If it is purely navigational or structural text with no content (e.g., "Page 1 of 10", "Table of Contents" with no actual
    entries).

    Guidelines for retaining text (return True):
    1. Keep all content that has any meaningful information.
    2. Keep content with minor formatting issues, typos, or occasional non-word characters.
    3. Retain metadata if it provides any context or information about the document.

    When in doubt, it is better to retain the text (return True) than to discard it.
    """  # noqa

    user_prompt = """
    Please analyze the following chunk of text and determine whether it should be retained or discarded. Return True if the text
     contains any meaningful content or information, even if it includes some citations or references. Return False ONLY if the
     text is exclusively citations (like the last page of a paper) or completely non-informative.

    Remember, we want to keep as much content as possible. Only discard text if it's purely citations or completely meaningless.
    Use the provided schema for your response.

    Text:
    {entry}
    """

    class DataModel(BaseModel):
        rationale: str = Field(description="The rationale for keeping the entry")
        retain: bool = Field(description="Whether to retain the entry, i.e. is it not just citations or nonsensical text")

    @classmethod
    async def format_prompt(cls, base_model: BaseModel, read=None):
        if isinstance(base_model, Ingestion):
            document = await read_content(base_model.file_path, read)  # Call the class method using cls
        elif isinstance(base_model, Entry):
            document = base_model.string
        return cls.system_prompt, cls.user_prompt.format(entry=document)

    @staticmethod
    def parse_response(base_models: list[BaseModel], parsed_items: dict[str, BaseModel]) -> list[BaseModel]:
        # add the "summary" field to the entry.string in the front
        new_base_models = []
        for i, basemodel in enumerate(base_models):
            model_response = parsed_items.get(i)
            retain = model_response.get("retain", False)
            rationale = model_response.get("rationale", "")
            if retain:
                if isinstance(basemodel, Ingestion):
                    basemodel.metadata["retain"] = True
                    basemodel.metadata["rationale"] = rationale
                elif isinstance(basemodel, Entry):
                    basemodel.ingestion.document_metadata["retain"] = True
                    basemodel.ingestion.document_metadata["rationale"] = rationale
                new_base_models.append(basemodel)
        return new_base_models


@PromptRegistry.register("keywords")
class KeywordPrompt(BasePrompt):
    """
    A prompt class for extracting keywords from text data, focusing on identifying the most relevant terms that capture the
     essence of the content.
    """

    system_prompt = """
    You are an advanced language model designed to assist in extracting keywords from text data. Your task is to analyze a chunk
     of text and identify the most relevant keywords that encapsulate the main ideas and themes of the content.

    Your primary objective is to return a list of 3 to 5 keywords that best represent the text. The keywords should be specific,
     meaningful, and relevant to the content.

    Guidelines for extracting keywords:
    1. Focus on nouns, noun phrases, and significant terms that convey the main topics of the text.
    2. Avoid common words or stop words that do not add value to the understanding of the content.
    3. Ensure that the keywords are concise and directly related to the text provided.

    When in doubt, prioritize keywords that reflect the core themes and subjects of the text.
    """

    user_prompt = """
    Please analyze the following chunk of text and extract 3 to 5 keywords that best represent the main ideas and themes.
     The keywords should be specific and relevant to the content. Avoid common words or phrases that do not add value.

    Text:
    {entry}
    """

    class DataModel(BaseModel):
        keywords: list[str] = Field(description="A list of keywords extracted from the entry")

    @classmethod
    def format_prompt(cls, entry: str) -> str:
        return cls.system_prompt, cls.user_prompt.format(entry=entry)

    @staticmethod
    def parse_response(entries: list[Entry], parsed_entries: dict[str, DataModel]) -> bool:
        for i, entry in enumerate(entries):
            try:
                entry.keywords = parsed_entries.get(i).get("keywords")
            except KeyError:
                entry.keywords = []
        return entries


@PromptRegistry.register("describe_image")
class ImageDescriptionPrompt(BasePrompt):
    """ """

    system_prompt = """
    You are an advanced vision language model who is excellent at describing images in detail.
    """

    user_prompt = """
    Please caption this image in detail.
    """

    class DataModel(BaseModel):
        description: str = Field(description="A description of the image")

    @classmethod
    def format_prompt(cls, entry: BaseModel, read=None):
        image = base_model_to_encoded_image(entry, read)
        return structure_image_prompt(cls.system_prompt, cls.user_prompt, image)


    @staticmethod
    def parse_response(entries: list[Entry], parsed_entries: dict[str, Entry]) -> bool:
        for i, entry in enumerate(entries):
            entry.string = parsed_entries.get(i).get("description")
        return entries


@PromptRegistry.register("extract_claims")
class ExtractClaimsPrompt(BasePrompt):
    """
    This prompt uses structured output to extract the key scientific claims from a podcast transcript.
    """

    system = (
        "You are an expert in biology, medicine, and health. Your task is to extract key scientific claims from the podcast"
        " transcript. Focus on identifying specific claims related to science, medicine, and personal health, and provide"
        " context for each claim."
    )

    user = (
        "You will be given a segment from a conversation."
        " Your job is to extract the key scientific claims that are made."
        " Please output the text in the provided structure."
        " Conversation Segment:\n{chunk}\n"
    )

    class DataModel(BaseModel):
        """
        DataModel for extracting thorough, scientifically rigorous information from a podcast transcript
        """

        claims: list[str] = Field(description="A list of claims from the conversation segment")

    @classmethod
    def format_prompt(cls, text):
        return {"system": cls.system, "user": cls.user.format(chunk=text)}

    @staticmethod
    def parse_response(response: DataModel) -> tuple[list[str], list[str], float]:
        text, cost = text_cost_parser(response)
        return text.claims, cost


@PromptRegistry.register("label_clusters")
class LabelClustersPrompt(BasePrompt):
    """
    This prompt labels the clusters of claims based on their relevance to a given topic.
    """

    system = "You are an expert in biology, medicine, and health."

    user = (
        "You are provided a set of scientific claims."
        " Your task is to analyze how the claims relate to the Topic of conversation and categorize their relevance."
        "\nClaims:\n{claims}"
    )

    class DataModel(BaseModel):
        description: str = Field(
            description="A summary describing the subtopic reflected in the theme of the claims in relation to the Topic.",
        )
        subtopic: str = Field(
            description="A title for the group of claims that represents a meaningful subtopic of the Topic."
        )

    @classmethod
    def format_prompt(cls, claims):
        claims = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(claims)])
        return {
            "system": cls.system,
            "user": cls.user.format(claims=claims),
        }

    @staticmethod
    def parse_response(response: DataModel, model: str) -> tuple[dict[str, str], float]:
        text, cost = text_cost_parser(response, model)
        return {"description": text.description, "subtopic": text.subtopic}, cost
