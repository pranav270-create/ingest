import json
from uuid import uuid4

from litellm import ModelResponse
from pydantic import BaseModel, Field

from src.llm_utils.utils import text_cost_parser
from src.pipeline.registry.prompt_registry import PromptRegistry
from src.prompts.base_prompt import BasePrompt
from src.schemas.schemas import Citation, EmbeddedFeatureType, Entry, ExtractedFeatureType, RelationshipType


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
            source_uuid=uuid,  # current
            target_uuid=entry.uuid,  # parent
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


class ExtractClaimsPrompt(BasePrompt[Entry]):
    pass


class LabelClustersPrompt(BasePrompt[Entry]):
    pass
