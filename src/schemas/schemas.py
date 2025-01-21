import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field, field_validator

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry.schema_registry import SchemaRegistry


class Scope(str, Enum):
    EXTERNAL = "external"
    INTERNAL = "internal"


class ContentType(str, Enum):
    # Legal
    ZONING_MAP = "zoning_map"
    MEETING_MINUTES = "meeting_minutes"
    LOCAL_ORDINANCE = "local_ordinance"
    STATE_LAW = "state_law"
    FEDERAL_LAW = "federal_law"
    OTHER_LAW = "other_law"
    # Articles
    GOLD_ARTICLES = "gold_articles"
    OTHER_ARTICLES = "other_articles"
    BLOG = "blog"
    # External Items
    RSS = "rss"
    MARKET_RESEARCH = "market_research"
    # Communication Items
    TEXT = "text"
    EMAIL = "email"
    LINKEDIN = "linkedin"
    TWEET = "tweet"
    YOUTUBE = "youtube"
    PODCAST = "podcast"
    # Zoom Items
    ZOOM_RECORDING_VIDEO = "zoom_recording_video"
    ZOOM_RECORDING_AUDIO = "zoom_recording_audio"
    ZOOM_RECORDING_TRANSCRIPT = "zoom_recording_transcript"
    # Catch all
    OTHER = "other"


class FileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    TXT = "txt"
    HTML = "html"
    MD = "md"
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    TSV = "tsv"
    SQL = "sql"
    YAML = "yaml"
    TOML = "toml"
    MP4 = "mp4"
    MP3 = "mp3"
    WAV = "wav"
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    SVG = "svg"
    WEBM = "webm"
    ODT = "odt"
    ODS = "ods"
    ODP = "odp"
    XLSX = "xlsx"


def mime_type_to_file_type(mime_type: str) -> FileType:
    mime_to_file_type = {
        "application/pdf": FileType.PDF,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileType.DOCX,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": FileType.PPTX,
        "text/plain": FileType.TXT,
        "text/html": FileType.HTML,
        "text/markdown": FileType.MD,
        "application/json": FileType.JSON,
        "application/xml": FileType.XML,
        "text/csv": FileType.CSV,
        "text/tab-separated-values": FileType.TSV,
        "application/sql": FileType.SQL,
        "application/x-yaml": FileType.YAML,
        "application/toml": FileType.TOML,
        "video/mp4": FileType.MP4,
        "audio/mpeg": FileType.MP3,
        "audio/wav": FileType.WAV,
        "image/jpeg": FileType.JPEG,
        "image/png": FileType.PNG,
        "image/gif": FileType.GIF,
        "image/svg+xml": FileType.SVG,
        "video/webm": FileType.WEBM,
        "application/vnd.oasis.opendocument.text": FileType.ODT,
        "application/vnd.oasis.opendocument.spreadsheet": FileType.ODS,
        "application/vnd.oasis.opendocument.presentation": FileType.ODP,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": FileType.XLSX,
        # Add more mappings as needed
    }
    return mime_to_file_type.get(mime_type, FileType.PDF)  # Default to PDF if unknown


class IngestionMethod(str, Enum):
    SLACK_HOOK = "slack_hook"
    ZOOM_HOOK = "zoom_hook"
    YOUTUBE_FEED = "youtube_feed"
    YOUTUBE_MP3 = "youtube_mp3"
    YOUTUBE_TRANSCRIPT = "youtube_transcript"
    URL_SCRAPE = "url_scrape"
    MINDSEARCH_SCRAPE = "mindsearch"
    SEMANTIC_SCHOLAR_API = "semantic_scholar_api"
    BRAVE_API = "brave_api"
    LOCAL_FILE = "local_file"
    S3 = "s3"
    GDRIVE_API = "gdrive_api"
    RSS = "rss"
    ARXIV_BIORXIV_API = "arxiv_biorxiv_api"
    TWITTER_API = "twitter_api"
    GMAIL_API = "gmail_api"
    LINKEDIN_API = "linkedin_api"
    VECTOR_DB = "vector_db"
    SQL_DB = "sql_db"
    BACKEND_API = "backend_api"
    OTHER = "other"


class ExtractionMethod(str, Enum):
    SIMPLE = "simple"
    TESSERACT = "tesseract"
    TEXTRACT = "textract"
    XRAY = "xray"
    GROBID = "grobid"
    MARKER = "marker"
    UNSTRUCTURED = "unstructured"
    MINERU = "mineru"
    OCR2_0 = "ocr2_0"
    WHISPER = "whisper"
    BS4 = "bs4"
    GOOGLE_LABS_HTML_CHUNKER = "google_labs_html_chunker"
    NONE = "none"


class ExtractedFeatureType(str, Enum):
    # Common content types (shared across extractors)
    text = "text"  # Basic text content (from Marker.Text and Textract.LAYOUT_TEXT)
    word = "word"  # Word-level text (from Textract extraction)
    line = "line"  # Line-level text (from Marker.Line and Textract extraction)
    image = "image"  # Basic image content
    table = "table"  # Tables (from Marker.Table and Textract.LAYOUT_TABLE)
    figure = "figure"  # Figures (from Marker.Figure and Textract.LAYOUT_FIGURE)
    code = "code"  # Code blocks (from Marker.Code)
    equation = "equation"  # Mathematical equations (from Marker.Equation)
    form = "form"  # Form elements (from Marker.Form)
    header = "header"  # Headers (from Marker.PageHeader and Textract.LAYOUT_HEADER)
    footer = "footer"  # Footers (from Marker.PageFooter and Textract.LAYOUT_FOOTER)
    section_header = "section_header"  # Section headers (from Marker.SectionHeader and Textract.LAYOUT_SECTION_HEADER)
    list = "list"  # List structures (from Marker.ListItem/ListGroup and Textract.LAYOUT_LIST)
    page_number = "page_number"  # Page numbers (from Textract.LAYOUT_PAGE_NUMBER)

    # Marker-specific types (original: PascalCase -> lowercase)
    span = "span"  # From Marker.Span
    figuregroup = "figuregroup"  # From Marker.FigureGroup
    tablegroup = "tablegroup"  # From Marker.TableGroup
    listgroup = "listgroup"  # From Marker.ListGroup
    picturegroup = "picturegroup"  # From Marker.PictureGroup
    picture = "picture"  # From Marker.Picture
    page = "page"  # From Marker.Page
    caption = "caption"  # From Marker.Caption
    footnote = "footnote"  # From Marker.Footnote
    handwriting = "handwriting"  # From Marker.Handwriting
    textinlinemath = "textinlinemath"  # From Marker.TextInlineMath
    tableofcontents = "tableofcontents"  # From Marker.TableOfContents
    document = "document"  # From Marker.Document
    complexregion = "complexregion"  # From Marker.ComplexRegion

    # MinerU-specific types
    table_body = "table_body"  # From MinerU.LAYOUT_TABLE_BODY
    table_caption = "table_caption"  # From MinerU.LAYOUT_TABLE_CAPTION
    table_footnote = "table_footnote"  # From MinerU.LAYOUT_TABLE_FOOTNOTE
    image_body = "image_body"  # From MinerU.LAYOUT_IMAGE_BODY
    image_caption = "image_caption"  # From MinerU.LAYOUT_IMAGE_CAPTION
    image_footnote = "image_footnote"  # From MinerU.LAYOUT_IMAGE_FOOTNOTE
    index = "index"  # From MinerU.LAYOUT_INDEX

    # Textract-specific types (original: LAYOUT_* -> lowercase)
    key_value = "key_value"  # From Textract.LAYOUT_KEY_VALUE

    # Catch-all types
    combined_text = "combined_text"  # Our aggregation type
    section_text = "section_text"  # Our aggregation type
    other = "other"  # Fallback type


class ChunkingMethod(str, Enum):
    TOPIC_SEGMENTATION = "topic_segmentation"
    SLIDING_WINDOW = "sliding_window"
    DISTANCE = "distance"
    GENERATIVE = "generative"
    TEXTRACT = "textract"
    NONE = "none"


class EmbeddedFeatureType(str, Enum):
    TEXT = "text"  # these words are either spoken or written (tables included)
    IMAGE = (
        "image"  # this is for directly embedded images or ColPali (joint latent space)
    )
    # these are synthetic features and are all text
    SYNTHETIC_FEATURE_DESCRIPTION = "synthetic_feature_description"  # this is for extracted images (traditional pipeline)
    SYNTHETIC_SUMMARY = "synthetic_summary"
    SYNTHETIC_CONTEXT_SUMMARY = "synthetic_context_summary"
    SYNTHETIC_TRENDS = "synthetic_trends"
    SYNTHETIC_COMPONENTS = "synthetic_components"
    SYNTHETIC_KEYWORDS = "synthetic_keywords"
    SYNTHETIC_ANSWERS = "synthetic_answers"
    SYNTHETIC_QUESTIONS = "synthetic_questions"
    SYNTHETIC_FACTS = "synthetic_facts"


class RelationshipType(str, Enum):
    CITES_GENERAL = "cites_general"
    CITES_BACKGROUND = "cites_background"
    CITES_METHOD = "cites_method"
    CITES_RESULT = "cites_result"
    CITED_BY = "cited_by"
    ATTACHMENT = "attachment"
    SECTION_REFERENCE = "section_reference"
    ONTOLOGICAL = "ontological"
    SNIPPET = "snippet"
    SYNTHETIC = "synthetic"
    FIGURE_CAPTION = "figure_caption"
    TABLE_CAPTION = "table_caption"
    TABLE_FOOTNOTE = "table_footnote"
    FIGURE_FOOTNOTE = "figure_footnote"


class Index(BaseModel):
    """
    This is a list of indices that can be used to index a larger object. (float for video/audio, int for everything else)
    We sort by primary, then secondary, then tertiary.
    NOTE: We start indexing at 1 for all indices.
    For a simple text extraction, we would only have a primary index.
    For a more complex PDF, we may have a primary and secondary index telling us where each chunk is within the page.
    For a table within a complex PDF, we may have a primary, secondary, and tertiary index telling us where each cell is
    within the table within the page.
    """

    primary: Annotated[int, Field(gt=0)]  # Ensure primary is greater than 0
    secondary: Optional[Annotated[int, Field(gt=0)]] = (
        None  # Ensure secondary is greater than 0 if provided
    )
    tertiary: Optional[Annotated[int, Field(gt=0)]] = (
        None  # Ensure tertiary is greater than 0 if provided
    )

    def __str__(self):
        return f"{self.primary}.{self.secondary}.{self.tertiary}"

    def __lt__(self, other):
        if not isinstance(other, Index):
            return NotImplemented

        # Compare primary keys
        if self.primary != other.primary:
            return self.primary < other.primary

        # If primary keys are equal, compare secondary keys
        if self.secondary is not None and other.secondary is not None:
            if self.secondary != other.secondary:
                return self.secondary < other.secondary
        elif self.secondary is not None:
            return False
        elif other.secondary is not None:
            return True

        # If secondary keys are equal or both None, compare tertiary keys
        if self.tertiary is not None and other.tertiary is not None:
            return self.tertiary < other.tertiary
        elif self.tertiary is not None:
            return False
        elif other.tertiary is not None:
            return True

        # If all keys are equal or None, consider them equal
        return False

    def __eq__(self, other):
        if not isinstance(other, Index):
            return NotImplemented
        return (
            self.primary == other.primary
            and self.secondary == other.secondary
            and self.tertiary == other.tertiary
        )

    def __hash__(self):
        return hash((self.primary, self.secondary, self.tertiary))


class BoundingBox(BaseModel):
    """
    This is the bounding box of the extracted data. This will be in the pixel space of a page image.
    0, 0 is the top left corner of the page.
    """

    left: float
    top: float
    width: float
    height: float
    page_width: float
    page_height: float

    @property
    def relative_coords(self) -> dict[str, float]:
        """Returns coordinates normalized to page dimensions (0-1 range)"""
        return {
            "left": self.left / self.page_width,
            "top": self.top / self.page_height,
            "width": self.width / self.page_width,
            "height": self.height / self.page_height,
        }


class Citation(BaseModel):
    relationship_type: RelationshipType
    target_uuid: str
    source_uuid: str


class ChunkLocation(BaseModel):
    """
    Represents the location of a chunk within a document, combining index information
    with physical location details and the type of content extracted at that location.
    """

    index: Index  # basically page number
    bounding_box: Optional[BoundingBox] = None  # Physical location on the page
    extracted_feature_type: Optional[ExtractedFeatureType] = (
        None  # Type of content at this location
    )
    page_file_path: Optional[str] = None  # This is the file path to the page screenshot
    extracted_file_path: Optional[str] = (
        None  # This is the file path to the extracted screenshot
    )


# ------------------------------ CORE MODELS ------------------------------ #


@SchemaRegistry.register("Ingestion")
class Ingestion(BaseModel):
    schema__: str = Field(default="Ingestion", alias="schema__")
    document_hash: Optional[str] = None  # This is the hash of the document
    # Processing fields
    ingestion_id: Optional[int] = None  # Needed for SQL mode
    pipeline_id: Optional[int] = None  # Needed for SQL mode
    # Document fields
    document_title: Optional[str] = None  # We may want a LLM to title the document
    scope: Scope  # You must know the scope of the document before you can ingest it.
    content_type: Optional[ContentType] = None  # We may eventually want to infer this
    creator_name: str  # This is the name of the entity who created the document (if known)
    creation_date: Optional[str] = None
    file_type: Optional[FileType] = None  # This is the file type of the document (pdf, docx, pptx, etc.)
    file_path: Optional[str] = None  # Cloud bucket path
    file_size: Optional[int] = None  # In bytes
    public_url: Optional[str] = None  # This is the public URL for the document (website, image, youtube, etc.)
    # Ingestion fields
    ingestion_method: IngestionMethod  # Source of ingestion (e.g., 'slack', 'youtube', 'wix', etc.)
    ingestion_date: Optional[str] = None
    # Added fields
    document_summary: Optional[str] = None  # This is a summary of the document
    document_keywords: Optional[list[str]] = None  # This is a list of keywords that we have extracted
    document_metadata: Optional[dict[str, Any]] = None  # This is for any metadata that we have not captured in other fields yet
    # Extraction fields
    extraction_method: Optional[ExtractionMethod] = None
    extraction_date: Optional[str] = None
    extracted_document_file_path: Optional[str] = None  # This is the path to the extracted file which we can use for more context
    # Chunking fields
    chunking_method: Optional[ChunkingMethod] = None
    chunking_metadata: Optional[dict[str, Any]] = None
    chunking_date: Optional[str] = None
    # Featurization fields
    feature_models: Optional[list[str]] = None  # feature_models = name of model used
    feature_dates: Optional[list[str]] = None  # feature_types = name of prompt from PromptRegistry
    feature_types: Optional[list[str]] = None  # feature_dates = date of prompt from PromptRegistry
    # Unprocessed citations
    unprocessed_citations: Optional[dict[str, Any]] = None  # This is for citations that have not been processed yet
    citations: Optional[list[Citation]] = None

    @field_validator("extracted_document_file_path")
    def validate_extracted_document_file_path(cls, v):
        # NOTE: The CONTENT is default of type .json with page_number and content?
        if v and not v.endswith(".json"):
            raise ValueError("extracted_document_file_path must end with .json")
        return v


@SchemaRegistry.register("Entry")
class Entry(BaseModel):
    schema__: str = Field(default="Entry", alias="schema__")
    # For unique hash identification
    uuid: str

    # Core fields
    ingestion: Optional[Ingestion] = None  # null for cross document only, citations must be there
    string: Optional[str] = None  # If we embed a document or image, we don't need the original text

    # Featurization fields
    entry_title: Optional[str] = None  # Title of a figure, table, or other content
    keywords: Optional[list[str]] = None  # This is for any keywords that we have not captured in other fields yet
    added_featurization: Optional[dict[str, Any]] = None  # This is for any additional features that we have added

    # Chunk location fields. Used for reconstruction
    consolidated_feature_type: Optional[ExtractedFeatureType] = None  # This is the type of feature that is being embedded
    chunk_locations: Optional[list[ChunkLocation]] = None  # Combined location information
    min_primary_index: Optional[int] = None  # Cached for quick access
    max_primary_index: Optional[int] = None  # Cached for quick access
    chunk_index: Optional[Annotated[int, Field(gt=0)]] = None  # Ensure chunk index is greater than 0
    table_number: Optional[int] = None  # This is for a table specifically
    figure_number: Optional[int] = None  # This is for a figure specifically

    # Embedding fields
    embedded_feature_type: Optional[EmbeddedFeatureType] = None  # embedded_feature_type = type of feature being embedded
    embedding_date: Optional[str] = None  # embedding_date = date of embedding
    embedding_model: Optional[str] = None  # embedding_model = name of model used
    embedding_dimensions: Optional[int] = None  # embedding_dimensions = dimensions of embedding

    # Graph DB
    citations: Optional[list[Citation]] = None

    # field for evaluation scores
    evaluation_scores: Optional[Dict[str, Union[int, str]]] = Field(
        default=None,
        description="Scores from chunk quality evaluation"
    )


@SchemaRegistry.register("Record")
class Record(Entry):
    schema__: str = Field(default="Record", alias="schema__")


@SchemaRegistry.register("Upsert")
class Upsert(BaseModel):
    schema__: str = Field(default="Upsert", alias="schema__")

    # Created during Qdrant Upsert
    # ------------------------------
    sparse_vector: dict[str, Union[list[float], list[int]]]

    # From EMBEDDING
    # ------------------------------
    dense_vector: Union[list[float], float]

    # From ENTRY
    # ------------------------------
    uuid: str
    string: Optional[str] = None
    # Featurization fields
    entry_title: Optional[str] = None  # From Entry - title of figure, table, or other content
    keywords: Optional[list[str]] = None  # From Entry
    # Chunk location fields from Entry
    consolidated_feature_type: Optional[ExtractedFeatureType] = None
    chunk_locations: Optional[list[ChunkLocation]] = None
    min_primary_index: Optional[int] = None
    max_primary_index: Optional[int] = None
    chunk_index: Optional[Annotated[int, Field(gt=0)]] = None
    table_number: Optional[int] = None
    figure_number: Optional[int] = None
    # Embedding specific fields
    embedded_feature_type: Optional[EmbeddedFeatureType] = None
    embedding_date: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    # Random
    added_featurization: Optional[dict[str, Any]] = None  # This is for any additional features that we have added
    # Graph DB
    citations: Optional[list[Citation]] = None

    # From INGESTION
    # ------------------------------
    document_hash: str
    # Processing fields from Ingestion
    ingestion_id: Optional[int] = None
    pipeline_id: Optional[int] = None
    # Document fields from Ingestion
    document_title: Optional[str] = None
    scope: Scope
    content_type: Optional[ContentType] = None
    creator_name: str
    creation_date: Optional[str] = None
    file_type: Optional[FileType] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    public_url: Optional[str] = None
    # Ingestion fields from Ingestion
    ingestion_method: IngestionMethod
    ingestion_date: str
    # Document analysis fields from Ingestion
    document_summary: Optional[str] = None
    document_keywords: Optional[list[str]] = None
    document_metadata: Optional[dict[str, Any]] = None
    # Extraction fields from Ingestion
    extraction_method: Optional[ExtractionMethod] = None
    extraction_date: Optional[str] = None
    extracted_document_file_path: Optional[str] = None
    # Chunking fields from Ingestion
    chunking_method: Optional[ChunkingMethod] = None
    chunking_metadata: Optional[dict[str, Any]] = None
    chunking_date: Optional[str] = None
    # Featurization fields from Ingestion
    feature_models: Optional[list[str]] = None
    feature_dates: Optional[list[str]] = None
    feature_types: Optional[list[str]] = None
    # Citations from Ingestion
    unprocessed_citations: Optional[dict[str, Any]] = None
    citations: Optional[list[Citation]] = None


@SchemaRegistry.register("Embedding")
class Embedding(Entry):
    schema__: str = Field(default="Embedding", alias="schema__")
    embedding: Optional[Union[list[float], float]] = None
    tokens: Optional[int] = None


@SchemaRegistry.register("chunk_comparison")
class ChunkComparison(BaseModel):
    schema__: str = Field(default="ChunkComparison", alias="schema__")
    document_title: str
    page_range: tuple[int, int]
    chunks_a: list[Entry]
    chunks_b: list[Entry]
    winner: Optional[str] = None
    reasoning: Optional[str] = None


@SchemaRegistry.register("chunk_evaluation")
class ChunkEvaluation(Entry):
    schema__: str = Field(default="ChunkEvaluation", alias="schema__")
    text_clarity: Optional[Annotated[float, Field(gt=0, lt=5)]] = Field(None, description="The score of the chunk")
    coherence: Optional[Annotated[float, Field(gt=0, lt=5)]] = Field(None, description="The score of the chunk")
    organization: Optional[Annotated[float, Field(gt=0, lt=5)]] = Field(None, description="The score of the chunk")
    score: Optional[Annotated[float, Field(gt=0, lt=15)]] = Field(None, description="The score of the chunk")
    explanation: Optional[str] = Field(None, description="The reasoning behind the score")


@SchemaRegistry.register("formatted_scored_points")
class FormattedScoredPoints(BaseModel):
    id: str
    ingestion: Optional[Ingestion] = None
    raw_text: str = ""
    score: float
    index: Optional[Any] = None
    title: str = ""
    date: str = ""
    rerank_score: Optional[float] = 0.0


BaseModelListType = TypeVar("BaseModelListType", list[Entry], list[Ingestion])
"""Type variable for list of database models (list[Entry] or list[Ingestion])"""

RegisteredSchemaListType = TypeVar("RegisteredSchemaListType", list[Ingestion], list[Entry], list[Embedding], list[Upsert])
"""Type variable for list of registered schemas (list[Ingestion], list[Entry], list[Embedding], list[Upsert])"""
