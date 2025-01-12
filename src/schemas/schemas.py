import sys
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.registry import SchemaRegistry


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


class ParsingMethod(str, Enum):
    SIMPLE = "simple"
    TESSERACT = "tesseract"
    TEXTRACT = "textract"
    XRAY = "xray"
    GROBID = "grobid"
    MARKER = "marker"
    UNSTRUCTURED = "unstructured"
    OCR2_0 = "ocr2_0"
    GLINER = "gliner"
    WHISPER = "whisper"
    BS4 = "bs4"
    GOOGLE_LABS_HTML_CHUNKER = "google_labs_html_chunker"
    NONE = "none"


class ParsedFeatureType(str, Enum):
    # Basic content types
    TEXT = "text"
    IMAGE = "image"
    FORM = "form"
    TABLE = "table"
    # Document structure
    PAGE = "page"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    TABLE_OF_CONTENTS = "table_of_contents"
    # Text elements
    PARAGRAPH = "paragraph"
    LIST_GROUP = "list_group"
    LIST_ITEM = "list_item"
    FOOTNOTE = "footnote"
    CAPTION = "caption"
    # Special content
    EQUATION = "equation"
    MATH_INLINE = "math_inline"
    CODE = "code"
    HANDWRITING = "handwriting"
    # Figures and media
    FIGURE = "figure"
    FIGURE_GROUP = "figure_group"
    PICTURE = "picture"
    # Complex elements
    COMPLEX_REGION = "complex_region"
    # Textract-specific elements
    WORD = "word"
    LINE = "line"
    TITLE = "title"  # Changed from "title"
    HEADER = "header"  # Changed from "header"
    FOOTER = "footer"  # Changed from "footer"
    LIST = "list"  # Changed from "list"
    PAGE_NUMBER = "page_number"  # Changed from "page_number"
    KEY_VALUE = "key_value"  # Changed from "key_value"
    SECTION_HEADER = "section_header"  # Added new
    # Catch-all
    COMBINED_TEXT = "combined_text"  # Changed from "combined_text"
    OTHER = "other"


class ChunkingMethod(str, Enum):
    REGEX = "regex"
    NLP_SENTENCE = "nlp_sentence"
    TOPIC_SEGMENTATION = "topic_segmentation"
    EMBEDDING = "embedding"
    SLIDING_WINDOW = "sliding_window"
    FIXED_LENGTH = "fixed_length"
    DISTANCE = "distance"
    GENERATIVE = "generative"
    TEXTRACT = "textract"
    NONE = "none"


class EmbeddedFeatureType(str, Enum):
    TEXT = "text"  # these words are either spoken or written (tables included)
    IMAGE = "image"  # this is for directly embedded images or ColPali (joint latent space)
    # these are synthetic features and are all text
    SYNTHETIC_FEATURE_DESCRIPTION = "synthetic_feature_description"  # this is for extracted images (traditional pipeline)
    SYNTHETIC_SUMMARY = "synthetic_summary"
    # these synthetic features are all exclusively for increasing RAG surface area
    SYNTHETIC_FACT = "synthetic_fact"
    SYNTHETIC_QUESTION = "synthetic_question"
    SYNTHETIC_KEYWORD = "synthetic_keyword"
    SYNTHETIC_ANSWER = "synthetic_answer"


class Index(BaseModel):
    """
    This is a list of indices that can be used to index a larger object. (float for video/audio, int for everything else)
    We sort by primary, then secondary, then tertiary.
    Note: We start indexing at 1 for all indices.
    For a simple text extraction, we would only have a primary index.
    For a more complex PDF, we may have a primary and secondary index telling us where each chunk is within the page.
    For a table within a complex PDF, we may have a primary, secondary, and tertiary index telling us where each cell is
    within the table within the page.
    """

    primary: Union[float, int]
    secondary: Optional[Union[float, int]] = None
    tertiary: Optional[Union[float, int]] = None

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
        return self.primary == other.primary and self.secondary == other.secondary and self.tertiary == other.tertiary

    def __hash__(self):
        return hash((self.primary, self.secondary, self.tertiary))


class BoundingBox(BaseModel):
    """
    This is the bounding box of the parsed data.
    """

    left: float
    top: float
    width: float
    height: float


"""
These are the core models
"""


@SchemaRegistry.register("Ingestion")
class Ingestion(BaseModel):
    schema__: str = Field(default="Ingestion", alias="schema__")
    # Ingestion fields
    ingestion_id: Optional[int] = None  # Needed for SQL mode
    pipeline_id: Optional[int] = None  # Needed for SQL mode
    document_title: Optional[str] = None  # We may want a LLM to title the document
    scope: Scope  # You must know the scope of the document before you can ingest it.
    content_type: Optional[ContentType] = None  # We may eventually want to infer this
    file_type: Optional[FileType] = None  # For alerts
    public_url: Optional[str] = None  # This is the public URL for the document (website, image, youtube, etc.)
    creator_name: str  # This is the name of the entity who created the document (if known)
    file_path: Optional[str] = None  # Cloud bucket path, so just grab the ending
    total_length: Optional[int] = None  # In characters if text, otherwise None
    creation_date: Optional[str] = None
    ingestion_method: IngestionMethod  # Source of ingestion (e.g., 'slack', 'youtube', 'wix', etc.)
    ingestion_date: str
    summary: Optional[str] = None  # This is a summary of the document
    keywords: Optional[list[str]] = None  # This is a list of keywords that we have extracted
    metadata: Optional[dict[str, Any]] = None  # This is for any metadata that we have not captured in other fields yet
    # Parsing fields
    parsing_method: Optional[ParsingMethod] = None
    parsing_date: Optional[str] = None
    parsed_feature_type: Optional[ParsedFeatureType] = None  # This is the type of feature that was extracted
    parsed_file_path: Optional[str] = None  # This is the path to the parsed file which we can use for more context
    # Chunking fields
    chunking_method: Optional[ChunkingMethod] = None
    chunking_metadata: Optional[dict[str, Any]] = None
    chunking_date: Optional[str] = None
    # Featurization fields
    feature_models: Optional[list[str]] = None
    feature_dates: Optional[list[str]] = None
    feature_types: Optional[list[str]] = None
    # Embedding fields
    embedded_feature_type: Optional[EmbeddedFeatureType] = None  # This is the type of feature that is being embedded
    embedding_date: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = None
    # Unprocessed citations
    unprocessed_citations: Optional[dict[str, Any]] = None  # This is for citations that have not been processed yet


@SchemaRegistry.register("Entry")
class Entry(BaseModel):
    schema__: str = Field(default="Entry", alias="schema__")
    ingestion: Optional[Ingestion] = None  # null for cross document only, citations must be there
    string: Optional[str] = None  # If we embed a document or image, we don't need the original text
    context_summary_string: Optional[str] = (
        None  # This is only if we are generating a summary of the entry wrt the broader document # noqa
    )
    added_featurization: Optional[dict[str, Any]] = None  # This is for any additional features that we have added
    keywords: Optional[list[str]] = None  # This is for any keywords that we have not captured in other fields yet
    index_numbers: Optional[list[Index]] = (
        None  # Null if we embed whole document or cross-doc summary. Represents range for continous time items; int for discrete. # noqa
    )
    bounding_box: Optional[list[BoundingBox]] = None   # since we may cross page boundaries
    parsed_feature_type: Optional[list[ParsedFeatureType]] = None  # since we may have multiple feature types in a single entry
    # Add fields for parent-child relationships in textract parsing
    id: Optional[str] = None  # Unique identifier for this entry
    parent_id: Optional[str] = None  # ID of the parent entry (e.g., table containing cells)
    child_ids: Optional[list[str]] = None  # IDs of child entries (e.g., cells in a table)


@SchemaRegistry.register("Embedding")
class Embedding(Entry):
    schema__: str = Field(default="Embedding", alias="schema__")
    embedding: Union[list[float], float]  # This is the actual embedding
    tokens: Optional[int] = None  # This is the number of tokens in the embedding


@SchemaRegistry.register("Upsert")
class Upsert(BaseModel):
    schema__: str = Field(default="Upsert", alias="schema__")
    # From Upsert
    uuid: str
    # From Entry
    keywords: Optional[list[str]] = None
    index_numbers: Optional[list[Index]] = (
        None  # Null if we embed whole document or cross-doc summary. Represents range for continous time items; int for discrete. # noqa
    )
    string: Optional[str] = None
    context_summary_string: Optional[str] = (
        None  # This is only if we are generating a summary of the entry wrt the broader document # noqa
    )
    added_featurization: Optional[dict[str, Any]] = None  # This is for any additional features that we have added
    # From Embedding
    sparse_vector: dict[str, Union[list[float], list[int]]]  # Changed this line
    dense_vector: Union[list[float], float]  # This is the actual embedding
    # From Ingestion, Also what is up in the VDB as Payload
    ingestion_id: Optional[int] = None  # Needed for SQL mode
    pipeline_id: Optional[int] = None  # Needed for SQL mode
    document_title: Optional[str] = None  # We may want a LLM to title the document
    scope: Scope  # You must know the scope of the document before you can ingest it.
    content_type: Optional[ContentType] = None  # We may eventually want to infer this
    file_type: Optional[FileType] = None  # For alerts
    public_url: Optional[str] = None  # This is the public URL for the document (website, image, youtube, etc.)
    creator_name: str  # This is the name of the entity who created the document (if known)
    file_path: Optional[str] = None  # Cloud bucket path, so just grab the ending
    total_length: Optional[int] = None  # In characters if text, otherwise None
    creation_date: Optional[str] = None
    ingestion_method: IngestionMethod  # Source of ingestion (e.g., 'slack', 'youtube', 'wix', etc.)
    ingestion_date: str
    # Parsing fields
    parsing_method: Optional[ParsingMethod] = None
    parsing_date: Optional[str] = None
    parsed_feature_type: Optional[ParsedFeatureType] = None  # This is the type of feature that was extracted
    parsed_file_path: Optional[str] = None  # This is the path to the parsed file which we can use for more context
    # Chunking fields
    chunking_method: Optional[ChunkingMethod] = None
    chunking_date: Optional[str] = None
    # Featurization fields
    feature_models: Optional[list[str]] = None
    feature_dates: Optional[list[str]] = None
    feature_types: Optional[list[str]] = None
    # Embedding fields
    embedded_feature_type: Optional[EmbeddedFeatureType] = None  # This is the type of feature that is being embedded
    embedding_date: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_dimensions: Optional[int] = None


class FormattedScoredPoints(BaseModel):
    id: str
    ingestion: Optional[Ingestion] = None
    raw_text: str = ""
    score: float
    index: Optional[Any] = None
    title: str = ""
    date: str = ""
    rerank_score: Optional[float] = 0.0
