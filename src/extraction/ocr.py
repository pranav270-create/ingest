from typing import Optional
from src.schemas.schemas import Document, Scope, ContentType
from src.extraction.ocr_service import main_ocr
from src.schemas.schemas import Ingestion, FileType
from datetime import datetime, timezone

async def ocr_parse(
    pdf_path: str, 
    scope: Optional[Scope] = Scope.EXTERNAL,
    content_type: Optional[ContentType] = ContentType.OTHER_ARTICLES,
    creator_name: str = "evaluation_pipeline",
    ingestion_method: str = "local_file",
    ingestion_date: Optional[str] = None
) -> Document:
    """Parse PDF using OCR and return a Document."""
    
    if ingestion_date is None:
        ingestion_date = datetime.now(timezone.utc).isoformat()
        
    # Create ingestion object
    ingestion = Ingestion(
        file_path=pdf_path,
        file_type=FileType.PDF,
        scope=scope,
        content_type=content_type,
        creator_name=creator_name,
        ingestion_method=ingestion_method,
        ingestion_date=ingestion_date
    )
    
    # Use existing OCR service
    documents = await main_ocr([ingestion])
    return documents[0]  # Return first document since we only process one 