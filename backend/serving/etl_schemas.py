from pydantic import BaseModel
from typing import List, Optional, Union
from datetime import datetime
from src.schemas.schemas import FormattedScoredPoints
from qdrant_client.http.models import FieldCondition, MatchValue


class ChatFilters(BaseModel):
    collection: Optional[str] = None
    pipeline_id: Optional[int] = None
    include_tags: Optional[List[str]] = None
    exclude_tags: Optional[List[str]] = None

    def to_qdrant_conditions(self) -> dict:
        must_conditions = []
        must_not_conditions = []

        if self.pipeline_id is not None:
            must_conditions.append(
                FieldCondition(
                    key="pipeline_id",
                    match=MatchValue(value=self.pipeline_id)
                )
            )

        if self.include_tags:
            for tag in self.include_tags:
                must_conditions.append(
                    FieldCondition(
                        key="tags",
                        match=MatchValue(value=tag)
                    )
                )

        if self.exclude_tags:
            for tag in self.exclude_tags:
                must_not_conditions.append(
                    FieldCondition(
                        key="tags",
                        match=MatchValue(value=tag)
                    )
                )

        return {
            "must": must_conditions,
            "must_not": must_not_conditions
        }


class TaggingRequest(BaseModel):
    collection_name: str
    ids: List[str]
    tag: str
    level: str


class EmbeddingFilters(BaseModel):
    ingestion_id: Optional[int] = None
    pipeline_id: Optional[int] = None
    document_title: Optional[str] = None
    scope: Optional[str] = None
    content_type: Optional[str] = None
    creator_name: Optional[str] = None
    ingestion_method: Optional[str] = None
    embedding_model: Optional[str] = None
    kmeans_clusters: Optional[int] = None


class Citation(FormattedScoredPoints):
    notes: Optional[str] = None


class ChatMessage(BaseModel):
    id: str
    type: str  # 'user' | 'assistant'
    content: str
    citations: Optional[List[Citation]] = None
    timestamp: Union[datetime, str]


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None
    collection: str
    filters: ChatFilters


class SearchRequest(BaseModel):
    query: str
