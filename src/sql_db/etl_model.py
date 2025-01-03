from sqlalchemy import DateTime, String, Text, func, BigInteger, Float, Integer, null, ForeignKey, JSON, Boolean, Enum as SQLAlchemyEnum, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.schema import UniqueConstraint, Index
from sqlalchemy.orm import validates
from sqlalchemy.ext.asyncio import AsyncAttrs
from datetime import datetime
from typing import Any, Optional, List
from sqlalchemy import Table, Column
from enum import Enum as PythonEnum
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))


class Base(DeclarativeBase):
    pass


class AbstractBase(Base):
    __abstract__ = True  # SQLAlchemy needs to know this method is abstract

    @classmethod
    def bytes_to_int(field: bytes):
        return int.from_bytes(field, "little")


class Ingest(AsyncAttrs, AbstractBase):
    __tablename__ = 'ingest'

    id: Mapped[int] = mapped_column(BigInteger, autoincrement=True, primary_key=True)
    document_title: Mapped[str] = mapped_column(Text, nullable=False, comment="Title of the document")
    public_url: Mapped[str] = mapped_column(Text, nullable=True, index=True, comment="URL to the public data")
    creator_name: Mapped[str] = mapped_column(String(100), nullable=True, comment="Name of the creator of the data")
    file_path: Mapped[str] = mapped_column(Text, nullable=True, index=True, comment="Cloud bucket path")
    total_length: Mapped[int] = mapped_column(Integer, nullable=True, comment="Total length of the data")
    creation_date: Mapped[datetime] = mapped_column(DateTime, nullable=True, comment="Date the data was created")
    ingestion_method: Mapped[str] = mapped_column(String(50), nullable=False, comment="Method used to ingest the data")
    ingestion_date: Mapped[datetime] = mapped_column(DateTime, default=func.now(), comment="Date the data was ingested")
    scope: Mapped[str] = mapped_column(String(50), nullable=False, comment="Scope of the data")
    content_type: Mapped[str] = mapped_column(String(50), nullable=True, comment="Type of content")
    file_type: Mapped[str] = mapped_column(String(100), nullable=True, comment="Type of file")
    summary: Mapped[str] = mapped_column(Text, nullable=True, comment="Summary of the data")
    keywords: Mapped[list[str]] = mapped_column(JSON, nullable=True, comment="Keywords for the data")
    metadata_field: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Additional metadata for the data")
    # TODO: Add the processed_file_path. Should be independant of VDB.
    processed_file_path: Mapped[str] = mapped_column(String(255), nullable=True, comment="Path to the processed file")

    unprocessed_citations: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Unprocessed citations")

    hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True, unique=True, comment="SHA256 hash of the data")
    __table_args__ = (
        UniqueConstraint('hash', name='uq_ingest_hash'),
    )
    processing_pipelines: Mapped[List["ProcessingPipeline"]] = relationship(
        "ProcessingPipeline",
        secondary="ingest_pipeline",
        back_populates="ingests"
    )
    entries: Mapped[List["Entry"]] = relationship("Entry", back_populates="ingest")

    outgoing_relationships: Mapped[List["DocumentRelationship"]] = relationship(
        "DocumentRelationship", 
        foreign_keys="[DocumentRelationship.source_id]", 
        back_populates="source"
    )
    incoming_relationships: Mapped[List["DocumentRelationship"]] = relationship(
        "DocumentRelationship", 
        foreign_keys="[DocumentRelationship.target_id]", 
        back_populates="target"
    )


class ProcessingPipeline(AsyncAttrs, AbstractBase):
    __tablename__ = 'processing_pipelines'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False, comment="Version of the processing pipeline")
    description: Mapped[str] = mapped_column(String(200), nullable=True, comment="Description of this processing pipeline")
    storage_type: Mapped[str] = mapped_column(String(50), nullable=False, default="local", comment="Type of storage")
    storage_path: Mapped[str] = mapped_column(String(255), nullable=True, comment="Path to the storage of this pipeline")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), comment="Creation date of this pipeline")
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index('idx_version_description', 'version', 'description'),
    )

    ingests: Mapped[List["Ingest"]] = relationship(
        "Ingest",
        secondary="ingest_pipeline",
        back_populates="processing_pipelines"
    )
    processing_steps: Mapped[List["ProcessingStep"]] = relationship("ProcessingStep", back_populates="pipeline")
    entries: Mapped[List["Entry"]] = relationship("Entry", back_populates="pipeline")


class StepType(str, PythonEnum):
    INGEST = "ingest"
    PARSE = "parse"
    CHUNK = "chunk"
    FEATURIZE = "featurize"
    EMBED = "embed"
    UPSERT = "upsert"


class Status(str, PythonEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStep(AsyncAttrs, AbstractBase):
    __tablename__ = 'processing_steps'

    id: Mapped[int] = mapped_column(BigInteger, autoincrement=True, primary_key=True)
    order: Mapped[int] = mapped_column(Integer, nullable=False)
    step_type: Mapped[str] = mapped_column(String(50), nullable=False, comment="Type of processing step")
    function_name: Mapped[str] = mapped_column(String(100), nullable=False, comment="Name of the function to run")
    date: Mapped[datetime] = mapped_column(DateTime, default=func.now(), comment="Date of the processing step")
    output_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment="Path to the output of this step")
    metadata_field: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Additional metadata for this step")
    is_optional: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default=Status.PENDING.value, comment="Status of the processing step")

    pipeline_id: Mapped[int] = mapped_column(ForeignKey('processing_pipelines.id'), nullable=False, index=True)
    pipeline: Mapped[ProcessingPipeline] = relationship('ProcessingPipeline', back_populates='processing_steps')

    previous_step_id: Mapped[Optional[int]] = mapped_column(ForeignKey('processing_steps.id'), nullable=True, index=True)
    previous_step: Mapped[Optional["ProcessingStep"]] = relationship("ProcessingStep", remote_side=[id], back_populates="next_steps")
    next_steps: Mapped[List["ProcessingStep"]] = relationship("ProcessingStep", back_populates="previous_step")

    __table_args__ = (
        UniqueConstraint('pipeline_id', 'order', name='uq_pipeline_step_order'),
    )

    @validates('step_type')
    def validate_step_type(self, key, value):
        if isinstance(value, StepType):
            return value.value
        if value not in StepType._value2member_map_:
            raise ValueError(f"Invalid step type: {value}")
        return value

    @validates('status')
    def validate_status(self, key, value):
        if isinstance(value, Status):
            return value.value
        if value not in Status._value2member_map_:
            raise ValueError(f"Invalid status: {value}")
        return value


class RelationshipType(str, PythonEnum):
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


class Entry(AsyncAttrs, AbstractBase):
    __tablename__ = 'entries'

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    unique_identifier: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, comment="Unique identifier for the entry (e.g., URL, UUID)")
    collection_name: Mapped[str] = mapped_column(String(100), index=True, nullable=False, comment="Name of the collection the entry belongs to")

    keywords: Mapped[list[str]] = mapped_column(JSON, nullable=True, comment="Keywords for the entry")
    string: Mapped[str] = mapped_column(Text, nullable=False, comment="The text of the entry")
    context_summary_string: Mapped[str] = mapped_column(Text, nullable=True, comment="A summary of the entry in the context of the document")
    synthetic_questions: Mapped[list[str]] = mapped_column(JSON, nullable=True, comment="Questions generated from the entry")
    added_featurization: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Additional features added to the entry")
    index_numbers: Mapped[str] = mapped_column(Text, nullable=True, comment="Index numbers for the entry")

    __table_args__ = (
       UniqueConstraint('collection_name', 'unique_identifier', name='uq_collection_unique_identifier'),
       Index('ix_collection_unique_identifier', 'collection_name', 'unique_identifier'),
    )

    pipeline_id: Mapped[Optional[int]] = mapped_column(ForeignKey('processing_pipelines.id'), nullable=True, index=True)
    pipeline: Mapped[Optional[ProcessingPipeline]] = relationship("ProcessingPipeline", back_populates="entries")

    ingestion_id: Mapped[Optional[int]] = mapped_column(ForeignKey('ingest.id'), nullable=True, index=True)
    ingest: Mapped[Optional[Ingest]] = relationship("Ingest", back_populates="entries")

    # Relationships
    outgoing_relationships: Mapped[List["EntryRelationship"]] = relationship("EntryRelationship", foreign_keys="[EntryRelationship.source_id]", back_populates="source")
    incoming_relationships: Mapped[List["EntryRelationship"]] = relationship("EntryRelationship", foreign_keys="[EntryRelationship.target_id]", back_populates="target")


class EntryRelationship(Base):
    __tablename__ = 'entry_relationships'

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey('entries.id'), nullable=False)
    target_id: Mapped[int] = mapped_column(ForeignKey('entries.id'), nullable=False)
    relationship_type: Mapped[str] = mapped_column(String(50), nullable=False)
    metadata_field: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, comment="Additional metadata for the relationship")

    linkage_level: Mapped[str] = mapped_column(String(50), nullable=False, comment="Level of linkage: 'entry' or 'ingestion'")

    source: Mapped["Entry"] = relationship("Entry", foreign_keys=[source_id], back_populates="outgoing_relationships")
    target: Mapped["Entry"] = relationship("Entry", foreign_keys=[target_id], back_populates="incoming_relationships")

    __table_args__ = (
        UniqueConstraint('source_id', 'target_id', 'relationship_type', name='uq_entry_relationship'),
        Index('ix_entry_relationship_source', 'source_id'),
        Index('ix_entry_relationship_target', 'target_id'),
    )

    @validates('relationship_type')
    def validate_relationship_type(self, key, value):
        if isinstance(value, RelationshipType):
            return value.value
        if value not in RelationshipType._value2member_map_:
            raise ValueError(f"Invalid relationship type: {value}")
        return value


class DocumentRelationship(Base):
    __tablename__ = 'document_relationships'

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey('ingest.id'), nullable=False)
    target_id: Mapped[int] = mapped_column(ForeignKey('ingest.id'), nullable=False)
    relationship_type: Mapped[str] = mapped_column(String(50), nullable=False)
    metadata_field: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, comment="Additional metadata for the relationship")

    source: Mapped["Ingest"] = relationship("Ingest", foreign_keys=[source_id], back_populates="outgoing_relationships")
    target: Mapped["Ingest"] = relationship("Ingest", foreign_keys=[target_id], back_populates="incoming_relationships")

    __table_args__ = (
        UniqueConstraint('source_id', 'target_id', 'relationship_type', name='uq_document_relationship'),
        Index('ix_document_relationship_source', 'source_id'),
        Index('ix_document_relationship_target', 'target_id'),
    )

    @validates('relationship_type')
    def validate_relationship_type(self, key, value):
        if isinstance(value, RelationshipType):
            return value.value
        if value not in RelationshipType._value2member_map_:
            raise ValueError(f"Invalid relationship type: {value}")
        return value


# New association table
ingest_pipeline = Table(
    'ingest_pipeline', Base.metadata,
    Column('ingest_id', ForeignKey('ingest.id'), primary_key=True),
    Column('pipeline_id', ForeignKey('processing_pipelines.id'), primary_key=True)
)


if __name__ == '__main__':
    import asyncio
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.future import select
    from sqlalchemy import text
    from src.sql_db.database_simple import get_engine

    engine = get_engine("tinypipeline")
    Session = sessionmaker(bind=engine)
    session = Session()

    async def alter_table():
        with session.begin():
            stmt = text("""
                ALTER TABLE ingest 
                ALTER COLUMN document_title TYPE TEXT,
                ALTER COLUMN summary TYPE TEXT,
                ALTER COLUMN public_url TYPE TEXT,
                ALTER COLUMN file_path TYPE TEXT
            """)
            session.execute(stmt)
            session.commit()

    async def check_ingestion():
        with session.begin():
            stmt = select(Ingest).where(Ingest.id > 500).limit(10)
            result = session.execute(stmt)
            ingests = result.scalars().all()
            for ingest in ingests:
                print(f"Ingest ID: {ingest.id}, Document Title: {ingest.document_title}")
                print(f"Associated Pipelines: {len(ingest.processing_pipelines)}")
                for pipeline in ingest.processing_pipelines:
                    print(f"  Pipeline ID: {pipeline.id}")
                print(f"Number of Entries: {len(ingest.entries)}")
                print("")

    asyncio.run(alter_table())
    # asyncio.run(check_ingestion())
