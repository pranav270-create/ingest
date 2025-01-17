import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import JSON, BigInteger, Boolean, Column, DateTime, ForeignKey, Integer, String, Table, Text, func, text
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, validates
from sqlalchemy.schema import Index, UniqueConstraint

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry.function_registry import StepType
from src.schemas.schemas import (
    ChunkingMethod,
    ContentType,
    EmbeddedFeatureType,
    ExtractedFeatureType,
    ExtractionMethod,
    FileType,
    IngestionMethod,
    RelationshipType,
    Scope,
)


class Status(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Base(DeclarativeBase):
    pass


class AbstractBase(Base):
    __abstract__ = True  # SQLAlchemy needs to know this method is abstract

    @classmethod
    def bytes_to_int(field: bytes):
        return int.from_bytes(field, "little")


class Ingest(AsyncAttrs, AbstractBase):
    __tablename__ = 'ingest'

    id: Mapped[int] = mapped_column(BigInteger, autoincrement=True, primary_key=True, unique=True)
    # document fields
    document_title: Mapped[str] = mapped_column(Text, nullable=False, comment="Title of the document")
    scope: Mapped[str] = mapped_column(String(50), nullable=False, comment="Scope of the data")
    content_type: Mapped[str] = mapped_column(String(50), nullable=True, comment="Type of content")
    creator_name: Mapped[str] = mapped_column(String(100), nullable=True, comment="Name of the creator of the data")
    creation_date: Mapped[datetime] = mapped_column(DateTime, nullable=True, comment="Date the data was created")
    file_type: Mapped[str] = mapped_column(String(100), nullable=True, comment="Type of file")
    file_path: Mapped[str] = mapped_column(Text, nullable=True, index=True, comment="Cloud bucket path")
    file_size: Mapped[int] = mapped_column(Integer, nullable=True, comment="Size of the file in bytes")
    public_url: Mapped[str] = mapped_column(Text, nullable=True, index=True, comment="URL to the public data")
    # Ingestion Fields
    # total_length: Mapped[int] = mapped_column(Integer, nullable=True, comment="Total length of the data")
    ingestion_method: Mapped[str] = mapped_column(String(50), nullable=False, comment="Method used to ingest the data")
    ingestion_date: Mapped[datetime] = mapped_column(DateTime, default=func.now(), comment="Date the data was ingested")
    # Added fields
    document_summary: Mapped[str] = mapped_column(Text, nullable=True, comment="Summary of the data")
    document_keywords: Mapped[list[str]] = mapped_column(JSON, nullable=True, comment="Keywords for the data")
    document_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Additional metadata for the data")
    # Extraction fields
    extraction_method: Mapped[str] = mapped_column(String(50), nullable=True, comment="Method used to extract the data")
    extraction_date: Mapped[datetime] = mapped_column(DateTime, default=func.now(), comment="Date the data was extracted")
    extracted_document_file_path: Mapped[str] = mapped_column(String(255), nullable=True, comment="Path to the processed file")
    # Chunking fields
    chunking_method: Mapped[str] = mapped_column(String(50), nullable=True, comment="Method used to chunk the data")
    chunking_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Additional metadata for the chunking")
    chunking_date: Mapped[datetime] = mapped_column(DateTime, default=func.now(), comment="Date the data was chunked")
    # Featurization fields
    feature_models: Mapped[list[str]] = mapped_column(JSON, nullable=True, comment="Models used to featurize the data")
    feature_types: Mapped[list[str]] = mapped_column(JSON, nullable=True, comment="Types of features used")
    feature_dates: Mapped[list[str]] = mapped_column(JSON, nullable=True, comment="Dates of the features used")
    # Unprocessed citations
    unprocessed_citations: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="Unprocessed citations")

    hash: Mapped[str] = mapped_column(String(64), primary_key=True, nullable=False, index=True, unique=True, comment="SHA256 hash of the data")
    __table_args__ = (UniqueConstraint('hash', name='uq_ingest_hash'),)

    # Relationships
    processing_pipelines: Mapped[list["ProcessingPipeline"]] = relationship("ProcessingPipeline", secondary="ingest_pipeline", back_populates="ingests")  # noqa
    entries: Mapped[list["Entry"]] = relationship("Entry", back_populates="ingest")
    outgoing_relationships: Mapped[list["IngestRelationship"]] = relationship("IngestRelationship", foreign_keys="[IngestRelationship.source_id]", back_populates="source")  # noqa
    incoming_relationships: Mapped[list["IngestRelationship"]] = relationship("IngestRelationship", foreign_keys="[IngestRelationship.target_id]", back_populates="target")  # noqa

    @validates('scope')
    def validate_scope(self, key, value):  # noqa
        if isinstance(value, Scope):
            return value.value
        if value not in Scope._value2member_map_:
            raise ValueError(f"Invalid scope: {value}")
        return value

    @validates('content_type')
    def validate_content_type(self, key, value):  # noqa
        if isinstance(value, ContentType):
            return value.value
        if value not in ContentType._value2member_map_:
            raise ValueError(f"Invalid content type: {value}")
        return value

    @validates('file_type')
    def validate_file_type(self, key, value):  # noqa
        if isinstance(value, FileType):
            return value.value
        if value not in FileType._value2member_map_:
            raise ValueError(f"Invalid file type: {value}")
        return value

    @validates('ingestion_method')
    def validate_ingestion_method(self, key, value):  # noqa
        if isinstance(value, IngestionMethod):
            return value.value
        if value not in IngestionMethod._value2member_map_:
            raise ValueError(f"Invalid ingestion method: {value}")
        return value

    @validates('extraction_method')
    def validate_extraction_method(self, key, value):  # noqa
        if isinstance(value, ExtractionMethod):
            return value.value
        if value not in ExtractionMethod._value2member_map_:
            raise ValueError(f"Invalid extraction method: {value}")
        return value

    @validates('chunking_method')
    def validate_chunking_method(self, key, value):  # noqa
        if isinstance(value, ChunkingMethod):
            return value.value
        if value not in ChunkingMethod._value2member_map_:
            raise ValueError(f"Invalid chunking method: {value}")
        return value


class ProcessingPipeline(AsyncAttrs, AbstractBase):
    __tablename__ = 'processing_pipelines'
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False, comment="Version of the processing pipeline")
    description: Mapped[str] = mapped_column(String(200), nullable=True, comment="Description of this processing pipeline")
    storage_type: Mapped[str] = mapped_column(String(50), nullable=False, default="local", comment="Type of storage, local or s3")
    storage_path: Mapped[str] = mapped_column(String(255), nullable=True, comment="Path to the storage of this pipeline")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), comment="Creation date of this pipeline")
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)

    __table_args__ = (Index('idx_version_description', 'version', 'description'),)

    # Relationships
    ingests: Mapped[list["Ingest"]] = relationship("Ingest", secondary="ingest_pipeline", back_populates="processing_pipelines")  # noqa
    processing_steps: Mapped[list["ProcessingStep"]] = relationship("ProcessingStep", back_populates="pipeline")
    entries: Mapped[list["Entry"]] = relationship("Entry", back_populates="pipeline")


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
    next_steps: Mapped[list["ProcessingStep"]] = relationship("ProcessingStep", back_populates="previous_step")

    __table_args__ = (UniqueConstraint('pipeline_id', 'order', name='uq_pipeline_step_order'),)

    @validates('step_type')
    def validate_step_type(self, key, value): # noqa
        if isinstance(value, StepType):
            return value.value
        if value not in StepType._value2member_map_:
            raise ValueError(f"Invalid step type: {value}")
        return value

    @validates('status')
    def validate_status(self, key, value):  # noqa
        if isinstance(value, Status):
            return value.value
        if value not in Status._value2member_map_:
            raise ValueError(f"Invalid status: {value}")
        return value


class Entry(AsyncAttrs, AbstractBase):
    __tablename__ = 'entries'

    uuid: Mapped[str] = mapped_column(String(255), unique=True, primary_key=True, nullable=False, comment="Unique identifier for the entry (e.g., URL, UUID)")  # noqa
    # Core fields
    string: Mapped[str] = mapped_column(Text, nullable=False, comment="The text of the entry")
    # Featurization fields
    entry_title: Mapped[str] = mapped_column(Text, nullable=True, comment="The title of the entry")
    keywords: Mapped[list[str]] = mapped_column(JSON, nullable=True, comment="Keywords for the entry")
    context_summary_string: Mapped[str] = mapped_column(Text, nullable=True, comment="The summary of the document")
    added_featurization: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, comment="The featurization added to the entry")
    index_numbers: Mapped[list[int]] = mapped_column(JSON, nullable=True, comment="The index numbers of the entry")
    # Chunk location fields. Used for reconstruction
    consolidated_feature_type: Mapped[str] = mapped_column(String(100), nullable=True, comment="The type of the feature being embedded")
    chunk_locations: Mapped[list[JSON]] = mapped_column(JSON, nullable=True, comment="The locations of the chunks in the entry")
    min_primary_index: Mapped[int] = mapped_column(Integer, nullable=True, comment="The minimum primary index of the entry")
    max_primary_index: Mapped[int] = mapped_column(Integer, nullable=True, comment="The maximum primary index of the entry")
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=True, comment="The index of the chunk in the entry")
    table_number: Mapped[int] = mapped_column(Integer, nullable=True, comment="The number of the table in the entry")
    figure_number: Mapped[int] = mapped_column(Integer, nullable=True, comment="The number of the figure in the entry")
    # Embedding Fields
    embedded_feature_type: Mapped[str] = mapped_column(String(100), nullable=True, comment="The type of the feature being embedded")
    embedding_date: Mapped[datetime] = mapped_column(DateTime, default=func.now(), comment="Date the embedding was created")
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=True, comment="The model used to embed the feature")
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=True, comment="The dimensions of the embedding")

    collection_name: Mapped[str] = mapped_column(String(100), nullable=True, comment="The name of the collection the entry belongs to")
    pipeline_id: Mapped[Optional[int]] = mapped_column(ForeignKey('processing_pipelines.id'), nullable=True, index=True)
    ingestion_id: Mapped[Optional[int]] = mapped_column(ForeignKey('ingest.id'), nullable=True, index=True)
    # Relationships
    pipeline: Mapped[Optional[ProcessingPipeline]] = relationship("ProcessingPipeline", back_populates="entries")
    ingest: Mapped[Optional[Ingest]] = relationship("Ingest", back_populates="entries")
    outgoing_relationships: Mapped[list["EntryRelationship"]] = relationship("EntryRelationship", foreign_keys="[EntryRelationship.source_uuid]", back_populates="source")  # noqa
    incoming_relationships: Mapped[list["EntryRelationship"]] = relationship("EntryRelationship", foreign_keys="[EntryRelationship.target_uuid]", back_populates="target")  # noqa

    @validates('consolidated_feature_type')
    def validate_consolidated_feature_type(self, key, value):  # noqa
        if isinstance(value, ExtractedFeatureType):
            return value.value
        if value not in ExtractedFeatureType._value2member_map_:
            raise ValueError(f"Invalid consolidated feature type: {value}")
        return value

    @validates('embedded_feature_type')
    def validate_embedded_feature_type(self, key, value):  # noqa
        if isinstance(value, EmbeddedFeatureType):
            return value.value
        if value not in EmbeddedFeatureType._value2member_map_:
            raise ValueError(f"Invalid embedded feature type: {value}")
        return value



class EntryRelationship(Base):
    __tablename__ = 'entry_relationships'

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_uuid: Mapped[str] = mapped_column(ForeignKey('entries.uuid'), nullable=False)
    target_uuid: Mapped[str] = mapped_column(ForeignKey('entries.uuid'), nullable=False)
    relationship_type: Mapped[str] = mapped_column(String(50), nullable=False)
    metadata_field: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, comment="Additional metadata for the relationship")

    # Relationships
    source: Mapped["Entry"] = relationship("Entry", foreign_keys=[source_uuid], back_populates="outgoing_relationships")
    target: Mapped["Entry"] = relationship("Entry", foreign_keys=[target_uuid], back_populates="incoming_relationships")

    __table_args__ = (
        UniqueConstraint('source_uuid', 'target_uuid', 'relationship_type', name='uq_entry_relationship'),
        Index('ix_entry_relationship_source', 'source_uuid'),
        Index('ix_entry_relationship_target', 'target_uuid'),
    )

    @validates('relationship_type')
    def validate_relationship_type(self, key, value):  # noqa
        if isinstance(value, RelationshipType):
            return value.value
        if value not in RelationshipType._value2member_map_:
            raise ValueError(f"Invalid relationship type: {value}")
        return value


class IngestRelationship(Base):
    __tablename__ = 'ingest_relationships'

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
    def validate_relationship_type(self, key, value): # noqa
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

async def recreate_tables(engine):
    """Drops all existing tables and recreates them from scratch."""
    # Drop all tables with CASCADE
    async with engine.begin() as conn:
        # Drop any remaining tables with CASCADE in correct order
        await conn.execute(text("DROP TABLE IF EXISTS document_relationships CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS ingest_pipeline CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS entry_relationships CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS ingest_relationships CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS entries CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS processing_steps CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS processing_pipelines CASCADE"))
        await conn.execute(text("DROP TABLE IF EXISTS ingest CASCADE"))

    # Recreate all tables
    async with engine.begin() as conn:
        # Create tables in correct order
        await conn.run_sync(lambda conn: Base.metadata.tables['ingest'].create(conn))
        await conn.run_sync(lambda conn: Base.metadata.tables['processing_pipelines'].create(conn))
        await conn.run_sync(lambda conn: Base.metadata.tables['processing_steps'].create(conn))
        await conn.run_sync(lambda conn: Base.metadata.tables['entries'].create(conn))
        await conn.run_sync(lambda conn: Base.metadata.tables['entry_relationships'].create(conn))
        await conn.run_sync(lambda conn: Base.metadata.tables['ingest_relationships'].create(conn))
        await conn.run_sync(lambda conn: Base.metadata.tables['ingest_pipeline'].create(conn))


if __name__ == '__main__':
    import asyncio

    from sqlalchemy import select, text
    from sqlalchemy.orm import sessionmaker

    from src.sql_db.database_simple import get_engine

    engine = get_engine()
    async def main():
        await recreate_tables(engine)
        print("Tables have been recreated successfully")

    asyncio.run(main())
    exit()


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
