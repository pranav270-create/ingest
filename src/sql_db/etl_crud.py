import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Union

from sqlalchemy import and_, delete, inspect, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.schemas import Embedding, Upsert
from src.schemas.schemas import Entry as EntrySchema
from src.sql_db.etl_model import Entry, EntryRelationship, Ingest, ProcessingPipeline, ProcessingStep, StepType, ingest_pipeline


async def get_insert_stmt(i: Union[str, AsyncSession]) -> Callable:
    if not isinstance(i, str):
        dialect = i.get_bind().dialect.name
    else:
        dialect = i

    if dialect == "mysql":
        return mysql_insert
    else:
        return pg_insert


async def create_or_update_ingest(session: AsyncSession, data: dict, pipeline=None) -> Ingest:
    # Get all column names of the Ingest model
    ingest_fields = [c.key for c in inspect(Ingest).columns if c.key != "id"]
    # Filter the input data to only include relevant fields
    ingest_data = {k: v for k, v in data.items() if k in ingest_fields}

    # Use the existing document_hash
    hash_value = data["document_hash"]

    # Check if an Ingest with this hash already exists
    stmt = select(Ingest).where(Ingest.document_hash == hash_value)
    result = await session.execute(stmt)
    existing_ingest = result.scalar_one_or_none()

    if existing_ingest:
        # Update the existing Ingest
        for key, value in ingest_data.items():
            setattr(existing_ingest, key, value)
        if pipeline:
            # Load the processing_pipelines relationship
            await session.refresh(existing_ingest, ["processing_pipelines"])
            # Check if the pipeline is already associated
            pipeline_ids = [p.id for p in existing_ingest.processing_pipelines]
            if pipeline.id not in pipeline_ids:
                existing_ingest.processing_pipelines.append(pipeline)
        await session.commit()
        return existing_ingest
    else:
        # Create a new Ingest
        new_ingest = Ingest(**ingest_data, hash=hash_value)
        if pipeline:
            new_ingest.processing_pipelines.append(pipeline)
        session.add(new_ingest)
        await session.commit()
        return new_ingest


async def create_or_update_ingest_batch(session: AsyncSession, items: list[dict], pipeline=None) -> dict[str, Ingest]:
    # Get all column names of the Ingest model
    ingest_fields = [c.key for c in inspect(Ingest).columns if c.key != "id"]

    # Process all items to prepare data
    hash_to_data = {}
    for data in items:
        # Filter and prepare the data
        ingest_data = {k: v for k, v in data.items() if k in ingest_fields}
        hash_value = data["document_hash"]  # Use the existing hash
        hash_to_data[hash_value] = ingest_data

    try:
        # Get existing ingests by hash
        existing_hashes = list(hash_to_data.keys())
        stmt = select(Ingest).where(Ingest.document_hash.in_(existing_hashes))
        result = await session.execute(stmt)
        existing_ingests = {i.document_hash: i for i in result.scalars().all()}

        # Update existing and create new ingests
        ingests = {}
        for hash_value, data in hash_to_data.items():
            if hash_value in existing_ingests:
                # Update existing ingest
                ingest = existing_ingests[hash_value]
                for key, value in data.items():
                    setattr(ingest, key, value)
            else:
                # Create new ingest
                ingest = Ingest(**data, hash=hash_value)
                session.add(ingest)

            ingests[hash_value] = ingest

        await session.flush()

        # Handle pipeline relationships if needed
        if pipeline:
            insert_stmt = await get_insert_stmt(session)

            for ingest in ingests.values():
                # Check if relationship exists
                stmt = select(ingest_pipeline).where(and_(ingest_pipeline.c.ingest_id == ingest.id, ingest_pipeline.c.pipeline_id == pipeline.id))
                result = await session.execute(stmt)

                if not result.first():
                    # Create new relationship
                    stmt = insert_stmt(ingest_pipeline).values({"ingest_id": ingest.id, "pipeline_id": pipeline.id})
                    await session.execute(stmt)

        await session.commit()
        return ingests

    except Exception as e:
        print(f"Error in create_or_update_ingest_batch: {str(e)}")
        await session.rollback()
        raise


async def create_or_get_processing_pipeline(session: AsyncSession, pipeline_config: dict = {}, storage_config: dict = {}) -> ProcessingPipeline:
    """
    Create or get a processing pipeline
    """
    pipeline_id = pipeline_config.get("pipeline_id", None)
    version = pipeline_config.get("version", "1.0")
    description = pipeline_config.get("description", "")

    storage_type = storage_config.get("type", "local")
    if storage_type == "s3":
        storage_path = storage_config.get("bucket_name", "")
    else:
        storage_path = storage_config.get("base_path", "")

    # Try to fetch existing pipeline
    if pipeline_id:
        stmt = select(ProcessingPipeline).where(ProcessingPipeline.id == pipeline_id)
        result = await session.execute(stmt)
        if pipeline := result.scalar_one_or_none():
            return pipeline

    # Create a new pipeline
    new_pipeline = ProcessingPipeline(
        version=version, description=description, storage_type=storage_type, storage_path=storage_path, created_at=datetime.now()
    )
    session.add(new_pipeline)
    await session.commit()
    await session.refresh(new_pipeline)

    return new_pipeline


async def clone_pipeline(
    session: AsyncSession, original: ProcessingPipeline, clone_till_step: int, new_description: str = None
) -> ProcessingPipeline:
    """
    Creates a new pipeline by cloning steps up to clone_till_step from the original pipeline.
    """
    # First get the ingests explicitly through a join query
    stmt = select(Ingest).join(ingest_pipeline, and_(ingest_pipeline.c.pipeline_id == original.id, ingest_pipeline.c.ingest_id == Ingest.id))
    result = await session.execute(stmt)
    ingests = result.scalars().all()

    # Create new pipeline with incremented version
    new_pipeline = ProcessingPipeline(
        version=f"{original.version}-branch",
        description=new_description or f"Branched from Pipeline {original.id} at step {clone_till_step}",
        storage_type=original.storage_type,
        storage_path=original.storage_path,
        config=original.config,
        created_at=datetime.now(),
    )
    session.add(new_pipeline)
    await session.flush()

    # Clone steps up to clone_till_step
    stmt = (
        select(ProcessingStep)
        .where((ProcessingStep.pipeline_id == original.id) & (ProcessingStep.order < clone_till_step))
        .order_by(ProcessingStep.order)
    )

    result = await session.execute(stmt)
    steps = result.scalars().all()

    # Clone each step
    step_map = {}  # maps old step IDs to new step IDs
    for step in steps:
        new_step = ProcessingStep(
            pipeline_id=new_pipeline.id,
            order=step.order,
            step_type=step.step_type,
            function_name=step.function_name,
            date=step.date,
            output_path=step.output_path,
            metadata_field=step.metadata_field,
            status=step.status,
        )
        session.add(new_step)
        await session.flush()

        step_map[step.id] = new_step.id

        # Update previous_step_id references
        if step.previous_step_id and step.previous_step_id in step_map:
            new_step.previous_step_id = step_map[step.previous_step_id]

    dialect = session.get_bind().dialect.name
    insert_stmt = await get_insert_stmt(dialect)

    # Clone ingest associations using explicit relationships
    for ingest in ingests:
        # Create the relationship using the association table
        stmt = insert_stmt(ingest_pipeline).values({"ingest_id": ingest.id, "pipeline_id": new_pipeline.id})

        # handle duplicates for both dialects
        if dialect == "mysql":
            stmt = stmt.prefix_with("IGNORE")
        else:
            stmt = stmt.on_conflict_do_nothing()
        await session.execute(stmt)

    await session.commit()
    await session.refresh(new_pipeline)

    return new_pipeline


async def create_processing_step(
    session: AsyncSession,
    pipeline_id: int,
    order: int,
    step_type: str,
    function_name: str,
    status: str,
    previous_step_id: int = None,
    output_path: str = None,
    metadata: dict = None,
) -> ProcessingStep:
    try:
        # see if step exists with that pipeline_id and order
        stmt = select(ProcessingStep).where(
            (ProcessingStep.pipeline_id == pipeline_id) &
            (ProcessingStep.order == order)
        )
        result = await session.execute(stmt)
        existing_step = result.scalar_one_or_none()

        if existing_step:
            print(
                f"\033[93mWARNING\033[0m: ProcessingStep already exists with pipeline_id {pipeline_id} and order {order}. This step will be overwritten."
            )
            existing_step.previous_step_id = previous_step_id
            existing_step.status = status
            existing_step.function_name = function_name
            existing_step.step_type = StepType[step_type.upper()]
            existing_step.date = datetime.now()
            existing_step.output_path = output_path
            existing_step.metadata_field = metadata
            session.add(existing_step)
            await session.commit()
            await session.refresh(existing_step)
            return existing_step

        new_step = ProcessingStep(
            pipeline_id=pipeline_id,
            previous_step_id=previous_step_id,
            order=order,
            status=status,
            function_name=function_name,
            step_type=StepType[step_type.upper()],
            date=datetime.now(),
            output_path=output_path,
            metadata_field=metadata,
        )
        session.add(new_step)
        await session.commit()
        await session.refresh(new_step)
        return new_step

    except Exception as e:
        print(f"Error creating processing step: {str(e)}")
        # Create a failed step if we encounter an error
        failed_step = ProcessingStep(
            pipeline_id=pipeline_id,
            order=order,
            status="failed",
            function_name=function_name,
            step_type=StepType.UNKNOWN,  # Default to UNKNOWN if we can't parse the step type
            date=datetime.now(),
            output_path=output_path,
            metadata_field={"error": str(e), **(metadata or {})}
        )
        session.add(failed_step)
        await session.commit()
        await session.refresh(failed_step)
        return failed_step


async def get_latest_processing_step(session: AsyncSession, pipeline_id: int) -> ProcessingStep:
    stmt = (
        select(ProcessingStep)
        .where((ProcessingStep.pipeline_id == pipeline_id) & (ProcessingStep.status == "completed"))
        .order_by(ProcessingStep.order.desc())
        .limit(1)
    )

    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_specific_processing_step(session: AsyncSession, pipeline_id: int, step_order: int) -> ProcessingStep:
    step_order = max(step_order - 1, 0)  # we want the results of the previous step
    stmt = select(ProcessingStep).where((ProcessingStep.pipeline_id == pipeline_id) & (ProcessingStep.order == step_order))

    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_next_processing_step(session: AsyncSession, pipeline_id: int, current_order: int) -> ProcessingStep:
    stmt = (
        select(ProcessingStep)
        .where((ProcessingStep.pipeline_id == pipeline_id) & (ProcessingStep.order > current_order))
        .order_by(ProcessingStep.order.asc())
    )

    result = await session.execute(stmt)
    return result.scalar_one_or_none()


@dataclass
class EntryCreationResult:
    total_processed: int
    new_entries: int
    skipped_duplicates: int
    updated_entries: int
    failed_entries: int
    error_messages: list[str]


def create_content_hash(data: dict, version: str) -> str:
    """Create a consistent hash from entry content."""
    content_to_hash = {
        "string": data.get("string", "").replace("\x00", "").encode("utf-8", "ignore").decode("utf-8"),
        "ingestion_id": data.get("ingestion_id", data.get("ingestion", {}).get("ingestion_id")),
        "chunk_locations": data.get("chunk_locations", []),
        "entry_title": data.get("entry_title"),
        "keywords": data.get("keywords"),
        "consolidated_feature_type": data.get("consolidated_feature_type"),
        "citations": data.get("citations", []),
        "version": version,
    }
    return hashlib.sha256(json.dumps(content_to_hash, sort_keys=True).encode()).hexdigest()


def map_entry_data(data: dict, content_hash: str, collection_name: str) -> dict:
    """Map schema data to SQL Entry model fields."""
    entry_data = {
        "content_hash": content_hash,
        "uuid": data.get("uuid"),
        "collection_name": collection_name,
        "pipeline_id": data.get("pipeline_id", data.get("ingestion", {}).get("pipeline_id", None)),
        "ingestion_id": data.get("ingestion_id", data.get("ingestion", {}).get("ingestion_id", None)),
        "string": data.get("string", "").replace("\x00", "").encode("utf-8", "ignore").decode("utf-8"),
        "entry_title": data.get("entry_title"),
        "keywords": json.dumps(data.get("keywords", [])),
        "added_featurization": json.dumps(data.get("added_featurization", {})),
        "consolidated_feature_type": data.get("consolidated_feature_type"),
        "chunk_locations": json.dumps(data.get("chunk_locations", [])),
        "min_primary_index": data.get("min_primary_index"),
        "max_primary_index": data.get("max_primary_index"),
        "chunk_index": data.get("chunk_index"),
        "table_number": data.get("table_number"),
        "figure_number": data.get("figure_number"),
        "embedded_feature_type": data.get("embedded_feature_type"),
        "embedding_date": data.get("embedding_date"),
        "embedding_model": data.get("embedding_model"),
        "embedding_dimensions": data.get("embedding_dimensions"),
    }
    return entry_data


async def process_single_entry(
    session: AsyncSession,
    entry_data: dict,
    update_on_collision: bool,
    result: EntryCreationResult
) -> Optional[Entry]:
    """Process a single entry and its relationships."""
    try:
        # Check if entry exists by content hash and pipeline_id
        stmt = select(Entry).where(
            and_(
                Entry.content_hash == entry_data["content_hash"],
                Entry.pipeline_id == entry_data["pipeline_id"]
            )
        )
        existing_entry = (await session.execute(stmt)).scalar_one_or_none()

        if existing_entry:
            if not update_on_collision:
                result.skipped_duplicates += 1
                return None
            # Update existing entry
            for key, value in entry_data.items():
                setattr(existing_entry, key, value)
            current_entry = existing_entry
            result.updated_entries += 1
        else:
            current_entry = Entry(**entry_data)
            session.add(current_entry)
            result.new_entries += 1

        await session.flush()  # Ensure we have an ID
        return current_entry

    except Exception as e:
        result.failed_entries += 1
        result.error_messages.append(f"Error processing entry: {str(e)}")
        return None


async def validate_data(data: dict, result: EntryCreationResult) -> bool:
    """Validate required fields based on data type."""
    if "sparse_vector" in data or "dense_vector" in data:
        # Upsert-specific validation
        required_fields = ["uuid", "pipeline_id", "ingestion_id", "document_hash", "scope", "creator_name", "ingestion_method"]

        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            result.failed_entries += 1
            result.error_messages.append(f"Missing required Upsert fields: {missing_fields} for entry {data.get('uuid')}")
            return False

        if not (data.get("dense_vector") or data.get("sparse_vector")):
            result.failed_entries += 1
            result.error_messages.append(f"Upsert must have either dense_vector or sparse_vector for entry {data.get('uuid')}")
            return False
    else:
        # Original Entry/Embedding validation
        if not data.get("ingestion", {}).get("pipeline_id") or not data.get("ingestion", {}).get("ingestion_id"):
            result.failed_entries += 1
            result.error_messages.append(f"Missing required fields for entry {data.get('uuid')}")
            return False

    return True


async def create_entries(
    session: AsyncSession,
    data_list: Union[list[EntrySchema], list[Embedding], list[Upsert]],
    collection_name: str,
    version: str,
    update_on_collision: bool = False,
    batch_size: int = 1000,
) -> EntryCreationResult:
    result = EntryCreationResult(
        total_processed=len(data_list), new_entries=0, skipped_duplicates=0, 
        updated_entries=0, failed_entries=0, error_messages=[]
    )

    try:
        # Phase 1: Prepare all entry data and collect existing entries
        entry_data_map = {}  # UUID to entry data mapping
        content_hashes = set()
        pipeline_ids = set()

        for data in data_list:
            if not await validate_data(data.model_dump(), result):
                continue

            data_dict = data.model_dump()
            content_hash = create_content_hash(data_dict, version)
            entry_data = map_entry_data(data_dict, content_hash, collection_name)

            entry_data_map[data.uuid] = entry_data
            content_hashes.add(content_hash)
            pipeline_ids.add(entry_data['pipeline_id'])

        # Bulk fetch existing entries
        stmt = select(Entry).where(
            and_(
                Entry.content_hash.in_(content_hashes),
                Entry.pipeline_id.in_(pipeline_ids)
            )
        )
        existing_entries = {
            (e.content_hash, e.pipeline_id): e 
            for e in (await session.execute(stmt)).scalars().all()
        }

        # Phase 2: Bulk create/update entries
        new_entries = []
        source_entries = {}

        for uuid, entry_data in entry_data_map.items():
            key = (entry_data['content_hash'], entry_data['pipeline_id'])
            if key in existing_entries:
                if update_on_collision:
                    entry = existing_entries[key]
                    for k, v in entry_data.items():
                        setattr(entry, k, v)
                    source_entries[uuid] = entry
                    result.updated_entries += 1
                else:
                    result.skipped_duplicates += 1
            else:
                entry = Entry(**entry_data)
                new_entries.append(entry)
                source_entries[uuid] = entry
                result.new_entries += 1

        if new_entries:
            session.add_all(new_entries)
            await session.flush()

        # Phase 3: Handle relationships
        citation_data = []
        missing_target_uuids = set()

        for data in data_list:
            if hasattr(data, 'citations') and data.citations:
                for citation in data.citations:
                    if citation.target_uuid not in source_entries:
                        missing_target_uuids.add(citation.target_uuid)
                    citation_data.append((data.uuid, citation))

        # Bulk fetch missing target entries
        target_entries = {}
        if missing_target_uuids:
            stmt = select(Entry).where(Entry.uuid.in_(missing_target_uuids))
            target_entries = {
                entry.uuid: entry
                for entry in (await session.execute(stmt)).scalars().all()
            }

        # Bulk create relationships
        relationships = []
        for source_uuid, citation in citation_data:
            source_entry = source_entries.get(source_uuid)
            target_entry = source_entries.get(citation.target_uuid) or target_entries.get(citation.target_uuid)

            if source_entry and target_entry:
                relationships.append(
                    EntryRelationship(
                        source_id=source_entry.id,
                        target_id=target_entry.id,
                        relationship_type=citation.relationship_type.value,
                    )
                )

        if relationships:
            # Use insert().prefix_with('IGNORE') for MySQL or on_conflict_do_nothing() for PostgreSQL
            insert_stmt = await get_insert_stmt(session)
            stmt = insert_stmt(EntryRelationship.__table__).values(
                [{'source_id': r.source_id, 'target_id': r.target_id, 'relationship_type': r.relationship_type} 
                 for r in relationships]
            )
            if session.get_bind().dialect.name == "mysql":
                stmt = stmt.prefix_with("IGNORE")
            else:
                stmt = stmt.on_conflict_do_nothing()

            await session.execute(stmt)

        await session.commit()
        return result

    except Exception as e:
        await session.rollback()
        result.error_messages.append(f"Unexpected error: {str(e)}")
        result.failed_entries += len(data_list)
        return result


async def update_ingests_from_results(session: AsyncSession, input_data: list[dict]):
    """Update ingest records with latest data from processing results."""
    ingestion_updates = {}

    # Collect all updates based on schema type
    for item in input_data:
        ingestion_id = None
        update_data = {}

        if item.schema__ == "Ingestion":
            ingestion_id = item.ingestion_id
            # Get all fields that exist in both the item and Ingest model
            ingest_fields = [c.key for c in inspect(Ingest).columns if c.key != "id"]
            update_data = {k: getattr(item, k) for k in ingest_fields if hasattr(item, k) and getattr(item, k) is not None}
        elif item.schema__ == "Entry":
            ingestion_id = item.ingestion.ingestion_id
            update_data = {
                k: getattr(item.ingestion, k)
                for k in inspect(Ingest).columns.keys()
                if hasattr(item.ingestion, k) and getattr(item.ingestion, k) is not None
            }
        if ingestion_id and update_data:
            ingestion_updates[ingestion_id] = update_data

    # Perform updates in batch if we have any
    if ingestion_updates:
        stmt = select(Ingest).where(Ingest.id.in_(ingestion_updates.keys()))
        result = await session.execute(stmt)
        ingests = result.scalars().all()

        for ingest in ingests:
            for key, value in ingestion_updates[ingest.id].items():
                setattr(ingest, key, value)

        await session.commit()

    return input_data
