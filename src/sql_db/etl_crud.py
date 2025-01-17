import hashlib
import json
from datetime import datetime
from typing import Callable, Union
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import and_, inspect, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.schemas import Entry as EntrySchema
from src.sql_db.etl_model import Entry, Ingest, ProcessingPipeline, ProcessingStep, StepType, ingest_pipeline


async def get_insert_stmt(i: Union[str,AsyncSession]) -> Callable:
    if not isinstance(i, str):
        dialect = i.get_bind().dialect.name
    else:
        dialect = i

    if dialect == 'mysql':
        return mysql_insert
    else:
        return pg_insert


async def create_or_update_ingest(session: AsyncSession, data: dict, pipeline=None) -> Ingest:
    # Get all column names of the Ingest model
    ingest_fields = [c.key for c in inspect(Ingest).columns if c.key != 'id']
    # Filter the input data to only include relevant fields
    ingest_data = {k: v for k, v in data.items() if k in ingest_fields}

    # Use the existing document_hash
    hash_value = data['document_hash']

    # Check if an Ingest with this hash already exists
    stmt = select(Ingest).where(Ingest.hash == hash_value)
    result = await session.execute(stmt)
    existing_ingest = result.scalar_one_or_none()

    if existing_ingest:
        # Update the existing Ingest
        for key, value in ingest_data.items():
            setattr(existing_ingest, key, value)
        if pipeline:
            # Load the processing_pipelines relationship
            await session.refresh(existing_ingest, ['processing_pipelines'])
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
    ingest_fields = [c.key for c in inspect(Ingest).columns if c.key != 'id']

    # Process all items to prepare data
    hash_to_data = {}
    for data in items:
        # Filter and prepare the data
        ingest_data = {k: v for k, v in data.items() if k in ingest_fields}
        hash_value = data['document_hash']  # Use the existing hash
        hash_to_data[hash_value] = ingest_data

    try:
        # Get existing ingests by hash
        existing_hashes = list(hash_to_data.keys())
        stmt = select(Ingest).where(Ingest.hash.in_(existing_hashes))
        result = await session.execute(stmt)
        existing_ingests = {i.hash: i for i in result.scalars().all()}

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
                stmt = select(ingest_pipeline).where(
                    and_(
                        ingest_pipeline.c.ingest_id == ingest.id,
                        ingest_pipeline.c.pipeline_id == pipeline.id
                    )
                )
                result = await session.execute(stmt)

                if not result.first():
                    # Create new relationship
                    stmt = insert_stmt(ingest_pipeline).values({
                        "ingest_id": ingest.id,
                        "pipeline_id": pipeline.id
                    })
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
    pipeline_id = pipeline_config.get('pipeline_id', None)
    version = pipeline_config.get('version', "1.0")
    description = pipeline_config.get('description', "")

    storage_type = storage_config.get('type', "local")
    if storage_type == "s3":
        storage_path = storage_config.get('bucket_name', "")
    else:
        storage_path = storage_config.get('base_path', "")

    # Try to fetch existing pipeline
    if pipeline_id:
        stmt = select(ProcessingPipeline).where(ProcessingPipeline.id == pipeline_id)
        result = await session.execute(stmt)
        if pipeline := result.scalar_one_or_none():
            return pipeline

    # Create a new pipeline
    new_pipeline = ProcessingPipeline(
        version=version,
        description=description,
        storage_type=storage_type,
        storage_path=storage_path,
        created_at=datetime.now()
    )
    session.add(new_pipeline)
    await session.commit()
    await session.refresh(new_pipeline)

    return new_pipeline


async def clone_pipeline(
        session: AsyncSession,
        original: ProcessingPipeline,
        clone_till_step: int,
        new_description: str = None
    ) -> ProcessingPipeline:
    """
    Creates a new pipeline by cloning steps up to clone_till_step from the original pipeline.
    """
    # First get the ingests explicitly through a join query
    stmt = select(Ingest).join(ingest_pipeline, and_(ingest_pipeline.c.pipeline_id == original.id,ingest_pipeline.c.ingest_id == Ingest.id))
    result = await session.execute(stmt)
    ingests = result.scalars().all()

    # Create new pipeline with incremented version
    new_pipeline = ProcessingPipeline(
        version=f"{original.version}-branch",
        description=new_description or f"Branched from Pipeline {original.id} at step {clone_till_step}",
        storage_type=original.storage_type,
        storage_path=original.storage_path,
        config=original.config,
        created_at=datetime.now()
    )
    session.add(new_pipeline)
    await session.flush()

    # Clone steps up to clone_till_step
    stmt = select(ProcessingStep).where(
        (ProcessingStep.pipeline_id == original.id) &
        (ProcessingStep.order < clone_till_step)
    ).order_by(ProcessingStep.order)

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
            status=step.status
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
        stmt = insert_stmt(ingest_pipeline).values({
            "ingest_id": ingest.id,
            "pipeline_id": new_pipeline.id
        })

        # handle duplicates for both dialects
        if dialect == 'mysql':
            stmt = stmt.prefix_with('IGNORE')
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
    metadata: dict = None
) -> ProcessingStep:
    # see if step exists with that pipeline_id and order
    stmt = select(ProcessingStep).where((ProcessingStep.pipeline_id == pipeline_id) & (ProcessingStep.order == order))
    result = await session.execute(stmt)
    existing_step = result.scalar_one_or_none()

    if existing_step:
        print(f"WARNING: ProcessingStep already exists with pipeline_id {pipeline_id} and order {order}. This step will be overwritten.")
        existing_step.previous_step_id = previous_step_id
        existing_step.status = status
        existing_step.function_name = function_name
        existing_step.step_type = step_type
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
        metadata_field=metadata
    )
    session.add(new_step)
    await session.commit()
    await session.refresh(new_step)
    return new_step


async def get_latest_processing_step(session: AsyncSession, pipeline_id: int) -> ProcessingStep:
    stmt = select(ProcessingStep).where(
        (ProcessingStep.pipeline_id == pipeline_id) &
        (ProcessingStep.status == "completed")
    ).order_by(ProcessingStep.order.desc()).limit(1)

    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_specific_processing_step(session: AsyncSession, pipeline_id: int, step_order: int) -> ProcessingStep:
    step_order = max(step_order - 1, 0)  # we want the results of the previous step
    stmt = select(ProcessingStep).where(
        (ProcessingStep.pipeline_id == pipeline_id) &
        (ProcessingStep.order == step_order)
    )

    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_next_processing_step(session: AsyncSession, pipeline_id: int, current_order: int) -> ProcessingStep:
    stmt = select(ProcessingStep).where(
        (ProcessingStep.pipeline_id == pipeline_id) &
        (ProcessingStep.order > current_order)
    ).order_by(ProcessingStep.order.asc())

    result = await session.execute(stmt)
    return result.scalar_one_or_none()


@dataclass
class EntryCreationResult:
    total_processed: int
    new_entries: int
    skipped_duplicates: int
    updated_entries: int
    failed_entries: int
    error_messages: List[str]


async def create_entries(
    session: AsyncSession,
    data_list: list[EntrySchema],
    collection_name: str,
    version: str,
    update_on_collision: bool = False,
    batch_size: int = 1000
) -> EntryCreationResult:
    result = EntryCreationResult(
        total_processed=len(data_list),
        new_entries=0,
        skipped_duplicates=0,
        updated_entries=0,
        failed_entries=0,
        error_messages=[]
    )

    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        values = []
        for data in batch:
            data = data.model_dump()

            # Get pipeline_id from ingestion
            pipeline_id = data.get("ingestion", {}).get("pipeline_id")
            if not pipeline_id:
                result.failed_entries += 1
                result.error_messages.append(f"Missing pipeline_id for entry with hash {data.get('content_hash')}")
                continue

            ingestion_id = data.get("ingestion", {}).get("ingestion_id")
            if not ingestion_id:
                result.failed_entries += 1
                result.error_messages.append(f"Missing ingestion_id for entry with hash {data.get('content_hash')}")
                continue

            # Sanitize string fields
            string_value = data.get("string")
            if string_value:
                string_value = string_value.replace('\x00', '').encode('utf-8', 'ignore').decode('utf-8')

            # Create content hash from relevant fields
            content_to_hash = {
                "string": string_value,
                "ingestion_id": ingestion_id,
                "chunk_locations": data.get("chunk_locations", []),
                "entry_title": data.get("entry_title"),
                "keywords": data.get("keywords"),
                "consolidated_feature_type": data.get("consolidated_feature_type"),
                "added_featurization": json.dumps(data.get("added_featurization", {})),
                "citations": data.get("citations", []),
                "version": version
            }
            content_hash = hashlib.sha256(
                json.dumps(content_to_hash, sort_keys=True).encode()
            ).hexdigest()

            # Map all fields from schema to SQL model
            entry_data = {
                # Core identification fields
                "content_hash": content_hash,
                "uuid": data.get("uuid"),
                "collection_name": collection_name,
                "pipeline_id": pipeline_id,
                "ingestion_id": ingestion_id,

                # Core content fields
                "string": string_value,
                "entry_title": data.get("entry_title"),
                "keywords": json.dumps(data.get("keywords", [])),
                "added_featurization": json.dumps(data.get("added_featurization", {})),

                # Chunk location fields
                "consolidated_feature_type": data.get("consolidated_feature_type"),
                "chunk_locations": json.dumps(data.get("chunk_locations", [])),
                "min_primary_index": data.get("min_primary_index"),
                "max_primary_index": data.get("max_primary_index"),
                "chunk_index": data.get("chunk_index"),
                "table_number": data.get("table_number"),
                "figure_number": data.get("figure_number"),

                # Embedding fields
                "embedded_feature_type": data.get("embedded_feature_type"),
                "embedding_date": data.get("embedding_date"),
                "embedding_model": data.get("embedding_model"),
                "embedding_dimensions": data.get("embedding_dimensions"),
            }
            values.append(entry_data)

        try:
            # Process each entry - check for hash collisions within same pipeline
            for entry_data in values:
                try:
                    stmt = select(Entry).where(
                        and_(
                            Entry.content_hash == entry_data["content_hash"],
                            Entry.pipeline_id == entry_data["pipeline_id"]
                        )
                    )
                    result_query = await session.execute(stmt)
                    existing_entry = result_query.scalar_one_or_none()

                    if existing_entry:
                        if update_on_collision:
                            # Update existing entry with new data
                            for key, value in entry_data.items():
                                setattr(existing_entry, key, value)
                            result.updated_entries += 1
                        else:
                            result.skipped_duplicates += 1
                        continue
                    else:
                        new_entry = Entry(**entry_data)
                        session.add(new_entry)
                        result.new_entries += 1

                except Exception as e:
                    result.failed_entries += 1
                    result.error_messages.append(f"Error processing entry: {str(e)}")

            await session.commit()

        except SQLAlchemyError as e:
            await session.rollback()
            result.error_messages.append(f"Error in batch commit: {str(e)}")
            result.failed_entries += len(values)
            print(f"SQLAlchemy error: {str(e)}")

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
            ingest_fields = [c.key for c in inspect(Ingest).columns if c.key != 'id']
            update_data = {
                k: getattr(item, k)
                for k in ingest_fields
                if hasattr(item, k) and getattr(item, k) is not None
            }
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
