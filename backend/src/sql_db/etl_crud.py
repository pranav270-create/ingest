from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from src.sql_db.etl_model import Ingest, ProcessingStep, Entry, ProcessingPipeline, ingest_pipeline, StepType
import hashlib
from datetime import datetime
import json
from sqlalchemy import insert, select, inspect, and_
from sqlalchemy.orm import selectinload
from sqlalchemy.dialects.postgresql import insert as pg_insert


async def create_or_update_ingest(session: AsyncSession, data: dict, pipeline=None) -> Ingest:
    # Get all column names of the Ingest model
    ingest_fields = [c.key for c in inspect(Ingest).columns if c.key != 'id']
    # Filter the input data to only include relevant fields
    ingest_data = {k: v for k, v in data.items() if k in ingest_fields}

    # Create a hash of the important fields
    hash_fields = ['public_url', 'document_title', 'creator_name']
    hash_items = {k: v for k, v in ingest_data.items() if k in hash_fields}
    hash_string = json.dumps(hash_items, sort_keys=True)
    hash_value = hashlib.sha256(hash_string.encode()).hexdigest()

    # Check if an Ingest with this hash already exists
    stmt = select(Ingest).where(Ingest.hash == hash_value)
    result = session.execute(stmt)
    existing_ingest = result.scalar_one_or_none()

    if existing_ingest:
        # Update the existing Ingest
        for key, value in ingest_data.items():
            setattr(existing_ingest, key, value)
        if pipeline:
            # Load the processing_pipelines relationship
            session.refresh(existing_ingest, ['processing_pipelines'])
            # Check if the pipeline is already associated
            pipeline_ids = [p.id for p in existing_ingest.processing_pipelines]
            if pipeline.id not in pipeline_ids:
                existing_ingest.processing_pipelines.append(pipeline)
        session.commit()
        return existing_ingest
    else:
        # Create a new Ingest
        new_ingest = Ingest(**ingest_data, hash=hash_value)
        if pipeline:
            new_ingest.processing_pipelines.append(pipeline)
        session.add(new_ingest)
        session.commit()
        return new_ingest

async def create_or_update_ingest_batch(session: AsyncSession, items: list[dict], pipeline=None) -> dict[str, Ingest]:
    # Get all column names of the Ingest model
    ingest_fields = [c.key for c in inspect(Ingest).columns if c.key != 'id']
    hash_fields = ['public_url', 'document_title', 'creator_name']

    # Process all items to create hashes and prepare data
    hash_to_data = {}
    for data in items:
        # Filter and prepare the data
        ingest_data = {k: v for k, v in data.items() if k in ingest_fields}
        
        # Create hash
        hash_items = {k: v for k, v in ingest_data.items() if k in hash_fields}
        hash_string = json.dumps(hash_items, sort_keys=True)
        hash_value = hashlib.sha256(hash_string.encode()).hexdigest()
        
        hash_to_data[hash_value] = ingest_data

    try:
        # Get existing ingests by hash
        existing_hashes = list(hash_to_data.keys())
        stmt = select(Ingest).where(Ingest.hash.in_(existing_hashes))
        result = session.execute(stmt)
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

        session.flush()

        # Handle pipeline relationships if needed
        if pipeline:
            for ingest in ingests.values():
                # Check if relationship exists
                stmt = select(ingest_pipeline).where(
                    and_(
                        ingest_pipeline.c.ingest_id == ingest.id,
                        ingest_pipeline.c.pipeline_id == pipeline.id
                    )
                )
                result = session.execute(stmt)
                if not result.first():
                    # Create new relationship
                    stmt = insert(ingest_pipeline).values({
                        "ingest_id": ingest.id,
                        "pipeline_id": pipeline.id
                    })
                    session.execute(stmt)

        session.commit()
        return ingests

    except Exception as e:
        print(f"Error in create_or_update_ingest_batch: {str(e)}")
        session.rollback()
        raise


async def create_or_get_processing_pipeline(session: AsyncSession, pipeline_config: dict = {}, storage_config: dict = {}) -> ProcessingPipeline:
    pipeline_id = pipeline_config.get('pipeline_id', None)
    version = pipeline_config.get('version', "1.0")
    description = pipeline_config.get('description', "")

    storage_type = storage_config.get('type', "local")
    if storage_type == "s3":
        storage_path = storage_config.get('bucket_name', "")
    else:
        storage_path = storage_config.get('base_path', "")

    # see if exists with that version and description
    if pipeline_id:
        stmt = select(ProcessingPipeline).where(
            (ProcessingPipeline.id == pipeline_id)
        )
        result = session.execute(stmt)
        existing_pipeline = result.scalar_one_or_none()
        if existing_pipeline:
            return existing_pipeline
    else:
        # Create a new pipeline
        new_pipeline = ProcessingPipeline(
            version=version,
            description=description,
            storage_type=storage_type,
            storage_path=storage_path,
            created_at=datetime.now()
        )
        session.add(new_pipeline)
        session.commit()
        return new_pipeline


async def clone_pipeline(session: AsyncSession,
                         original: ProcessingPipeline,
                         clone_till_step: int,
                         new_description: str = None) -> ProcessingPipeline:
    """
    Creates a new pipeline by cloning steps up to clone_till_step from the original pipeline.
    """
    # First get the ingests explicitly through a join query
    stmt = select(Ingest).join(
        ingest_pipeline,
        and_(
            ingest_pipeline.c.pipeline_id == original.id,
            ingest_pipeline.c.ingest_id == Ingest.id
        )
    )
    result = session.execute(stmt)
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
    session.flush()

    # Clone steps up to clone_till_step
    stmt = select(ProcessingStep).where(
        (ProcessingStep.pipeline_id == original.id) &
        (ProcessingStep.order < clone_till_step)
    ).order_by(ProcessingStep.order)

    steps = (session.execute(stmt)).scalars().all()

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
        session.flush()
        step_map[step.id] = new_step.id

        # Update previous_step_id references
        if step.previous_step_id and step.previous_step_id in step_map:
            new_step.previous_step_id = step_map[step.previous_step_id]

    # Clone ingest associations using explicit relationships
    for ingest in ingests:
        # Create the relationship using the association table
        stmt = insert(ingest_pipeline).values({
            "ingest_id": ingest.id,
            "pipeline_id": new_pipeline.id
        }).on_conflict_do_nothing()
        session.execute(stmt)
    
    session.commit()
    return new_pipeline


async def create_processing_step(session: AsyncSession, pipeline_id: int, order: int, step_type: str, function_name: str, status: str, previous_step_id: int = None, output_path: str = None, metadata: dict = None) -> ProcessingStep:
    # see if step exists with that pipeline_id and order
    stmt = select(ProcessingStep).where(
        (ProcessingStep.pipeline_id == pipeline_id) &
        (ProcessingStep.order == order)
    )
    result = session.execute(stmt)
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
        session.commit()
        return existing_step
    else:
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
        session.commit()
        return new_step


async def get_latest_processing_step(session: AsyncSession, pipeline_id: int) -> ProcessingStep:
    stmt = select(ProcessingStep).where(
        (ProcessingStep.pipeline_id == pipeline_id) &
        (ProcessingStep.status == "completed")
    ).order_by(ProcessingStep.order.desc()).limit(1)
    result = session.execute(stmt)
    return result.scalar_one_or_none()


async def get_specific_processing_step(session: AsyncSession, pipeline_id: int, step_order: int) -> ProcessingStep:
    step_order = max(step_order - 1, 0)  # we want the results of the previous step
    stmt = select(ProcessingStep).where(
        (ProcessingStep.pipeline_id == pipeline_id) &
        (ProcessingStep.order == step_order)
    )
    result = session.execute(stmt)
    return result.scalar_one_or_none()


async def get_next_processing_step(session: AsyncSession, pipeline_id: int, current_order: int) -> ProcessingStep:
    stmt = select(ProcessingStep).where(
        (ProcessingStep.pipeline_id == pipeline_id) &
        (ProcessingStep.order > current_order)
    ).order_by(ProcessingStep.order.asc())
    result = session.execute(stmt)
    return result.scalar_one_or_none()


async def create_entries(session: AsyncSession, data_list: list[dict], collection_name: str, batch_size: int = 1000) -> list[Entry]:
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        # Prepare the data for insertion
        values = []
        for data in batch:
            data = data.model_dump()

            # Sanitize string fields
            string_value = data.get("string")
            if string_value:
                string_value = string_value.replace('\x00', '').encode('utf-8', 'ignore').decode('utf-8')

            context_summary = data.get("context_summary_string")
            if context_summary:
                context_summary = context_summary.replace('\x00', '').encode('utf-8', 'ignore').decode('utf-8')

            entry_data = {
                "unique_identifier": data.get("uuid"),
                "collection_name": collection_name,
                "keywords": json.dumps(data.get("keywords", [])),
                "string": string_value,
                "context_summary_string": context_summary,
                "added_featurization": json.dumps(data.get("added_featurization", {})),
                "index_numbers": json.dumps(data.get("index_numbers", [])),
                "pipeline_id": data.get("pipeline_id"),
                "ingestion_id": data.get("ingestion_id")
            }
            values.append(entry_data)

        try:
            # Process each entry - check if exists and update, or create new
            for entry_data in values:
                # Check if entry exists
                stmt = select(Entry).where(Entry.unique_identifier == entry_data["unique_identifier"])
                result = session.execute(stmt)
                existing_entry = result.scalar_one_or_none()

                if existing_entry:
                    # Update existing entry
                    for key, value in entry_data.items():
                        setattr(existing_entry, key, value)
                else:
                    # Create new entry
                    new_entry = Entry(**entry_data)
                    session.add(new_entry)

            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            print(f"Error inserting batch: {e}")


async def update_ingests_from_results(input_data, session):
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
        elif item.schema__ == "Document":
            if item.entries:
                ingestion_id = item.entries[0].ingestion.ingestion_id
                update_data = {
                    k: getattr(item.entries[0].ingestion, k) 
                    for k in inspect(Ingest).columns.keys() 
                    if hasattr(item.entries[0].ingestion, k) and getattr(item.entries[0].ingestion, k) is not None
                }
        
        if ingestion_id and update_data:
            ingestion_updates[ingestion_id] = update_data

    # Perform updates in batch if we have any
    if ingestion_updates:
        stmt = select(Ingest).where(Ingest.id.in_(ingestion_updates.keys()))
        result = session.execute(stmt)
        ingests = result.scalars().all()
        
        for ingest in ingests:
            for key, value in ingestion_updates[ingest.id].items():
                setattr(ingest, key, value)
        
        session.commit()
    
    return input_data
