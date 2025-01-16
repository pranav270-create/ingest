import hashlib
import json

from sqlalchemy import inspect, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.sql_db.etl_model import Entry, Ingest


async def create_or_update_ingest(session: AsyncSession, data: dict, pipeline=None) -> Ingest:
    # Get all column names of the Ingest model
    ingest_fields = [c.key for c in inspect(Ingest).columns if c.key != "id"]
    # Filter the input data to only include relevant fields
    ingest_data = {k: v for k, v in data.items() if k in ingest_fields}

    # Create a hash of the important fields
    hash_fields = ["public_url", "document_title", "creator_name"]
    hash_items = {k: v for k, v in ingest_data.items() if k in hash_fields}
    hash_string = json.dumps(hash_items, sort_keys=True)
    hash_value = hashlib.sha256(hash_string.encode()).hexdigest()

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
    hash_fields = ["public_url", "document_title", "creator_name"]

    # Process all items to create hashes and prepare data
    ingest_records = []
    hash_to_data = {}
    for data in items:
        # Filter and prepare the data
        ingest_data = {k: v for k, v in data.items() if k in ingest_fields}

        # Create hash
        hash_items = {k: v for k, v in ingest_data.items() if k in hash_fields}
        hash_string = json.dumps(hash_items, sort_keys=True)
        hash_value = hashlib.sha256(hash_string.encode()).hexdigest()

        ingest_data["hash"] = hash_value
        ingest_records.append(ingest_data)
        hash_to_data[hash_value] = ingest_data

    try:
        # Use MySQL's INSERT ... ON DUPLICATE KEY UPDATE
        for ingest_data in ingest_records:
            insert_stmt = mysql_insert(Ingest).values(ingest_data)

            # Specify which columns to update on duplicate
            update_dict = {
                "document_title": ingest_data["document_title"],
                "public_url": ingest_data["public_url"],
                "creator_name": ingest_data["creator_name"],
                "file_path": ingest_data.get("file_path"),
                "total_length": ingest_data.get("total_length"),
                "creation_date": ingest_data.get("creation_date"),
                "ingestion_method": ingest_data.get("ingestion_method"),
                "ingestion_date": ingest_data.get("ingestion_date"),
                "scope": ingest_data.get("scope"),
                "content_type": ingest_data.get("content_type"),
                "file_type": ingest_data.get("file_type"),
                "summary": ingest_data.get("summary"),
                "keywords": ingest_data.get("keywords"),
                "metadata_field": ingest_data.get("metadata_field"),
                "processed_file_path": ingest_data.get("processed_file_path"),
                "unprocessed_citations": ingest_data.get("unprocessed_citations"),
            }

            # Remove None values from update dict
            update_dict = {k: v for k, v in update_dict.items() if v is not None}

            upsert_stmt = insert_stmt.on_duplicate_key_update(**update_dict)
            await session.execute(upsert_stmt)

        await session.flush()

        # Get all inserted/updated records
        hashes = list(hash_to_data.keys())
        stmt = select(Ingest).where(Ingest.hash.in_(hashes))
        result = await session.execute(stmt)
        ingests = {i.hash: i for i in result.scalars().all()}

        # Handle pipeline relationships if needed
        if pipeline:
            from src.sql_db.etl_model import ingest_pipeline

            for ingest in ingests.values():
                rel_stmt = mysql_insert(ingest_pipeline).values({"ingest_id": ingest.id, "pipeline_id": pipeline.id})
                rel_stmt = rel_stmt.on_duplicate_key_update(ingest_id=rel_stmt.inserted.ingest_id, pipeline_id=rel_stmt.inserted.pipeline_id)
                await session.execute(rel_stmt)

            await session.flush()

        await session.commit()
        return ingests

    except Exception as e:
        print(f"Error in create_or_update_ingest_batch: {str(e)}")
        await session.rollback()
        raise




async def create_entries(session: AsyncSession, data_list: list[dict], collection_name: str, batch_size: int = 1000) -> list[Entry]:
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i : i + batch_size]
        # Prepare the data for insertion
        values = []
        for data in batch:
            data = data.model_dump()

            # Sanitize string fields
            string_value = data.get("string")
            if string_value:
                string_value = string_value.replace("\x00", "").encode("utf-8", "ignore").decode("utf-8")

            context_summary = data.get("context_summary_string")
            if context_summary:
                context_summary = context_summary.replace("\x00", "").encode("utf-8", "ignore").decode("utf-8")

            entry_data = {
                "unique_identifier": data.get("uuid"),
                "collection_name": collection_name,
                "keywords": json.dumps(data.get("keywords", [])),
                "string": string_value,
                "context_summary_string": context_summary,
                "added_featurization": json.dumps(data.get("added_featurization", {})),
                "index_numbers": json.dumps(data.get("index_numbers", [])),
                "pipeline_id": data.get("pipeline_id"),
                "ingestion_id": data.get("ingestion_id"),
            }
            values.append(entry_data)

        try:
            # Use MySQL's INSERT ... ON DUPLICATE KEY UPDATE
            for entry_data in values:
                stmt = mysql_insert(Entry).values(entry_data)  # Changed this line
                stmt = stmt.on_duplicate_key_update(
                    keywords=entry_data["keywords"],
                    string=entry_data["string"],
                    context_summary_string=entry_data["context_summary_string"],
                    added_featurization=entry_data["added_featurization"],
                    index_numbers=entry_data["index_numbers"],
                    pipeline_id=entry_data["pipeline_id"],
                    ingestion_id=entry_data["ingestion_id"],
                )
                await session.execute(stmt)
            await session.commit()
        except SQLAlchemyError as e:
            await session.rollback()
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
            ingest_fields = [c.key for c in inspect(Ingest).columns if c.key != "id"]
            update_data = {k: getattr(item, k) for k in ingest_fields if hasattr(item, k) and getattr(item, k) is not None}
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
        result = await session.execute(stmt)
        ingests = result.scalars().all()

        for ingest in ingests:
            for key, value in ingestion_updates[ingest.id].items():
                setattr(ingest, key, value)

        await session.commit()

    return input_data
