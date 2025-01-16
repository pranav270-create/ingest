import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.pipeline import PipelineOrchestrator
from src.pipeline.registry.function_registry import FunctionRegistry
from src.pipeline.registry.schema_registry import SchemaRegistry
from src.pipeline.storage_backend import StorageBackend
from src.schemas.schemas import BaseModelListType, RegisteredSchemaListType
from src.sql_db.database_simple import get_session
from src.sql_db.etl_crud import (
    clone_pipeline,
    create_entries,
    create_or_get_processing_pipeline,
    create_or_update_ingest_batch,
    create_processing_step,
    get_latest_processing_step,
    get_specific_processing_step,
    update_ingests_from_results,
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run ETL pipeline")
    parser.add_argument("--config", type=str, default="youtube_labels", help="config file name (default: youtube_labels)")
    return parser.parse_args()


async def validate_collection_name(stages: list[dict], collection_name: str):
    """
    Ensure Qdrant collection name in Upsert stage is same as in Pipeline Config
    """
    for stage_config in stages:
        if stage_config["name"] == "upsert":
            if stage_config["functions"][0]["params"]["collection_name"] != collection_name:
                raise ValueError("Collection name in upsert stage params must match pipeline collection name")


async def get_step_results(session, pipeline_id: int, resume_from_step: int, storage: StorageBackend) -> list[RegisteredSchemaListType]:
    """
    Get results from a specific step
    """
    step = await get_specific_processing_step(session, pipeline_id, resume_from_step)
    if not step:
        return []

    results = json.loads(await storage.read(step.output_path))
    return [SchemaRegistry.get(item["schema__"]).model_validate(item) for item in results]


async def get_last_step_results(session, pipeline_id: int, storage: StorageBackend) -> tuple[int, list[RegisteredSchemaListType]]:
    """
    Get results from the last successful step and begin the next step
    """
    latest_step = await get_latest_processing_step(session, pipeline_id)
    if not latest_step:
        return 0, []
    results = json.loads(await storage.read(latest_step.output_path))
    results = [SchemaRegistry.get(item["schema__"]).model_validate(item) for item in results]
    return latest_step.order + 1, results


async def update_pipeline_ids(results: list[BaseModelListType], new_pipeline_id: int) -> list[BaseModelListType]:
    """
    Update pipeline_id in Ingestions and Entries.
    """
    for item in results:
        if hasattr(item, "pipeline_id"):  # Ingestion
            item.pipeline_id = new_pipeline_id
        elif hasattr(item, "ingestion"):  # Entry
            item.ingestion.pipeline_id = new_pipeline_id
    return results


async def process_batch(session, items, pipeline):

    # Process items in batches of 100
    batch_size = 100
    processed_results = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_data = [item.model_dump() for item in batch]
        # Process batch
        ingests = await create_or_update_ingest_batch(session, batch_data, pipeline)
        # Update items with their IDs
        for item in batch:
            item_hash = 0  # TODO: Change this to use the hash from Ingest that is a part of it
            ingest = ingests[item_hash]
            item.ingestion_id = ingest.id
            item.pipeline_id = pipeline.id
            processed_results.append(item)
    return processed_results


async def pipeline_step(session, pipeline, storage: StorageBackend, stage: str, order: int, function_name: str, **params):
    try:
        output_path = f"{pipeline.id}_{stage}_{function_name}_{order}.json"
        step = await create_processing_step(
            session,
            pipeline_id=pipeline.id,
            order=order,
            step_type=stage,
            function_name=function_name,
            status="running",
            output_path=output_path,
            metadata={"params": params},
        )
        session.commit()

        func = FunctionRegistry.get(stage, function_name)

        results = await func(**params, simple_mode=False)
        logger.info(f"Function {function_name} Done Running")

        is_ingestion_step = order == 0
        if is_ingestion_step:
            results = await process_batch(session, results, pipeline)

        # Save results and handle special cases
        result_dicts = [item.model_dump() for item in results]
        await storage.write(output_path, json.dumps(result_dicts))

        # Update SQL database with latest information
        await update_ingests_from_results(results, session)

        step.status = "completed"
        session.commit()
        return results

    except Exception as e:
        session.rollback()
        step.status = "failed"
        session.commit()  # Commit the failure status
        raise e


async def etl_pipeline(orchestrator: PipelineOrchestrator):
    config = orchestrator.config
    pipeline_config = config["pipeline"]

    for session in get_session():
        await validate_collection_name(config["stages"], pipeline_config["collection_name"])

        # Create or load existing processing pipeline
        pipeline = await create_or_get_processing_pipeline(session, pipeline_config, config["storage"])

        resume_from_step = pipeline_config.get("resume_from_step", None)
        fork_pipeline = pipeline_config.get("fork_pipeline", None)

        if not resume_from_step:
            # start from last successful step
            step_order, current_results = await get_last_step_results(session, pipeline.id, orchestrator.storage)

        # resume from specific step or start from the first step
        current_results = await get_step_results(session, pipeline.id, resume_from_step, orchestrator.storage)
        step_order = resume_from_step if current_results else 0

        if fork_pipeline:
            # Create new pipeline branch
            pipeline = await clone_pipeline(session, pipeline, resume_from_step, new_description=pipeline_config.get("description"))  # noqa
            current_results = await update_pipeline_ids(current_results, pipeline.id)

        logger.info(f"Resuming from step {step_order} with Pipeline ID: {pipeline.id}")

        # process all stages
        for stage_config in config["stages"][step_order:]:
            stage = stage_config["name"].split("_")[0] if "_" in stage_config["name"] else stage_config["name"]

            # process all functions in the stage
            stage_results = []
            for function in stage_config["functions"]:

                step_results = await pipeline_step(
                    session,
                    pipeline,
                    orchestrator.storage,
                    stage,
                    step_order,
                    function["name"],
                    current_results if step_order > 0 else None,
                    **function.get("params", {}),
                )
                stage_results.extend(step_results)

            # move to the next step and update current results
            step_order += 1
            current_results = stage_results

        logger.info("Creating Entries. No more processing steps to run")
        if current_results and current_results[0].schema__ == "Entry":
            await create_entries(session, current_results, pipeline_config.get("collection_name"))


if __name__ == "__main__":
    args = parse_args()
    config = args.config

    if not config.endswith(".yaml"):
        config = f"{config}.yaml"

    # Initialize the orchestrator to register functions
    config_path = Path(__file__).resolve().parent.parent / "config" / config
    orchestrator = PipelineOrchestrator(str(config_path))
    storage = orchestrator.storage
    FunctionRegistry.set_storage_backend(storage)

    asyncio.run(etl_pipeline())
