import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.pipeline import PipelineOrchestrator
from src.pipeline.registry import FunctionRegistry
from src.schemas.registry import SchemaRegistry
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
from src.sql_db.database_simple import get_async_session


def parse_args():
    parser = argparse.ArgumentParser(description="Run ETL pipeline")
    parser.add_argument("--config", type=str, default="youtube_labels", help="Configuration name (default: youtube_labels)")
    return parser.parse_args()


args = parse_args()
CONFIG = args.config

# Initialize the orchestrator to register functions
config_path = Path(__file__).resolve().parent.parent / "config" / f"{CONFIG}.yaml"
orchestrator = PipelineOrchestrator(str(config_path))
storage = orchestrator.storage
FunctionRegistry.set_storage_backend(storage)


async def process_batch(session, items, pipeline):
    def create_item_hash(item):
        """Create a hash for an item based on key fields."""
        hash_fields = {k: getattr(item, k) for k in ["public_url", "document_title", "creator_name"]}
        return hashlib.sha256(json.dumps(hash_fields, sort_keys=True).encode()).hexdigest()

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
            item_hash = create_item_hash(item)
            ingest = ingests[item_hash]
            item.ingestion_id = ingest.id
            item.pipeline_id = pipeline.id
            processed_results.append(item)
    return processed_results


async def pipeline_step(session, pipeline, stage: str, order: int, function_name: str, input_data=None, **params):
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
        session.commit()  # Commit step creation separately

        func = FunctionRegistry.get(stage, function_name)
        if order == 0:  # Ingestion step
            results = await func(**params, simple_mode=False)
            print("Function Done Running")
            processed_results = await process_batch(session, results, pipeline)
            results = processed_results
        else:  # Processing step
            results = await func(input_data, **params, simple_mode=False)
            print("Function Done Running")

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


async def get_latest_step_results(session, pipeline_id: int) -> tuple[int, list[dict[str, Any]]]:
    latest_step = await get_latest_processing_step(session, pipeline_id)
    if latest_step:
        results = json.loads(await storage.read(latest_step.output_path))
        results = [SchemaRegistry.get(item["schema__"]).parse_obj(item) for item in results]
        return latest_step.order + 1, results  # start from the next step with the results from the last successful step
    return 0, []


async def get_step_results(session, pipeline_id: int, resume_from_step: int) -> tuple[int, list[dict[str, Any]]]:
    step = await get_specific_processing_step(session, pipeline_id, resume_from_step)
    if step:
        results = json.loads(await storage.read(step.output_path))
        results = [SchemaRegistry.get(item["schema__"]).parse_obj(item) for item in results]
        return resume_from_step, results
    return 0, []


async def update_pipeline_ids(results: list, new_pipeline_id: int) -> list:
    """Update pipeline_id in results based on their schema type."""
    for item in results:
        if hasattr(item, "pipeline_id"):  # Ingestion
            item.pipeline_id = new_pipeline_id
        elif hasattr(item, "ingestion"):  # Entry
            item.ingestion.pipeline_id = new_pipeline_id
        if hasattr(item, "entries"):  # Document
            for entry in item.entries:
                if hasattr(entry, "pipeline_id"):
                    entry.pipeline_id = new_pipeline_id
                if hasattr(entry, "ingestion"):
                    entry.ingestion.pipeline_id = new_pipeline_id
    return results


async def etl_pipeline():
    config = orchestrator.config
    for session in get_async_session():
        for stage_config in config["stages"]:
            if stage_config["name"] == "upsert":
                assert (
                    stage_config["functions"][0]["params"]["collection_name"] == config["pipeline"]["collection_name"]
                ), "Collection name in upsert stage params must match pipeline collection name"
        # Create or get existing processing pipeline
        pipeline = await create_or_get_processing_pipeline(session, config["pipeline"], config["storage"])
        resume_from_step = config["pipeline"].get("resume_from_step", None)
        fork_pipeline = config["pipeline"].get("fork_pipeline", None)

        # Note: The options are to clone a pipeline, overwrite and continue an old pipeline, or start a new one (step 0)
        if resume_from_step:
            # old pipeline to get the intermediate results
            resume_from_step, latest_results = await get_step_results(session, pipeline.id, resume_from_step)
            if fork_pipeline:  # fork the pipeline starting at 'resume_from_step'
                pipeline = await clone_pipeline(
                    session, pipeline, resume_from_step, new_description=config["pipeline"].get("description")
                )  # noqa
                latest_results = await update_pipeline_ids(latest_results, pipeline.id)
        else:  # pick up from last successful step
            latest_step, latest_results = await get_latest_step_results(session, pipeline.id)
            resume_from_step = latest_step

        step_order = resume_from_step
        current_results = latest_results
        print(f"Resuming from step {step_order} with Pipeline ID: {pipeline.id}", flush=True)

        # Process all stages including ingestion
        for stage_config in config["stages"][step_order:]:
            stage = stage_config["name"].split("_")[0] if "_" in stage_config["name"] else stage_config["name"]
            processed_results = []
            for function in stage_config["functions"]:
                result = await pipeline_step(
                    session,
                    pipeline,
                    stage,
                    step_order,
                    function["name"],
                    current_results if step_order > 0 else None,
                    **function.get("params", {}),
                )
                processed_results.extend(result)
            step_order += 1
            current_results = processed_results

        print("Creating Entries. No more processing steps to run", flush=True)
        if current_results[0].schema__ == "Upsert":
            await create_entries(session, current_results, config["pipeline"].get("collection_name"))


if __name__ == "__main__":
    asyncio.run(etl_pipeline())
