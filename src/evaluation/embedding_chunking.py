import matplotlib.pyplot as plt
import numpy as np
import json
import time
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.sql import func, cast
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.sql_db.database_simple import get_async_session
from src.sql_db.etl_model import ProcessingStep, StepType, Status, ProcessingPipeline
from src.pipeline.storage_backend import StorageFactory


async def get_latest_embedding_chunking_step(session: AsyncSession) -> ProcessingStep:
    stmt = select(ProcessingStep)
    result = session.execute(stmt)    
    all_steps = result.scalars().all()
    chunk_step = [step for step in all_steps if step.step_type == StepType.CHUNK.value and step.status == Status.COMPLETED.value]
    chunk_step = sorted(chunk_step, key=lambda x: x.date, reverse=False) if chunk_step else None
    chunk_step = chunk_step[-1] if chunk_step else None
    # get the ProcessingPipeline that ProcessingStep belongs to
    if chunk_step:
        stmt = select(ProcessingPipeline).where(ProcessingPipeline.id == chunk_step.pipeline_id)
        result = session.execute(stmt)
        pipeline = result.scalar_one_or_none()
    return chunk_step, pipeline


async def get_chunking_step_by_pipeline_id(session: AsyncSession, pipeline_id: int) -> ProcessingStep:
    pipeline_id = int(pipeline_id)
    stmt = select(ProcessingStep).where(
        and_(
            ProcessingStep.pipeline_id == pipeline_id,
            ProcessingStep.step_type == StepType.CHUNK.value,
            ProcessingStep.status == Status.COMPLETED.value,
        )
    )
    result = session.execute(stmt)
    chunk_step = result.scalars().first()
    # get the ProcessingPipeline that ProcessingStep belongs to
    if chunk_step:
        stmt = select(ProcessingPipeline).where(ProcessingPipeline.id == chunk_step.pipeline_id)
        result = session.execute(stmt)
        pipeline = result.scalar_one_or_none()
    return chunk_step, pipeline


def plot_similarity_data(output_data):
    plt.figure(figsize=(15, 8))
    
    # Create a colormap for different documents
    colors = plt.cm.rainbow(np.linspace(0, 1, len(output_data)))
    
    for doc_idx, doc in enumerate(output_data):
        if doc.get('metadata') and 'sentence_indices' in doc['metadata'] and 'similarities' in doc['metadata']:
            plt.plot(
                doc['metadata']['sentence_indices'],
                doc['metadata']['similarities'],
                label=f'Document {doc_idx + 1}',
                color=colors[doc_idx],
                alpha=0.7  # Some transparency to help see overlapping lines
            )
    
    plt.title('Sentence Similarities Across All Documents')
    plt.xlabel('Sentence Index')
    plt.ylabel('Similarity')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside
    plt.tight_layout()  # Adjust layout to prevent legend cutoff
    plt.show()


async def main(pipeline_id=None):
    for session in get_async_session():
        if pipeline_id:
            step, pipeline = await get_chunking_step_by_pipeline_id(session, pipeline_id)
        else:
            step, pipeline = await get_latest_embedding_chunking_step(session)

        if step is None:
            print("No embedding chunking step found.")
            return

        if step.output_path is None:
            print("No output path found for the embedding chunking step.")
            return
        
        # print the pipeline attributes
        for attr, value in pipeline.__dict__.items():
            print(f"{attr}: {value}")

        if hasattr(pipeline, 'storage_path') and pipeline.storage_path:
            if pipeline.storage_type and pipeline.storage_type == 'local':
                storage = StorageFactory.create(pipeline.storage_type, base_path=pipeline.storage_path)
            else:
                storage = StorageFactory.create(pipeline.storage_type, bucket_name=pipeline.storage_path)
        else:
            storage = StorageFactory.create('s3', bucket_name='astralis-data-4170a4f6')

        output_data = json.loads(await storage.read(step.output_path))
        print(f"Loaded {len(output_data)} documents from the output.")

        if output_data and isinstance(output_data, list):
            # Pass the entire output_data to plot all documents
            plot_similarity_data(output_data)
        else:
            print("No valid output data found.")


if __name__ == '__main__':
    import sys
    pipeline_id = int(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(main(pipeline_id))
