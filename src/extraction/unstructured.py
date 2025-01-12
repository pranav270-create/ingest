import unstructured_client
from unstructured_client.models import operations, shared
import json
import asyncio
from pathlib import Path
from typing import List, Dict
import aiofiles
import os


async def process_single_file(
    client: unstructured_client.UnstructuredClient, filepath: str
) -> Dict:
    """Process a single file through the unstructured API"""
    async with aiofiles.open(filepath, "rb") as f:
        data = await f.read()

    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(
                content=data,
                file_name=filepath,
            ),
            strategy=shared.Strategy.HI_RES,
            languages=["eng"],
            split_pdf_page=True,
            split_pdf_allow_failed=True,
            split_pdf_concurrency_level=15,
            coordinates=True,
            # chunking_strategy=shared.ChunkingStrategy.BY_TITLE,
            # max_characters=1024,
        ),
    )

    try:
        res = client.general.partition(request=req)
        return res
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


async def process_folder(folder_path: str, max_concurrent: int = 10) -> List[Dict]:
    """
    Process all files in a folder through the unstructured API with concurrent processing

    Args:
        folder_path: Path to folder containing files to process
        max_concurrent: Maximum number of concurrent API calls

    Returns:
        List of processing results in the same order as input files
    """
    client = unstructured_client.UnstructuredClient(
        api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
        server_url="https://api.unstructuredapp.io",
    )

    # Get all files in the folder
    files = sorted([str(f) for f in Path(folder_path).glob("*") if f.is_file()])

    # Process files in batches
    results = [None] * len(files)
    for i in range(0, len(files), max_concurrent):
        batch = files[i : i + max_concurrent]
        tasks = [process_single_file(client, f) for f in batch]
        batch_results = await asyncio.gather(*tasks)

        # Store results in the same order as input files
        for j, result in enumerate(batch_results):
            results[i + j] = result

    return results


# Example usage:
async def main():
    folder_path = "/Users/pranaviyer/Desktop/AstralisData"
    results = await process_folder(folder_path, max_concurrent=10)

    # Save results to JSON
    with open("output.json", "w") as f:
        for result in results:
            f.write(json.dumps(result, default=str, indent=4))
            f.write("\n")


if __name__ == "__main__":
    asyncio.run(main())
