import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import Entry, Ingestion, ExtractedFeatureType, ExtractionMethod
from src.pipeline.registry.function_registry import FunctionRegistry


@FunctionRegistry.register("extract", "dummy")
async def parse_dummy(ingestions: list[Ingestion], read=None, write=None, **kwargs) -> list[Entry]:
    all_entries = []
    for ingestion in ingestions:
        if read:
            all_text = await read(ingestion.file_path)
        else:
            with open(ingestion.file_path) as f:
                all_text = f.read()
        entry = Entry(ingestion=ingestion, string=all_text, chunk_locations=None, citations=None)
        all_entries.append(entry)
    return all_entries


if __name__ == "__main__":
    pass
