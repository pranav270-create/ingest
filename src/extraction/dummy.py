import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import Document, Entry, Ingestion, ParsedFeatureType, ParsingMethod
from src.pipeline.registry import FunctionRegistry


@FunctionRegistry.register("parse", "dummy")
async def parse_dummy(ingestions: list[Ingestion], read=None, write=None, **kwargs):
    documents = []
    for ingestion in ingestions:
        if read:
            all_text = await read(ingestion.file_path)
        else:
            with open(ingestion.file_path) as f:
                all_text = f.read()
        entry = Entry(ingestion=ingestion, string=all_text, index_numbers=None, citations=None)
        document = Document(
            entries=[entry],
        )
        documents.append(document)
    return documents


if __name__ == "__main__":
    pass
