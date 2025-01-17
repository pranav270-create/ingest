from google_labs_html_chunker.html_chunker import HtmlChunker
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import Entry, Ingestion, ExtractedFeatureType, ExtractionMethod
from src.pipeline.registry.function_registry import FunctionRegistry
from src.utils.datetime_utils import get_current_utc_datetime


@FunctionRegistry.register("extract", "google_labs_html_chunker")
async def parse_html(ingestions: list[Ingestion], max_words_per_aggregate_passage=220, greedily_aggregate_sibling_nodes=True, read=None, write=None, **kwargs) -> list[Entry]:
    all_entries = []
    for ingestion in ingestions:
        if read:
            html = await read(ingestion.file_path)
        else:
            with open(ingestion.file_path) as f:
                html = f.read()
        chunker = HtmlChunker(
            max_words_per_aggregate_passage=max_words_per_aggregate_passage,
            greedily_aggregate_sibling_nodes=greedily_aggregate_sibling_nodes,
        )
        passages = chunker.chunk(html)
        all_text = " ".join(passages)
        parsed_file_path = ingestion.file_path.replace(".html", ".txt")
        if write:
            await write(parsed_file_path, all_text)
        else:
            with open(parsed_file_path, "w") as f:
                f.write(all_text)
        ingestion.extraction_method = ExtractionMethod.GOOGLE_LABS_HTML_CHUNKER
        ingestion.extraction_date = get_current_utc_datetime()
        ingestion.parsed_feature_type = [ExtractedFeatureType.TEXT]
        ingestion.extracted_document_file_path = parsed_file_path
        entry = Entry(ingestion=ingestion, string=all_text, index_numbers=None, citations=None)
        all_entries.append(entry)
    return all_entries


if __name__ == "__main__":
    pass
