import os 
import sys
from pathlib import Path 
import asyncio 
from crawl4ai import AsyncWebCrawler 
from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonCssExtractionStrategy, CosineStrategy
from crawl4ai.chunking_strategy import RegexChunking
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from urllib.parse import urlparse
import random

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.schemas.schemas import ContentType, Entry, FileType, Ingestion, IngestionMethod, ExtractedFeatureType, ExtractionMethod, Scope
from src.pipeline.registry import FunctionRegistry
from src.utils.datetime_utils import get_current_utc_datetime, parse_datetime
from src.utils.ingestion_utils import update_ingestion_with_metadata

class PageSummary(BaseModel): 
    title: str = Field(..., description="Title of the page.") 
    summary: str = Field(..., description="Summary of the page.") 
    keywords: list = Field(..., description="Keywords assigned to the page.")

class CrawlResult(BaseModel):
    url: str
    html: str
    success: bool
    cleaned_html: Optional[str] = None
    media: dict[str, list[dict]] = {}
    links: dict[str, list[dict]] = {}
    screenshot: Optional[str] = None
    markdown: Optional[str] = None
    extracted_content: Optional[str] = None
    metadata: Optional[dict] = None
    error_message: Optional[str] = None
    session_id: Optional[str] = None
    responser_headers: Optional[dict] = None
    status_code: Optional[int] = None

def create_llm_extraction_strategy(api_token):
    return LLMExtractionStrategy(
        provider="openai/gpt-4o-mini",
        api_token=api_token,
        schema=PageSummary.model_json_schema(),
        extraction_type="schema",
        apply_chunking=False,
        instruction=(
            "From the crawled content, extract the following details: "
            "1. Title of the page "
            "2. Summary of the page, which is a detailed summary "
            "3. Keywords assigned to the page, which is a list of keywords. "
            'The extracted JSON format should look like this: '
            '{ '
            '  "title": "Page Title", '
            '  "summary": "Detailed summary of the page.", '
            '  "keywords": ["keyword1", "keyword2", "keyword3"] '
            '}'
        )
    )

def create_css_selector_extraction_strategy():
    # Define the extraction schema
    schema = {
        "name": "Title",
        "baseSelector": ".cds-tableRow-t45thuk",
        "fields": [
            {
                "name": "crypto",
                "selector": "td:nth-child(1) h2",
                "type": "text",
            },
            {
                "name": "symbol",
                "selector": "td:nth-child(1) p",
                "type": "text",
            },
            {
                "name": "price",
                "selector": "td:nth-child(2)",
                "type": "text",
            }
        ],
    }
    return JsonCssExtractionStrategy(schema, verbose=True)

def create_cosine_extraction_strategy():
    strategy = CosineStrategy(
            semantic_filter="finance economy stock market",
            word_count_threshold=10,
            max_dist=0.2,
            linkage_method='ward',
            top_k=3,
            model_name='BAAI/bge-small-en-v1.5'
        )
    return strategy

async def crawl_and_summarize(url, extraction_strategy):
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url=url,
            word_count_threshold=10,  # Minimum number of words required for a page to be processed
            extraction_strategy=extraction_strategy,
            chunking_strategy=RegexChunking(),
            bypass_cache=True,  # Bypass the cache to get the latest content
            js_code=None,  # JavaScript code to execute on the page
            css_selector=None,  # CSS selector to extract data from the page
            wait_for=None  # Wait for a specific element to be present on the page
        )
    return result

def create_ingestion(url, result, creator_name):
    current_time = get_current_utc_datetime()
    html_path = f"/tmp/{urlparse(url).netloc}-{random.randint(1000, 9999)}.html"
    
    with open(html_path, "w") as f:
        f.write(result.html)

    return Ingestion(
        document_title=result.extracted_content.get('title', ''),
        scope=Scope.EXTERNAL,
        content_type=ContentType.TEXT,
        file_type=FileType.HTML,
        file_path=html_path,
        total_length=len(result.html),
        public_url=url,
        creator_name=creator_name,
        metadata={
            'extracted_data': result.extracted_content,
        },
        creation_date=current_time,
        ingestion_date=current_time,
        ingestion_method=IngestionMethod.URL_SCRAPE,
    )

async def process_url(url, extraction_strategy, creator_name):
    try:
        result = await crawl_and_summarize(url, extraction_strategy)
        if result.success:
            ingestion = create_ingestion(url, result, creator_name)
            return ingestion
        else:
            return None
    except Exception as e:
        return None

@FunctionRegistry.register("ingest", "webcrawl_ai")
async def ingest(url_configs: List[Dict], added_metadata: Dict = {}) -> List[Ingestion]:
    api_token = os.getenv('OPENAI_API_KEY')
    extraction_strategy = create_llm_extraction_strategy(api_token)
    
    async with AsyncWebCrawler(verbose=True) as crawler:
        async def process_url_with_crawler(url, creator_name):
            try:
                result = await crawler.arun(
                    url=url,
                    word_count_threshold=10,
                    extraction_strategy=extraction_strategy,
                    chunking_strategy=RegexChunking(),
                    bypass_cache=True,
                )
                return result
                # if result.success:
                #     return create_ingestion(url, result, creator_name)
                # else:
                #     return None
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                return None

        tasks = [
            process_url_with_crawler(config['url'], config.get('creator_name', 'AI Web Crawler'))
            for config in url_configs
        ]

        results = await asyncio.gather(*tasks)
        ingestions = [result for result in results if result is not None]
        # ingestions = [update_ingestion_with_metadata(ingestion, added_metadata) for ingestion in ingestions]
        return ingestions


if __name__ == "__main__":
    url_configs = [
        {
            "url": "https://ancestors.fandom.com/wiki/Ancestors:_The_Humankind_Odyssey_Wiki",
            "creator_name": "AI Web Crawler",
        },
        {
            "url": "https://docs.rapids.ai/api/cuspatial/stable/api_docs/spatial/#measurement-functions",
            "creator_name": "AI Web Crawler",
        },
    ]
    import json
    ingestions = asyncio.run(ingest(url_configs))
    for i, ingestion in enumerate(ingestions):
        # dump each to a json file
        with open("/tmp/ingestion_{}.json".format(i), "w") as f:
            json.dump(ingestion.dict(), f, indent=2)