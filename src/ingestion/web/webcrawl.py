"""
This script is used to crawl a website and save it to a file.
Produces a HTML Ingestion ready for parsing and an Image Entry ready for featurization.
"""

import asyncio
import io
import json
import logging
import random
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import aiofiles
import fitz
import requests
from crawlee.playwright_crawler import PlaywrightCrawler, PlaywrightCrawlingContext
from crawlee.proxy_configuration import ProxyConfiguration

# from crawlee.storages import KeyValueStore
# from crawlee.storages import Dataset
from fake_useragent import UserAgent

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import ContentType, Entry, ExtractedFeatureType, ExtractionMethod, FileType, Ingestion, IngestionMethod, Scope
from src.utils.datetime_utils import get_current_utc_datetime, parse_datetime
from src.utils.ingestion_utils import update_ingestion_with_metadata

headless = True

# Configure logging
logging.basicConfig(
    filename="scraper.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Change to DEBUG for detailed logs
)

# ANSI color codes
COLORS = {
    "RESET": "\033[0m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "PURPLE": "\033[95m",
    "CYAN": "\033[96m",
}

def log_event(event_type, message, **kwargs):
    """
    Logs events in JSON format with color codes for better structure and analysis.

    Args:
        event_type (str): Type of the event (e.g., INFO, ERROR).
        message (str): The log message.
        **kwargs: Additional contextual data.
    """
    color = COLORS.get(event_type, COLORS["RESET"])
    reset = COLORS["RESET"]

    log_entry = {"event_type": f"{color}{event_type}{reset}", "message": message, "data": kwargs}
    logging.info(f"{color}{json.dumps(log_entry)}{reset}")

async def handle_captcha(context: PlaywrightCrawlingContext):
    """
    Handles CAPTCHA detection. Can be extended to integrate CAPTCHA-solving services.

    Args:
        context (PlaywrightCrawlingContext): The crawling context.
    """
    log_event("YELLOW", f"Captcha detected at {context.request.url}", url=context.request.url)
    # Implement CAPTCHA solving logic here if needed
    # For now, we'll skip processing this page
    await context.skip()

async def handle_pdf(url: str, write=None, content_type=ContentType.OTHER, creator_name="Unknown"):
    # wget the url
    pdf_content = requests.get(url).content
    pdf_file = io.BytesIO(pdf_content)
    pdf_reader = fitz.open(stream=pdf_file, filetype="pdf")
    metadata = pdf_reader.metadata
    pdf_path = f"{urlparse(url).netloc}-{random.randint(1000, 9999)}.pdf"
    if write:
        await write(pdf_path, pdf_content, mode="wb")
    else:
        async with aiofiles.open(pdf_path, "wb") as f:
            await f.write(pdf_content)
    pdf_ingestion = Ingestion(
        document_title=metadata.get('/Title', 'Untitled PDF'),
        scope=Scope.EXTERNAL,
        content_type=content_type,
        file_type=FileType.PDF,
        file_path=pdf_path,
        public_url=url,
        creator_name=creator_name,
        metadata={
            'pdf_metadata': dict(metadata),
            'author': metadata.get('/Author', ''),
        },
        creation_date=parse_datetime(metadata.get('/CreationDate', '')),
        ingestion_date=get_current_utc_datetime(),
        ingestion_method=IngestionMethod.URL_SCRAPE,
    )
    return pdf_ingestion

# Initialize the lock at the module level
visited_urls_lock = asyncio.Lock()

async def run_crawler(config, write=None, visited_urls=None, lock=None):
    """
    Runs the crawler based on the provided configuration and returns the results.

    Args:
        config (dict): Configuration for the crawler.
        write (callable, optional): Optional write function.
        visited_urls (set, optional): Shared set of visited URLs.
        lock (asyncio.Lock, optional): Lock for synchronizing access to visited_urls.

    Returns:
        list[Ingestion]: A list of Ingestion objects generated by the crawler.
    """
    urls = config['url']
    recursive = config.get('recursive', False)
    screenshot = config.get('screenshot', False)
    extract_params = config.get('extract_params', {})
    select_params = config.get('select_params', ['a[href]'])  # New parameter
    timeout = config.get('timeout', 30)
    proxy_urls = config.get('proxy_configuration', [])
    content_type = config.get('content_type', ContentType.OTHER)
    creator_name = config.get('creator_name', "Unknown")

    log_event("GREEN", f"Starting crawler for {urls} with recursive={recursive}, screenshot={screenshot}, extract_params={extract_params}, timeout={timeout}, proxy_urls={proxy_urls}", url=urls)

    # Configure proxy rotation if proxies are provided
    if proxy_urls:
        proxy_config = ProxyConfiguration(
            tiered_proxy_urls=[
                proxy_urls[: len(proxy_urls) // 2],  # Lower tier proxies
                proxy_urls[len(proxy_urls) // 2 :],  # Higher tier proxies
            ]
        )
    else:
        proxy_config = None
    proxy_config = None  # TMP: no proxy

    crawler_kwargs = {
        "max_requests_per_crawl": config.get("max_requests_per_crawl", 100),
        "headless": headless,
    }

    if proxy_config:
        crawler_kwargs["proxy_configuration"] = proxy_config

    crawler = PlaywrightCrawler(**crawler_kwargs)
    ua = UserAgent()

    collected_ingestions = []  # list to collect Ingestion objects

    def normalize_url(url):
        parsed = urlparse(url)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

    @crawler.router.default_handler
    async def request_handler(context: PlaywrightCrawlingContext) -> None:
        """
        Handles each request based on its label and configuration.

        Args:
            context (PlaywrightCrawlingContext): The crawling context.
        """
        url = context.request.url
        normalized = normalize_url(url)

        async with lock:
            if normalized in visited_urls:
                return
            visited_urls.add(normalized)

        # Check if the URL ends with .pdf
        if "/pdf" in url.lower() or url.lower().endswith(".pdf"):
            pdf_ingestion = await handle_pdf(url, write, content_type, creator_name)
            collected_ingestions.append(pdf_ingestion)
            return
        if config.get('wait_selector'):
            await context.page.wait_for_selector(config.get('wait_selector'), timeout=10000)  # 10 seconds timeout
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Sleep between 0.1 to 0.5 seconds
        if config.get('scroll', False):
            await context.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)  # Wait for any lazy-loaded content
        # Set a random User-Agent header
        await context.page.set_extra_http_headers({"User-Agent": ua.random})
        # Random delay to mimic human behavior
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Sleep between 0.1 to 0.5 seconds
        log_event("BLUE", f"Processing URL: {context.request.url}")

        # Handle CAPTCHA if detected
        if await context.page.locator("captcha").count() > 0:
            await handle_captcha(context)

        data = {
            "url": context.request.url,
            "title": await context.page.title(),
            "content": await context.page.content(),
        }
        current_time = get_current_utc_datetime()

        async def extract_data(element, config):
            result = {}
            for attr in config.get('attributes', []):
                result[attr] = await element.get_attribute(attr)

            if 'extract' in config:
                if config['extract'] == 'text':
                    result['text'] = await element.inner_text()
                elif config['extract'] == 'html':
                    result['html'] = await element.inner_html()

            for key, nested_config in config.get('nested', {}).items():
                nested_elements = await element.query_selector_all(nested_config['selector'])
                if nested_config.get('multiple', False):
                    result[key] = [await extract_data(ne, nested_config) for ne in nested_elements]
                elif nested_elements:
                    result[key] = await extract_data(nested_elements[0], nested_config)

            return result

        for param_name, param_config in extract_params.items():
            elements = await context.page.query_selector_all(param_config['selector'])
            if elements:
                data[param_name] = [await extract_data(element, param_config) for element in elements]
            else:
                data[param_name] = None

        # Save the html to the file path
        html_path = f"{urlparse(context.request.url).netloc}-{random.randint(1000, 9999)}.html"
        if write:
            await write(html_path, data["content"])
        else:
            async with aiofiles.open(html_path, "w") as f:
                await f.write(data["content"])

        # Create Ingestion object for the webpage
        webpage_ingestion = Ingestion(
            document_title=data["title"],
            scope=Scope.EXTERNAL,
            content_type=content_type,
            file_type=FileType.HTML,
            file_path=html_path,
            public_url=context.request.url,
            creator_name=creator_name,
            metadata={
                'extracted_data': {},
            },
            creation_date=current_time,
            ingestion_date=current_time,
            ingestion_method=IngestionMethod.URL_SCRAPE,
        )

        # Add all extracted data to Ingestion.metadata
        for key, value in data.items():
            if key not in ['url', 'title', 'content']:
                webpage_ingestion.document_metadata["extracted_data"][key] = value

        # Add summary information if available
        for key, value in webpage_ingestion.document_metadata["extracted_data"].items():
            if isinstance(value, list):
                webpage_ingestion.document_metadata[f"{key}_count"] = len(value)
                if value and isinstance(value[0], dict):
                    for sub_key in value[0].keys():
                        # Ignore entries with empty or None values
                        webpage_ingestion.document_metadata[f"{key}_{sub_key}_list"] = [
                            item.get(sub_key) for item in value if sub_key in item and item.get(sub_key)
                        ]

        # Collect the Ingestion object
        collected_ingestions.append(webpage_ingestion)

        # If screenshot was taken, create Ingestion and Entry for it
        if screenshot:
            screenshot_path = f"{urlparse(context.request.url).netloc}-{random.randint(1000, 9999)}.png"
            screenshot_content = await context.page.screenshot(full_page=True)
            if write:
                await write(screenshot_path, screenshot_content, mode="wb")
            else:
                async with aiofiles.open(screenshot_path, "wb") as f:
                    await f.write(screenshot_content)

            screenshot_ingestion = Ingestion(
                document_title=f"Screenshot of {data['title']}",
                scope=Scope.EXTERNAL,
                content_type=ContentType.OTHER,
                file_type=FileType.PNG,
                file_path=screenshot_path,
                public_url=context.request.url,
                creator_name=creator_name,
                creation_date=current_time,
                ingestion_date=current_time,
                ingestion_method=IngestionMethod.URL_SCRAPE,
                parsing_method=ExtractionMethod.NONE,
                parsing_date=current_time,
                parsed_feature_type=[ExtractedFeatureType.IMAGE],
                unprocessed_citations=None,
                embedded_feature_type=None,
            )

            screenshot_entry = Entry(
                ingestion=screenshot_ingestion,
                string=None,
                index_numbers=None,
                citations=None,
            )

            # Collect the Screenshot Entry
            collected_ingestions.append(screenshot_entry)

        # If recursive, enqueue links based on select_params
        if recursive:
            # TODO: not fully proper yet
            for selector in select_params:
                next_links = await context.page.query_selector_all(selector)
                if next_links:
                    await context.enqueue_links(
                        include=[next_links],
                        label='RECURSIVE',
                    )
                    # async with lock:
                        # enqueued = {normalize_url(u) for u in next_links}
                        # visited_urls.update(enqueued)

    try:
        urls = [urls] if isinstance(urls, str) else urls
        pdf_urls = [url for url in urls if "/pdf" in url.lower() or url.lower().endswith(".pdf")]
        normal_urls = [url for url in urls if url not in pdf_urls]
        for url in pdf_urls:
            pdf_ingestion = await handle_pdf(url, write, content_type, creator_name)
            collected_ingestions.append(pdf_ingestion)
        if normal_urls:
            await asyncio.wait_for(crawler.run(normal_urls), timeout=timeout)
    except asyncio.TimeoutError:
        log_event("YELLOW", f"Crawler run for {urls} timed out after {timeout} seconds.", url=urls)
    except Exception as e:
        log_event("RED", f"Unexpected error for {urls}: {e}", url=urls)
    return collected_ingestions

async def run_all_crawlers(url_configs, write=None):
    """
    Runs all crawlers based on the provided URL configurations.

    Args:
        url_configs (list[dict]): A list of configurations for each crawler.
        write (callable, optional): Optional write function.

    Returns:
        list[Ingestion]: A combined list of Ingestion objects from all crawlers.
    """
    all_ingestions = []
    tasks = []

    # Shared visited_urls set and lock
    visited_urls = set()
    lock = asyncio.Lock()

    for config in url_configs:
        tasks.append(run_crawler(config, write, visited_urls, lock))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            # Handle exceptions as needed
            continue
        all_ingestions.extend(result)
        log_event("CYAN", f"Crawler has finished with {len(result)} ingestions.")

    log_event("GREEN", "All crawlers have finished and results have been aggregated.")
    return all_ingestions

@FunctionRegistry.register("ingest", "webcrawl_ingestion")
async def scrape_ingestion(ingestions: list[Ingestion], added_metadata: dict={}, write=None, **kwargs) -> list[Ingestion]:
    """
    Asynchronous main function to run all crawlers.

    Args:
        ingestions (list[Ingestion]): List of initial ingestions.
        added_metadata (dict, optional): Additional metadata to add.
        write (callable, optional): Optional write function.
        **kwargs: Additional keyword arguments.

    Returns:
        list[Ingestion]: A combined list of Ingestion objects from all crawlers.
    """
    url_configs = {
            "url": [ingestion.public_url for ingestion in ingestions],
            "recursive": kwargs.get("recursive", False),
            "screenshot": kwargs.get("screenshot", False),
            "extract_params": {},
            "select_params": [],
            "timeout": kwargs.get("timeout", 30),
            "max_requests_per_crawl": kwargs.get("max_requests_per_crawl", 100),
        }

    visited_urls = set()
    lock = asyncio.Lock()
    new_ingestions = await run_crawler(url_configs, write, visited_urls, lock)
    new_entries = [ing for ing in new_ingestions if isinstance(ing, Entry)]
    # get unique entries by public_url
    url_to_entry = {ing.ingestion.public_url: ing for ing in new_entries}
    new_entries = list(url_to_entry.values())
    # Create a mapping of public_url to new file_path
    url_to_file_path = {ing.public_url: ing for ing in new_ingestions if isinstance(ing, Ingestion)}
    # Update original ingestions with new file paths
    updated_ingestions = []
    for ingestion in ingestions:
        if ingestion.public_url in url_to_file_path:
            ingestion.file_path = url_to_file_path[ingestion.public_url].file_path
            ingestion.file_type = url_to_file_path[ingestion.public_url].file_type
            ingestion.document_metadata['extracted_data'] = url_to_file_path[ingestion.public_url].metadata
            ingestion = update_ingestion_with_metadata(ingestion, added_metadata)
            updated_ingestions.append(ingestion)
        else:
            log_event("YELLOW", f"No new file path found for URL: {ingestion.public_url}")
            updated_ingestions.append(ingestion)
    # Add new ingestions that weren't in the original list
    new_ingestions_to_add = [ing for ing in new_ingestions if isinstance(ing, Ingestion) and ing.public_url not in [i.public_url for i in ingestions]]
    updated_ingestions.extend(new_ingestions_to_add)
    return updated_ingestions + new_entries

@FunctionRegistry.register("ingest", "webcrawl_url")
async def scrape_urls(url_configs: list[dict], added_metadata: dict={}, write=None) -> list[Ingestion]:
    """
    Asynchronous main function to run all crawlers.

    Args:
        url_configs (list[dict]): A list of configurations for each crawler.
        added_metadata (dict, optional): Additional metadata to add.
        write (callable, optional): Optional write function.

    Returns:
        list[Ingestion]: A combined list of Ingestion objects from all crawlers.
    """
    ingestions = await run_all_crawlers(url_configs, write=write)
    ingestions = [update_ingestion_with_metadata(ingestion, added_metadata) for ingestion in ingestions]
    return ingestions

if __name__ == "__main__":
    ingestions = ["https://www.whatisemerging.com/opinions", "https://www.whatisemerging.com/"]
    asyncio.run(scrape_ingestion(ingestions))
    url_configs = [
        {
            "url": "https://www.whatisemerging.com/opinions",
            "recursive": True,
            "screenshot": False,
            "extract_params": {},
            "select_params": ["a[href]"],
            "timeout": 1000,
            "max_requests_per_crawl": 5,
        },
    ]
    ingestions = asyncio.run(scrape_urls(url_configs))
    for i, ingestion in enumerate(ingestions):
        print(f"Ingestion {i + 1}: {ingestion}")
