import asyncio
import os
import sys
import uuid
from asyncio import Queue
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import aiohttp
import backoff
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.pipeline.registry.function_registry import FunctionRegistry
from src.schemas.schemas import (
    ContentType,
    FileType,
    Ingestion,
    IngestionMethod,
    Scope,
)
from src.utils.datetime_utils import get_current_utc_datetime
from src.utils.ingestion_utils import update_ingestion_with_metadata


async def get_urls(urls: list[str], write=None, added_metadata: dict = {}):
    master_list = []
    all_ingestions = []
    temp_dir = get_temp_dir()

    for url in urls:
        try:
            download_filename = f"{uuid.uuid4()}.html"
            download_path = os.path.join(temp_dir, download_filename)

            os.system(f'wget -O "{download_path}" {url}')

            with open(download_path) as f:
                html = f.read()

            soup = BeautifulSoup(html, "html.parser")
            document_title = soup.title.string if soup.title else str(uuid.uuid4())  # Get title from BeautifulSoup if available, else use uuid
            new_local_path = f"{download_path}.html"
            os.rename(download_path, new_local_path)  # Rename the file
            local_path = new_local_path

            if write:
                new_local_path = local_path.replace(f"{temp_dir}/", "")
                await write(new_local_path, html, mode="w")
                os.system(f"rm {local_path}")
                local_path = new_local_path

            ingestion = Ingestion(
                document_title=document_title,
                scope=Scope.EXTERNAL,
                content_type=ContentType.WEBSITE,
                file_type=FileType.HTML,
                file_path=local_path,  # Now we have the correct local path
                public_url=url,
                creator_name="Unknown",
                creation_date=None,
                ingestion_date=get_current_utc_datetime(),
                ingestion_method=IngestionMethod.URL_SCRAPE,
                unprocessed_citations=None,
                embedded_feature_type=None,
            )
            ingestion = update_ingestion_with_metadata(ingestion, added_metadata)
            all_ingestions.append(ingestion)
            ahrefs = soup.find_all("a")
            hrefs = [a.get("href") for a in ahrefs]
            master_list.extend(hrefs)
        except Exception as e:
            print(f"Error processing {url}: {e}")
            continue

    return master_list, all_ingestions


def get_temp_dir():
    temp_dir = "temp_downloads"
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


class DownloadQueue:
    def __init__(self, max_concurrent: int = 5, timeout: int = 30):
        self.queue = Queue()
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=3)
    async def download_single(self, url: str) -> Optional[bytes]:
        try:
            async with self.session.get(url, timeout=self.timeout) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    print(f"Failed to download {url}: Status {response.status}")
                    return None
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            print(f"Error downloading {url}: {e}")
            return None

    async def worker(self):
        while True:
            url, callback = await self.queue.get()
            try:
                content = await self.download_single(url)
                if content and callback:
                    await callback(url, content)
            except Exception as e:
                print(f"Worker error for {url}: {e}")
            finally:
                self.queue.task_done()

    async def process_urls(self, urls: list[str], callback) -> list[tuple[str, bool]]:
        # Start workers
        workers = [asyncio.create_task(self.worker()) for _ in range(self.max_concurrent)]

        # Add URLs to queue
        for url in urls:
            await self.queue.put((url, callback))

        # Wait for queue to be processed
        await self.queue.join()

        # Clean up workers
        for w in workers:
            w.cancel()

        return workers


async def download_pdfs(pdf_links: list[str], write=None) -> list[tuple[str, str]]:
    downloaded_pairs = []
    temp_dir = get_temp_dir()

    async def process_download(url: str, content: bytes):
        if not content:
            return

        file_name = os.path.basename(urlparse(url).path)
        if write:
            await write(file_name, content, mode="wb")
            downloaded_pairs.append((url, file_name))
        else:
            local_path = os.path.join(file_name)
            with open(local_path, "wb") as f:
                f.write(content)
            downloaded_pairs.append((url, local_path))

    async with DownloadQueue(max_concurrent=5, timeout=30) as queue:
        await queue.process_urls(pdf_links, process_download)

    # Clean up temp directory
    try:
        import shutil

        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Error cleaning up temp directory: {e}")

    return downloaded_pairs


def clean_pdf_links(master_list: list[str], urls: list[str]) -> None:
    # Filter and clean PDF links
    pdf_links = set()
    for link in master_list:
        if not link:  # Skip empty links
            continue

        # Normalize the link
        clean_link = link.strip().lower()
        if "#" in clean_link:  # Only split if # exists
            clean_link = clean_link.split("#")[0]

        # Skip invalid protocols or obviously broken links
        if clean_link.startswith(("javascript:", "mailto:", "tel:", "data:")):
            continue

        # Check for PDF indicators
        is_pdf = any(
            [
                clean_link.endswith(".pdf"),
                "application/pdf" in clean_link,
                "/pdf/" in clean_link,
                "download=pdf" in clean_link,
                "type=pdf" in clean_link,
                "format=pdf" in clean_link,
            ]
        )

        if is_pdf:
            # Handle relative URLs by joining with base URL if needed
            try:
                if not urlparse(link).netloc:  # No domain = relative URL
                    base_url = urls[0]  # Using first URL as base
                    link = urljoin(base_url, link)
                pdf_links.add(link)
            except Exception as e:
                print(f"Error processing PDF link {link}: {e}")
                continue
    return pdf_links


@FunctionRegistry.register("ingest", "manual_links")
async def manual_ingest(urls: list[str], read=None, write=None, added_metadata: dict = {}, **kwargs):
    master_list, all_ingestions = await get_urls(urls, write, added_metadata)

    # Filter and clean PDF links
    pdf_links = clean_pdf_links(master_list, urls)

    # Download PDFs with queue system
    downloaded_pairs = await download_pdfs(pdf_links, write)

    # Process successful downloads
    for link, local_path in downloaded_pairs:
        document_title = os.path.basename(urlparse(link).path)
        ingestion = Ingestion(
            document_title=document_title,
            scope=Scope.EXTERNAL,
            content_type=ContentType.PDF_DOCUMENTS,
            file_type=FileType.PDF,
            file_path=local_path,
            public_url=link,
            creator_name="Bridgewater Township",
            creation_date=None,
            ingestion_date=get_current_utc_datetime(),
            ingestion_method=IngestionMethod.URL_SCRAPE,
            unprocessed_citations=None,
            embedded_feature_type=None,
        )
        ingestion = update_ingestion_with_metadata(ingestion, added_metadata)
        all_ingestions.append(ingestion)
    return all_ingestions


if __name__ == "__main__":
    urls = [
        # "http://www.bridgewaternj.gov/land-development-applications-instructions/",
        # "https://www.bridgewaternj.gov/ammendments-plans-reports/",
        # "https://www.bridgewaternj.gov/department-of-code-enforcement/",
        # "https://www.bridgewaternj.gov/zoning-board-minutes/",
        # "https://www.bridgewaternj.gov/council-meeting-minutes/",
        "https://www.bridgewaternj.gov/planning-board-minutes/"
    ]
    urls = ["http://ww2.abilenetx.com/ordinances/2011%20Ordinances/"]
    all_ingestions = asyncio.run(manual_ingest(urls, added_metadata={"creator_name": "City of Abilene"}))
