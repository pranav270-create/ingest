import json
import logging
import os
import time
import urllib

from groundx import Groundx

# Configure logging to show up in terminal
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

groundx = Groundx(api_key=os.environ.get("GROUNDX_API_KEY"))


def ingest_papers_to_groundx(
    bucket_id: str, base_path: str, pdf_filepaths: list[str], papers_metadata: list[dict[str, str]]
) -> list[str]:
    """
    Ingest downloaded documents with GroundX

    :param bucket_id: GroundX bucket ID to ingest the papers into
    :param pdf_filepaths: list of directory paths to PDF files
    :param papers_metadata: list of metadata dictionaries for each paper
    :return: list of ingestion responses
    """
    ingestion_results = []

    pdf_filepaths_sort = sorted(pdf_filepaths)
    papers_metadata = sorted(papers_metadata, key=lambda x: x["link"])

    for pdf_path, paper_metadata in zip(pdf_filepaths_sort, papers_metadata):
        file_name = os.path.basename(pdf_path)
        code = paper_metadata["link"].split("/")[-1]
        assert file_name.rsplit(".", 1)[0] == code, f"file name {file_name} does not match code {code}"

        search_data = {
            "title": paper_metadata["title"],
            "link": paper_metadata["link"],
            "authors": paper_metadata["authors"],
            "tags": paper_metadata["tags"],
            "abstract": paper_metadata["abstract"],
            "arxiv_id": paper_metadata["id"],
        }

        try:
            response = groundx.documents.ingest_local(
                body=[
                    {
                        "blob": open(f"{base_path}/{file_name}", "rb"),
                        "metadata": {
                            "bucketId": bucket_id,
                            "fileName": file_name,
                            "fileType": "pdf",
                            "searchData": search_data,
                        },
                    },
                ]
            )
            if response.status == 200:  # save ids for later
                ingestion_results.append(response.body["ingest"]["processId"])
                logging.info(f"Ingested {file_name}")
            elif response.status == 413:
                logging.error(f"{file_name}: ")
        except Exception as e:
            logging.error(f"Error ingesting {file_name}: {str(e)}")

    return ingestion_results


def wait_for_ingestion_completion(ingestion_responses: list[str]):
    """
    Check the status of ingestion requests and provide a summary.
    """
    for process_id in ingestion_responses:
        while True:
            logging.info(f"Checking status of {process_id}")
            ingest = groundx.documents.get_processing_status_by_id(process_id=process_id)
            if ingest.body["ingest"]["status"] == "complete":
                break
            if ingest.body["ingest"]["status"] == "error":
                raise ValueError("Error Ingesting Document")
            print(ingest.body["ingest"]["status"])
            time.sleep(5)


def make_groundx_bucket(bucket_name: str):
    """
    Create a new bucket in GroundX, returns bucket_id
    """
    bucket_response = groundx.buckets.create(name=bucket_name)
    bucket_id = bucket_response.body["bucket"]["bucketId"]
    return bucket_id


def get_xray_responses(ingestion_ids: list[str]):
    ingestion_id_set = set(ingestion_ids)
    xray_urls = [r["xrayUrl"] for r in documents_response.body["documents"] if r["processId"] in ingestion_id_set]
    data = []
    for url in xray_urls:
        with urllib.request.urlopen(url) as url:
            xray_data = json.loads(url.read().decode())
            data.append(xray_data)
            logging.info(f"Got X-Ray response for {url}")
    return data


def xray_highlight(data):
    # print(f"fileType: {data['fileType']}")
    # print(f"language: {data['language']}")
    print(f"fileName: {data['fileName']}")
    print(f"fileKeywords: {data['fileKeywords']}")
    print(f"fileSummary: {data['fileSummary']}")
    print(f"Total Pages: {len(data['documentPages'])}")


def xray_page(page):
    print(f"\033[91m\nPage {page['pageNumber']}\033[0m -> Dim: {page['height']} x {page['width']}, {len(page['chunks'])} Chunks")
    # print(f"Page URL: {page['pageUrl']}")

    for chunk in page["chunks"]:
        xray_chunk(chunk)


def xray_chunk(chunk, show_bounding_boxes: bool = False):
    print(f"\033[94mChunk {chunk['chunk']}:\033[0m")
    print(f"Section Summary: {chunk['sectionSummary']}")
    # print(f"Suggested Text: {chunk['suggestedText']}")
    # print(f"Text: {chunk['text'][:100]}...")

    if show_bounding_boxes:
        for bounding_box in chunk["boundingBoxes"]:
            print(f"Top Left: {bounding_box['topLeftX']}, {bounding_box['topLeftY']}")
            print(f"Bottom Right: {bounding_box['bottomRightX']}, {bounding_box['bottomRightY']}")
            print(f"Page Number: {bounding_box['pageNumber']}")

    for content_type in chunk["contentType"]:
        print(f"  Content Type: {content_type}")
        if "paragraph" in content_type:
            print(f"    Text: {chunk['text'][:10]}...")  # Print first 100 characters
        else:
            print(f"    URL: {chunk['multimodalUrl']}")
            print(f"    narrative: {chunk['narrative']}")
            if "table" in content_type:
                print(f"    Table Data: {chunk['json']}")
            elif "figure" in content_type:
                print(f"    Figure Description: {chunk['narrative']}")


def process_xray_data(data):
    xray_highlight(data)

    for page in data["documentPages"]:
        xray_page(page)

    # Example of extracting all tables
    # tables = [chunk for page in data['documentPages'] for chunk in page['chunks'] if 'table' in chunk['contentType']]
    # print(f"\nTotal Tables Found: {len(tables)}")

    # Example of extracting all figures
    # figures = [chunk for page in data['documentPages'] for chunk in page['chunks'] if 'figure' in chunk['contentType']]
    # print(f"Total Figures Found: {len(figures)}")


if __name__ == "__main__":
    # fetch new paper ids
    new_papers = [{"": ""}]  # list of dicts containing metadata and an id for each document
    # download papers
    pdf_localpaths = [""]  # download the pdfs
    download_folder = ""  # root folder
    # get groundx bucket id or create bucket
    bucket_name = "al-internal"
    bucket_id = next((b["bucketId"] for b in groundx.buckets.list().body["buckets"] if b["name"] == bucket_name), None)
    if bucket_id is None:
        bucket_id = make_groundx_bucket(bucket_name)
    # Ingest papers into GroundX
    ingestion_responses = ingest_papers_to_groundx(bucket_id, download_folder, pdf_localpaths, new_papers)
    # get rid of erroneous ingestion responses
    pruned_ingestion_responses = [i for i in ingestion_responses if isinstance(i, str)]
    # Check status of ingestion
    wait_for_ingestion_completion(pruned_ingestion_responses)
    # Getting parsed documents from the bucket
    documents_response = groundx.documents.lookup(id=bucket_id)
    # Getting the X-Ray parsing results for documents
    xray_responses = get_xray_responses(pruned_ingestion_responses)
    process_xray_data(xray_responses[0])