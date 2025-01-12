import json
import logging
import os
import time
import urllib
from groundx import Document, GroundX

# Configure logging to show up in terminal
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

groundx = GroundX(api_key=os.environ.get("GROUNDX_API_KEY"))


def ingest_pdfs_to_groundx(bucket_id: str, pdf_folder: str) -> list[str]:
    """
    Ingest PDF files from a folder into GroundX
    """
    ingestion_results = []
    for filename in os.listdir(pdf_folder):
        if not filename.lower().endswith('.pdf'):
            continue
        file_path = os.path.join(pdf_folder, filename)
        try:
            document = Document(
                bucket_id=bucket_id,
                file_name=filename,
                file_path=file_path,
                file_type="pdf"
            )
            response = groundx.ingest(documents=[document])
            ingestion_results.append(response.ingest.process_id)
            logging.info(f"Ingested {filename}")
        except Exception as e:
            logging.error(f"Error ingesting {filename}: {str(e)}")
    return ingestion_results


def wait_for_ingestion_completion(ingestion_responses: list[str]):
    """
    Check the status of ingestion requests and provide a summary.
    """
    for process_id in ingestion_responses:
        while True:
            logging.info(f"Checking status of {process_id}")
            response = groundx.documents.get_processing_status_by_id(process_id=process_id)
            # Check the main status field
            status = response.ingest.status
            if status == "complete":
                break
            if status == "error":
                raise ValueError(f"Error Ingesting Document: {response.ingest.status_message}")
            # Print current status for monitoring
            print(f"Current status: {status}")
            if response.ingest.progress and response.ingest.progress.processing:
                docs = response.ingest.progress.processing.documents
                if docs:
                    print(f"Document status: {docs[0].status}")
            time.sleep(5)


def make_groundx_bucket(bucket_name: str):
    """
    Create a new bucket in GroundX, returns bucket_id
    """
    # Updated to match documentation format
    bucket_response = groundx.buckets.create(name=bucket_name)
    return bucket_response.bucket.bucket_id


def delete_groundx_bucket(bucket_id: str):
    """
    Delete a bucket in GroundX
    """
    groundx.buckets.delete(bucket_id=bucket_id)


def get_xray_responses(documents_response):
    """
    Extract and fetch X-Ray data from document responses
    Args:
        documents_response: DocumentLookupResponse object containing document details
    Returns:
        list of parsed X-Ray JSON data
    """
    data = []
    for doc in documents_response.documents:
        with urllib.request.urlopen(doc.xray_url) as url:
            xray_data = json.loads(url.read().decode())
            data.append(xray_data)
            logging.info(f"Got X-Ray response for {doc.file_name}")
    return data


def get_bucket_id(bucket_name: str):
    buckets_response = groundx.buckets.list()
    bucket_id = next((b.bucket_id for b in buckets_response.buckets if b.name == bucket_name), None)
    return bucket_id


def ingest_pdfs_from_folder(file_path: str, bucket_name: str):
    bucket_id = get_bucket_id(bucket_name)
    if bucket_id is None:
        bucket_id = make_groundx_bucket(bucket_name)
    else:
        print(f"Bucket {bucket_name} already exists with ID {bucket_id}")
    # Ingest PDFs to GroundX
    ingestion_responses = ingest_pdfs_to_groundx(bucket_id, file_path)
    # Wait for processing to complete
    wait_for_ingestion_completion(ingestion_responses)


def process_xray_data_from_bucket(bucket_name: str):
    bucket_id = get_bucket_id(bucket_name)
    documents_response = groundx.documents.lookup(id=bucket_id)
    xray_responses = get_xray_responses(documents_response)
    for xray_data in xray_responses:
        with open(f"{xray_data['fileName']}.json", "w") as f:
            json.dump(xray_data, f, indent=4)
        # process_xray_data(xray_data)


# RENDERING FUNCTIONS
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


def xray_highlight(data):
    print(f"fileType: {data['fileType']}")
    print(f"language: {data['language']}")
    print(f"fileName: {data['fileName']}")
    print(f"fileKeywords: {data['fileKeywords']}")
    print(f"fileSummary: {data['fileSummary']}")
    print(f"Total Pages: {len(data['documentPages'])}")


def xray_page(page):
    print(f"\033[91m\nPage {page['pageNumber']}\033[0m -> Dim: {page['height']} x {page['width']}, {len(page['chunks'])} Chunks")
    print(f"Page URL: {page['pageUrl']}")

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


if __name__ == "__main__":
    file_path = "/Users/pranaviyer/Desktop/AstralisData"
    bucket_name = "pdf-analysis"
    # ingest_pdfs_from_folder(file_path, bucket_name)
    process_xray_data_from_bucket(bucket_name)
