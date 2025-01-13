from groundx import GroundX
import os
from typing import Optional

client = GroundX(
    api_key=os.environ["GROUNDX_API_KEY"],
)


def get_bucket_id(bucket_name: str):
    buckets_response = client.buckets.list()
    bucket_id = next((b.bucket_id for b in buckets_response.buckets if b.name == bucket_name), None)
    return bucket_id


def search_bucket(bucket_name: str, num_results: int, verbosity: int, query: str, relevance_threshold: float, next_token: Optional[str] = None):
    bucket_id = get_bucket_id(bucket_name)
    response = client.search.content(
        id=bucket_id,
        n=num_results,
        next_token=next_token,
        verbosity=verbosity,
        query=query,
        relevance=relevance_threshold,
    )
    return response


if __name__ == "__main__":
    bucket_name = "pdf-analysis"  # bucketId, groupId, projectId, or documentId to be searched
    num_results = 20
    verbosity = 2  # 0 == no search results, only the recommended context. 1 == search results but no searchData. 2 == search results and searchData.
    query = "What are the datasets that E5 paper uses and which are open source?"
    relevance_threshold = 10.0  # minimum relevance score to return
    # next_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"
    response = search_bucket(bucket_name, num_results, verbosity, query, relevance_threshold)
    print(response)
 