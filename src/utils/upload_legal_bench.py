import argparse
import asyncio

from src.utils.conversion_utils import convert_and_upload_to_s3


async def main():
    parser = argparse.ArgumentParser(description='Convert and upload LegalBench data to S3')
    parser.add_argument('--txt_dir', type=str, required=True, help='Directory containing text files')
    parser.add_argument('--bucket', type=str, default="astralis-data-4170a4f6", help='S3 bucket name')
    parser.add_argument('--prefix', type=str, default="legal_bench", help='S3 prefix/folder')

    args = parser.parse_args()

    await convert_and_upload_to_s3(
        corpus_directory=args.txt_dir,
        bucket_name=args.bucket,
        prefix=args.prefix
    )

if __name__ == "__main__":
    asyncio.run(main())