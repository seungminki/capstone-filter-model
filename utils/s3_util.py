import os
import boto3

from settings import BUCKET_NAME, AWS_S3_KEY_ID, AWS_S3_SECRET_KEY


def download_from_s3_if_not_exists(local_dir, local_filename, s3_key):
    # 전체 로컬 경로
    local_path = os.path.join(local_dir, local_filename)

    # 파일이 이미 있으면 다운로드하지 않음
    if os.path.exists(local_path):
        # print(f"'{filename}' already exists in '{local_dir}'")
        return

    # S3 클라이언트 생성
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_S3_KEY_ID,
        aws_secret_access_key=AWS_S3_SECRET_KEY,
        region_name="ap-northeast-2",
    )

    try:
        # 다운로드
        # print(f"Downloading '{filename}' from S3 bucket '{bucket_name}'...")
        s3.download_file(BUCKET_NAME, s3_key, local_path)
        # print("Download completed.")
    except Exception as e:
        print(f"Failed to download: {e}")
