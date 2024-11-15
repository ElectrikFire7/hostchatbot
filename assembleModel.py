import boto3
import os
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv

load_dotenv()

def download_model_from_s3(bucket_name, s3_model_path, local_model_path):
    """
    Downloads a model directory from S3 to the local system.

    :param bucket_name: Name of the S3 bucket
    :param s3_model_path: Path to the model in the S3 bucket
    :param local_model_path: Local path where the model should be saved
    """
    s3 = boto3.client('s3')

    try:
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_model_path)

        if 'Contents' in objects:
            for obj in objects['Contents']:
                s3_file_path = obj['Key']
                local_file_path = os.path.join(local_model_path, os.path.relpath(s3_file_path, s3_model_path))

                # Ensure local directory exists
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the file
                print(f"Downloading {s3_file_path} to {local_file_path}")
                s3.download_file(bucket_name, s3_file_path, local_file_path)
        else:
            print("No objects found in the specified S3 path.")
    
    except NoCredentialsError:
        print("AWS credentials not found.")
        raise
    except PartialCredentialsError:
        print("Incomplete AWS credentials configuration.")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise