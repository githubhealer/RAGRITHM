from fastapi import APIRouter
import PyPDF2
import os
from google.cloud import storage
import dotenv

dotenv.load_dotenv()

router = APIRouter(prefix="/documents", tags=["documents"])

def clear_bucket(bucket_name):
    """Delete all objects in the bucket"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()
    for blob in blobs:
        blob.delete()
    print(f"All objects deleted from bucket: {bucket_name}")

def upload_to_gcs(bucketname,destination_blob_name,content):
    client = storage.Client()
    bucket = client.bucket(bucketname)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(content)
    print(f"File {destination_blob_name} uploaded to {bucketname}.")
    return f"gs://{bucketname}/{destination_blob_name}"

@router.get("/upload")
async def read_document():
    path = os.path.join(os.getcwd(), "documents")
    f = []
    gcs_links = []
    
    documents_bucket = os.getenv('GCS_DOCUMENTS_BUCKET')
    clear_bucket(documents_bucket)
    
    for r, d, files in os.walk(path):
        for file in files:
            print(f"Found file: {file}")
            f.append(file)
            if file.lower().endswith('.pdf'):
                with open(os.path.join(r,file),'rb') as pdf:
                    reader = PyPDF2.PdfReader(pdf)
                    text = ""
                    for page in reader.pages:
                        text+=page.extract_text() or ""
                    chunks = text.split('\n\n')
                    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
                    chunks = [chunk.replace('\n', '') for chunk in chunks]

                    if len(chunks) == 1:
                        chunk_size = 3000
                        overlap = 500 
                        chunks = []
                        for i in range(0, len(text), chunk_size - overlap):
                            chunk = text[i:i + chunk_size]
                            if chunk.strip():
                                chunks.append(chunk.strip())
                    
                    for idx, chunk in enumerate(chunks):
                        chunk_path = f"{file}_chunk_{idx}.txt"
                        gcs_path = upload_to_gcs(documents_bucket, chunk_path, chunk)
                        print(f"Chunk uploaded to: {gcs_path}")
                        gcs_links.append(gcs_path)
    return {"files": f, "gcs_links": gcs_links}
