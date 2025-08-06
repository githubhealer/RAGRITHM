from fastapi import APIRouter
import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.cloud import storage
import json
import time
import os
import dotenv

dotenv.load_dotenv()

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

def convert_to_vector(text):
    try:
        vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location=os.getenv("GOOGLE_CLOUD_REGION"))
        model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
        embeddings = model.get_embeddings([text])
        return embeddings[0].values
    except Exception as e:
        print(f"Error initializing Vertex AI: {e}")
        return None
def read_blob():
    try:
        client = storage.Client()
        bucket_name = os.getenv('GCS_DOCUMENTS_BUCKET')
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs()
        for blob in blobs:
            if blob.name.endswith('.txt'):
                content = blob.download_as_text()
                yield blob.name, content
    except Exception as e:
        print(f"Error reading blob: {e}")
        return []
def store_embeddings(vector, filename, chunk_number):
    try:
        client = storage.Client()
        bucket_name = os.getenv('GCS_EMBEDDINGS_BUCKET')
        bucket = client.bucket(bucket_name)
        base_name = filename.replace('.txt', '')
        embedding_name = f"{base_name}_embedding.json"
        blob = bucket.blob(embedding_name)
        
        embedding_data = {
            "source_file": filename,
            "chunk_number": chunk_number,
            "embedding": vector,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        blob.upload_from_string(json.dumps(embedding_data, indent=2))
        print(f"Stored embedding: {embedding_name}")
        
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        return None
@router.get("/generate")
async def generate_embeddings():
    contents = []
    
    client = storage.Client()
    bucket_name = os.getenv('GCS_EMBEDDINGS_BUCKET')
    bucket = client.bucket(bucket_name)

    for filename, content in read_blob():
        print(f"Processing file: {filename}, content: {content[:50]}...")
        
        # Extract chunk number from filename
        if "_chunk_" in filename:
            chunk_number = int(filename.split("_chunk_")[1].split(".")[0])
        else:
            chunk_number = 0
            
        # ✅ FIX: Use filename as base, don't add chunk again
        base_name = filename.replace('.txt', '')
        embedding_name = f"{base_name}_embedding.json"  # Simple naming
        
        blob = bucket.blob(embedding_name)
        
        if blob.exists():
            print(f"Embedding already exists: {embedding_name}, skipping...")
            continue
            
        vector = convert_to_vector(content)
        if vector is not None:
            # Pass the extracted chunk_number to store function
            store_embeddings(vector, filename, chunk_number)
            contents.append({
                "source_file": filename,
                "chunk_number": chunk_number,
                "embedding": vector
            })
            
            time.sleep(2)
            print(f"NEW embedding created: {embedding_name}, sleeping 2 seconds...")
        else:
            print(f"Failed to generate embedding for {filename}")
    
    return {
        "total_new_embeddings": len(contents),
        "processed_files": len(contents),
        "newly_created_embeddings": contents
    }
@router.get("/list_models")
async def list_available_models():
    """List available embedding models"""
    try:
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        vertexai.init(project=project_id)

        models_to_test = [
            "text-embedding-004",
            "gemini-embedding-001",
            "textembedding-gecko@003", 
            "textembedding-gecko@001"
        ]
        
        available_models = []
        for model_name in models_to_test:
            try:
                model = TextEmbeddingModel.from_pretrained(model_name)
                available_models.append(f"{model_name}")
            except Exception as e:
                available_models.append(f"No {model_name}: {str(e)}")
        
        return {"available_models": available_models}
    except Exception as e:
        return {"error": str(e)} 