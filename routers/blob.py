import requests
import PyPDF2
import io
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import os
import dotenv
from neo4j import GraphDatabase
from urllib.parse import urlparse
import time

dotenv.load_dotenv()

router = APIRouter(prefix="/blob")

# Neo4j connection
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# Global embedding model
_embedding_model = None
_chat_model = None

class BlobUrlRequest(BaseModel):
    blob_url: str

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

class ChatRequest(BaseModel):
    question: str
    mode: str = "insurance"
    limit: int = 3

def get_embedding_model():
    """Get or create embedding model instance"""
    global _embedding_model
    if _embedding_model is None:
        try:
            vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location=os.getenv("GOOGLE_CLOUD_REGION"))
            _embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
            print("Embedding model initialized")
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
            return None
    return _embedding_model

def generate_embeddings(text: str):
    """Generate embeddings for text"""
    try:
        model = get_embedding_model()
        if model is None:
            return None
        
        embeddings = model.get_embeddings([text])
        return embeddings[0].values
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def get_chat_model():
    """Get or create chat model instance"""
    global _chat_model
    if _chat_model is None:
        try:
            vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location=os.getenv("GOOGLE_CLOUD_REGION"))
            _chat_model = GenerativeModel("gemini-2.5-flash-lite")
            print("Chat model initialized")
        except Exception as e:
            print(f"Error initializing chat model: {e}")
            return None
    return _chat_model

def store_in_neo4j(document_name: str, text: str, embeddings: list, file_type: str, blob_url: str):
    """Store document, text, and embeddings in Neo4j"""
    try:
        with neo4j_driver.session() as session:
            session.run("""
                MERGE (doc:Document {name: $doc_name})
                SET doc.file_type = $file_type,
                    doc.blob_url = $blob_url,
                    doc.text_length = $text_length,
                    doc.created_at = datetime(),
                    doc.full_text = $text,
                    doc.embedding = $embedding
            """, doc_name=document_name, file_type=file_type, 
                blob_url=blob_url, text_length=len(text), 
                text=text, embedding=embeddings)
            
            print(f"✅ Stored document '{document_name}' in Neo4j")
            return True
            
    except Exception as e:
        print(f"❌ Error storing in Neo4j: {e}")
        return False

def create_vector_index():
    """Create vector index in Neo4j for similarity search"""
    try:
        with neo4j_driver.session() as session:
            # Create vector index for embeddings - gemini-embedding-001 has 3072 dimensions
            session.run("""
                CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
                FOR (d:Document) ON (d.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 3072,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """)
            print("✅ Vector index created")
            
    except Exception as e:
        print(f"Error creating vector index: {e}")

@router.post("/setup-neo4j")
async def setup_neo4j():
    """Initialize Neo4j database with vector indexes"""
    create_vector_index()
    return {"message": "✅ Neo4j setup completed with vector index"}

@router.post("/reset-index")
async def reset_vector_index():
    """Drop and recreate the vector index with correct dimensions"""
    try:
        with neo4j_driver.session() as session:
            # Drop existing index
            session.run("DROP INDEX document_embeddings IF EXISTS")
            print("🗑️ Dropped existing index")
            
        # Create new index with correct dimensions
        create_vector_index()
        return {"message": "✅ Vector index reset with 3072 dimensions for gemini-embedding-001"}
        
    except Exception as e:
        return {"error": f"Failed to reset index: {str(e)}"}

@router.get("/test-neo4j")
async def test_neo4j():
    """Test Neo4j connection"""
    try:
        with neo4j_driver.session() as session:
            result = session.run("RETURN 'Neo4j connected!' as message")
            record = result.single()
            return {"status": "Connected", "message": record["message"]}
    except Exception as e:
        return {"status": "Failed", "error": str(e)}

@router.post("/search")
async def semantic_search(request: SearchRequest):
    """
    Semantic search through stored documents using vector similarity
    """
    try:
        # Generate embedding for the search query
        print(f"🔍 Searching for: {request.query}")
        query_embedding = generate_embeddings(request.query)
        
        if not query_embedding:
            return {"error": "Failed to generate query embedding"}
        
        with neo4j_driver.session() as session:
            # Vector similarity search
            result = session.run("""
                CALL db.index.vector.queryNodes('document_embeddings', $limit, $query_embedding)
                YIELD node, score
                RETURN 
                    node.name as document_name,
                    node.file_type as file_type,
                    node.text_length as text_length,
                    node.full_text as full_text,
                    node.blob_url as blob_url,
                    score
                ORDER BY score DESC
            """, limit=request.limit, query_embedding=query_embedding)
            
            search_results = []
            for record in result:
                search_results.append({
                    "document_name": record["document_name"],
                    "file_type": record["file_type"],
                    "text_length": record["text_length"],
                    "text_preview": record["full_text"][:300] + "..." if len(record["full_text"]) > 300 else record["full_text"],
                    "similarity_score": record["score"]
                })
        
        return {
            "query": request.query,
            "results_found": len(search_results),
            "results": search_results
        }
        
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

def extract_pdf_text(pdf_content: bytes) -> str:
    """Extract text from PDF bytes"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

@router.post("/extract-text")
async def extract_text_from_blob(request: BlobUrlRequest):
    """
    Extract text content from blob URL
    """
    try:
        print(f"📥 Downloading from: {request.blob_url}")
        
        # Download the file
        response = requests.get(request.blob_url, timeout=30)
        response.raise_for_status()
        
        # Extract text based on content type
        content_type = response.headers.get('content-type', '').lower()
        
        if 'pdf' in content_type or request.blob_url.lower().endswith('.pdf'):
            extracted_text = extract_pdf_text(response.content)
            file_type = "pdf"
        else:
            extracted_text = response.text
            file_type = "text"
        
        if not extracted_text:
            return {"error": "Could not extract any text", "file_type": file_type}
        
        return {
            "success": True,
            "file_type": file_type,
            "text_length": len(extracted_text),
            "extracted_text": extracted_text[:20] + "..." if len(extracted_text) > 20 else extracted_text,
        }
        
    except Exception as e:
        return {"error": f"Failed to process blob URL: {str(e)}"}

@router.post("/extract-text-with-embeddings")
async def extract_text_with_embeddings(request: BlobUrlRequest):
    """
    Extract text content from blob URL and generate embeddings
    """
    try:
        print(f"📥 Downloading from: {request.blob_url}")
        
        # Download the file
        response = requests.get(request.blob_url, timeout=30)
        response.raise_for_status()
        
        # Extract text based on content type
        content_type = response.headers.get('content-type', '').lower()
        
        if 'pdf' in content_type or request.blob_url.lower().endswith('.pdf'):
            extracted_text = extract_pdf_text(response.content)
            file_type = "pdf"
        else:
            extracted_text = response.text
            file_type = "text"
        
        if not extracted_text:
            return {"error": "Could not extract any text", "file_type": file_type}
        
        # Generate embeddings
        print("🔄 Generating embeddings...")
        embeddings = generate_embeddings(extracted_text)
        
        # Store in Neo4j
        parsed_url = urlparse(request.blob_url)
        doc_name = os.path.basename(parsed_url.path) or f"document_{int(time.time())}"
        
        if embeddings:
            print("💾 Storing in Neo4j...")
            stored = store_in_neo4j(doc_name, extracted_text, embeddings, file_type, request.blob_url)
        else:
            stored = False
        
        return {
            "success": True,
            "file_type": file_type,
            "document_name": doc_name,
            "text_length": len(extracted_text),
            "extracted_text": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
            "embeddings": embeddings[:5] if embeddings else None,  # First 5 values
            "embeddings_length": len(embeddings) if embeddings else 0,
            "has_embeddings": embeddings is not None,
            "stored_in_neo4j": stored
        }
        
    except Exception as e:
        return {"error": f"Failed to process blob URL: {str(e)}"}

@router.post("/chat")
async def chat_with_documents(request: ChatRequest):
    """Chat with your documents using semantic search"""
    try:
        print(f"🤖 Processing question: {request.question}")
        
        # Perform semantic search to find relevant content
        search_results = await semantic_search(SearchRequest(query=request.question, limit=5))
        
        if not search_results.get("results"):
            return {"answer": "I couldn't find any relevant information in your documents to answer that question."}
        
        # Get chat model
        chat_model = get_chat_model()
        if not chat_model:
            raise HTTPException(status_code=500, detail="Chat model not available")
        
        # Prepare context from search results
        context = ""
        for i, result in enumerate(search_results["results"], 1):
            context += f"Document {i} (from {result['document_name']}):\n{result['text_preview']}\n\n"
        
        # Create prompt
        prompt = f"""Based on the following document excerpts, please answer the user's question. If the information isn't available in the documents, say so clearly.

Context from documents:
{context}

User question: {request.question}

Please provide a helpful, accurate answer based on the document content:"""
        
        # Generate response
        response = chat_model.generate_content(prompt)
        
        return {
            "answer": response.text,
            "sources": [{"document_name": r["document_name"], "similarity": r["similarity_score"]} for r in search_results["results"]]
        }
        
    except Exception as e:
        print(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))