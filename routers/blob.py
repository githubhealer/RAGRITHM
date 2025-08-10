# Document Processing System - Exact Specification Implementation

import requests
import PyPDF2
import io
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Header
from pydantic import BaseModel
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import os
import dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError, ServiceUnavailable, SessionExpired
from urllib.parse import urlparse
import time
import re
import uuid
import asyncio
from typing import Dict, Any
import random

try:
    from google.api_core.exceptions import ResourceExhausted, TooManyRequests  # type: ignore
except Exception:  # pragma: no cover
    ResourceExhausted = TooManyRequests = Exception

dotenv.load_dotenv()

router = APIRouter(prefix="/blob")

def _normalize_neo4j_uri(uri: str | None) -> str | None:
    if not uri:
        return uri
    # For Aura (neo4j.io), enforce encrypted scheme
    if "neo4j.io" in uri and "+s" not in uri:
        uri = uri.replace("neo4j://", "neo4j+s://").replace("bolt://", "bolt+s://")
    return uri

# Neo4j connection
NEO4J_URI = _normalize_neo4j_uri(os.getenv("NEO4J_URI"))
neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# Neo4j vector index configuration
INDEX_NAME = os.getenv("NEO4J_VECTOR_INDEX", "chunk_embeddings")
INDEX_PROP = os.getenv("NEO4J_VECTOR_PROPERTY", "embedding")
NEO4J_DB = os.getenv("NEO4J_DATABASE")  # optional; Aura often uses 'neo4j'
DEFAULT_DIMS = int(os.getenv("EMBEDDING_DIMS", "768"))

# Embedding configuration
EMBEDDING_TARGET_DIMS = int(os.getenv("EMBEDDING_TARGET_DIMS", str(DEFAULT_DIMS)))  # enforce 768 by default
EMBEDDING_MODELS = [
    m.strip() for m in os.getenv(
        "EMBEDDING_MODELS",
        "text-embedding-004,text-embedding-005,text-multilingual-embedding-002"
    ).split(",") if m.strip()
]

# Global models
_embedding_model = None  # retained for health check compatibility
_embedding_clients: Dict[str, TextEmbeddingModel] = {}
_chat_model = None

# In-memory progress tracking for long-running jobs (ephemeral per instance)
_jobs: Dict[str, Dict[str, Any]] = {}

def _set_progress(job_id: str, **kwargs):
    job = _jobs.get(job_id, {})
    job.update(kwargs)
    job["updated_at"] = time.time()
    _jobs[job_id] = job

class BlobUrlRequest(BaseModel):
    blob_url: str

class ChatRequest(BaseModel):
    question: str
    limit: int = 5

class HackRxRunRequest(BaseModel):
    documents: str  # Single document URL for now
    questions: list[str]

class HackRxJobStartResponse(BaseModel):
    job_id: str
    status_url: str


# --- Simple Bearer token auth dependency ---
def require_bearer_token(authorization: str | None = Header(default=None)):
    expected = os.getenv("API_BEARER_TOKEN", "").strip()
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing or invalid authorization header")
    token = authorization.split(" ", 1)[1].strip()
    # If expected token is configured, enforce exact match
    if expected:
        if token != expected:
            raise HTTPException(status_code=401, detail="invalid token")
    # If expected is empty, accept any non-empty bearer token
    return True


def _neo4j_session():
    """Helper to open a session, honoring optional database name."""
    if NEO4J_DB:
        return neo4j_driver.session(database=NEO4J_DB)
    return neo4j_driver.session()


def _clean_answer_text(text: str) -> str:
    """Normalize model output to a single-line plain sentence without bullets or markdown."""
    if not text:
        return ""
    # Replace line breaks with spaces
    cleaned = re.sub(r"[\r\n]+", " ", text)
    # Remove common markdown/list markers and excessive punctuation spacing
    cleaned = re.sub(r"\s*([*•\-]+)\s+", " ", cleaned)
    # Remove ordered list markers like "1. ", "2) " at word boundaries
    cleaned = re.sub(r"(?:(?<=\s)|^)(\d+\.|\d+\))\s+", "", cleaned)
    # Strip backticks and hash headers
    cleaned = cleaned.replace("`", "").replace("#", "")
    # Collapse multiple spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _index_exists(session) -> bool:
    res = session.run("SHOW INDEXES YIELD name WHERE name = $name RETURN name", name=INDEX_NAME)
    return res.single() is not None


def _try_create_index(session, dims: int) -> str | None:
    """Try DDL first, then procedure. Return the method used or None if failed."""
    # DDL
    try:
        session.run(
            f"""
            CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
            FOR (c:Chunk) ON (c.{INDEX_PROP})
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: $dims,
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """,
            dims=dims,
        ).consume()
        return "ddl"
    except ClientError as e:
        # Fall through to procedure
        print(f"⚠️ DDL create failed: {e}")
    except Exception as e:
        print(f"⚠️ DDL create error: {e}")

    # Procedure
    try:
        session.run(
            "CALL db.index.vector.createNodeIndex($name,'Chunk',$prop,$dims,'cosine')",
            name=INDEX_NAME,
            prop=INDEX_PROP,
            dims=dims,
        ).consume()
        return "procedure"
    except ClientError as e:
        print(f"⚠️ Procedure create failed: {e}")
    except Exception as e:
        print(f"⚠️ Procedure create error: {e}")
    return None

def _init_vertex_if_needed():
    try:
        vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location=os.getenv("GOOGLE_CLOUD_REGION"))
    except Exception as e:
        print(f"⚠️ Vertex init warning: {e}")


def _get_embedding_client(model_name: str) -> TextEmbeddingModel | None:
    global _embedding_clients
    if model_name in _embedding_clients:
        return _embedding_clients[model_name]
    try:
        _init_vertex_if_needed()
        client = TextEmbeddingModel.from_pretrained(model_name)
        _embedding_clients[model_name] = client
        print(f"✅ Embedding client ready ({model_name})")
        return client
    except Exception as e:
        print(f"❌ Failed to init embedding client {model_name}: {e}")
        return None


def get_embedding_model():
    """Legacy getter for health check; prefers first configured model."""
    global _embedding_model
    if _embedding_model is None:
        # Try to create a client for the first configured model to mark 'available'
        client = _get_embedding_client(EMBEDDING_MODELS[0] if EMBEDDING_MODELS else "gemini-embedding-001")
        _embedding_model = client
    return _embedding_model

def get_chat_model():
    """Get or create chat model instance"""
    global _chat_model
    if _chat_model is None:
        try:
            vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location=os.getenv("GOOGLE_CLOUD_REGION"))
            
            # Try different model versions in order of preference
            models_to_try = [
                "gemini-2.5-flash-exp"
            ]
            
            for model_name in models_to_try:
                try:
                    _chat_model = GenerativeModel(model_name)
                    print(f"✅ Chat model initialized ({model_name})")
                    return _chat_model
                except Exception as model_error:
                    print(f"⚠️ Failed to load {model_name}: {model_error}")
                    continue
            
            print("❌ No compatible model found")
            return None
            
        except Exception as e:
            print(f"❌ Error initializing chat model: {e}")
            return None
    return _chat_model

def extract_pdf_text(blob_url):
    """Extract text from PDF blob URL"""
    try:
        print(f"📄 Downloading PDF from: {blob_url}")
        response = requests.get(blob_url, timeout=30)
        response.raise_for_status()
        
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text += page_text + "\n"
            
        print(f"📝 Extracted {len(text)} characters from {len(pdf_reader.pages)} pages")
        return text.strip()
        
    except Exception as e:
        print(f"❌ Error extracting PDF text: {e}")
        return None

def chunk_text(text, chunk_size=1000, overlap=150):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_size
        
        # If this isn't the last chunk, try to end at a word boundary
        if end < len(text):
            # Look for the last space within reasonable distance
            last_space = text.rfind(' ', start, end)
            if last_space > start + chunk_size * 0.8:  # Only if we don't lose too much
                end = last_space
        
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append({
                'text': chunk,
                'start_pos': start,
                'end_pos': end,
                'chunk_index': len(chunks)
            })
        
        # Move start position for next chunk (with overlap)
        start = end - overlap
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    print(f"🔪 Split text into {len(chunks)} chunks (size: {chunk_size}, overlap: {overlap})")
    return chunks

def _get_tokenizer():
    """Lazy-import tiktoken tokenizer; fall back to whitespace if unavailable."""
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return enc
    except Exception:
        return None


def chunk_text_tokens(text: str, chunk_tokens: int = None, overlap_tokens: int = None):
    """Split text into token-based overlapping chunks.    """
    if chunk_tokens is None:
        chunk_tokens = int(os.getenv("CHUNK_TOKENS", "2900"))
    if overlap_tokens is None:
        overlap_tokens = int(os.getenv("OVERLAP_TOKENS", "150"))

    enc = _get_tokenizer()
    if enc is None:
        # Fallback to character-based splitter
        print("⚠️ Tokenizer not available; using character-based splitting")
        return chunk_text(text, chunk_size=chunk_tokens * 4, overlap=overlap_tokens * 4)

    tokens = enc.encode(text)
    chunks = []
    start = 0
    idx = 0
    n = len(tokens)
    while start < n:
        end = min(n, start + chunk_tokens)
        token_slice = tokens[start:end]
        chunk_str = enc.decode(token_slice)
        if chunk_str.strip():
            chunks.append({
                'text': chunk_str.strip(),
                'start_pos': start,  # token index
                'end_pos': end,      # token index
                'chunk_index': idx
            })
            idx += 1
        if end >= n:
            break
        start = end - overlap_tokens
        if start < 0:
            start = 0
    print(f"🔪 Split text into {len(chunks)} token-chunks (tokens: {chunk_tokens}, overlap: {overlap_tokens})")
    return chunks

def _is_rate_limit_error(err: Exception) -> bool:
    s = str(err).lower()
    return (
        isinstance(err, (ResourceExhausted, TooManyRequests))
        or "429" in s
        or "rate limit" in s
        or "quota" in s
        or "resource has been exhausted" in s
    )


def _truncate_or_pad(vec: list[float], target: int) -> list[float]:
    if len(vec) == target:
        return vec
    if len(vec) > target:
        return vec[:target]
    return vec + [0.0] * (target - len(vec))


def embed_text(text, max_retries=5):
    if not text:
        return None
    models = EMBEDDING_MODELS if EMBEDDING_MODELS else ["text-embedding-004", "gemini-embedding-001"]
    mcount = len(models)
    last_err = None
    start_idx = 0
    for attempt in range(1, max_retries + 1):
        # Try a full round of all models before backing off
        for step in range(mcount):
            model_name = models[(start_idx + step) % mcount]
            client = _get_embedding_client(model_name)
            if client is None:
                last_err = RuntimeError(f"embedding client unavailable: {model_name}")
                continue
            try:
                if model_name == "text-embedding-004":
                    res = client.get_embeddings([text], output_dimensionality=EMBEDDING_TARGET_DIMS)
                else:
                    res = client.get_embeddings([text])
                if not res:
                    raise RuntimeError("empty embedding result")
                vec = list(res[0].values)
                vec = _truncate_or_pad(vec, EMBEDDING_TARGET_DIMS)
                return vec
            except Exception as e:
                last_err = e
                if _is_rate_limit_error(e):
                    print(f"⚠️ Rate limited on {model_name}. Trying next model immediately (attempt {attempt}/{max_retries}).")
                    continue  # move to next model without delay
                else:
                    print(f"⚠️ Embedding error on {model_name}: {e}. Trying next model.")
                    continue

        # After a full round, backoff with jitter and rotate the start index
        if attempt < max_retries:
            base = 2 ** (attempt - 1)
            time.sleep(base + random.uniform(0, 0.5))
            start_idx = (start_idx + 1) % mcount
    print(f"❌ Failed to embed after {max_retries} attempts. Last error: {last_err}")
    return None


def generate_embeddings(text, max_retries=5):
    """Generate 768-dim embeddings with alternation and truncation/padding."""
    try:
        return embed_text(text, max_retries=max_retries)
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return None

def store_document_with_chunks(file_name, full_text, blob_url, chunks_with_embeddings):
    """Store document and chunks in Neo4j with proper graph structure"""
    try:
        with neo4j_driver.session() as session:
            # First, create the Document node (retryable via execute_write)
            def _create_doc(tx):
                return tx.run(
                    """
                    MERGE (d:Document {filename: $filename})
                    SET d.blob_url = $blob_url,
                        d.full_text_length = $text_length,
                        d.total_chunks = $total_chunks,
                        d.created_at = coalesce(d.created_at, datetime())
                    RETURN d
                    """,
                    filename=file_name,
                    blob_url=blob_url,
                    text_length=len(full_text),
                    total_chunks=len(chunks_with_embeddings)
                ).single()

            document_node = session.execute_write(_create_doc)
            if not document_node:
                raise Exception("Failed to create Document node")

            print(f"📊 Created Document node: {file_name}")

            # Now create Chunk nodes and relationships
            chunks_created = 0
            for chunk_data in chunks_with_embeddings:
                if chunk_data['embedding'] is None:
                    print(f"⚠️ Skipping chunk {chunk_data['chunk_index']} - no embedding")
                    continue

                def _upsert_chunk(tx):
                    return tx.run(
                        """
                        MATCH (d:Document {filename: $filename})
                        MERGE (c:Chunk {chunk_id: $chunk_id})
                        SET c.text = $text,
                            c.embedding = $embedding,
                            c.chunk_index = $chunk_index,
                            c.start_pos = $start_pos,
                            c.end_pos = $end_pos,
                            c.text_length = $text_length
                        MERGE (d)-[:HAS_CHUNK]->(c)
                        RETURN c
                        """,
                        filename=file_name,
                        chunk_id=f"{file_name}:{chunk_data['chunk_index']}",
                        text=chunk_data['text'],
                        embedding=chunk_data['embedding'],
                        chunk_index=chunk_data['chunk_index'],
                        start_pos=chunk_data['start_pos'],
                        end_pos=chunk_data['end_pos'],
                        text_length=len(chunk_data['text'])
                    ).single()

                session.execute_write(_upsert_chunk)
                chunks_created += 1

            print(f"✅ Created {chunks_created} Chunk nodes with HAS_CHUNK relationships")
            return True
            
    except Exception as e:
        print(f"❌ Error storing document with chunks: {e}")
        return False

async def _async_hackrx_job(job_id: str, documents: str, questions: list[str]):
    """Background job that processes a document and answers questions, updating progress."""
    try:
        _set_progress(job_id, status="running", percent=1, stage="starting")

        # Step 1: Process the document
        _set_progress(job_id, percent=10, stage="processing_document")
        blob_request = BlobUrlRequest(blob_url=documents)
        process_result = await process_blob_document(blob_request)
        _set_progress(job_id, percent=60, stage="document_processed", doc=process_result.get("filename"))

        # Step 2: Answer all questions
        answers: list[str] = []
        total = max(1, len(questions))
        for i, question in enumerate(questions):
            _set_progress(job_id, percent=60 + int((i / total) * 40), stage="answering", current=i+1, total=total)
            chat_request = ChatRequest(question=question, limit=3)
            chat_result = await chat_with_documents(chat_request)
            answers.append(chat_result.get("answer", ""))
            await asyncio.sleep(0)  # yield control

        result = {
            "answers": answers,
            "document_processed": process_result.get("filename"),
            "total_questions": len(questions),
            "total_chunks": process_result.get("total_chunks", 0)
        }
        _set_progress(job_id, status="completed", percent=100, stage="done", result=result)

    except Exception as e:
        _set_progress(job_id, status="failed", percent=100, stage="error", error=str(e))

def _run_hackrx_job(job_id: str, payload: dict):
    """Run the async job in a fresh event loop (for BackgroundTasks)."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_async_hackrx_job(job_id, payload["documents"], payload["questions"]))
    finally:
        try:
            loop.close()
        except Exception:
            pass

def _setup_neo4j_core(force: bool = False):
    """Core logic to ensure the vector index exists. Returns a status dict."""
    try:
        with _neo4j_session() as session:
            if _index_exists(session):
                if not force:
                    return {"ok": True, "index": INDEX_NAME, "present": True, "message": "Index already exists"}
                # Drop then recreate
                print("⚠️ Force requested. Dropping existing index before recreate…")
                session.run(f"DROP INDEX {INDEX_NAME} IF EXISTS").consume()
                print("🗑️ Dropped old vector index")

            # Determine dimensions via sample embedding
            dims_probe = DEFAULT_DIMS
            try:
                sample = generate_embeddings("vector index setup dimension probe")
                if sample and isinstance(sample, list):
                    dims_probe = len(sample)
                    print(f"ℹ️ Detected embedding dimensions: {dims_probe}")
                else:
                    print(f"ℹ️ Could not detect embedding dimensions, using default {DEFAULT_DIMS}")
            except Exception as dim_err:
                print(f"⚠️ Dimension detection failed, using default {DEFAULT_DIMS}: {dim_err}")

            dims_candidates = [dims_probe]
            if 768 not in dims_candidates:
                dims_candidates.append(768)

            last_err = None
            for dims in dims_candidates:
                method = _try_create_index(session, dims)
                if method and _index_exists(session):
                    print(f"✅ Vector index '{INDEX_NAME}' created via {method} (dims={dims})")
                    return {"ok": True, "index": INDEX_NAME, "present": True, "created_with": method, "dims": dims}
                last_err = f"creation failed for dims={dims}"

            raise HTTPException(status_code=500, detail={"message": "Failed to create vector index", "last_error": last_err})
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error creating vector index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create vector index: {str(e)}")


@router.post("/setup-neo4j")
async def setup_neo4j(force: bool = False):
    """Setup/verify Neo4j vector index for chunks. Use force=true to drop and recreate."""
    return _setup_neo4j_core(force)


@router.get("/setup-neo4j")
async def setup_neo4j_get(force: bool = False):
    """GET alias for setup; safe by default (no drop)."""
    return _setup_neo4j_core(force)

@router.post("/process-blob")
async def process_blob_document(request: BlobUrlRequest):
    """
    Process a blob URL document according to exact specifications:
    1. Download PDF and extract text
    2. Split into 1000-char chunks with 150-char overlap
    3. Generate embeddings for each chunk
    4. Store in Neo4j with Document->Chunk structure
    """
    try:
        print(f"🚀 Processing blob document: {request.blob_url}")
        
        # Extract filename from URL
        parsed_url = urlparse(request.blob_url)
        file_name = parsed_url.path.split('/')[-1] or "unknown_document.pdf"
        
        # Step 1: Download PDF and extract text
        full_text = extract_pdf_text(request.blob_url)
        if not full_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF")

        # Step 2: Split text into token-based chunks to reduce 429/quota risk
        chunks = chunk_text_tokens(full_text)
        
        # Step 3: Generate embeddings for each chunk with robust retry logic
        print(f"🔢 Generating embeddings for {len(chunks)} chunks...")
        chunks_with_embeddings = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Try to generate embedding with multiple attempts for this specific chunk
            embedding = None
            chunk_attempts = 0
            max_chunk_attempts = 5
            
            while embedding is None and chunk_attempts < max_chunk_attempts:
                chunk_attempts += 1
                print(f"  Attempt {chunk_attempts}/{max_chunk_attempts} for chunk {i+1}")
                
                # Generate embeddings with retry logic
                embedding = generate_embeddings(chunk['text'])
                
                if embedding is None and chunk_attempts < max_chunk_attempts:
                    # Wait longer for chunk-level retries
                    wait_time = 10 + (chunk_attempts * 5)  # 15s, 20s for chunk retries
                    print(f"  ❌ Chunk {i+1} failed attempt {chunk_attempts}. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
            
            if embedding is None:
                print(f"⚠️ Failed to generate embedding for chunk {i+1} after {max_chunk_attempts} attempts")
            else:
                print(f"✅ Successfully processed chunk {i+1}")
            
            chunks_with_embeddings.append({
                'text': chunk['text'],
                'chunk_index': chunk['chunk_index'],
                'start_pos': chunk['start_pos'],
                'end_pos': chunk['end_pos'],
                'embedding': embedding
            })
            
            # No fixed delay between chunks; backoff occurs only on failures/rate limits
        
        # Step 4: Store in Neo4j with proper graph structure
        success = store_document_with_chunks(file_name, full_text, request.blob_url, chunks_with_embeddings)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store document in Neo4j")
        
        # Count successful embeddings
        successful_embeddings = sum(1 for chunk in chunks_with_embeddings if chunk['embedding'] is not None)
        
        return {
            "status": "success",
            "message": "Document processed successfully",
            "filename": file_name,
            "full_text_length": len(full_text),
            "total_chunks": len(chunks),
            "chunks_with_embeddings": successful_embeddings,
            "blob_url": request.blob_url
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing blob document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@router.post("/chat")
async def chat_with_documents(request: ChatRequest):
    """
    Chat with documents using chunk-based vector similarity search
    """
    try:
        print(f"🤖 Processing question: {request.question}")

        # Step 1: Generate embedding for user question
        question_embedding = generate_embeddings(request.question)
        if not question_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate question embedding")

        # Step 2: Vector similarity search against chunk_embeddings index
        with neo4j_driver.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes('chunk_embeddings', $limit, $question_embedding)
                YIELD node, score
                MATCH (d:Document)-[:HAS_CHUNK]->(node)
                WHERE d.filename IS NOT NULL AND node.text IS NOT NULL
                RETURN 
                    node.text as chunk_text,
                    node.chunk_index as chunk_index,
                    d.filename as document_name,
                    score as similarity_score
                ORDER BY score DESC
                """,
                limit=request.limit,
                question_embedding=question_embedding,
            )

            # Step 3: Construct context from retrieved chunks
            retrieved_chunks = []
            context_parts = []
            for record in result:
                chunk_info = {
                    "chunk_text": record["chunk_text"],
                    "chunk_index": record["chunk_index"],
                    "document_name": record["document_name"],
                    "similarity_score": record["similarity_score"],
                }
                retrieved_chunks.append(chunk_info)
                context_parts.append(
                    f"[From {record['document_name']}, Chunk {record['chunk_index']}]: {record['chunk_text']}"
                )

        if not retrieved_chunks:
            return {
                "answer": "I couldn't find any relevant information in your documents to answer that question.",
                "sources": [],
                "chunks_used": 0,
            }

        # Step 4: Create detailed prompt for Gemini
        context_string = "\n\n".join(context_parts)
        prompt = f"""Based on the following document chunks, please provide a comprehensive answer to the user's question.

DOCUMENT CONTEXT:
{context_string}

USER QUESTION: {request.question}

Please provide a detailed answer based on the document content above. If the information is not sufficient, please state what additional information would be helpful.

Output format rules:
- Provide a single plain English paragraph.
- Do not use lists, bullets, numbering, or markdown.
- Do not include headings or prefixes.
- Avoid line breaks; keep it as one line if possible."""

        # Step 5: Generate response using Gemini
        chat_model = get_chat_model()
        if not chat_model:
            raise HTTPException(status_code=500, detail="Chat model not available")

        response = chat_model.generate_content(prompt)
        answer_text = _clean_answer_text(getattr(response, "text", "") or "")

        # Prepare source information
        sources = []
        for chunk in retrieved_chunks:
            source_info = {
                "document_name": chunk["document_name"],
                "chunk_index": chunk["chunk_index"],
                "similarity_score": round(chunk["similarity_score"], 4),
            }
            if source_info not in sources:  # Avoid duplicates
                sources.append(source_info)

        return {
            "answer": answer_text,
            "sources": sources,
            "chunks_used": len(retrieved_chunks),
            "question": request.question,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@router.get("/documents")
async def list_documents():
    """List all processed documents with chunk information"""
    try:
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                RETURN 
                    d.filename as filename,
                    d.blob_url as blob_url,
                    d.full_text_length as text_length,
                    d.total_chunks as total_chunks,
                    count(c) as chunks_stored,
                    d.created_at as created_at
                ORDER BY d.created_at DESC
            """)
            
            documents = []
            for record in result:
                documents.append({
                    "filename": record["filename"],
                    "blob_url": record["blob_url"],
                    "text_length": record["text_length"],
                    "total_chunks": record["total_chunks"],
                    "chunks_stored": record["chunks_stored"],
                    "created_at": str(record["created_at"])
                })
        
        return {
            "status": "success",
            "documents": documents,
            "total_documents": len(documents)
        }
        
    except Exception as e:
        print(f"❌ Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hackrx/run")
async def hackrx_run(request: HackRxRunRequest, _auth: bool = Depends(require_bearer_token)):
    """
    HackRx endpoint: Process a document and answer multiple questions
    """
    try:
        print(f"🚀 HackRx Run: Processing document and {len(request.questions)} questions")

        # Step 1: Process the document first
        blob_request = BlobUrlRequest(blob_url=request.documents)
        process_result = await process_blob_document(blob_request)

        if process_result.get("status") != "success":
            raise HTTPException(status_code=500, detail="Failed to process document")

        print(f"✅ Document processed: {process_result['filename']}")

        # Step 2: Answer all questions
        answers: list[str] = []
        for i, question in enumerate(request.questions):
            print(f"🤖 Answering question {i+1}/{len(request.questions)}: {question[:50]}...")
            try:
                chat_request = ChatRequest(question=question, limit=3)
                chat_result = await chat_with_documents(chat_request)
                answer = chat_result.get("answer", "Could not generate answer for this question.")
                answers.append(_clean_answer_text(answer))
                if i < len(request.questions) - 1:
                    time.sleep(0.5)
            except Exception as e:
                print(f"❌ Error answering question {i+1}: {e}")
                answers.append(f"Error generating answer: {str(e)}")

        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in HackRx run: {e}")
        raise HTTPException(status_code=500, detail=f"HackRx run failed: {str(e)}")

@router.post("/reset-database")
async def reset_database():
    """Complete database reset - removes all documents and chunks"""
    try:
        with neo4j_driver.session() as session:
            # Remove all relationships first
            result1 = session.run("MATCH ()-[r:HAS_CHUNK]-() DELETE r RETURN count(r) as deleted")
            deleted_rels = result1.single()["deleted"]
            
            # Remove all chunks
            result2 = session.run("MATCH (c:Chunk) DELETE c RETURN count(c) as deleted")
            deleted_chunks = result2.single()["deleted"]
            
            # Remove all documents
            result3 = session.run("MATCH (d:Document) DELETE d RETURN count(d) as deleted")
            deleted_docs = result3.single()["deleted"]
            
            return {
                "status": "success",
                "message": f"Database reset: {deleted_docs} documents, {deleted_chunks} chunks, {deleted_rels} relationships removed"
            }
            
    except Exception as e:
        print(f"❌ Error during reset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup")
async def cleanup_database():
    """Clean up duplicate or corrupted data in Neo4j"""
    try:
        with neo4j_driver.session() as session:
            # First, remove relationships for documents without proper filenames
            result1 = session.run("""
                MATCH (d:Document)-[r:HAS_CHUNK]->(c:Chunk) 
                WHERE d.filename IS NULL 
                DELETE r, c 
                RETURN count(r) as deleted_relationships
            """)
            deleted_rels = result1.single()["deleted_relationships"]
            
            # Then remove documents without proper filenames
            result2 = session.run("MATCH (d:Document) WHERE d.filename IS NULL DELETE d RETURN count(d) as deleted")
            deleted_docs = result2.single()["deleted"]
            
            # Remove any remaining orphaned chunks
            result3 = session.run("""
                MATCH (c:Chunk) 
                WHERE NOT EXISTS((c)<-[:HAS_CHUNK]-(:Document))
                DELETE c 
                RETURN count(c) as deleted
            """)
            deleted_chunks = result3.single()["deleted"]
            
            return {
                "status": "success",
                "message": f"Cleanup completed: {deleted_docs} documents, {deleted_rels} relationships, and {deleted_chunks} orphaned chunks removed"
            }
            
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hackrx/run-async", response_model=HackRxJobStartResponse)
async def hackrx_run_async(request: HackRxRunRequest, background_tasks: BackgroundTasks):
    """Start HackRx run in the background and return a job id for progress polling."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "queued",
        "percent": 0,
        "stage": "queued",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    background_tasks.add_task(_run_hackrx_job, job_id, request.model_dump())
    return HackRxJobStartResponse(job_id=job_id, status_url=f"/blob/hackrx/status/{job_id}")

@router.get("/hackrx/status/{job_id}")
async def hackrx_status(job_id: str):
    """Get status/progress for a background HackRx job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Neo4j connection
        neo4j_driver.verify_connectivity()
        with _neo4j_session() as session:
            session.run("RETURN 1")
            idx = session.run("SHOW INDEXES YIELD name WHERE name = $name", name=INDEX_NAME).single()
            vector_index_status = "present" if idx else "absent"
        
        # Test embedding model
        embedding_model = get_embedding_model()
        
        return {
            "status": "healthy",
            "neo4j": "connected",
            "embedding_model": "available" if embedding_model else "unavailable",
            "vector_index": vector_index_status
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
