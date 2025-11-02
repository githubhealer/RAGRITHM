# Document Processing System - Exact Specification Implementation

import requests
import PyPDF2
import io
from fastapi import APIRouter, HTTPException, Header, UploadFile, File
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
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "documents")  # local folder for PDFs

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

class BlobUrlRequest(BaseModel):
    blob_url: str

class ChatRequest(BaseModel):
    question: str
    limit: int = 5
    document_filename: str | None = None  # optional: restrict retrieval to a single processed document


def _neo4j_session():
    """Helper to open a session, honoring optional database name."""
    if NEO4J_DB:
        return neo4j_driver.session(database=NEO4J_DB)
    return neo4j_driver.session()


def _policy_scope_mismatch(question: str, doc_filename: str | None) -> bool:
    """Heuristic: decide if we should ignore document scoping for this question.

    Rationale: If the question references a policy different from the currently processed
    document (e.g., mentions another policy name), the scoped vector search will miss.

    Simple heuristic rules (cheap, no extra DB hits):
    1. If no doc filename -> cannot mismatch.
    2. Extract base name tokens from doc filename (strip extension, split on non-letters).
    3. If question contains the word 'policy' and none of the meaningful base tokens (len>=5)
       appear in the question (case-insensitive), treat as mismatch.
    4. Additionally, if question contains a multi-word phrase ending with 'policy' whose
       first word is not among doc tokens, treat as mismatch.
    This keeps it conservative; we only expand when clearly divergent.
    """
    if not doc_filename:
        return False
    ql = question.lower()
    if 'policy' not in ql:
        return False  # generic question likely fine within scope
    base = doc_filename.rsplit('.', 1)[0].lower()
    # Tokenize base filename
    base_tokens = [t for t in re.split(r"[^a-zA-Z]+", base) if len(t) >= 5]
    if base_tokens and any(bt in ql for bt in base_tokens):
        return False  # at least one significant token matches
    # Look for a phrase ending with 'policy'
    m = re.search(r"([a-z][a-z\s]{3,}?)policy", ql)
    if m:
        phrase = m.group(1).strip()
        # If phrase first token (>=4 chars) not in base tokens -> mismatch
        first = phrase.split()[0] if phrase else ''
        if len(first) >= 4 and first not in base_tokens:
            return True
    # Default: if we reached here and no overlap, treat as mismatch
    return True


def _clean_answer_text(text: str) -> str:
    """Normalize model output while preserving structure for elaborate responses."""
    if not text:
        return ""
    
    # Remove all markdown formatting
    cleaned = text
    
    # Remove bold markers (**text**)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    
    # Remove italic markers (*text*)
    cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)
    
    # Remove bullet point markers (* • -)
    cleaned = re.sub(r"^\s*[*•\-]\s*", "", cleaned, flags=re.MULTILINE)
    
    # Remove excessive bullet point markers in middle of lines
    cleaned = re.sub(r"\s*[*•\-]{2,}\s*", " ", cleaned)
    
    # Remove hash headers (# ## ###)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    
    # Remove backticks for code formatting
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    
    # Clean up excessive whitespace and line breaks
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r"\n ", "\n", cleaned)
    
    # Clean up ordered list markers - keep single ones but remove excessive
    cleaned = re.sub(r"(\d+\.)\s*(\d+\.)\s*", r"\1 ", cleaned)
    
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
        pass
    except Exception as e:
        pass

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
        pass
    except Exception as e:
        pass
    return None

def _init_vertex_if_needed():
    try:
        vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location=os.getenv("GOOGLE_CLOUD_REGION"))
    except Exception as e:
        pass


def _get_embedding_client(model_name: str) -> TextEmbeddingModel | None:
    global _embedding_clients
    if model_name in _embedding_clients:
        return _embedding_clients[model_name]
    try:
        _init_vertex_if_needed()
        client = TextEmbeddingModel.from_pretrained(model_name)
        _embedding_clients[model_name] = client
        return client
    except Exception:
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
    """Get or create chat model instance with deterministic config."""
    global _chat_model
    if _chat_model is None:
        try:
            vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location=os.getenv("GOOGLE_CLOUD_REGION"))
            models_to_try = ["gemini-2.5-pro"]
            for model_name in models_to_try:
                try:
                    _chat_model = GenerativeModel(model_name)
                    break
                except Exception:
                    continue
        except Exception:
            _chat_model = None
    return _chat_model


_GENERATION_CONFIG = {
    "temperature": float(os.getenv("GEN_TEMPERATURE", "0.6")),  # increased creativity for more elaborate responses
    "top_p": float(os.getenv("GEN_TOP_P", "0.9")),
    "top_k": int(os.getenv("GEN_TOP_K", "40")),
    "max_output_tokens": int(os.getenv("MAX_OUTPUT_TOKENS", "2048")),  # doubled for more elaborate responses
}

def _generate_answer(prompt: str) -> str:
    model = get_chat_model()
    if not model:
        return "Model unavailable to answer the question."
    try:
        response = model.generate_content(prompt, generation_config=_GENERATION_CONFIG)
        return _clean_answer_text(getattr(response, "text", "") or "")
    except Exception:
        return "Unable to generate answer due to a generation error."  # keep generic, deterministic

def extract_pdf_text(blob_url):
    """Extract text from PDF blob URL"""
    try:
        response = requests.get(blob_url, timeout=30)
        response.raise_for_status()
        
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text += page_text + "\n"
            
        return text.strip()
        
    except Exception as e:
        return None

def extract_pdf_text_from_path(file_path: str) -> str | None:
    """Extract text from a local PDF file path."""
    try:
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                text += (page_text or "") + "\n"
            return text.strip()
    except Exception:
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
        chunk_tokens = int(os.getenv("CHUNK_TOKENS", "2922"))
    if overlap_tokens is None:
        overlap_tokens = int(os.getenv("OVERLAP_TOKENS", "150"))

    enc = _get_tokenizer()
    if enc is None:
        # Fallback to character-based splitter
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


# --- Fallback retrieval helpers ---
_STOP_WORDS = {"what","is","the","for","and","are","any","with","under","plus","policy","plan","does","this","cover","of","to","a","an","in","on","by","be","it"}

def _extract_keywords(question: str, min_len: int = 4, max_keywords: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z]{3,}", question.lower())
    kws: list[str] = []
    for t in tokens:
        if t in _STOP_WORDS or len(t) < min_len:
            continue
        if t not in kws:
            kws.append(t)
        if len(kws) >= max_keywords:
            break
    return kws

def _keyword_search(session, keywords: list[str], doc: str | None, limit: int) -> list[dict[str, any]]:
    if not keywords:
        return []
    if doc:
        cypher = """
        UNWIND $keywords AS kw
        MATCH (d:Document {filename:$doc})-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.text CONTAINS kw
        RETURN c.text as chunk_text,
               c.chunk_index as chunk_index,
               d.filename as document_name,
               0.0 as similarity_score
        LIMIT $limit
        """
        params = {"keywords": keywords, "doc": doc, "limit": limit}
    else:
        cypher = """
        UNWIND $keywords AS kw
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
        WHERE c.text CONTAINS kw
        RETURN c.text as chunk_text,
               c.chunk_index as chunk_index,
               d.filename as document_name,
               0.0 as similarity_score
        LIMIT $limit
        """
        params = {"keywords": keywords, "limit": limit}
    result = session.run(cypher, **params)
    out: list[dict[str, any]] = []
    for r in result:
        out.append({
            "chunk_text": r["chunk_text"],
            "chunk_index": r["chunk_index"],
            "document_name": r["document_name"],
            "similarity_score": r["similarity_score"],
        })
    return out


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
                    continue  # move to next model without delay
                else:
                    continue

        # After a full round, backoff with jitter and rotate the start index
        if attempt < max_retries:
            base = 2 ** (attempt - 1)
            time.sleep(base + random.uniform(0, 0.5))
            start_idx = (start_idx + 1) % mcount
    return None


def generate_embeddings(text, max_retries=5):
    """Generate 768-dim embeddings with alternation and truncation/padding."""
    try:
        return embed_text(text, max_retries=max_retries)
    except Exception as e:
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

            

            # Now create Chunk nodes and relationships
            chunks_created = 0
            for chunk_data in chunks_with_embeddings:
                if chunk_data['embedding'] is None:
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

            
            return True
            
    except Exception as e:
        return False



def _setup_neo4j_core(force: bool = False):
    """Core logic to ensure the vector index exists. Returns a status dict."""
    try:
        with _neo4j_session() as session:
            if _index_exists(session):
                if not force:
                    return {"ok": True, "index": INDEX_NAME, "present": True, "message": "Index already exists"}
                # Drop then recreate
                session.run(f"DROP INDEX {INDEX_NAME} IF EXISTS").consume()

            # Determine dimensions via sample embedding
            dims_probe = DEFAULT_DIMS
            try:
                sample = generate_embeddings("vector index setup dimension probe")
                if sample and isinstance(sample, list):
                    dims_probe = len(sample)
                else:
                    pass
            except Exception as dim_err:
                pass

            dims_candidates = [dims_probe]
            if 768 not in dims_candidates:
                dims_candidates.append(768)

            last_err = None
            for dims in dims_candidates:
                method = _try_create_index(session, dims)
                if method and _index_exists(session):
                    return {"ok": True, "index": INDEX_NAME, "present": True, "created_with": method, "dims": dims}
                last_err = f"creation failed for dims={dims}"

            raise HTTPException(status_code=500, detail={"message": "Failed to create vector index", "last_error": last_err})
    except HTTPException:
        raise
    except Exception as e:
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
        print(f"\n{'='*60}")
        print(f"🚀 Starting document processing...")
        print(f"{'='*60}")
        
        # Extract filename from URL
        parsed_url = urlparse(request.blob_url)
        file_name = parsed_url.path.split('/')[-1] or "unknown_document.pdf"
        print(f"📄 Filename: {file_name}")
        
        # Step 1: Download PDF and extract text
        print(f"⬇️  Step 1: Downloading and extracting PDF text...")
        start_time = time.time()
        full_text = extract_pdf_text(request.blob_url)
        if not full_text:
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF")
        print(f"   ✅ Text extracted: {len(full_text)} characters in {time.time() - start_time:.2f}s")

        # Step 2: Split text into token-based chunks to reduce 429/quota risk
        print(f"✂️  Step 2: Splitting text into chunks...")
        start_time = time.time()
        chunks = chunk_text_tokens(full_text)
        print(f"   ✅ Created {len(chunks)} chunks in {time.time() - start_time:.2f}s")
        print(f"   ✅ Created {len(chunks)} chunks in {time.time() - start_time:.2f}s")
        
        # Step 3: Generate embeddings for each chunk with robust retry logic
        print(f"🧠 Step 3: Generating embeddings for {len(chunks)} chunks...")
        print(f"   ⚠️  This may take a while depending on API rate limits...")
        overall_start = time.time()
        
        chunks_with_embeddings = []
        successful_count = 0
        failed_count = 0
        
        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            print(f"   📊 Processing chunk {i+1}/{len(chunks)}...", end=" ", flush=True)
            
            # Try to generate embedding with multiple attempts for this specific chunk
            embedding = None
            chunk_attempts = 0
            max_chunk_attempts = 5
            
            while embedding is None and chunk_attempts < max_chunk_attempts:
                chunk_attempts += 1
                
                # Generate embeddings with retry logic
                embedding = generate_embeddings(chunk['text'])
                
                if embedding is None and chunk_attempts < max_chunk_attempts:
                    # Wait longer for chunk-level retries
                    wait_time = 10 + (chunk_attempts * 5)  # 15s, 20s for chunk retries
                    print(f"⚠️  Retry {chunk_attempts}, waiting {wait_time}s...", end=" ", flush=True)
                    time.sleep(wait_time)
            
            if embedding is None:
                failed_count += 1
                print(f"❌ FAILED after {time.time() - chunk_start:.2f}s")
            else:
                successful_count += 1
                print(f"✅ Success in {time.time() - chunk_start:.2f}s")
            
            chunks_with_embeddings.append({
                'text': chunk['text'],
                'chunk_index': chunk['chunk_index'],
                'start_pos': chunk['start_pos'],
                'end_pos': chunk['end_pos'],
                'embedding': embedding
            })
            
            # No fixed delay between chunks; backoff occurs only on failures/rate limits
        
        print(f"   ✅ Embeddings complete: {successful_count} successful, {failed_count} failed")
        print(f"   ⏱️  Total embedding time: {time.time() - overall_start:.2f}s")
        
        # Step 4: Store in Neo4j with proper graph structure
        print(f"💾 Step 4: Storing document and chunks in Neo4j...")
        start_time = time.time()
        success = store_document_with_chunks(file_name, full_text, request.blob_url, chunks_with_embeddings)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store document in Neo4j")
        
        print(f"   ✅ Stored in Neo4j in {time.time() - start_time:.2f}s")
        
        # Count successful embeddings
        successful_embeddings = sum(1 for chunk in chunks_with_embeddings if chunk['embedding'] is not None)
        
        print(f"\n{'='*60}")
        print(f"✨ PROCESSING COMPLETE!")
        print(f"   📄 Filename: {file_name}")
        print(f"   📊 Total chunks: {len(chunks)}")
        print(f"   ✅ Successful embeddings: {successful_embeddings}")
        print(f"   ❌ Failed embeddings: {len(chunks) - successful_embeddings}")
        print(f"{'='*60}\n")
        
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
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@router.post("/upload-file")
async def upload_file_document(file: UploadFile = File(...)):
    """
    Upload and process a PDF file directly through the interface.
    Accepts multipart file upload and processes it similar to blob processing.
    """
    try:
        print(f"\n{'='*60}")
        print(f"📤 Starting file upload processing...")
        print(f"{'='*60}")
        
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        print(f"📄 Filename: {file.filename}")
        
        # Read file content with size limit (10MB max)
        print(f"📥 Reading uploaded file...")
        start_time = time.time()
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB")
        
        print(f"   ✅ File read: {len(file_content)} bytes in {time.time() - start_time:.2f}s")
        
        if file.content_type and not file.content_type.startswith('application/pdf'):
            # Allow common PDF MIME types
            if file.content_type not in ['application/pdf', 'application/x-pdf', 'application/acrobat', 'applications/vnd.pdf', 'text/pdf', 'text/x-pdf']:
                raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed")
        
        # Extract text from PDF content
        print(f"📖 Extracting text from PDF...")
        start_time = time.time()
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            full_text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                full_text += (page_text or "") + "\n"
            
            full_text = full_text.strip()
            if not full_text:
                raise HTTPException(status_code=400, detail="No text content found in PDF")
                
        except Exception as pdf_error:
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(pdf_error)}")

        print(f"   ✅ Text extracted: {len(full_text)} characters in {time.time() - start_time:.2f}s")

        # Generate a clean filename
        file_name = file.filename or "uploaded_document.pdf"
        
        # Step 2: Split text into token-based chunks
        print(f"✂️  Splitting text into chunks...")
        start_time = time.time()
        chunks = chunk_text_tokens(full_text)
        print(f"   ✅ Created {len(chunks)} chunks in {time.time() - start_time:.2f}s")
        
        # Step 3: Generate embeddings for each chunk
        print(f"🧠 Generating embeddings for {len(chunks)} chunks...")
        print(f"   ⚠️  This may take a while depending on API rate limits...")
        overall_start = time.time()
        chunks_with_embeddings = []
        successful_count = 0
        failed_count = 0
        
        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            print(f"   📊 Processing chunk {i+1}/{len(chunks)}...", end=" ", flush=True)
            
            # Try to generate embedding with retry logic
            embedding = None
            chunk_attempts = 0
            max_chunk_attempts = 5
            
            while embedding is None and chunk_attempts < max_chunk_attempts:
                chunk_attempts += 1
                embedding = generate_embeddings(chunk['text'])
                
                if embedding is None and chunk_attempts < max_chunk_attempts:
                    wait_time = 10 + (chunk_attempts * 5)
                    print(f"⚠️  Retry {chunk_attempts}, waiting {wait_time}s...", end=" ", flush=True)
                    time.sleep(wait_time)
            
            if embedding is None:
                failed_count += 1
                print(f"❌ FAILED after {time.time() - chunk_start:.2f}s")
            else:
                successful_count += 1
                print(f"✅ Success in {time.time() - chunk_start:.2f}s")
            
            chunks_with_embeddings.append({
                'text': chunk['text'],
                'chunk_index': chunk['chunk_index'],
                'start_pos': chunk['start_pos'],
                'end_pos': chunk['end_pos'],
                'embedding': embedding
            })
        
        print(f"   ✅ Embeddings complete: {successful_count} successful, {failed_count} failed")
        print(f"   ⏱️  Total embedding time: {time.time() - overall_start:.2f}s")
        
        # Step 4: Store in Neo4j with file:// URL
        print(f"💾 Storing document and chunks in Neo4j...")
        start_time = time.time()
        file_url = f"file://uploaded/{file_name}"
        success = store_document_with_chunks(file_name, full_text, file_url, chunks_with_embeddings)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store document in Neo4j")
        
        print(f"   ✅ Stored in Neo4j in {time.time() - start_time:.2f}s")
        
        # Count successful embeddings
        successful_embeddings = sum(1 for chunk in chunks_with_embeddings if chunk['embedding'] is not None)
        
        print(f"\n{'='*60}")
        print(f"✨ UPLOAD PROCESSING COMPLETE!")
        print(f"   📄 Filename: {file_name}")
        print(f"   📊 Total chunks: {len(chunks)}")
        print(f"   ✅ Successful embeddings: {successful_embeddings}")
        print(f"   ❌ Failed embeddings: {len(chunks) - successful_embeddings}")
        print(f"{'='*60}\n")
        
        return {
            "status": "success",
            "message": "File uploaded and processed successfully",
            "filename": file_name,
            "file_size": len(file_content),
            "full_text_length": len(full_text),
            "total_chunks": len(chunks),
            "chunks_with_embeddings": successful_embeddings,
            "file_url": file_url
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload and process file: {str(e)}")

@router.post("/process-documents-folder")
async def process_documents_folder(force: bool = False, limit: int | None = None):
    """Ingest all local PDFs from the documents folder into Neo4j.

    - force=true will re-embed even if the document already exists in DB.
    - limit can restrict how many files to process in one call.
    """
    base_dir = os.path.abspath(DOCUMENTS_DIR)
    if not os.path.isdir(base_dir):
        raise HTTPException(status_code=400, detail=f"documents folder not found: {base_dir}")

    # Gather PDF files
    all_files = [f for f in os.listdir(base_dir) if f.lower().endswith(".pdf")]
    if limit is not None and limit > -1:
        all_files = all_files[:limit]

    processed = []
    skipped = []
    failed = []

    try:
        with neo4j_driver.session() as session:
            for fname in all_files:
                # Skip if already present (unless force)
                if not force:
                    rec = session.run(
                        """
                        MATCH (d:Document {filename: $filename})-[:HAS_CHUNK]->(c:Chunk)
                        RETURN count(c) as chunk_count
                        """,
                        filename=fname,
                    ).single()
                    if rec and rec["chunk_count"] and rec["chunk_count"] > 0:
                        skipped.append({"filename": fname, "reason": "already embedded"})
                        continue

                fpath = os.path.join(base_dir, fname)
                text = extract_pdf_text_from_path(fpath)
                if not text:
                    failed.append({"filename": fname, "reason": "text extraction failed"})
                    continue

                chunks = chunk_text_tokens(text)
                chunks_with_embeddings = []
                for ch in chunks:
                    emb = generate_embeddings(ch['text'])
                    chunks_with_embeddings.append({
                        'text': ch['text'],
                        'chunk_index': ch['chunk_index'],
                        'start_pos': ch['start_pos'],
                        'end_pos': ch['end_pos'],
                        'embedding': emb,
                    })

                ok = store_document_with_chunks(
                    file_name=fname,
                    full_text=text,
                    blob_url=f"file://{fpath}",
                    chunks_with_embeddings=chunks_with_embeddings,
                )
                if ok:
                    processed.append({
                        "filename": fname,
                        "total_chunks": len(chunks),
                        "embedded": sum(1 for c in chunks_with_embeddings if c['embedding'] is not None),
                    })
                else:
                    failed.append({"filename": fname, "reason": "neo4j storage failed"})

        return {
            "status": "success",
            "folder": base_dir,
            "processed_count": len(processed),
            "skipped_count": len(skipped),
            "failed_count": len(failed),
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Folder ingest failed: {str(e)}")

@router.post("/chat")
async def chat_with_documents(request: ChatRequest):
    """
    Chat with documents using chunk-based vector similarity search
    """
    try:

        # Step 1: Generate embedding for user question
        question_embedding = generate_embeddings(request.question)
        if not question_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate question embedding")

        # Step 2: Vector similarity search against chunk_embeddings index (optionally scoped to a single document)
        with neo4j_driver.session() as session:
            if request.document_filename:
                cypher = """
                CALL db.index.vector.queryNodes('chunk_embeddings', $raw_limit, $question_embedding)
                YIELD node, score
                MATCH (d:Document)-[:HAS_CHUNK]->(node)
                WHERE d.filename = $doc AND node.text IS NOT NULL
                RETURN node.text as chunk_text,
                       node.chunk_index as chunk_index,
                       d.filename as document_name,
                       score as similarity_score
                ORDER BY score DESC
                LIMIT $limit
                """
                params = {
                    "raw_limit": max(request.limit * 3, request.limit),  # overfetch then limit for resilience
                    "limit": request.limit,
                    "question_embedding": question_embedding,
                    "doc": request.document_filename,
                }
            else:
                cypher = """
                CALL db.index.vector.queryNodes('chunk_embeddings', $limit, $question_embedding)
                YIELD node, score
                MATCH (d:Document)-[:HAS_CHUNK]->(node)
                WHERE d.filename IS NOT NULL AND node.text IS NOT NULL
                RETURN node.text as chunk_text,
                       node.chunk_index as chunk_index,
                       d.filename as document_name,
                       score as similarity_score
                ORDER BY score DESC
                """
                params = {
                    "limit": request.limit,
                    "question_embedding": question_embedding,
                }
            result = session.run(cypher, **params)

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
        prompt = f"""You are a comprehensive document assistant with deep expertise in document analysis. Your task is to provide detailed, complete, and well-structured responses based on the document context provided.

CRITICAL INSTRUCTIONS:
1. Provide a COMPLETE and THOROUGH explanation (aim for 800-1200 words for complex questions)
2. Cover ALL relevant aspects of the question - do not cut off or truncate your response
3. Include specific details such as numbers, percentages, definitions, conditions, and examples when available
4. Explain background context and related concepts to give a full understanding
5. Structure your response with clear paragraphs and logical flow
6. Use the exact information from the documents - do not speculate or add information not present
7. If certain details are not available in the context, clearly state what information is missing
8. Continue explaining until you've fully addressed the question - DO NOT stop mid-sentence

The system will automatically add a Yes/No prefix, so focus on providing a comprehensive explanation without starting with Yes or No.

DOCUMENT CONTEXT:
{context_string}

USER QUESTION: {request.question}

Provide a detailed, comprehensive, and COMPLETE explanation that thoroughly addresses the question. Ensure your response is not cut off:"""

        # Step 5: Generate comprehensive response
        answer_text = _generate_answer(prompt)
        if not answer_text:
            answer_text = "Information not found in the provided document."
        # Clean and structure the response while preserving detail
        answer_text = _clean_answer_text(answer_text)
        if answer_text and not answer_text.endswith(('.', '!', '?')):
            answer_text += '.'

        # Prepend Yes/No based on availability
        if answer_text.lower().startswith("information not found"):
            answer_text = f"No. {answer_text}"
        else:
            answer_text = f"Yes. {answer_text}"

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

        # --- Fallback: broaden retrieval if answer came back as Not Found ---
        if answer_text.lower().startswith("no. information not found"):
            keywords = _extract_keywords(request.question)
            combined: list[dict[str, any]] = []
            seen: set[tuple[str,int]] = set()
            # Expanded vector search
            with neo4j_driver.session() as session:
                if request.document_filename:
                    cypher_fb = """
                    CALL db.index.vector.queryNodes('chunk_embeddings', $raw_limit, $qemb)
                    YIELD node, score
                    MATCH (d:Document)-[:HAS_CHUNK]->(node)
                    WHERE node.text IS NOT NULL AND d.filename = $doc
                    RETURN node.text as chunk_text,
                           node.chunk_index as chunk_index,
                           d.filename as document_name,
                           score as similarity_score
                    ORDER BY score DESC
                    LIMIT $limit
                    """
                    params_fb = {
                        "raw_limit": max(request.limit * 6, 36),
                        "limit": max(request.limit * 2, 16),
                        "qemb": question_embedding,
                        "doc": request.document_filename,
                    }
                else:
                    cypher_fb = """
                    CALL db.index.vector.queryNodes('chunk_embeddings', $raw_limit, $qemb)
                    YIELD node, score
                    MATCH (d:Document)-[:HAS_CHUNK]->(node)
                    WHERE node.text IS NOT NULL
                    RETURN node.text as chunk_text,
                           node.chunk_index as chunk_index,
                           d.filename as document_name,
                           score as similarity_score
                    ORDER BY score DESC
                    LIMIT $limit
                    """
                    params_fb = {
                        "raw_limit": max(request.limit * 6, 36),
                        "limit": max(request.limit * 2, 16),
                        "qemb": question_embedding,
                    }
                fb_result = session.run(cypher_fb, **params_fb)
                for rec in fb_result:
                    key = (rec["document_name"], rec["chunk_index"])
                    if key not in seen:
                        seen.add(key)
                        combined.append({
                            "chunk_text": rec["chunk_text"],
                            "chunk_index": rec["chunk_index"],
                            "document_name": rec["document_name"],
                            "similarity_score": rec["similarity_score"],
                        })
                # Keyword search (document scoped first)
                kw_chunks = _keyword_search(session, keywords, request.document_filename, 20)
                if not kw_chunks and not request.document_filename:
                    kw_chunks = _keyword_search(session, keywords, None, 20)
                for ch in kw_chunks:
                    key = (ch["document_name"], ch["chunk_index"])
                    if key not in seen:
                        seen.add(key)
                        combined.append(ch)
            # Add original retrieved chunks to priority front
            for ch in retrieved_chunks:
                key = (ch["document_name"], ch["chunk_index"])
                if key not in seen:
                    seen.add(key)
                    combined.insert(0, ch)
            if combined:
                context_parts_fb = [
                    f"[From {c['document_name']}, Chunk {c['chunk_index']}]: {c['chunk_text']}" for c in combined[:40]
                ]
                context_string_fb = "\n\n".join(context_parts_fb)
                prompt_fb = (
                    "You are a comprehensive document assistant with deep expertise in document analysis. "
                    "Provide detailed, complete, and well-structured responses based on the document context.\n\n"
                    "CRITICAL INSTRUCTIONS:\n"
                    "1. Provide a COMPLETE explanation (aim for 800-1200 words for complex topics)\n"
                    "2. Include specific details, numbers, percentages, conditions, and examples\n"
                    "3. Cover background context, definitions, and related concepts fully\n"
                    "4. Structure your response with clear paragraphs and logical flow\n"
                    "5. Use only facts from the provided context - do not speculate\n"
                    "6. If information is not available, state: Information not found in the provided document\n"
                    "7. Continue until you've FULLY addressed the question - do NOT stop mid-sentence\n"
                    "8. Do NOT add Yes or No yourself\n\n"
                    f"DOCUMENT CONTEXT:\n{context_string_fb}\n\n"
                    f"USER QUESTION: {request.question}\n\nDetailed and COMPLETE explanation:"
                )
                second = _generate_answer(prompt_fb)
                second = _clean_answer_text(second)
                if second and not second.lower().startswith("information not found"):
                    if not second.endswith((".","!","?")):
                        second += "."
                    answer_text = f"Yes. {second}"
                # Update sources to combined (top N)
                sources = []
                for ch in combined[:40]:
                    si = {
                        "document_name": ch["document_name"],
                        "chunk_index": ch["chunk_index"],
                        "similarity_score": round(ch["similarity_score"], 4),
                    }
                    if si not in sources:
                        sources.append(si)

        # Cross-document fallback: if still not found in a scoped search, try unscoped across all documents
        if (
            answer_text.lower().startswith("no. information not found")
            and request.document_filename
        ):
            with neo4j_driver.session() as session:
                # Broad vector + keyword search without document filter
                cypher_all = """
                CALL db.index.vector.queryNodes('chunk_embeddings', $raw_limit, $qemb)
                YIELD node, score
                MATCH (d:Document)-[:HAS_CHUNK]->(node)
                WHERE node.text IS NOT NULL
                RETURN node.text as chunk_text,
                       node.chunk_index as chunk_index,
                       d.filename as document_name,
                       score as similarity_score
                ORDER BY score DESC
                LIMIT $limit
                """
                params_all = {
                    "raw_limit": max(request.limit * 8, 48),
                    "limit": max(request.limit * 3, 24),
                    "qemb": question_embedding,
                }
                all_result = session.run(cypher_all, **params_all)
                combined_all: list[dict[str, any]] = []
                seen_all: set[tuple[str,int]] = set()
                for rec in all_result:
                    key = (rec["document_name"], rec["chunk_index"])
                    if key not in seen_all:
                        seen_all.add(key)
                        combined_all.append({
                            "chunk_text": rec["chunk_text"],
                            "chunk_index": rec["chunk_index"],
                            "document_name": rec["document_name"],
                            "similarity_score": rec["similarity_score"],
                        })
                # Keyword expansion unscoped
                kw_all = _keyword_search(session, _extract_keywords(request.question), None, 30)
                for ch in kw_all:
                    key = (ch["document_name"], ch["chunk_index"])
                    if key not in seen_all:
                        seen_all.add(key)
                        combined_all.append(ch)
            if combined_all:
                context_parts_all = [
                    f"[From {c['document_name']}, Chunk {c['chunk_index']}]: {c['chunk_text']}" for c in combined_all[:60]
                ]
                context_joined_all = "\n\n".join(context_parts_all)
                prompt_all = (
                    "You are a comprehensive document assistant with deep expertise in document analysis. "
                    "Provide detailed, complete, and well-structured responses based on the document context.\n\n"
                    "CRITICAL INSTRUCTIONS:\n"
                    "1. Provide a COMPLETE explanation (aim for 800-1200 words for complex topics)\n"
                    "2. Include specific details, numbers, percentages, conditions, and examples\n"
                    "3. Cover background context, definitions, and related concepts fully\n"
                    "4. Structure your response with clear paragraphs and logical flow\n"
                    "5. Use only facts from the provided context - do not speculate\n"
                    "6. If information is not available, state: Information not found in the provided document\n"
                    "7. Continue until you've FULLY addressed the question - do NOT stop mid-sentence\n"
                    "8. Do NOT add Yes or No yourself\n\n"
                    f"DOCUMENT CONTEXT:\n{context_joined_all}\n\n"
                    f"USER QUESTION: {request.question}\n\nDetailed and COMPLETE explanation:"
                )
                third = _generate_answer(prompt_all)
                third = _clean_answer_text(third)
                if third and not third.lower().startswith("information not found"):
                    if not third.endswith((".","!","?")):
                        third += "."
                    answer_text = f"Yes. {third}"
                    # Replace sources with unscoped ones
                    sources = []
                    for ch in combined_all[:60]:
                        si = {
                            "document_name": ch["document_name"],
                            "chunk_index": ch["chunk_index"],
                            "similarity_score": round(ch["similarity_score"], 4),
                        }
                        if si not in sources:
                            sources.append(si)

        return {
            "answer": answer_text,
            "sources": sources,
            "chunks_used": len(retrieved_chunks),
            "question": request.question,
            "document": request.document_filename,
        }

    except HTTPException:
        raise
    except Exception as e:
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
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))

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
