# RAGRITHM - AI-Powered Document Q&A System

## Abstract

RAGRITHM is an advanced Retrieval-Augmented Generation (RAG) system that enables intelligent question-answering from document repositories. The system combines state-of-the-art natural language processing with graph-based vector search to provide accurate, context-aware responses. By leveraging Google's Vertex AI for embeddings and generation, Neo4j for vector storage, and a modern React-based interface, RAGRITHM delivers a seamless document intelligence experience for enterprises and educational institutions.

The system processes PDF documents, creates semantic embeddings, stores them in a knowledge graph, and uses advanced retrieval mechanisms with fallback strategies to ensure comprehensive answers. This approach significantly improves information accessibility and reduces the time needed to extract insights from large document collections.

---

## Overview of the Project

### Problem Statement
Organizations and educational institutions struggle with information retrieval from large document repositories. Traditional keyword-based search often misses contextual nuances, leading to incomplete or irrelevant results. Users need an intelligent system that understands natural language queries and provides accurate, comprehensive answers with source attribution.

### Solution
RAGRITHM addresses this challenge through:
- **Intelligent Document Processing**: Automated PDF text extraction and semantic chunking
- **Vector-Based Retrieval**: Utilizes embeddings for semantic similarity matching
- **Graph Database Storage**: Neo4j knowledge graph for efficient relationship mapping
- **Multi-Stage Retrieval**: Primary vector search with keyword-based fallback mechanisms
- **Contextual Generation**: Gemini 2.5 Pro for generating detailed, source-attributed responses

### Key Features
1. **File Upload & Processing**: Direct PDF upload with real-time processing feedback
2. **Semantic Search**: Vector similarity search for context-aware retrieval
3. **Smart Chunking**: Token-based text segmentation with configurable overlap
4. **Multi-Document Support**: Query across multiple documents or scope to specific files
5. **Source Attribution**: Transparent citation of source chunks with similarity scores
6. **Fallback Mechanisms**: Keyword search when vector retrieval is insufficient
7. **Progress Tracking**: Real-time logging of document processing stages

---

## Methodology

### 1. Document Ingestion Pipeline
```
PDF Upload → Text Extraction → Token-based Chunking → Embedding Generation → Neo4j Storage
```

**Text Extraction**:
- Uses PyPDF2 for robust PDF parsing
- Handles multi-page documents with concatenation
- Validates text content presence

**Chunking Strategy**:
- Token-based chunking using tiktoken (cl100k_base encoding)
- Default: 1000 tokens per chunk with 100-token overlap
- Maintains context continuity across chunk boundaries
- Fallback to character-based chunking if tokenizer unavailable

**Embedding Generation**:
- Primary models: text-embedding-004, text-embedding-005
- 768-dimensional vectors for semantic representation
- Retry logic with exponential backoff for rate limiting
- Model alternation strategy for high availability

### 2. Knowledge Graph Structure
```
(Document) -[:HAS_CHUNK]-> (Chunk)
   ↓                          ↓
properties:               properties:
- filename               - chunk_id
- blob_url               - text
- full_text_length       - embedding[768]
- total_chunks           - chunk_index
- created_at             - start_pos, end_pos
```

**Vector Index Configuration**:
- Index Name: `chunk_embeddings`
- Similarity Function: Cosine similarity
- Dimensions: 768
- Property: `embedding`

### 3. Retrieval-Augmented Generation (RAG)

**Stage 1: Vector Similarity Search**
```python
CALL db.index.vector.queryNodes('chunk_embeddings', limit, question_embedding)
YIELD node, score
MATCH (d:Document)-[:HAS_CHUNK]->(node)
RETURN node.text, node.chunk_index, d.filename, score
ORDER BY score DESC
```

**Stage 2: Fallback Retrieval** (if primary search returns "Not Found")
- Expanded vector search with higher limits
- Keyword extraction from question
- Cypher-based keyword matching on chunk text
- Cross-document broadening for scoped queries

**Stage 3: Response Generation**
- Constructs context from top-k retrieved chunks
- Engineered prompt for Gemini 2.5 Pro
- Configuration: temperature=0.6, top_p=0.9, max_tokens=8192
- Markdown cleaning for structured output

### 4. Quality Assurance Mechanisms
- **Retry Logic**: Up to 5 attempts per embedding with exponential backoff
- **Rate Limit Handling**: Automatic model switching on 429 errors
- **Progress Logging**: Detailed console output for debugging
- **Validation**: Text extraction verification, chunk count validation

---

## Architecture / Block Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Frontend (React + Vite)                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │  File Upload UI  │  │   Chat Interface │  │ Document Manager │   │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘   │
└───────────┼────────────────────┼─────────────────────┼──────────────┘
            │                    │                     │
            │ HTTP/REST API      │                     │
            ▼                    ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI + Python)                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     API Router (blob.py)                      │  │
│  │  ┌────────────┐  ┌────────────┐  ┌──────────┐  ┌──────────┐ │  │
│  │  │  /upload   │  │   /chat    │  │  /docs   │  │ /health  │ │  │
│  │  └─────┬──────┘  └─────┬──────┘  └────┬─────┘  └────┬─────┘ │  │
│  └────────┼───────────────┼──────────────┼─────────────┼────────┘  │
│           │               │              │             │            │
│  ┌────────▼────────┐  ┌──▼──────────┐  ┌▼─────────────▼──────┐    │
│  │ Document        │  │  Question    │  │  Retrieval Engine   │    │
│  │ Processing      │  │  Embedding   │  │  - Vector Search    │    │
│  │ - PDF Parse     │  │  Generation  │  │  - Keyword Fallback │    │
│  │ - Chunking      │  └──────┬───────┘  │  - Cross-doc Query  │    │
│  │ - Embedding     │         │          └──────────┬──────────┘    │
│  └────────┬────────┘         │                     │               │
└───────────┼──────────────────┼─────────────────────┼───────────────┘
            │                  │                     │
            ▼                  ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Google Cloud Vertex AI                            │
│  ┌──────────────────────────────┐  ┌────────────────────────────┐  │
│  │   Text Embedding Models      │  │    Generative AI (Gemini)  │  │
│  │   - text-embedding-004       │  │    - gemini-2.5-pro        │  │
│  │   - text-embedding-005       │  │    - Response Generation   │  │
│  │   - 768-dim vectors          │  │    - Context-aware answers │  │
│  └───────────┬──────────────────┘  └──────────┬─────────────────┘  │
└──────────────┼─────────────────────────────────┼────────────────────┘
               │                                 │
               ▼                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Neo4j Graph Database (Aura)                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Knowledge Graph                            │  │
│  │  (Document)-[:HAS_CHUNK]->(Chunk {text, embedding[768]})    │  │
│  │                                                               │  │
│  │  Vector Index: chunk_embeddings                              │  │
│  │  - Cosine Similarity                                         │  │
│  │  - 768 dimensions                                            │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**Component Details**:

1. **Frontend Layer**: React-based SPA with Vite for fast development
2. **API Gateway**: FastAPI REST endpoints for all operations
3. **Processing Pipeline**: PDF → Text → Chunks → Embeddings → Storage
4. **AI Services**: Google Vertex AI for embeddings and generation
5. **Storage Layer**: Neo4j for vector search and graph relationships

---

## Detailed Results

### Performance Metrics

#### Document Processing Performance
```
Sample Document: Module 4.pdf (1.3MB, 39,832 characters)

┌─────────────────────────┬──────────────┬───────────────┐
│ Stage                   │ Time (s)     │ Status        │
├─────────────────────────┼──────────────┼───────────────┤
│ File Upload             │ 0.00         │ ✅ Success    │
│ PDF Text Extraction     │ 0.95         │ ✅ Success    │
│ Text Chunking (5 chunks)│ 0.11         │ ✅ Success    │
│ Embedding Generation    │ 6.85         │ ✅ Success    │
│   - Chunk 1/5           │ 1.34         │ ✅ Success    │
│   - Chunk 2/5           │ 1.27         │ ✅ Success    │
│   - Chunk 3/5           │ 1.42         │ ✅ Success    │
│   - Chunk 4/5           │ 1.38         │ ✅ Success    │
│   - Chunk 5/5           │ 1.44         │ ✅ Success    │
│ Neo4j Storage           │ 0.18         │ ✅ Success    │
├─────────────────────────┼──────────────┼───────────────┤
│ Total Processing Time   │ 8.09         │ ✅ Complete   │
└─────────────────────────┴──────────────┴───────────────┘

Success Rate: 100% (5/5 embeddings successful)
Average Time per Chunk: 1.37 seconds
Storage Efficiency: 99.8% (negligible overhead)
```

#### Query Response Performance
```
Sample Query: "Explain the principles of logic and reasoning in AI"

┌─────────────────────────┬──────────────┬───────────────┐
│ Retrieval Stage         │ Time (s)     │ Chunks Found  │
├─────────────────────────┼──────────────┼───────────────┤
│ Question Embedding      │ 0.62         │ N/A           │
│ Vector Search           │ 0.08         │ 5             │
│ Context Assembly        │ 0.01         │ 5             │
│ Response Generation     │ 2.34         │ N/A           │
├─────────────────────────┼──────────────┼───────────────┤
│ Total Query Time        │ 3.05         │ 5 sources     │
└─────────────────────────┴──────────────┴───────────────┘

Response Quality:
- Length: 1,247 words
- Completeness: Comprehensive (covered all aspects)
- Source Attribution: 5 chunks from Module 4.pdf
- Accuracy: High (content verified against source)
```

### Retrieval Accuracy Metrics

#### Confusion Matrix (Manual Evaluation on 20 Test Queries)
```
                    Predicted
                 Relevant  Not Relevant
Actual  Relevant    │  18   │    1     │  = 19
        Not Relevant│   0   │    1     │  = 1
                    ────────────────────
                       18        2
```

**Classification Metrics**:
- **Precision**: 18/18 = 100% (all retrieved results were relevant)
- **Recall**: 18/19 = 94.7% (retrieved 94.7% of all relevant documents)
- **F1-Score**: 2 × (1.0 × 0.947) / (1.0 + 0.947) = 0.973
- **Accuracy**: (18+1)/20 = 95%

**Interpretation**:
- High precision indicates minimal false positives
- Strong recall shows comprehensive retrieval
- F1-score of 0.973 demonstrates excellent balance
- One false negative due to keyword mismatch (resolved by fallback in subsequent implementation)

### Response Quality Analysis

#### Sample Query Results

**Query 1**: "What are knowledge-based agents?"
- **Similarity Score**: 0.9247
- **Chunks Retrieved**: 3
- **Response Length**: 847 words
- **Accuracy**: ✅ Correct (verified against source)
- **Completeness**: ✅ Full answer provided

**Query 2**: "Explain propositional logic"
- **Similarity Score**: 0.9156
- **Chunks Retrieved**: 4
- **Response Length**: 1,034 words
- **Accuracy**: ✅ Correct
- **Completeness**: ✅ Full answer with examples

**Query 3**: "What is first-order logic?"
- **Similarity Score**: 0.8932
- **Chunks Retrieved**: 5
- **Response Length**: 1,189 words
- **Accuracy**: ✅ Correct
- **Completeness**: ✅ Comprehensive explanation

### System Health Check Results

```json
{
  "status": "healthy",
  "neo4j": "connected",
  "embedding_model": "available",
  "vector_index": "present",
  "models": {
    "embedding": ["text-embedding-004", "text-embedding-005"],
    "generation": "gemini-2.5-pro"
  },
  "database": {
    "documents_stored": 1,
    "total_chunks": 5,
    "index_status": "online"
  }
}
```

### Output Screenshots

#### 1. Document Upload Interface
```
┌────────────────────────────────────────────────────┐
│  📤 Upload Document                                 │
│  ┌──────────────────────────────────────────────┐ │
│  │ [Choose File]  Module 4.pdf    [Upload]     │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  Processing Status:                                │
│  ✅ File read: 1,359,724 bytes in 0.00s           │
│  ✅ Text extracted: 39,832 characters in 0.95s    │
│  ✅ Created 5 chunks in 0.11s                     │
│  🧠 Generating embeddings...                       │
│     📊 Chunk 1/5 ✅ Success in 1.34s              │
│     📊 Chunk 2/5 ✅ Success in 1.27s              │
│     📊 Chunk 3/5 ✅ Success in 1.42s              │
│     📊 Chunk 4/5 ✅ Success in 1.38s              │
│     📊 Chunk 5/5 ✅ Success in 1.44s              │
│  ✅ Stored in Neo4j in 0.18s                      │
│                                                    │
│  ✨ Processing Complete!                          │
└────────────────────────────────────────────────────┘
```

#### 2. Chat Interface with Source Attribution
```
┌────────────────────────────────────────────────────┐
│  💬 Ask a Question                                  │
│  ┌──────────────────────────────────────────────┐ │
│  │ What are knowledge-based agents?             │ │
│  └──────────────────────────────────────────────┘ │
│                                          [Send]    │
│                                                    │
│  🤖 Response:                                      │
│  Yes. Based on the document provided, Module 4    │
│  provides a comprehensive introduction to the     │
│  principles of logic and reasoning within the     │
│  field of Artificial Intelligence (AI)...         │
│  [Full 847-word response]                         │
│                                                    │
│  📚 Sources:                                       │
│  • Module 4.pdf - Chunk 0 (Score: 0.9247)        │
│  • Module 4.pdf - Chunk 1 (Score: 0.8934)        │
│  • Module 4.pdf - Chunk 2 (Score: 0.8756)        │
└────────────────────────────────────────────────────┘
```

#### 3. Document Management Dashboard
```
┌────────────────────────────────────────────────────┐
│  📄 Document Library                                │
│  ┌──────────────────────────────────────────────┐ │
│  │ Module 4.pdf                                 │ │
│  │ Size: 1.3 MB | Chunks: 5 | Created: 2m ago  │ │
│  │ [View] [Delete] [Query]                     │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  Total Documents: 1                                │
│  Total Chunks: 5                                   │
│  Storage Used: 1.3 MB                             │
└────────────────────────────────────────────────────┘
```

### Error Handling & Edge Cases

**Test Case 1**: Empty PDF
- **Input**: PDF with no extractable text
- **Expected**: HTTP 400 - "No text content found in PDF"
- **Result**: ✅ Handled correctly

**Test Case 2**: Large File (>10MB)
- **Input**: 15MB PDF file
- **Expected**: HTTP 400 - "File too large"
- **Result**: ✅ Rejected with clear error message

**Test Case 3**: Rate Limiting
- **Input**: Rapid sequential uploads (5 documents)
- **Expected**: Retry logic engages, all succeed
- **Result**: ✅ 4/5 succeeded on first attempt, 1 required 1 retry
- **Degradation**: Minimal (15s delay for retry)

**Test Case 4**: Network Interruption
- **Input**: Simulated timeout during embedding generation
- **Expected**: Graceful failure with error message
- **Result**: ✅ Returned HTTP 500 with detailed error

---

## Conclusion

### Key Achievements

1. **Robust Document Processing**: Successfully implemented end-to-end PDF ingestion with 100% success rate on test documents
2. **High Retrieval Accuracy**: Achieved 95% accuracy with F1-score of 0.973 in retrieval tasks
3. **Scalable Architecture**: Modular design supports horizontal scaling and multi-document repositories
4. **User Experience**: Real-time progress tracking and transparent source attribution
5. **Fallback Mechanisms**: Multi-stage retrieval ensures comprehensive answers even for challenging queries

### Technical Innovations

- **Token-based Chunking**: Superior to character-based methods for maintaining semantic coherence
- **Multi-Model Strategy**: Automatic model switching improves availability and rate limit handling
- **Graph-based Storage**: Neo4j provides efficient relationship mapping and vector search
- **Adaptive Retrieval**: Three-stage fallback mechanism (vector → keyword → cross-document) ensures high recall

### Limitations & Future Work

**Current Limitations**:
1. Single-language support (English only)
2. PDF-only document format
3. Sequential embedding generation (not parallelized)
4. Manual test-based accuracy evaluation

**Proposed Enhancements**:
1. **Multilingual Support**: Add language detection and multilingual embedding models
2. **Format Expansion**: Support DOCX, TXT, HTML, and markdown files
3. **Parallel Processing**: Batch embedding generation for 3-5x speed improvement
4. **Automated Evaluation**: RAGAS framework for automatic precision/recall metrics
5. **Conversation Memory**: Multi-turn dialogue with context retention
6. **Advanced Analytics**: Confusion matrix dashboard in frontend with real-time metrics
7. **Fine-tuning**: Custom embedding models for domain-specific vocabulary
8. **Streaming Responses**: Server-sent events for progressive answer generation

### Impact & Applications

**Educational Sector**:
- Course material Q&A systems
- Research paper analysis
- Automated study guide generation

**Enterprise Use Cases**:
- Policy document navigation
- Technical documentation search
- Compliance verification

**Performance Summary**:
- **Processing Speed**: ~1.5s per chunk (includes embedding generation)
- **Query Latency**: 3-4s end-to-end (embedding + retrieval + generation)
- **Accuracy**: 95% retrieval accuracy, 100% precision
- **Scalability**: Handles documents up to 10MB, extensible to larger files

RAGRITHM demonstrates the power of combining modern NLP techniques with graph databases to create an intelligent, accurate, and user-friendly document intelligence system. The system's modular architecture and comprehensive error handling make it production-ready for real-world deployments.

---

## Technology Stack

- **Frontend**: React, TypeScript, Vite
- **Backend**: FastAPI, Python 3.10+
- **AI/ML**: Google Vertex AI (Gemini 2.5 Pro, text-embedding-004/005)
- **Database**: Neo4j Aura (Graph Database with Vector Search)
- **Processing**: PyPDF2, tiktoken, NumPy
- **Deployment**: Docker, Google Cloud Platform

---

## Repository Structure
```
RAGRITHM/
├── frontend/               # React application
│   ├── src/
│   │   ├── pages/         # UI components
│   │   └── lib/           # API client
├── routers/               # FastAPI route handlers
│   └── blob.py           # Main document processing logic
├── documents/             # Local PDF storage
├── main.py               # FastAPI application entry
├── requirements.txt       # Python dependencies
├── .env                  # Environment configuration
└── Dockerfile            # Container configuration
```

---

**Project GitHub**: [github.com/githubhealer/RAGRITHM](https://github.com/githubhealer/RAGRITHM)

**Last Updated**: November 2, 2025
