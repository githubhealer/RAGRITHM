export type Health = {
  status: string
  neo4j?: string
  embedding_model?: string
  vector_index?: string
  error?: string
}

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export type SimpleStatus = { status: string; message?: string }

export type ProcessBlobResponse = {
  status: string
  message: string
  filename: string
  full_text_length: number
  total_chunks: number
  chunks_with_embeddings: number
  blob_url: string
}

export type ChatResponse = {
  answer: string
  sources: { document_name: string; chunk_index: number; similarity_score: number }[]
  chunks_used: number
  question: string
}

export type DocumentInfo = {
  filename: string
  blob_url: string
  text_length: number
  total_chunks: number
  chunks_stored: number
  created_at: string
}

export type DocumentsResponse = {
  status: string
  documents: DocumentInfo[]
  total_documents: number
}

export type FileUploadResponse = {
  status: string
  message: string
  filename: string
  file_size: number
  full_text_length: number
  total_chunks: number
  chunks_with_embeddings: number
  file_url: string
}

export const api = {
  health: async (): Promise<Health> => json<Health>(await fetch(`${API_BASE}/blob/health`)),
  setupNeo4j: async (): Promise<SimpleStatus> => json<SimpleStatus>(await fetch(`${API_BASE}/blob/setup-neo4j`, { method: 'POST' })),
  processBlob: async (blob_url: string): Promise<ProcessBlobResponse> => json<ProcessBlobResponse>(await fetch(`${API_BASE}/blob/process-blob`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ blob_url })
  })),
  uploadFile: async (file: File): Promise<FileUploadResponse> => {
    const formData = new FormData()
    formData.append('file', file)
    const response = await fetch(`${API_BASE}/blob/upload-file`, {
      method: 'POST',
      body: formData
    })
    return json<FileUploadResponse>(response)
  },
  chat: async (question: string, limit = 5): Promise<ChatResponse> => json<ChatResponse>(await fetch(`${API_BASE}/blob/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, limit })
  })),
  documents: async (): Promise<DocumentsResponse> => json<DocumentsResponse>(await fetch(`${API_BASE}/blob/documents`)),
  resetDb: async (): Promise<SimpleStatus> => json<SimpleStatus>(await fetch(`${API_BASE}/blob/reset-database`, { method: 'POST' })),
  cleanup: async (): Promise<SimpleStatus> => json<SimpleStatus>(await fetch(`${API_BASE}/blob/cleanup`, { method: 'POST' }))
}
