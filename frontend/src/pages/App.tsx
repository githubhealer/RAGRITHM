import { useEffect, useMemo, useState } from 'react'
import { api, DocumentsResponse } from '../lib/api'

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={{ 
      border: '1px solid #e5e7eb', 
      borderRadius: 12, 
      padding: 24, 
      marginBottom: 20,
      backgroundColor: '#fafafa',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
    }}>
      <h2 style={{ 
        marginTop: 0, 
        marginBottom: 16, 
        color: '#374151',
        fontSize: '1.25rem',
        fontWeight: '600'
      }}>{title}</h2>
      {children}
    </section>
  )
}

function AnswerCard({ answer, sources }: { answer: string; sources?: any[] }) {
  // Parse the answer to extract Yes/No and explanation
  const isYesNo = answer.startsWith('Yes.') || answer.startsWith('No.')
  const answerType = isYesNo ? answer.substring(0, 3) : null
  const explanation = isYesNo ? answer.substring(4).trim() : answer

  return (
    <div style={{
      background: 'white',
      border: `2px solid ${answerType === 'Yes' ? '#10b981' : answerType === 'No' ? '#ef4444' : '#6b7280'}`,
      borderRadius: 12,
      padding: 20,
      marginTop: 16
    }}>
      {answerType && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          marginBottom: 12,
          gap: 8
        }}>
          <span style={{
            backgroundColor: answerType === 'Yes' ? '#10b981' : '#ef4444',
            color: 'white',
            padding: '4px 12px',
            borderRadius: 20,
            fontSize: '0.875rem',
            fontWeight: '600'
          }}>
            {answerType}
          </span>
          <span style={{ color: '#6b7280', fontSize: '0.875rem' }}>
            {answerType === 'Yes' ? 'Information found' : 'Information not available'}
          </span>
        </div>
      )}
      
      <div style={{
        fontSize: '1rem',
        lineHeight: '1.6',
        color: '#374151',
        marginBottom: sources && sources.length > 0 ? 16 : 0
      }}>
        {explanation}
      </div>

      {sources && sources.length > 0 && (
        <details style={{ marginTop: 12 }}>
          <summary style={{ 
            cursor: 'pointer', 
            fontSize: '0.875rem', 
            color: '#6b7280',
            fontWeight: '500'
          }}>
            View Sources ({sources.length})
          </summary>
          <div style={{ marginTop: 8 }}>
            {sources.map((source, idx) => (
              <div key={idx} style={{
                backgroundColor: '#f3f4f6',
                padding: 8,
                borderRadius: 6,
                marginBottom: 4,
                fontSize: '0.75rem'
              }}>
                <strong>{source.document_name}</strong> (Chunk {source.chunk_index})
                {source.similarity_score && (
                  <span style={{ color: '#6b7280', marginLeft: 8 }}>
                    Score: {source.similarity_score.toFixed(3)}
                  </span>
                )}
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  )
}

function LoadingSpinner() {
  return (
    <div style={{
      display: 'inline-block',
      width: 16,
      height: 16,
      border: '2px solid #f3f3f3',
      borderTop: '2px solid #3498db',
      borderRadius: '50%',
      animation: 'spin 1s linear infinite',
      marginRight: 8
    }}>
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}

function ProgressBar({ progress, status }: { progress: number; status?: string }) {
  return (
    <div style={{ width: '100%', marginTop: 12 }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 8
      }}>
        <span style={{ fontSize: '0.875rem', color: '#374151', fontWeight: '500' }}>
          {status || 'Processing...'}
        </span>
        <span style={{ fontSize: '0.875rem', color: '#6b7280', fontWeight: '600' }}>
          {Math.round(progress)}%
        </span>
      </div>
      <div style={{
        width: '100%',
        backgroundColor: '#f3f4f6',
        borderRadius: 8,
        height: 8,
        overflow: 'hidden'
      }}>
        <div style={{
          width: `${progress}%`,
          backgroundColor: '#3b82f6',
          height: '100%',
          borderRadius: 8,
          transition: 'width 0.3s ease',
          background: progress === 100 
            ? 'linear-gradient(90deg, #10b981, #059669)' 
            : 'linear-gradient(90deg, #3b82f6, #2563eb)'
        }} />
      </div>
    </div>
  )
}

function Button({ 
  children, 
  onClick, 
  disabled = false, 
  variant = 'primary',
  loading = false 
}: { 
  children: React.ReactNode; 
  onClick?: () => void; 
  disabled?: boolean;
  variant?: 'primary' | 'secondary' | 'danger';
  loading?: boolean;
}) {
  const styles = {
    primary: { backgroundColor: '#3b82f6', color: 'white' },
    secondary: { backgroundColor: '#6b7280', color: 'white' },
    danger: { backgroundColor: '#ef4444', color: 'white' }
  }

  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      style={{
        ...styles[variant],
        padding: '10px 16px',
        border: 'none',
        borderRadius: 8,
        fontSize: '0.875rem',
        fontWeight: '500',
        cursor: disabled || loading ? 'not-allowed' : 'pointer',
        opacity: disabled || loading ? 0.6 : 1,
        display: 'inline-flex',
        alignItems: 'center',
        marginRight: 8,
        marginBottom: 8
      }}
    >
      {loading && <LoadingSpinner />}
      {children}
    </button>
  )
}

export default function App() {
  const [health, setHealth] = useState<any>(null)
  const [processing, setProcessing] = useState(false)
  const [blobUrl, setBlobUrl] = useState('')
  const [chatQ, setChatQ] = useState('')
  const [chatHistory, setChatHistory] = useState<{ question: string; answer: any; timestamp: string }[]>([])
  const [chatLoading, setChatLoading] = useState(false)
  const [docs, setDocs] = useState<any[]>([])
  const [folderLoading, setFolderLoading] = useState(false)
  const [fileUploading, setFileUploading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadStatus, setUploadStatus] = useState('')
  const [processingProgress, setProcessingProgress] = useState(0)
  const [processingStatus, setProcessingStatus] = useState('')
  const API = useMemo(() => api, [])

  useEffect(() => {
    API.health().then(setHealth).catch((err: unknown) => setHealth({ status: 'unhealthy', error: String(err) }))
    API.documents().then((r: DocumentsResponse) => setDocs(r.documents ?? [])).catch(() => {})
  }, [API])

  async function runSetup() {
    try {
      const r = await API.setupNeo4j()
      alert(r.message ?? 'Setup completed successfully!')
    } catch (e: any) {
      alert('Setup failed: ' + (e?.message ?? String(e)))
    }
  }

  async function doProcess() {
    setProcessing(true)
    setProcessingProgress(0)
    setProcessingStatus('Initializing...')
    
    try {
      // Simulate progress stages
      setProcessingProgress(10)
      setProcessingStatus('Downloading PDF...')
      
      // Add a small delay to show the progress
      await new Promise(resolve => setTimeout(resolve, 500))
      
      setProcessingProgress(30)
      setProcessingStatus('Extracting text from PDF...')
      
      const r = await API.processBlob(blobUrl)
      
      setProcessingProgress(70)
      setProcessingStatus('Generating embeddings...')
      
      await new Promise(resolve => setTimeout(resolve, 500))
      
      setProcessingProgress(90)
      setProcessingStatus('Storing in database...')
      
      await new Promise(resolve => setTimeout(resolve, 300))
      
      setProcessingProgress(100)
      setProcessingStatus('Complete!')
      
      alert(r.message ?? 'Document processed successfully!')
      const d = await API.documents()
      setDocs(d.documents ?? [])
      setBlobUrl('') // Clear input after success
      
      // Reset progress after a short delay
      setTimeout(() => {
        setProcessingProgress(0)
        setProcessingStatus('')
      }, 1000)
      
    } catch (e: any) {
      setProcessingProgress(0)
      setProcessingStatus('Error occurred')
      alert('Processing failed: ' + (e?.message ?? String(e)))
    } finally {
      setProcessing(false)
    }
  }

  async function doProcessFolder() {
    setFolderLoading(true)
    try {
      const r = await fetch('http://localhost:8000/blob/process-documents-folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      }).then(res => res.json())
      
      alert(`Folder processing complete!\nProcessed: ${r.processed_count}\nSkipped: ${r.skipped_count}\nFailed: ${r.failed_count}`)
      const d = await API.documents()
      setDocs(d.documents ?? [])
    } catch (e: any) {
      alert('Folder processing failed: ' + (e?.message ?? String(e)))
    } finally {
      setFolderLoading(false)
    }
  }

  async function handleFileUpload(file: File) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      alert('Please select a PDF file')
      return
    }

    setFileUploading(true)
    setUploadProgress(0)
    setUploadStatus('Preparing file...')
    
    try {
      setUploadProgress(5)
      setUploadStatus('Validating file...')
      
      const formData = new FormData()
      formData.append('file', file)

      setUploadProgress(15)
      setUploadStatus('Uploading file...')

      // Simulate upload progress
      const uploadInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev < 40) return prev + 5
          return prev
        })
      }, 100)

      const result = await API.uploadFile(file)

      clearInterval(uploadInterval)
      setUploadProgress(50)
      setUploadStatus('Processing PDF...')

      setUploadProgress(70)
      setUploadStatus('Extracting text and generating embeddings...')
      
      setUploadProgress(90)
      setUploadStatus('Storing in database...')
      
      // Small delay to show final progress
      await new Promise(resolve => setTimeout(resolve, 500))
      
      setUploadProgress(100)
      setUploadStatus('Upload complete!')

      alert(`File uploaded successfully!\nFilename: ${result.filename}\nChunks: ${result.chunks_with_embeddings}/${result.total_chunks}`)
      
      // Refresh documents list
      const d = await API.documents()
      setDocs(d.documents ?? [])
      
      // Reset progress after a short delay
      setTimeout(() => {
        setUploadProgress(0)
        setUploadStatus('')
      }, 1000)
      
    } catch (e: any) {
      setUploadProgress(0)
      setUploadStatus('Upload failed')
      alert('File upload failed: ' + (e?.message ?? String(e)))
    } finally {
      setFileUploading(false)
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(false)
    
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFileUpload(files[0])
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(true)
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault()
    setDragOver(false)
  }

  async function doChat() {
    if (!chatQ.trim()) return
    
    setChatLoading(true)
    try {
      const r = await API.chat(chatQ, 5)
      const newChatEntry = {
        question: chatQ,
        answer: r,
        timestamp: new Date().toLocaleString()
      }
      setChatHistory(prev => [...prev, newChatEntry])
      setChatQ('') // Clear the input after successful submission
    } catch (e: any) {
      const errorEntry = {
        question: chatQ,
        answer: { answer: 'Error: ' + (e?.message ?? String(e)) },
        timestamp: new Date().toLocaleString()
      }
      setChatHistory(prev => [...prev, errorEntry])
    } finally {
      setChatLoading(false)
    }
  }

  async function doReset() {
    if (!confirm('⚠️ This will permanently delete all documents and embeddings. Are you sure?')) return
    try {
      await API.resetDb()
      const d = await API.documents()
      setDocs(d.documents ?? [])
      alert('Database reset successfully!')
    } catch (e: any) {
      alert('Reset failed: ' + (e?.message ?? String(e)))
    }
  }

  async function doCleanup() {
    try {
      const r = await API.cleanup()
      alert(r.message ?? 'Cleanup completed!')
    } catch (e: any) {
      alert('Cleanup failed: ' + (e?.message ?? String(e)))
    }
  }

  function clearChatHistory() {
    setChatHistory([])
  }

  return (
    <div style={{ 
      fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif', 
      maxWidth: 1200, 
      margin: '0 auto', 
      padding: 24,
      backgroundColor: '#f8fafc',
      minHeight: '100vh'
    }}>
      <header style={{ 
        textAlign: 'center', 
        marginBottom: 32,
        padding: 24,
        backgroundColor: 'white',
        borderRadius: 16,
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
      }}>
        <h1 style={{ 
          margin: 0, 
          fontSize: '2.5rem', 
          color: '#1f2937',
          marginBottom: 8
        }}>🤖 RAGRITHM</h1>
        <p style={{ 
          color: '#6b7280', 
          fontSize: '1.125rem',
          margin: 0
        }}>
          AI-Powered Document Processing & Question Answering System
        </p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: 24 }}>
        <Section title="🏥 System Health">
          <div style={{
            background: health?.status === 'healthy' ? '#dcfce7' : '#fef2f2',
            border: `1px solid ${health?.status === 'healthy' ? '#16a34a' : '#dc2626'}`,
            borderRadius: 8,
            padding: 12,
            marginBottom: 16
          }}>
            <div style={{ 
              fontSize: '0.875rem', 
              fontWeight: '600',
              color: health?.status === 'healthy' ? '#15803d' : '#dc2626'
            }}>
              Status: {health?.status === 'healthy' ? '✅ System Healthy' : '❌ System Issues'}
            </div>
            {health?.neo4j && (
              <div style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: 4 }}>
                Neo4j: {health.neo4j} | Vector Index: {health.vector_index} | Embedding: {health.embedding_model}
              </div>
            )}
          </div>
          <Button onClick={runSetup} variant="secondary">
            🔧 Setup Neo4j Vector Index
          </Button>
        </Section>

        <Section title="📄 Process Documents">
          <div style={{ marginBottom: 20 }}>
            <h3 style={{ fontSize: '1rem', marginBottom: 8, color: '#374151' }}>From URL</h3>
            <input
              placeholder="https://example.com/document.pdf"
              value={blobUrl}
              onChange={(e) => setBlobUrl(e.target.value)}
              style={{ 
                width: '100%', 
                padding: 12, 
                marginBottom: 12,
                border: '1px solid #d1d5db',
                borderRadius: 8,
                fontSize: '0.875rem'
              }}
            />
            <Button 
              disabled={!blobUrl.trim()} 
              onClick={doProcess}
              loading={processing}
            >
              🔄 Process from URL
            </Button>
            {processing && processingProgress > 0 && (
              <ProgressBar progress={processingProgress} status={processingStatus} />
            )}
          </div>

          <div style={{ marginBottom: 20 }}>
            <h3 style={{ fontSize: '1rem', marginBottom: 8, color: '#374151' }}>Upload File</h3>
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              style={{
                border: `2px dashed ${dragOver ? '#3b82f6' : '#d1d5db'}`,
                borderRadius: 12,
                padding: 32,
                textAlign: 'center',
                backgroundColor: dragOver ? '#eff6ff' : 'white',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                marginBottom: 12
              }}
              onClick={() => {
                const input = document.createElement('input')
                input.type = 'file'
                input.accept = '.pdf'
                input.onchange = (e) => {
                  const file = (e.target as HTMLInputElement).files?.[0]
                  if (file) handleFileUpload(file)
                }
                input.click()
              }}
            >
              {fileUploading ? (
                <div>
                  <LoadingSpinner />
                  <div style={{ color: '#6b7280', fontSize: '0.875rem' }}>
                    Uploading and processing...
                  </div>
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: '2rem', marginBottom: 8 }}>📤</div>
                  <div style={{ fontSize: '1rem', fontWeight: '600', marginBottom: 4, color: '#374151' }}>
                    Drop PDF file here or click to browse
                  </div>
                  <div style={{ fontSize: '0.875rem', color: '#6b7280' }}>
                    Supports PDF files up to 10MB
                  </div>
                </div>
              )}
            </div>
            {fileUploading && uploadProgress > 0 && (
              <ProgressBar progress={uploadProgress} status={uploadStatus} />
            )}
          </div>


        </Section>
      </div>

      <Section title="💬 Ask Questions">
        <div style={{ marginBottom: 16 }}>
          <textarea
            placeholder="Ask any question about your documents... (e.g., What is the grace period for premium payment?)"
            value={chatQ}
            onChange={(e) => setChatQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                doChat()
              }
            }}
            style={{ 
              width: '100%', 
              padding: 16, 
              marginBottom: 16, 
              minHeight: 100,
              border: '1px solid #d1d5db',
              borderRadius: 8,
              fontSize: '0.875rem',
              resize: 'vertical'
            }}
          />
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <Button 
              disabled={!chatQ.trim()} 
              onClick={doChat}
              loading={chatLoading}
            >
              🤔 Ask Question
            </Button>
            {chatHistory.length > 0 && (
              <Button 
                onClick={clearChatHistory}
                variant="secondary"
              >
                🗑️ Clear History
              </Button>
            )}
          </div>
        </div>

        {/* Chat History */}
        {chatHistory.length > 0 && (
          <div style={{ marginTop: 20 }}>
            <h3 style={{ fontSize: '1rem', marginBottom: 16, color: '#374151' }}>
              💬 Chat History ({chatHistory.length} questions)
            </h3>
            <div style={{ 
              maxHeight: 400, 
              overflowY: 'auto',
              border: '1px solid #e5e7eb',
              borderRadius: 8,
              backgroundColor: 'white'
            }}>
              {chatHistory.map((chat, idx) => (
                <div key={idx} style={{
                  padding: 16,
                  borderBottom: idx < chatHistory.length - 1 ? '1px solid #f3f4f6' : 'none'
                }}>
                  <div style={{
                    fontSize: '0.875rem',
                    fontWeight: '600',
                    color: '#374151',
                    marginBottom: 8,
                    padding: 8,
                    backgroundColor: '#f8fafc',
                    borderRadius: 6,
                    border: '1px solid #e5e7eb'
                  }}>
                    <span style={{ color: '#6b7280' }}>Q{idx + 1}:</span> {chat.question}
                    <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginTop: 4 }}>
                      {chat.timestamp}
                    </div>
                  </div>
                  <AnswerCard answer={chat.answer.answer} sources={chat.answer.sources} />
                </div>
              )).reverse()}
            </div>
          </div>
        )}

        {chatHistory.length === 0 && (
          <div style={{
            textAlign: 'center',
            padding: 40,
            color: '#6b7280',
            backgroundColor: 'white',
            borderRadius: 8,
            border: '1px solid #e5e7eb',
            marginTop: 16
          }}>
            💭 No questions asked yet. Start a conversation by asking about your documents!
          </div>
        )}
      </Section>

      <Section title="📚 Document Library">
        <div style={{ marginBottom: 16 }}>
          <strong>Total Documents: {docs.length}</strong>
        </div>
        
        {docs.length > 0 ? (
          <div style={{ 
            maxHeight: 300, 
            overflowY: 'auto',
            border: '1px solid #e5e7eb',
            borderRadius: 8
          }}>
            {docs.map((doc, idx) => (
              <div key={idx} style={{
                padding: 12,
                borderBottom: idx < docs.length - 1 ? '1px solid #f3f4f6' : 'none',
                backgroundColor: 'white'
              }}>
                <div style={{ fontWeight: '600', fontSize: '0.875rem', marginBottom: 4 }}>
                  📄 {doc.filename}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>
                  Chunks: {doc.chunks_stored}/{doc.total_chunks} | 
                  Size: {doc.text_length?.toLocaleString()} chars |
                  Created: {new Date(doc.created_at).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div style={{
            textAlign: 'center',
            padding: 40,
            color: '#6b7280',
            backgroundColor: 'white',
            borderRadius: 8,
            border: '1px solid #e5e7eb'
          }}>
            📭 No documents uploaded yet. Process some documents to get started!
          </div>
        )}
        
        <div style={{ marginTop: 16, display: 'flex', gap: 8 }}>
          <Button onClick={doReset} variant="danger">
            🗑️ Reset Database
          </Button>
          <Button onClick={doCleanup} variant="secondary">
            🧹 Cleanup
          </Button>
        </div>
      </Section>

      <footer style={{ 
        textAlign: 'center',
        color: '#9ca3af', 
        marginTop: 32,
        padding: 16,
        fontSize: '0.875rem'
      }}>
        Backend API: <code style={{ backgroundColor: '#f3f4f6', padding: '2px 6px', borderRadius: 4 }}>
          http://localhost:8000
        </code>
      </footer>
    </div>
  )
}
