import { useEffect, useMemo, useState } from 'react'
import { api, DocumentsResponse } from '../lib/api'

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <h2 style={{ marginTop: 0 }}>{title}</h2>
      {children}
    </section>
  )
}

export default function App() {
  const [health, setHealth] = useState<any>(null)
  const [processing, setProcessing] = useState(false)
  const [blobUrl, setBlobUrl] = useState('')
  const [chatQ, setChatQ] = useState('')
  const [chatA, setChatA] = useState<string | null>(null)
  const [docs, setDocs] = useState<any[]>([])
  const [hackDoc, setHackDoc] = useState('')
  const [hackQs, setHackQs] = useState('What is the policy number?\nWhat is the premium?')
  const [hackOut, setHackOut] = useState<any>(null)
  const API = useMemo(() => api, [])

  useEffect(() => {
    API.health().then(setHealth).catch((err: unknown) => setHealth({ status: 'unhealthy', error: String(err) }))
    API.documents().then((r: DocumentsResponse) => setDocs(r.documents ?? [])).catch(() => {})
  }, [API])

  async function runSetup() {
  const r = await API.setupNeo4j()
  alert(r.message ?? 'Setup done')
  }

  async function doProcess() {
    setProcessing(true)
    try {
  const r = await API.processBlob(blobUrl)
  alert(r.message ?? 'Processed')
  const d = await API.documents()
  setDocs(d.documents ?? [])
    } catch (e: any) {
      alert(e?.message ?? String(e))
    } finally {
      setProcessing(false)
    }
  }

  async function doChat() {
    setChatA(null)
  const r = await API.chat(chatQ, 5)
  setChatA(r.answer)
  }

  async function doReset() {
    if (!confirm('Are you sure to reset database?')) return
  await API.resetDb()
  const d = await API.documents()
  setDocs(d.documents ?? [])
  }

  async function doCleanup() {
  const r = await API.cleanup()
  alert(r.message ?? 'Cleanup done')
  }

  async function runHackRx() {
    setHackOut(null)
    const questions = hackQs.split('\n').map(s => s.trim()).filter(Boolean)
    const r = await API.hackrxRun(hackDoc, questions)
    setHackOut(r)
  }

  return (
    <div style={{ fontFamily: 'system-ui, Arial, sans-serif', maxWidth: 1000, margin: '24px auto', padding: 16 }}>
      <h1>RAGRITHM Console</h1>
      <p style={{ color: '#6b7280' }}>Simple UI to process PDFs, chat, and run the HackRx flow.</p>

      <Section title="Health">
        <pre style={{ background: '#f9fafb', padding: 12, borderRadius: 6, overflowX: 'auto' }}>
          {JSON.stringify(health, null, 2)}
        </pre>
        <button onClick={runSetup}>Setup Neo4j Vector Index</button>
      </Section>

      <Section title="Process PDF (Blob URL)">
        <input
          placeholder="https://.../file.pdf"
          value={blobUrl}
          onChange={(e) => setBlobUrl(e.target.value)}
          style={{ width: '100%', padding: 8, marginBottom: 8 }}
        />
        <button disabled={!blobUrl || processing} onClick={doProcess}>
          {processing ? 'Processing...' : 'Process'}
        </button>
      </Section>

      <Section title="Chat with Documents">
        <textarea
          placeholder="Ask a question"
          value={chatQ}
          onChange={(e) => setChatQ(e.target.value)}
          style={{ width: '100%', padding: 8, marginBottom: 8, minHeight: 80 }}
        />
        <button disabled={!chatQ} onClick={doChat}>Ask</button>
        {chatA && (
          <pre style={{ background: '#f9fafb', padding: 12, borderRadius: 6, marginTop: 8 }}>{chatA}</pre>
        )}
      </Section>

      <Section title="Documents in Neo4j">
        <pre style={{ background: '#f9fafb', padding: 12, borderRadius: 6, overflowX: 'auto' }}>
          {JSON.stringify(docs, null, 2)}
        </pre>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={doReset}>Reset DB</button>
          <button onClick={doCleanup}>Cleanup</button>
        </div>
      </Section>

      <Section title="HackRx Runner">
        <input
          placeholder="Document URL (PDF)"
          value={hackDoc}
          onChange={(e) => setHackDoc(e.target.value)}
          style={{ width: '100%', padding: 8, marginBottom: 8 }}
        />
        <textarea
          placeholder={"One question per line"}
          value={hackQs}
          onChange={(e) => setHackQs(e.target.value)}
          style={{ width: '100%', padding: 8, marginBottom: 8, minHeight: 80 }}
        />
        <button disabled={!hackDoc || !hackQs.trim()} onClick={runHackRx}>Run</button>
        {hackOut && (
          <pre style={{ background: '#f9fafb', padding: 12, borderRadius: 6, marginTop: 8 }}>
            {JSON.stringify(hackOut, null, 2)}
          </pre>
        )}
      </Section>

      <footer style={{ color: '#9ca3af', marginTop: 24 }}>
        Backend: <code>http://localhost:8000</code> (override with VITE_API_BASE)
      </footer>
    </div>
  )
}
