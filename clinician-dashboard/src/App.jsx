import React, { useMemo, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

async function apiFetch(path, { method = 'GET', headers = {}, body, isForm = false } = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: isForm ? undefined : { 'Content-Type': 'application/json', ...headers },
    body: isForm ? body : body ? JSON.stringify(body) : undefined,
  })
  const text = await res.text()
  const data = text ? JSON.parse(text) : {}
  const requestId = res.headers.get('X-Request-ID') || data?.request_id || null
  const durMs = res.headers.get('X-Duration-MS') || data?.duration_ms || null
  return { ok: res.ok, status: res.status, data, headers: res.headers, requestId, durMs }
}

function fmtMs(ms) {
  if (!ms && ms !== 0) return '-'
  const n = Number(ms)
  if (Number.isNaN(n)) return String(ms)
  if (n < 1000) return `${n} ms`
  return `${(n / 1000).toFixed(2)} s`
}

function DownloadButton({ filename, mime = 'application/json', getBlob }) {
  const onClick = () => {
    const blob = getBlob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }
  return (
    <button onClick={onClick} className="button-secondary">
      Download {filename}
    </button>
  )
}

export default function App() {
  const [patientId, setPatientId] = useState('P001')
  const [variantFile, setVariantFile] = useState(null)
  const [ehrFile, setEhrFile] = useState(null)
  const [uploadMsg, setUploadMsg] = useState('')
  const [topN, setTopN] = useState(5)

  const [analysis, setAnalysis] = useState(null) // { patient_id, prioritized: [...], policy_version, ... }
  const [analysisMeta, setAnalysisMeta] = useState({ requestId: null, durationMs: null })

  const [summary, setSummary] = useState(null)   // { model, summary, generated_at, ... }
  const [summaryMeta, setSummaryMeta] = useState({ requestId: null, durationMs: null })

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const generatedAt = useMemo(() => new Date().toISOString(), [summary?.summary])

  async function uploadVariants() {
    if (!variantFile) return setError('Please choose a VCF/CSV file for variants.')
    setError(null); setUploadMsg('Uploading variants...')
    const fd = new FormData()
    fd.append('file', variantFile)
    fd.append('patient_id', patientId)
    const r = await apiFetch('/upload/variants', { method: 'POST', body: fd, isForm: true })
    if (!r.ok) {
      const details = r.data?.detail || r.data?.message
      return setError(`Upload variants failed: ${r.status}${details ? ` — ${JSON.stringify(details)}` : ''}`)
    }
    setUploadMsg('Variants uploaded ✓')
  }

  async function uploadEhr() {
    if (!ehrFile) return setError('Please choose a JSON/CSV file for EHR.')
    setError(null); setUploadMsg('Uploading EHR...')
    const fd = new FormData()
    fd.append('file', ehrFile)
    fd.append('patient_id', patientId)
    const r = await apiFetch('/upload/ehr', { method: 'POST', body: fd, isForm: true })
    if (!r.ok) {
      const details = r.data?.detail || r.data?.message
      return setError(`Upload EHR failed: ${r.status}${details ? ` — ${JSON.stringify(details)}` : ''}`)
    }
    setUploadMsg('EHR uploaded ✓')
  }

  async function runAnalyze() {
    setBusy(true); setError(null); setSummary(null)
    const t0 = Date.now()
    const r = await apiFetch('/analyze', { method: 'POST', body: { patient_id: patientId } })
    const t1 = Date.now()
    setBusy(false)

    if (!r.ok) {
      const details = r.data?.detail || r.data?.message
      return setError(`Analyze failed: ${r.status}${details ? ` — ${JSON.stringify(details)}` : ''}`)
    }
    setAnalysis(r.data)
    setAnalysisMeta({
      requestId: r.requestId,
      durationMs: r.durMs || (t1 - t0)
    })
  }

  async function runSummary() {
    if (!analysis?.prioritized?.length) return setError('No analysis results to summarize.')
    const selected = analysis.prioritized.slice(0, Math.max(1, Number(topN) || 5))
    const payload = {
      patient_id: patientId,
      variants: selected.map(v => ({
        variant: v.variant,
        priority_score: v.priority_score,
        priority_label: v.priority_label
      })),
      ehr: analysis.ehr || null
    }
    setBusy(true); setError(null)
    const t0 = Date.now()
    const r = await apiFetch('/llm_summary', { method: 'POST', body: payload })
    const t1 = Date.now()
    setBusy(false)

    if (!r.ok) {
      if (r.status === 429) {
        return setError(`Too many requests (429). Please slow down: ${r.data?.message || 'rate limit'}`)
      }
      const details = r.data?.detail || r.data?.message
      return setError(`Summary failed: ${r.status}${details ? ` — ${JSON.stringify(details)}` : ''}`)
    }

    setSummary({ ...r.data })
    setSummaryMeta({
      requestId: r.requestId,
      durationMs: r.durMs || (t1 - t0)
    })
  }

  function exportJSON() {
    const obj = {
      analysis,
      summary,
      meta: {
        analysis_request_id: analysisMeta.requestId,
        analysis_duration_ms: analysisMeta.durationMs,
        summary_request_id: summaryMeta.requestId,
        summary_duration_ms: summaryMeta.durationMs,
        generated_at: new Date().toISOString()
      }
    }
    return new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' })
  }

  function exportCSV() {
    if (!analysis?.prioritized?.length) {
      return new Blob([`variant_id,gene,cadd,polyphen,sift,clinvar,priority_score,priority_label\n`], { type: 'text/csv' })
    }
    const rows = [
      ['variant_id','gene','cadd','polyphen','sift','clinvar','priority_score','priority_label','knowledge_links'].join(',')
    ]
    for (const item of analysis.prioritized) {
      const v = item.variant || {}
      const knowledge = (v.extra?.knowledge || [])
        .map(k => k.url || k.link || '')
        .filter(Boolean)
        .join('|')
      const line = [
        safe(v.variant_id),
        safe(v.gene),
        safe(v.cadd),
        safe(v.polyphen),
        safe(v.sift),
        safe(v.clinvar),
        safe(item.priority_score),
        safe(item.priority_label),
        safe(knowledge)
      ].join(',')
      rows.push(line)
    }
    const csv = rows.join('\n')
    return new Blob([csv], { type: 'text/csv' })
  }

  function safe(x) {
    if (x === undefined || x === null) return ''
    const s = String(x).replaceAll('"', '""')
    if (s.includes(',') || s.includes('"') || s.includes('\n')) return `"${s}"`
    return s
  }

  return (
    <div className="page">
      <h1 className="h1">Genomic CDSS — Clinician Dashboard</h1>

      {/* Transparency Bar */}
      <div className="transparency">
        <div><b>Model LLM:</b> {summary?.model || '—'}</div>
        <div><b>Policy version:</b> {analysis?.policy_version || '—'}</div>
        <div><b>Analyze time:</b> {fmtMs(analysisMeta.durationMs)}</div>
        <div><b>Summary time:</b> {fmtMs(summaryMeta.durationMs)}</div>
        <div><b>X-Request-ID (analyze):</b> {analysisMeta.requestId || '—'}</div>
        <div><b>X-Request-ID (summary):</b> {summaryMeta.requestId || '—'}</div>
        <div><b>generated_at:</b> {summary ? new Date().toISOString() : '—'}</div>
      </div>

      {/* Upload panel */}
      <section className="card">
        <h2 className="h2">1) Upload files</h2>

        <div className="row" style={{ marginBottom: 8 }}>
          <label className="label">Patient ID</label>
          <input
            type="text"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            className="input text"
            placeholder="e.g., P001"
          />
        </div>

        <div className="row">
          <div className="box">
            <label className="label">Variants (VCF/CSV)</label>
            <input type="file" accept=".vcf,.csv" onChange={e => setVariantFile(e.target.files?.[0] || null)} />
            <button onClick={uploadVariants} className="button">Upload variants</button>
          </div>
          <div className="box">
            <label className="label">EHR (JSON/CSV)</label>
            <input type="file" accept=".json,.csv" onChange={e => setEhrFile(e.target.files?.[0] || null)} />
            <button onClick={uploadEhr} className="button">Upload EHR</button>
          </div>
        </div>
        {uploadMsg ? <p className="ok">{uploadMsg}</p> : null}
      </section>

      {/* Analyze */}
      <section className="card">
        <h2 className="h2">2) Analyze</h2>
        <button onClick={runAnalyze} disabled={busy} className="button-primary">
          {busy ? 'Running...' : 'Run analysis'}
        </button>
        {error ? <p className="err">{error}</p> : null}
      </section>

      {/* Results table */}
      <section className="card">
        <h2 className="h2">3) Results</h2>
        {analysis?.prioritized?.length ? (
          <>
            <Table results={analysis.prioritized} />
            <div className="row-between">
              <div className="row">
                <label>Top-N for summary:&nbsp;</label>
                <input
                  type="number"
                  min="1"
                  className="input"
                  value={topN}
                  onChange={(e) => setTopN(e.target.value)}
                />
                <button onClick={runSummary} disabled={busy} className="button-primary">
                  {busy ? 'Generating...' : 'Generate summary'}
                </button>
              </div>
              <div className="row">
                <DownloadButton filename="results.json" getBlob={exportJSON} />
                <span style={{ width: 8 }} />
                <DownloadButton filename="results.csv" mime="text/csv" getBlob={exportCSV} />
              </div>
            </div>
          </>
        ) : (
          <p>No results yet.</p>
        )}
      </section>

      {/* Summary */}
      <section className="card">
        <h2 className="h2">4) Summary</h2>
        {summary?.summary ? (
          <pre className="summary-box">{summary.summary}</pre>
        ) : (
          <p>—</p>
        )}
      </section>

      <footer className="footer">
        <small>API base: {API_BASE}</small>
      </footer>
    </div>
  )
}

function Table({ results }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="table">
        <thead>
          <tr>
            <th>Variant</th>
            <th>Gene</th>
            <th>CADD</th>
            <th>PolyPhen</th>
            <th>SIFT</th>
            <th>ClinVar</th>
            <th>Priority</th>
            <th>Score</th>
            <th>Knowledge</th>
            <th>Explanation</th>
          </tr>
        </thead>
        <tbody>
          {results.map((row, idx) => {
            const v = row.variant || {}
            const knowledge = (v.extra?.knowledge || [])
            const badgeClass =
              row.priority_label === 'HIGH' ? 'badge high' :
              row.priority_label === 'MEDIUM' ? 'badge medium' :
              row.priority_label === 'LOW' ? 'badge low' :
              'badge default'
            return (
              <tr key={idx}>
                <td>{v.variant_id || '—'}</td>
                <td>{v.gene || '—'}</td>
                <td>{v.cadd ?? '—'}</td>
                <td>{v.polyphen ?? '—'}</td>
                <td>{v.sift ?? '—'}</td>
                <td>{v.clinvar ?? '—'}</td>
                <td><span className={badgeClass}>{row.priority_label || '—'}</span></td>
                <td>{row.priority_score != null ? row.priority_score.toFixed(3) : '—'}</td>
                <td>
                  {knowledge.length ? knowledge.map((k, i) => (
                    <div key={i}>
                      {k.url ? <a href={k.url} target="_blank" rel="noreferrer">{k.url}</a> : (k.link || '—')}
                    </div>
                  )) : '—'}
                </td>
                <td style={{ maxWidth: 360 }}>{row.rationale || '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
