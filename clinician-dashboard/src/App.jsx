import React, { useEffect, useMemo, useRef, useState } from 'react'

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
  const durMs = res.headers.get('X-Response-Time-MS') || data?.duration_ms || null
  return { ok: res.ok, status: res.status, data, headers: res.headers, requestId, durMs }
}

function fmtMs(ms) {
  if (!ms && ms !== 0) return '-'
  const n = Number(ms)
  if (Number.isNaN(n)) return String(ms)
  if (n < 1000) return `${n} ms`
  return `${(n / 1000).toFixed(2)} s`
}

function Banner({ type='info', children }) {
  const bg = type === 'error' ? '#3b1111' : type === 'warn' ? '#3b2d11' : '#11263b'
  const br = type === 'error' ? '#a33' : type === 'warn' ? '#a87' : '#79a9dd'
  return (
    <div style={{ background:bg, border:`1px solid ${br}`, padding:'8px 12px', borderRadius:8, margin:'8px 0' }}>
      {children}
    </div>
  )
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

function InfoTip({ text }) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState({ top: 24, left: 0, transform: '' })
  const btnRef = useRef(null)
  const tipRef = useRef(null)

  useEffect(() => {
    if (!open) return
    const btn = btnRef.current
    const tip = tipRef.current
    if (!btn || !tip) return

    const b = btn.getBoundingClientRect()
    const t = tip.getBoundingClientRect()
    const vw = window.innerWidth
    const vh = window.innerHeight

    let top = b.bottom + 6
    let left = b.left
    let transform = 'translateX(0)'

    if (left + t.width > vw - 8) left = Math.max(8, b.right - t.width)
    if (left < 8) left = Math.max(8, b.left + b.width / 2 - t.width / 2)
    if (top + t.height > vh - 8) top = Math.max(8, b.top - t.height - 6)

    setPos({ top, left, transform })
  }, [open, text])

  useEffect(() => {
    if (!open) return
    const onKey = (e) => e.key === 'Escape' && setOpen(false)
    const onClick = (e) => {
      if (btnRef.current?.contains(e.target) || tipRef.current?.contains(e.target)) return
      setOpen(false)
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('mousedown', onClick)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('mousedown', onClick)
    }
  }, [open])

  return (
    <span style={{ position: 'relative', display: 'inline-block' }}>
      <span
        ref={btnRef}
        className="info"
        title="More info"
        onClick={() => setOpen(v => !v)}
      >i</span>

      {open && (
        <div
          ref={tipRef}
          className="info-tip"
          style={{
            position: 'fixed',
            top: pos.top,
            left: pos.left,
            transform: pos.transform,
            maxWidth: 360,
            zIndex: 20
          }}
        >
          {text}
        </div>
      )}
    </span>
  )
}

export default function App() {
  const [patientId, setPatientId] = useState('P001')
  const [variantFile, setVariantFile] = useState(null)
  const [ehrFile, setEhrFile] = useState(null)
  const [topN, setTopN] = useState(5)

  const [analysis, setAnalysis] = useState(null)
  const [analysisMeta, setAnalysisMeta] = useState({ requestId: null, durationMs: null })

  const [summary, setSummary] = useState(null)
  const [summaryMeta, setSummaryMeta] = useState({ requestId: null, durationMs: null })

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const [llmHealth, setLlmHealth] = useState({ ok: null, provider: '—', models: [], error: null, host: '', selected_model: null, candidates: [], min_vram_gb: null, reachable: false })

  useEffect(() => {
    (async () => {
      const h = await apiFetch('/health/llm')
      if (h.ok) {
        setLlmHealth({
          ok: h.data.ok,
          provider: h.data.provider,
          models: h.data.models || [],
          error: h.data.error || null,
          host: h.data.host || '',
          selected_model: h.data.selected_model || null,
          candidates: h.data.candidates || [],
          min_vram_gb: h.data.min_vram_gb ?? null,
          reachable: !!h.data.reachable
        })
      } else {
        setLlmHealth({ ok: false, provider: '—', models: [], error: `HTTP ${h.status}`, host: '', selected_model: null, candidates: [], min_vram_gb: null, reachable: false })
      }
    })()
  }, [])

  async function maybeUploadFiles() {
    if (!variantFile && !ehrFile) return { ok: true }
    if (variantFile) {
      const fd = new FormData()
      fd.append('file', variantFile)
      fd.append('patient_id', patientId)
      const upV = await apiFetch('/upload/variants', { method: 'POST', body: fd, isForm: true })
      if (!upV.ok) return upV
    }
    if (ehrFile) {
      const fd = new FormData()
      fd.append('file', ehrFile)
      fd.append('patient_id', patientId)
      const upE = await apiFetch('/upload/ehr', { method: 'POST', body: fd, isForm: true })
      if (!upE.ok) return upE
    }
    return { ok: true }
  }

  async function runAnalyze() {
    setBusy(true); setError(null); setSummary(null)
    const uploadRes = await maybeUploadFiles()
    if (!uploadRes.ok) {
      setBusy(false)
      const details = uploadRes.data?.detail || uploadRes.data?.message
      setError(`Upload failed: ${uploadRes.status}${details ? ` — ${JSON.stringify(details)}` : ''}`)
      return
    }

    const t0 = Date.now()
    const r = await apiFetch('/analyze', { method: 'POST', body: { patient_id: patientId } })
    const t1 = Date.now()
    setBusy(false)

    if (!r.ok) {
      const details = r.data?.detail || r.data?.message
      return setError(`Analyze failed: ${r.status}${details ? ` — ${JSON.stringify(details)}` : ''}`)
    }
    setAnalysis(r.data)
    setAnalysisMeta({ requestId: r.requestId, durationMs: r.durMs || (t1 - t0) })
  }

  async function runSummary() {
    if (!analysis?.prioritized?.length) return setError('No analysis results to summarize.')
    const selected = analysis.prioritized.slice(0, Math.max(1, Number(topN) || 5))
    const payload = {
      patient_id: analysis.patient_id,
      variants: selected.map(v => ({
        variant: v.variant,
        priority_score: v.priority_score,
        priority_label: v.priority_label,
        rationale: v.rationale || ''
      })),
      ehr: analysis.ehr ? analysis.ehr : null
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
    setSummaryMeta({ requestId: r.requestId, durationMs: r.durMs || (t1 - t0) })
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

  function copy(text) {
    if (!text) return
    navigator.clipboard.writeText(text).catch(() => {})
  }

  return (
    <div className="page">
      {busy && (
        <div className="overlay" aria-busy="true" aria-live="polite">
          <div className="spinner" />
        </div>
      )}

      <h1 className="h1">Genomic CDSS — Clinician Dashboard</h1>

      {/* LLM health banner */}
      {llmHealth.provider === 'OLLAMA' && llmHealth.ok === false && (
        <Banner type="error">
          <b>LLM unavailable.</b> The system is using the rule-based fallback.
          <div style={{ marginTop: 6 }}>
            <div><b>Host:</b> <code>{llmHealth.host || '(unset)'}</code></div>
            <div><b>Reachable:</b> {llmHealth.reachable ? 'yes' : 'no'}</div>
            <div><b>Selected (by VRAM):</b> {llmHealth.selected_model || '—'}</div>
            <div><b>Candidates:</b> {Array.isArray(llmHealth.candidates) ? llmHealth.candidates.join(', ') : String(llmHealth.candidates || '—')}</div>
            <div><b>Min VRAM:</b> {llmHealth.min_vram_gb ?? '—'} GB</div>
            <div style={{ marginTop: 6 }}>
              Backend error: <code>{llmHealth.error || 'connection to Ollama failed'}</code>
            </div>
            <div>Detected models: {llmHealth.models?.length ? llmHealth.models.join(', ') : '—'}</div>
            <div style={{ opacity: 0.85, marginTop: 6 }}>
              Tip: ensure Ollama is running and accessible at the host above
              (e.g., <code>ollama serve</code>), and that a model is pulled (e.g., <code>ollama pull {llmHealth.selected_model || 'qwen2.5:7b-instruct'}</code>).
            </div>
          </div>
        </Banner>
      )}

      <div className="transparency">
        <div><b>Model LLM:</b> {summary?.model || (llmHealth.ok ? '—' : 'fallback-rule-based')}</div>
        <div><b>Policy version:</b> {analysis?.policy_version || '—'}</div>
        <div><b>Analyze time:</b> {fmtMs(analysisMeta.durationMs)}</div>
        <div><b>Summary time:</b> {fmtMs(summaryMeta.durationMs)}</div>
        <div><b>X-Request-ID (analyze):</b> {analysisMeta.requestId || '—'}</div>
        <div><b>X-Request-ID (summary):</b> {summaryMeta.requestId || '—'}</div>
        {summary?.used_fallback && <div><b>LLM status:</b> Fallback used</div>}
      </div>

      <section className="card">
        <h2 className="h2">1) Upload files</h2>
        <div className="row">
          <div className="box" style={{ minWidth: 360 }}>
            <label className="label">Variants (VCF/CSV)
              &nbsp;<InfoTip text={
                `VCF/CSV with columns or fields that can be normalized to {variant_id, gene}.
- VCF: CHROM,POS,REF,ALT (gene can be derived).
- CSV: variant_id,gene or CHROM,POS,REF,ALT.
Backend deduplicates rows and requires gene present.`
              } />
            </label>
            <input type="file" accept=".vcf,.csv" onChange={e => setVariantFile(e.target.files?.[0] || null)} />
          </div>

          <div className="box" style={{ minWidth: 360 }}>
            <label className="label">EHR (JSON/CSV)
              &nbsp;<InfoTip text={
                `Minimal patient EHR with {age, sex, cancer_type, stage}.
JSON example: {"age":71,"sex":"M","cancer_type":"CRC","stage":"III"}
Fields listed in REDACT_EHR_FIELDS are redacted from LLM prompts.`
              } />
            </label>
            <input type="file" accept=".json,.csv" onChange={e => setEhrFile(e.target.files?.[0] || null)} />
          </div>
        </div>

        <div className="row" style={{ marginTop: 12 }}>
          <button onClick={runAnalyze} disabled={busy} className="button-primary">
            {busy ? 'Working…' : 'Run analysis'}
          </button>
          {error ? <p className="err" style={{ marginLeft: 8 }}>{error}</p> : null}
        </div>
      </section>

      <section className="card">
        <h2 className="h2">2) Results</h2>
        {analysis?.prioritized?.length ? (
          <>
            <Table results={analysis.prioritized} />
            <div className="row-between" style={{ marginTop: 12 }}>
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
                  {busy ? 'Generating…' : 'Generate AI overview'}
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

      <section className="card">
        <h2 className="h2">3) AI overview</h2>
        {summary?.summary ? (
          <>
            {summary?.used_fallback && (
              <Banner type="warn">
                Using fallback (LLM unavailable). Start Ollama and pull a model to enable AI text.
              </Banner>
            )}
            <pre className="summary-box">{summary.summary}</pre>

            {summary?.prompt && (
              <details style={{ marginTop: 10 }}>
                <summary style={{ cursor: 'pointer' }}>
                  Show prompt used (debug)
                </summary>
                <div style={{ marginTop: 8 }}>
                  <button className="button-secondary" onClick={() => copy(summary.prompt)}>Copy prompt</button>
                </div>
                <pre className="summary-box" style={{ marginTop: 8 }}>{summary.prompt}</pre>
              </details>
            )}
          </>
        ) : (
          <p>—</p>
        )}
      </section>

      <footer className="footer">
        <small>© 2025 AI-Powered Clinical Decision Support System. All rights reserved.</small>
      </footer>
    </div>
  )
}

function Table({ results }) {
  return (
    <div>
      <table className="table">
        <thead>
          <tr>
            <th>Variant</th>
            <th>Gene</th>
            <th className="right">CADD</th>
            <th>PolyPhen</th>
            <th>SIFT</th>
            <th>ClinVar</th>
            <th className="center">Priority</th>
            <th className="right">Score</th>
            <th>Knowledge (OncoKB/CIViC)</th>
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
                <td className="right num">{v.cadd ?? '—'}</td>
                <td className="num">{v.polyphen ?? '—'}</td>
                <td className="num">{v.sift ?? '—'}</td>
                <td className="num">{v.clinvar ?? '—'}</td>
                <td className="center"><span className={badgeClass}>{row.priority_label || '—'}</span></td>
                <td className="right num">{row.priority_score != null ? row.priority_score.toFixed(3) : '—'}</td>
                <td>
                  {knowledge.length ? knowledge.slice(0,3).map((k, i) => (
                    <div key={i} style={{ marginBottom: 6 }}>
                      <span className="badge default">{k.source || 'SRC'}{k.evidence_level ? ` • ${k.evidence_level}` : ''}</span>
                      {k.url ? <a className="ml-2 underline" href={k.url} target="_blank" rel="noreferrer" style={{ marginLeft: 8 }}>link</a> : null}
                      <div className="text-xs" style={{ opacity: 0.8, marginTop: 2 }}>
                        {k.statement || k.evidence_note || k.url || '—'}
                      </div>
                    </div>
                  )) : '—'}
                </td>
                <td>{row.rationale || '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
