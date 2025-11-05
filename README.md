# Genomic CDSS — Minimal, Demo-Ready (no Docker)

This build focuses on **clear transparency** and a **single health check**.  
It supports uploads (VCF/CSV + EHR), rule-based prioritization, and a local LLM summary with a safe fallback.

## Quickstart

```bash
./setup_and_run.sh
# API → http://127.0.0.1:8000
```

> If you use a separate UI, keep its `.env` (e.g., `VITE_API_BASE=http://127.0.0.1:8000`) unchanged.

## .env keys (minimal)

```env
# Provider & model
LLM_PROVIDER=OLLAMA
OLLAMA_HOST=http://127.0.0.1:11434
LLM_CANDIDATES=qwen2.5:7b-instruct, qwen2.5:3b-instruct, phi3:mini
LLM_VRAM_CATALOG=qwen2.5:7b-instruct=8, qwen2.5:3b-instruct=4, phi3:mini=3
LLM_MIN_VRAM_GB=6
LLM_TEMPERATURE=0.1
LLM_TIMEOUT_SECONDS=15
SUMMARY_MAX_WORDS=0
SEND_EHR_TO_LLM=true

# Files
VARIANT_SCORES_PATH=./variant_scores.csv
POLICY_FILE=./policies/default.yaml
GENE_KNOWLEDGE_CSV=./data/gene_knowledge.csv
SYSTEM_PROMPT_FILE=./prompts/system_prompt_en.txt
SUMMARY_INSTRUCTIONS_FILE=./prompts/summary_instructions_en.txt

# Limits & audit
MAX_VARIANTS_PER_PATIENT=400
AUDIT_ENABLED=true
AUDIT_DIR=./audit
REDACT_EHR_FIELDS=name,patient_id
```

## Endpoints

- `POST /upload/variants` — form-data: `patient_id`, `file` (VCF/CSV).  
- `POST /upload/ehr` — form-data: `patient_id`, either `file` (CSV) **or** `ehr_json` string.  
- `POST /analyze` — body: `{ "patient_id": "...", "variants": [ {variant_id, gene}? ], "ehr": {...}? }`  
- `POST /llm_summary` — body: `{ "patient_id": "...", "variants": [...prioritized from /analyze...], "ehr": {...}? }`  
- `GET /health/all` — one-shot check for model & files.  
- `GET /config` — current configuration view.  
- `GET /` — service banner.

## Transparency badges (for your UI)

Display these from `/analyze` + `/llm_summary` responses/headers:
- **Model** (from `/llm_summary.model`)
- **generated_at**
- **policy_version** (from `/analyze.policy_version`)
- **X-Request-ID** (response header)
- **analyze_time/summary_time** (use `duration_ms`)

## Demo flow (no curl needed if you have the UI)

1. Upload variants + EHR.  
2. Click **Analyze** → see table (priority, score, rationale, links).  
3. Click **Generate summary** → clinician-friendly points.

## Troubleshooting

- **LLM not installed** → summaries fall back to rule-based text.  
- **Check files & model** → `GET /health/all` should show `ok: true`.  
- **Env names** — this build accepts both new (`LLM_CANDIDATES`, `LLM_VRAM_CATALOG`, `VARIANT_SCORES_PATH`) and legacy (`MODEL_CANDIDATES`, `MODEL_CATALOG`, `SCORES_CSV`) keys.
