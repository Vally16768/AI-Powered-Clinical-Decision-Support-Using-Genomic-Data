# Genomic Variant Prioritization API — v2 (configurable models & prompts)

This version lets you:

- Choose from **multiple LLM models** via `.env` (ordered candidates)
- Gate model choice by **minimum VRAM (GB)** you specify
- **Switch providers** (local **Ollama** or **OpenAI**) without changing code
- Configure **all prompts/guidance from `.env`** (or external files) — nothing hardcoded

## Quick Start

```bash
cp .env.example .env  # or edit your existing .env using the keys below
./setup_and_run.sh
```

Then call the API:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "patient_id":"P001",
    "variants":[
      {"variant_id":"chr17:7579472:C:T","gene":"TP53"},
      {"variant_id":"chr12:25398284:G:A","gene":"KRAS"}
    ],
    "ehr":{"age":71,"sex":"M","cancer_type":"CRC","stage":"III"}
  }'
```

Health check for LLM (shows provider, selected model, candidates, latency):

```bash
curl http://127.0.0.1:8000/health/llm
```

## Environment Keys

### Provider & Model Selection

* `LLM_PROVIDER` = `OLLAMA` (default) | `OPENAI`
* `MODEL_CANDIDATES` — comma-separated ordered list of models to try (e.g. `llama3:8b-instruct, qwen2.5:7b-instruct, llama3.2:3b-instruct`)
* `MODEL_CATALOG` — map model→min VRAM GB (e.g. `llama3:8b-instruct=8, qwen2.5:7b-instruct=8, llama3.2:3b-instruct=4`)
* `LLM_MIN_VRAM_GB` — your available/desired VRAM threshold (e.g. `6`)
* `OLLAMA_HOST` — Ollama server URL (default `http://localhost:11434`)
* `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` — if using provider `OPENAI`

### Guidance / Prompts (no hardcoding)

* `SYSTEM_PROMPT` or `SYSTEM_PROMPT_FILE`
* `SUMMARY_INSTRUCTIONS` or `SUMMARY_INSTRUCTIONS_FILE`
* `SUMMARY_MAX_WORDS` (default `180`)

### Other

* `PORT` (default `8000`)
* `SCORES_CSV` (default `variant_scores.csv`)
* `MAX_VARIANTS_PER_PATIENT` (cap for per-request list)

## Scoring (transparent heuristic)

We compute a weighted score ∈ [0,1] from available annotations (CADD, PolyPhen, SIFT, ClinVar) and label variants as **HIGH / MEDIUM / LOW**. Edit the weights in `VariantAnnotator._score_variant` if you want a different policy.

## Notes

* `variant_scores.csv` is unchanged; keep your file as-is.
* For large prompts, prefer `SYSTEM_PROMPT_FILE`/`SUMMARY_INSTRUCTIONS_FILE`.
* If no candidate matches `LLM_MIN_VRAM_GB` strictly, the first candidate is used as a fallback.
