#!/usr/bin/env bash
set -euo pipefail

echo "[+] Creating virtualenv (if missing) ..."
python3 -m venv .venv || true
source .venv/bin/activate

echo "[+] Installing deps ..."
pip install --upgrade pip
pip install fastapi uvicorn pandas requests pydantic python-dotenv python-multipart pyyaml

# Pull Ollama models (optional, env-driven only)
if command -v ollama >/dev/null 2>&1; then
  echo "[+] Pulling Ollama candidates from env (if provided) ..."
  python - <<'PY'
import os, json, subprocess, sys
cands = []
if os.getenv("LLM_CANDIDATES"):
    try:
        cands = json.loads(os.getenv("LLM_CANDIDATES","[]"))
        if not isinstance(cands, list):
            cands = []
    except Exception:
        cands = []
if not cands and os.getenv("MODEL_CANDIDATES"):
    cands = [x.strip() for x in os.getenv("MODEL_CANDIDATES","").split(",") if x.strip()]
for m in cands:
    try:
        subprocess.run(["ollama","pull",m], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
PY
else
  echo "[!] Ollama not found on PATH; /llm_summary will not work until it's installed and running."
fi

echo "[+] Loading environment from .env (if present) ..."
set -a; [ -f .env ] && source .env || true; set +a

# Preflight: prompt files must exist and be non-empty
: "${SYSTEM_PROMPT_FILE:?SYSTEM_PROMPT_FILE is required in .env}"
: "${SUMMARY_INSTRUCTIONS_FILE:?SUMMARY_INSTRUCTIONS_FILE is required in .env}"
if [ ! -s "$SYSTEM_PROMPT_FILE" ]; then
  echo "[!] SYSTEM_PROMPT_FILE does not exist or is empty: $SYSTEM_PROMPT_FILE" >&2
  exit 1
fi
if [ ! -s "$SUMMARY_INSTRUCTIONS_FILE" ]; then
  echo "[!] SUMMARY_INSTRUCTIONS_FILE does not exist or is empty: $SUMMARY_INSTRUCTIONS_FILE" >&2
  exit 1
fi

echo "[+] Starting API at http://127.0.0.1:8000 ..."
echo "    Endpoints: GET /config, GET /audit/ping, POST /analyze, POST /llm_summary"
uvicorn app:app --host 0.0.0.0 --port 8000
