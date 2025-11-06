#!/usr/bin/env bash
set -euo pipefail

echo "[+] Creating virtualenv (if missing) ..."
python3 -m venv .venv || true
source .venv/bin/activate

echo "[+] Installing dependencies ..."
pip install --upgrade pip
pip install fastapi uvicorn pandas requests pydantic python-dotenv python-multipart pyyaml httpx

# -------------------------------
# OLLAMA SECTION (auto start)
# -------------------------------

if command -v ollama >/dev/null 2>&1; then
  echo "[+] Checking if Ollama is running..."
  if ! pgrep -x "ollama" >/dev/null 2>&1; then
    echo "[+] Starting Ollama in background..."
    nohup ollama serve > ~/.ollama/ollama.log 2>&1 &
    sleep 3
  else
    echo "[+] Ollama already running."
  fi

  echo "[+] Waiting for Ollama to respond..."
  for i in {1..10}; do
    if curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
      echo "[+] Ollama is up and responding."
      break
    fi
    echo "   ... retry $i"
    sleep 2
  done

  if ! curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    echo "[ERROR] Ollama did not start correctly; aborting backend startup."
    exit 1
  fi

  # Pull models if not yet pulled (env-driven)
  echo "[+] Ensuring Ollama models are available ..."
  python - <<'PY'
import os, json, subprocess
cands=[]
for key in ("LLM_CANDIDATES","MODEL_CANDIDATES"):
    val=os.getenv(key)
    if val:
        try:
            if val.strip().startswith("["): cands=json.loads(val)
            else: cands=[x.strip() for x in val.split(",") if x.strip()]
            break
        except Exception: pass
for m in cands:
    print(f"[+] Checking model {m} ...")
    subprocess.run(["ollama","pull",m], check=False)
PY

else
  echo "[!] Ollama not found on PATH; /llm_summary will use rule-based fallback."
fi

# -------------------------------
# ENV + POLICY VALIDATION
# -------------------------------

echo "[+] Loading environment from .env (if present) ..."
set -a; [ -f .env ] && source .env || true; set +a

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

# -------------------------------
# START BACKEND
# -------------------------------
echo "[+] Starting API at http://127.0.0.1:8000 ..."
echo "    Endpoints: GET /config, GET /audit/ping, POST /analyze, POST /llm_summary"
uvicorn app:app --host 0.0.0.0 --port 8000
