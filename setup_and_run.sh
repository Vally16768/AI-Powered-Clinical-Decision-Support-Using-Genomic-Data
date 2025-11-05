#!/usr/bin/env bash
set -euo pipefail

echo "[+] Creating virtualenv (if missing) ..."
python3 -m venv .venv || true
source .venv/bin/activate

echo "[+] Installing deps ..."
pip install --upgrade pip
pip install fastapi uvicorn pandas requests pydantic

# Pull Ollama models (optional)
if command -v ollama >/dev/null 2>&1; then
  echo "[+] Pulling Ollama candidates (if available) ..."
  python - <<'PY'
import os, json, subprocess
cands = json.loads(os.getenv("LLM_CANDIDATES",'["qwen2.5:3b-instruct","phi3:mini","qwen2.5:7b-instruct","llama3:8b-instruct"]'))
for m in cands:
    try:
        import subprocess, os
        subprocess.run(["ollama","pull",m], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
PY
else
  echo "[!] Ollama not found on PATH; /llm_summary will not work until it's installed and running."
fi

echo "[+] Starting API at http://127.0.0.1:8000 ..."
echo "    New endpoints: GET /config, GET /audit/ping"
uvicorn app:app --host 0.0.0.0 --port 8000
