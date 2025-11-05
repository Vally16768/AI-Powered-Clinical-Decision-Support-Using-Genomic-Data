#!/usr/bin/env bash
set -euo pipefail

#!/usr/bin/env bash
set -euo pipefail

# --- Safe .env loader (supports spaces and commas in values)
if [[ -f .env ]]; then
  echo "• Loading environment from .env"
  while IFS='=' read -r key value; do
    # skip empty lines or comments
    [[ -z "$key" || "$key" =~ ^# ]] && continue
    # trim whitespace and export correctly
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | sed 's/^ *//;s/ *$//')
    export "$key"="$value"
  done < .env
fi


PORT="${PORT:-8000}"
LLM_PROVIDER="${LLM_PROVIDER:-OLLAMA}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
MODEL_CANDIDATES="${MODEL_CANDIDATES:-}"

# ------- Optional: start Ollama if provider is OLLAMA
if [[ "${LLM_PROVIDER}" == "OLLAMA" ]]; then
  if ! pgrep -x "ollama" >/dev/null 2>&1; then
    echo "• Starting Ollama daemon..."
    nohup ollama serve >/dev/null 2>&1 &
    sleep 2
  fi

  # Pre-pull all candidates to avoid cold-start latency
  if [[ -n "${MODEL_CANDIDATES}" ]]; then
    IFS=',' read -ra CANDS <<< "${MODEL_CANDIDATES}"
    echo "• Pre-pulling candidates: ${CANDS[*]}"
    for m in "${CANDS[@]}"; do
      m_trimmed="$(echo "$m" | xargs)"
      [[ -z "$m_trimmed" ]] && continue
      ollama pull "$m_trimmed" || true
    done
  fi
fi

# ---------- Python deps
python3 -m venv .venv || true
source .venv/bin/activate
python -m pip install --upgrade pip
pip install fastapi uvicorn python-dotenv requests

# ---------- Run API
echo "• Starting API on :$PORT"
exec uvicorn app:app --host 0.0.0.0 --port "$PORT" --reload
