#!/usr/bin/env bash
set -euo pipefail

# ===========================================
# CDSS Genomics – local setup & run (no Docker)
# - No file creation
# - No CLI parameters
# - Reads config from .env (PORT, MODEL, optional OLLAMA_HOST)
# - Auto-configure LLM (Ollama): install (Linux/macOS), start, pull model
# ===========================================

# ---------- helpers ----------
have() { command -v "$1" >/dev/null 2>&1; }
die() { echo "ERROR: $*" >&2; exit 1; }

ensure_python() {
  if ! have python3; then
    die "python3 is not installed. Install it (Ubuntu: 'sudo apt install python3 python3-venv', macOS: 'brew install python')."
  fi
  PYVER=$(python3 - <<'PY'
import sys
print(".".join(map(str, sys.version_info[:3])))
PY
)
  echo "• Python3 detected: v$PYVER"
}

mk_venv() {
  if [[ ! -d .venv ]]; then
    echo "• Creating venv ..."
    python3 -m venv .venv || die "python3-venv missing? (Ubuntu: 'sudo apt install python3-venv')"
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip >/dev/null
}

wait_for_http() {
  local url="$1" retries="${2:-20}" delay="${3:-1}"
  for _ in $(seq 1 "$retries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then return 0; fi
    sleep "$delay"
  done
  return 1
}

install_ollama() {
  # Try to install Ollama automatically (Linux/macOS). Windows not handled here.
  if have ollama; then
    return 0
  fi
  echo "• Ollama not found. Attempting automatic install ..."
  OS=$(uname -s || echo "Unknown")
  case "$OS" in
    Linux)
      if curl -fsSL https://ollama.com/install.sh | sh; then
        echo "• Ollama installed (Linux)."
      else
        echo "!! Could not install Ollama automatically on Linux. Continuing without LLM."
        return 1
      fi
      ;;
    Darwin)
      if have brew; then
        if brew list --cask ollama >/dev/null 2>&1 || brew list ollama >/dev/null 2>&1; then
          echo "• Ollama already installed via Homebrew."
        else
          if brew install --cask ollama; then
            echo "• Ollama installed (macOS)."
          else
            echo "!! Could not install Ollama via Homebrew. Continuing without LLM."
            return 1
          fi
        fi
      else
        echo "!! Homebrew not found; install Homebrew or Ollama manually. Continuing without LLM."
        return 1
      fi
      ;;
    *)
      echo "!! Unsupported OS for auto-install ($OS). Please install Ollama manually. Continuing without LLM."
      return 1
      ;;
  esac
  have ollama
}

start_ollama() {
  local host="${1:-http://localhost:11434}"
  echo "• Checking Ollama service at $host ..."
  if curl -fsS "$host/api/tags" >/dev/null 2>&1; then
    echo "  - Ollama is already running."
    return 0
  fi

  # Try to start the service
  if [[ "$(uname -s || echo Unknown)" == "Darwin" ]]; then
    # On macOS the app usually auto-starts; fallback to background serve
    (nohup ollama serve >/dev/null 2>&1 &) || true
  else
    (nohup ollama serve >/dev/null 2>&1 &) || true
  fi

  if wait_for_http "$host/api/tags" 30 1; then
    echo "  - Ollama started."
    return 0
  else
    echo "!! Ollama service did not respond. Continuing without LLM."
    return 1
  fi
}

pull_model() {
  local model="$1"
  echo "• Ensuring model is available: $model"
  # 'ollama pull' is idempotent
  if ! ollama pull "$model"; then
    echo "!! Could not pull model '$model'. You can pull it manually: 'ollama pull $model'."
    return 1
  fi
  return 0
}

# ---------- load .env ----------
[[ -f .env ]] || die ".env not found in current directory. Please create it with at least: PORT=8000 and MODEL=llama3:8b-instruct"

# shellcheck disable=SC2046
export $(grep -v '^\s*#' .env | sed -E 's/\s*$//' | xargs -I {} echo {})

# Validate required envs
[[ -n "${PORT:-}"  ]] || die "PORT is not set in .env"
[[ -n "${MODEL:-}" ]] || die "MODEL is not set in .env"

# Optional envs with defaults
export OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
# Map MODEL -> LLM_MODEL for the app
export LLM_MODEL="$MODEL"

echo "==> Using configuration"
echo "   PORT        = $PORT"
echo "   MODEL       = $MODEL"
echo "   OLLAMA_HOST = $OLLAMA_HOST (optional; app has default too)"

# ---------- sanity checks for project files ----------
[[ -f requirements.txt ]] || die "requirements.txt not found in project root."
[[ -f app.py ]] || die "app.py not found in project root."
[[ -f variant_scores.csv ]] || echo "⚠️  variant_scores.csv not found — the API will still run, but annotator may return empty fields."

# ---------- Python setup ----------
ensure_python
mk_venv
echo "• Installing Python dependencies ..."
pip install -r requirements.txt >/dev/null

# ---------- LLM auto-configuration (best-effort) ----------
LLM_READY=0
if install_ollama; then
  if start_ollama "$OLLAMA_HOST"; then
    if pull_model "$MODEL"; then
      LLM_READY=1
    fi
  fi
else
  echo "• Skipping Ollama setup (not installed)."
fi

if [[ "$LLM_READY" -eq 1 ]]; then
  echo "==> LLM is ready via Ollama ($MODEL) at $OLLAMA_HOST"
else
  echo "==> Proceeding without LLM (llm_summary will be null)."
fi

# ---------- Start API ----------
echo "==> Starting API on port $PORT ..."
echo "   Quick test (in another terminal):"
echo "curl -X POST http://127.0.0.1:$PORT/analyze -H 'Content-Type: application/json' -d '{\"patient_id\":\"P001\",\"variants\":[{\"variant_id\":\"chr17:7579472:C:T\",\"gene\":\"TP53\"},{\"variant_id\":\"chr12:25398284:G:A\",\"gene\":\"KRAS\"}],\"ehr\":{\"age\":71,\"sex\":\"M\",\"cancer_type\":\"CRC\",\"stage\":\"III\"}}'"

exec uvicorn app:app --host 0.0.0.0 --port "$PORT" --reload
