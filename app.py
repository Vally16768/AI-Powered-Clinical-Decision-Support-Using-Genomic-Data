import os
import csv
import math
from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# =========================
# ENV & CONFIG HELPERS
# =========================
load_dotenv()

def _getlist(env_key: str, default: str = "") -> List[str]:
    raw = os.getenv(env_key, default)
    parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
    return parts

def _getcatalog(env_key: str = "MODEL_CATALOG") -> Dict[str, float]:
    """
    Format in .env:
      MODEL_CATALOG="llama3.2:3b-instruct=4, qwen2.5:7b-instruct=8, llama3:8b-instruct=8"
    Values represent min VRAM (GB) needed.
    """
    out: Dict[str, float] = {}
    raw = os.getenv(env_key, "")
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        name, vram = item.split("=", 1)
        try:
            out[name.strip()] = float(vram.strip())
        except ValueError:
            # ignore malformed entries
            pass
    return out

# Basic app settings
PORT = int(os.getenv("PORT", "8000"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
SCORES_CSV = os.getenv("SCORES_CSV", "variant_scores.csv")
MAX_VARIANTS = int(os.getenv("MAX_VARIANTS_PER_PATIENT", "50"))

# Model selection settings
MODEL_CANDIDATES = _getlist("MODEL_CANDIDATES", os.getenv("MODEL", ""))
MODEL_CATALOG = _getcatalog("MODEL_CATALOG")
LLM_MIN_VRAM_GB = float(os.getenv("LLM_MIN_VRAM_GB", "0"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "OLLAMA").upper()  # OLLAMA | OPENAI

# Prompt / guidance (configurable via env or file, no hardcoding)

def _load_text_from_env_or_file(env_key: str, file_key: Optional[str] = None, fallback: str = "") -> str:
    val = os.getenv(env_key)
    if val:
        return val
    if file_key:
        path = os.getenv(file_key)
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return fallback

DEFAULT_SYSTEM_PROMPT = (
    "You are a clinical genomics assistant. Summarize prioritized variants for oncology. "
    "Use cautious language, avoid treatment advice, mention uncertainty and suggest validation."
)
SYSTEM_PROMPT_TXT = _load_text_from_env_or_file(
    "SYSTEM_PROMPT", "SYSTEM_PROMPT_FILE", DEFAULT_SYSTEM_PROMPT
)

DEFAULT_SUMMARY_INSTR = (
    "Please provide a short clinician-friendly summary that highlights key variants, "
    "states uncertainty, and avoids treatment advice."
)
SUMMARY_INSTRUCTIONS_TXT = _load_text_from_env_or_file(
    "SUMMARY_INSTRUCTIONS", "SUMMARY_INSTRUCTIONS_FILE", DEFAULT_SUMMARY_INSTR
)
SUMMARY_MAX_WORDS = int(os.getenv("SUMMARY_MAX_WORDS", "180"))


# =========================
# MODEL CHOICE
# =========================

def choose_llm_model(candidates: List[str], min_vram_gb: float, catalog: Dict[str, float]) -> Optional[str]:
    """
    Pick the first candidate whose catalog VRAM requirement <= min_vram_gb.
    If min_vram_gb == 0, don't filter. If nothing matches, return first candidate.
    """
    if not candidates:
        return None
    if min_vram_gb > 0 and catalog:
        for m in candidates:
            req = catalog.get(m)
            if req is None:
                # Unknown requirement -> allow optimistically
                return m
            if req <= min_vram_gb:
                return m
        # fallback if none matched
    return candidates[0]


CHOSEN_MODEL = choose_llm_model(MODEL_CANDIDATES, LLM_MIN_VRAM_GB, MODEL_CATALOG)


# =========================
# LLM PROVIDERS
# =========================
class LLMClient:
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self, host: str, model: str):
        self.url = f"{host.rstrip('/')}/api/generate"
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        import requests
        payload = {
            "model": self.model,
            "prompt": f"<s>[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n[USER]\n{user_prompt}\n[/USER]\n[ASSISTANT]",
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }
        r = requests.post(self.url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("response", "")


class OpenAIClient(LLMClient):
    """Optional provider using the Chat Completions API. Requires OPENAI_API_KEY."""

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LLM_PROVIDER=OPENAI")

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        import requests
        url_root = self.base_url.rstrip("/") if self.base_url else "https://api.openai.com/v1"
        url = url_root + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        r = requests.post(url, json=data, headers=headers, timeout=60)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]


def build_llm_client() -> Optional[LLMClient]:
    try:
        if LLM_PROVIDER == "OPENAI":
            model = os.getenv("OPENAI_MODEL") or (CHOSEN_MODEL or "gpt-4o-mini")
            return OpenAIClient(model)
        # default: OLLAMA
        model = CHOSEN_MODEL or (os.getenv("LLM_MODEL") or "llama3.2:3b-instruct")
        return OllamaClient(OLLAMA_HOST, model)
    except Exception:
        return None


# =========================
# DATA MODELS
# =========================
class VariantIn(BaseModel):
    variant_id: str
    gene: Optional[str] = None


class PatientEHR(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    cancer_type: Optional[str] = None
    stage: Optional[str] = None


class VariantDetails(BaseModel):
    variant_id: str
    gene: Optional[str]
    cadd: Optional[float] = None
    polyphen: Optional[str] = None
    sift: Optional[str] = None
    clinvar: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class PrioritizedVariant(BaseModel):
    variant: VariantDetails
    priority_score: float
    priority_label: str
    rationale: str


class AnalyzeRequest(BaseModel):
    patient_id: str
    variants: List[VariantIn]
    ehr: Optional[PatientEHR] = None


class AnalyzeResponse(BaseModel):
    patient_id: str
    prioritized: List[PrioritizedVariant]
    llm_summary: Optional[str] = None


# =========================
# ANNOTATOR & PRIORITIZATION
# =========================
class VariantAnnotator:
    def __init__(self, csv_path: str = SCORES_CSV) -> None:
        self.db: Dict[str, Dict[str, Any]] = {}
        if os.path.isfile(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # expect at least variant_id, gene, cadd, polyphen, sift, clinvar
                    key = (row.get("variant_id") or "").strip()
                    if not key:
                        continue
                    self.db[key] = {
                        "gene": (row.get("gene") or None),
                        "cadd": _safe_float(row.get("cadd")),
                        "polyphen": _norm_str(row.get("polyphen")),
                        "sift": _norm_str(row.get("sift")),
                        "clinvar": _norm_str(row.get("clinvar")),
                    }

    def annotate(self, v: VariantIn) -> VariantDetails:
        rec = self.db.get(v.variant_id, {})
        return VariantDetails(
            variant_id=v.variant_id,
            gene=v.gene or rec.get("gene"),
            cadd=rec.get("cadd"),
            polyphen=rec.get("polyphen"),
            sift=rec.get("sift"),
            clinvar=rec.get("clinvar"),
            extra=None,
        )

    def prioritize(self, variants: List[VariantIn]) -> List[PrioritizedVariant]:
        out: List[PrioritizedVariant] = []
        for v in variants[:MAX_VARIANTS]:
            det = self.annotate(v)
            score, rationale = self._score_variant(det)
            label = (
                "HIGH" if score >= 0.75 else
                "MEDIUM" if score >= 0.50 else
                "LOW"
            )
            out.append(PrioritizedVariant(variant=det, priority_score=score, priority_label=label, rationale=rationale))
        # sort desc by score
        out.sort(key=lambda pv: pv.priority_score, reverse=True)
        return out

    def _score_variant(self, d: VariantDetails) -> (float, str):
        # Heuristic scoring; keep simple & transparent
        parts = []
        weights = []

        # CADD (0..50+): normalize to 0..1 via min( cadd / 50, 1)
        if d.cadd is not None:
            cadd_norm = min(max(d.cadd, 0.0), 50.0) / 50.0
            parts.append(cadd_norm)
            weights.append(0.35)
        else:
            parts.append(0.0); weights.append(0.0)

        # PolyPhen: probably_damaging=1, possibly_damaging=0.7, benign=0.1, unknown=0.4
        poly_map = {
            "probably_damaging": 1.0,
            "possibly_damaging": 0.7,
            "benign": 0.1,
        }
        if d.polyphen is not None:
            parts.append(poly_map.get(d.polyphen, 0.4))
            weights.append(0.25)
        else:
            parts.append(0.0); weights.append(0.0)

        # SIFT: damaging=1, tolerated=0.2, unknown=0.4
        sift_map = {
            "damaging": 1.0,
            "deleterious": 1.0,
            "tolerated": 0.2,
        }
        if d.sift is not None:
            parts.append(sift_map.get(d.sift, 0.4))
            weights.append(0.20)
        else:
            parts.append(0.0); weights.append(0.0)

        # ClinVar: pathogenic=1, likely_pathogenic=0.85, vusc=0.5, likely_benign=0.2, benign=0.1
        clinvar_map = {
            "pathogenic": 1.0,
            "likely_pathogenic": 0.85,
            "vus": 0.5,
            "uncertain_significance": 0.5,
            "likely_benign": 0.2,
            "benign": 0.1,
        }
        if d.clinvar is not None:
            parts.append(clinvar_map.get(d.clinvar, 0.5))
            weights.append(0.20)
        else:
            parts.append(0.0); weights.append(0.0)

        # weighted average
        wsum = sum(weights) or 1.0
        score = sum(p * w for p, w in zip(parts, weights)) / wsum
        score = float(max(0.0, min(1.0, score)))

        rationale = (
            f"CADD={d.cadd if d.cadd is not None else 'NA'}, "
            f"PolyPhen={d.polyphen or 'NA'}, SIFT={d.sift or 'NA'}, ClinVar={d.clinvar or 'NA'}"
        )
        return score, rationale


def _safe_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _norm_str(x: Optional[str]) -> Optional[str]:
    return (x or None).lower() if x else None


# =========================
# PROMPTS (no hardcoding)
# =========================

def system_prompt() -> str:
    return SYSTEM_PROMPT_TXT


def user_prompt(pid: str, items: List[PrioritizedVariant], ehr: Optional[PatientEHR]) -> str:
    ehr_lines = []
    if ehr:
        if ehr.age is not None:
            ehr_lines.append(f"Age: {ehr.age}")
        if ehr.sex:
            ehr_lines.append(f"Sex: {ehr.sex}")
        if ehr.cancer_type:
            ehr_lines.append(f"Cancer type: {ehr.cancer_type}")
        if ehr.stage:
            ehr_lines.append(f"Stage: {ehr.stage}")

    lines = [f"Patient: {pid}", "", "Prioritized variants:"]
    for pv in items[:10]:
        v = pv.variant
        lines.append(
            f"- {v.gene or ''} {v.variant_id}: {pv.priority_label} (score {pv.priority_score:.2f}); "
            f"CADD={v.cadd}, PolyPhen={v.polyphen}, SIFT={v.sift}, ClinVar={v.clinvar}. Rationale: {pv.rationale}"
        )

    guidance = f"\n{SUMMARY_INSTRUCTIONS_TXT}\nLimit: <= {SUMMARY_MAX_WORDS} words."
    return ("\n".join(ehr_lines) + "\n\n" + "\n".join(lines) + guidance)


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Genomic Variant Prioritization API", version="2.0.0")
annotator = VariantAnnotator(SCORES_CSV)
llm = build_llm_client()


@app.get("/health")
def health():
    return {"ok": True, "provider": LLM_PROVIDER, "model": CHOSEN_MODEL}


@app.get("/health/llm")
def health_llm():
    info = {
        "provider": LLM_PROVIDER,
        "model": CHOSEN_MODEL,
        "min_vram_gb": LLM_MIN_VRAM_GB,
        "candidates": MODEL_CANDIDATES,
    }
    import time
    try:
        t0 = time.time()
        if isinstance(llm, OllamaClient):
            import requests
            r = requests.post(
                OLLAMA_HOST.rstrip("/") + "/api/generate",
                json={"model": CHOSEN_MODEL, "prompt": "ok", "options": {"num_predict": 4}},
                timeout=15,
            )
            r.raise_for_status()
        elif isinstance(llm, OpenAIClient):
            llm.generate("ping", "ok", max_tokens=4)
        else:
            return {"ok": False, "error": "LLM client unavailable", **info}
        latency_ms = int((time.time() - t0) * 1000)
        return {"ok": True, "latency_ms": latency_ms, **info}
    except Exception as e:
        return {"ok": False, "error": str(e), **info}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(req: AnalyzeRequest):
    # Prioritize
    prioritized = annotator.prioritize(req.variants)

    # Optional LLM summary
    summary = None
    if llm is not None and prioritized:
        try:
            sp = system_prompt()
            up = user_prompt(req.patient_id, prioritized, req.ehr)
            summary = llm.generate(sp, up)
        except Exception:
            summary = None

    return AnalyzeResponse(patient_id=req.patient_id, prioritized=prioritized, llm_summary=summary)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
