# app.py
# FastAPI service for rule-based genomic variant prioritization + optional LLM summary (Ollama)
# Prompts MUST come from files specified in env; no fallbacks.

from __future__ import annotations

import os
import time
import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

APP_VERSION = "2.3.0"
POLICY_VERSION = "1.0.0"

# Load .env early (ENV_FILE can override path)
load_dotenv(os.getenv("ENV_FILE", ".env"), override=False)

# ----------------------
# Strict prompt loading
# ----------------------

def _require_file_env(key: str) -> str:
    path = os.getenv(key)
    if not path:
        raise RuntimeError(f"Missing required environment variable: {key}")
    p = Path(path)
    if not (p.exists() and p.is_file()):
        raise RuntimeError(f"{key} points to a missing file: {path}")
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        raise RuntimeError(f"{key} file is empty: {path}")
    return txt

# Load prompts from files ONLY; no fallback
SYSTEM_PROMPT = _require_file_env("SYSTEM_PROMPT_FILE")
SUMMARY_INSTRUCTIONS = _require_file_env("SUMMARY_INSTRUCTIONS_FILE")

# ----------------------
# Environment
# ----------------------

def _parse_candidates() -> List[str]:
    v_json = os.getenv("LLM_CANDIDATES")
    v_csv = os.getenv("MODEL_CANDIDATES")
    if v_json:
        try:
            arr = json.loads(v_json)
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return arr
        except Exception:
            pass
    if v_csv:
        return [x.strip() for x in v_csv.split(",") if x.strip()]
    return []

def _parse_catalog() -> Dict[str, float]:
    j = os.getenv("LLM_VRAM_CATALOG")
    if j:
        try:
            obj = json.loads(j)
            return {k: float(v) for k, v in obj.items()}
        except Exception:
            pass
    c = os.getenv("MODEL_CATALOG")
    out: Dict[str, float] = {}
    if c:
        for item in c.split(","):
            item = item.strip()
            if not item or "=" not in item:
                continue
            name, val = item.split("=", 1)
            try:
                out[name.strip()] = float(val.strip())
            except ValueError:
                continue
    return out

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()

ENV = {
    "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "OLLAMA").upper(),
    "LLM_MIN_VRAM_GB": float(os.getenv("LLM_MIN_VRAM_GB", "6")),
    "LLM_CANDIDATES": _parse_candidates(),
    "LLM_VRAM_CATALOG": _parse_catalog(),
    "OLLAMA_URL": os.getenv("OLLAMA_URL", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")),
    "LLM_TIMEOUT_SECONDS": int(os.getenv("LLM_TIMEOUT_SECONDS", "15")),
    "LLM_TEMPERATURE": float(os.getenv("LLM_TEMPERATURE", "0.1")),
    "VARIANT_SCORES_PATH": os.getenv("VARIANT_SCORES_PATH", os.getenv("SCORES_CSV", "./variant_scores.csv")),
    "SEND_EHR_TO_LLM": os.getenv("SEND_EHR_TO_LLM", "true").lower() == "true",
    "CADD_MAX": float(os.getenv("CADD_MAX", "40")),
    "W_CADD": float(os.getenv("W_CADD", "0.45")),
    "W_POLYPHEN": float(os.getenv("W_POLYPHEN", "0.25")),
    "W_SIFT": float(os.getenv("W_SIFT", "0.10")),
    "W_CLINVAR": float(os.getenv("W_CLINVAR", "0.20")),
    "SUMMARY_MAX_WORDS": int(os.getenv("SUMMARY_MAX_WORDS", "0")),  # 0 = no trimming
}

# ----------------------
# Utility
# ----------------------

def choose_llm_model(candidates: List[str], min_vram_gb: float, catalog: Dict[str, float]) -> Optional[str]:
    if not candidates:
        return None
    fits = [m for m in candidates if catalog.get(m, float("inf")) <= min_vram_gb]
    if fits:
        return fits[0]
    return min(candidates, key=lambda m: catalog.get(m, float("inf")))

# ----------------------
# Data Models
# ----------------------

class VariantInput(BaseModel):
    variant_id: str
    gene: Optional[str] = None

class EHR(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    cancer_type: Optional[str] = None
    stage: Optional[str] = None
    treatment: Optional[str] = None

class AnalyzeRequest(BaseModel):
    patient_id: str
    variants: List[VariantInput]
    ehr: Optional[EHR] = None

class VariantEvidence(BaseModel):
    cadd_norm: Optional[float] = None
    polyphen_score: Optional[float] = None
    sift_score: Optional[float] = None
    clinvar_score: Optional[float] = None
    rules_fired: List[str] = []

class VariantExtra(BaseModel):
    warnings: List[str] = []
    evidence: VariantEvidence = VariantEvidence()
    knowledge: List[Dict[str, Any]] = []
    audit: Dict[str, Any] = {}

class VariantDetails(BaseModel):
    variant_id: str
    gene: Optional[str] = None
    cadd: Optional[float] = None
    polyphen: Optional[str] = None
    sift: Optional[str] = None
    clinvar: Optional[str] = None
    extra: VariantExtra = VariantExtra()

class PrioritizedVariant(BaseModel):
    variant: VariantDetails
    priority_score: float
    priority_label: str
    rationale: str

class AnalyzeResponse(BaseModel):
    patient_id: str
    policy_version: str
    prioritized: List[PrioritizedVariant]

class LLMRequest(BaseModel):
    patient_id: str
    variants: List[PrioritizedVariant]
    ehr: Optional[EHR] = None

class LLMResponse(BaseModel):
    patient_id: str
    model: Optional[str] = None
    summary: str
    generated_at: str

# ----------------------
# Variant Annotator
# ----------------------

REQUIRED_COLUMNS = ["variant_id", "gene", "cadd", "polyphen", "sift", "clinvar"]

class VariantAnnotator:
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Variant score file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"variant_scores.csv is missing columns: {missing}")
        self.table: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            vid = str(row["variant_id"]).strip()
            if not vid:
                continue
            self.table[vid] = {
                "variant_id": vid,
                "gene": None if pd.isna(row["gene"]) else str(row["gene"]),
                "cadd": None if pd.isna(row["cadd"]) else float(row["cadd"]),
                "polyphen": None if pd.isna(row["polyphen"]) else str(row["polyphen"]).lower(),
                "sift": None if pd.isna(row["sift"]) else str(row["sift"]).lower(),
                "clinvar": None if pd.isna(row["clinvar"]) else str(row["clinvar"]).lower(),
            }

    def annotate(self, v: VariantInput) -> VariantDetails:
        base = self.table.get(v.variant_id)
        warnings: List[str] = []
        if not base:
            warnings.append("unknown_variant_in_scores_table")
            return VariantDetails(
                variant_id=v.variant_id,
                gene=v.gene,
                cadd=None, polyphen=None, sift=None, clinvar=None,
                extra=VariantExtra(warnings=warnings, audit={"annotated_at": datetime.now(timezone.utc).isoformat()}),
            )
        gene = v.gene or base.get("gene")
        return VariantDetails(
            variant_id=base["variant_id"],
            gene=gene,
            cadd=base["cadd"],
            polyphen=base["polyphen"],
            sift=base["sift"],
            clinvar=base["clinvar"],
            extra=VariantExtra(warnings=warnings, audit={"annotated_at": datetime.now(timezone.utc).isoformat()}),
        )

# ----------------------
# Scoring & Rules
# ----------------------

_POLYPHEN_MAP = {"probably_damaging": 1.0, "possibly_damaging": 0.7, "benign": 0.0}
_SIFT_MAP = {"damaging": 1.0, "tolerated": 0.0}
_CLINVAR_MAP = {
    "pathogenic": 1.0,
    "likely_pathogenic": 0.8,
    "uncertain_significance": 0.3,
    "likely_benign": 0.1,
    "benign": 0.0,
}

def norm_cadd(cadd: Optional[float], max_cadd: float) -> Optional[float]:
    if cadd is None or math.isnan(cadd):
        return None
    return max(0.0, min(1.0, cadd / max_cadd))

def _score_variant(v: VariantDetails) -> Tuple[float, str, str, VariantEvidence, List[str]]:
    rules_fired: List[str] = []
    cadd_n = norm_cadd(v.cadd, ENV["CADD_MAX"])
    polyphen_score = _POLYPHEN_MAP.get((v.polyphen or "").lower())
    sift_score = _SIFT_MAP.get((v.sift or "").lower())
    clinvar_score = _CLINVAR_MAP.get((v.clinvar or "").lower())

    score = 0.0
    if cadd_n is not None:
        score += ENV["W_CADD"] * cadd_n
        if v.cadd is not None and v.cadd >= 20:
            rules_fired.append(f"R1:CADD>=20:+{ENV['W_CADD'] * cadd_n}")
    if polyphen_score is not None:
        score += ENV["W_POLYPHEN"] * polyphen_score
        if (v.polyphen or "").lower() in ("probably_damaging", "possibly_damaging"):
            rules_fired.append(f"R2:PolyPhen_damaging:+{ENV['W_POLYPHEN'] * polyphen_score}")
    if sift_score is not None:
        score += ENV["W_SIFT"] * sift_score
        if (v.sift or "").lower() == "damaging":
            rules_fired.append(f"R3:SIFT_damaging:+{ENV['W_SIFT'] * sift_score}")
    if clinvar_score is not None:
        score += ENV["W_CLINVAR"] * clinvar_score
        if (v.clinvar or "").lower() in ("pathogenic", "likely_pathogenic"):
            rules_fired.append(f"R4:ClinVar_pathogenic:+{ENV['W_CLINVAR'] * clinvar_score}")

    if score >= 0.75:
        label = "HIGH"
    elif score >= 0.4:
        label = "MEDIUM"
    else:
        label = "LOW"

    bits = []
    if v.cadd is not None:
        bits.append(f"CADD {v.cadd}")
    if v.polyphen:
        bits.append(f"PolyPhen {v.polyphen}")
    if v.sift:
        bits.append(f"SIFT {v.sift}")
    if v.clinvar:
        bits.append(f"ClinVar {v.clinvar}")
    rationale = "Variant prioritized as {} based on: {}".format(label, ", ".join(bits)) if bits else f"Variant prioritized as {label} with limited evidence."

    evidence = VariantEvidence(
        cadd_norm=cadd_n,
        polyphen_score=polyphen_score,
        sift_score=sift_score,
        clinvar_score=clinvar_score,
        rules_fired=rules_fired,
    )
    return score, label, rationale, evidence, rules_fired

# ----------------------
# LLM Orchestration (Ollama)
# ----------------------

def _ollama_generate(model: str, system: str, prompt: str, timeout_s: int, temperature: float) -> str:
    url = f"{ENV['OLLAMA_URL']}/api/generate"
    payload = {"model": model, "system": system, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

def build_llm_prompt(ehr: Optional[EHR], pv: List[PrioritizedVariant]) -> str:
    lines = []
    if ehr:
        lines.append(f"EHR: {json.dumps(ehr.dict(), ensure_ascii=False)}")
    lines.append("Prioritized variants:")
    for item in pv[:10]:
        v = item.variant
        lines.append(f"- {v.gene or 'NA'} {v.variant_id} | priority={item.priority_label} | score={round(item.priority_score,3)} | CADD={v.cadd} | PolyPhen={v.polyphen} | SIFT={v.sift} | ClinVar={v.clinvar}")
    lines.append(SUMMARY_INSTRUCTIONS)
    return "\n".join(lines)

# ----------------------
# FastAPI App
# ----------------------

app = FastAPI(title="Genomic AI-Orchestrator", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANNOTATOR: 'VariantAnnotator'
MODEL_SELECTED: Optional[str]

@app.on_event("startup")
def _startup() -> None:
    global ANNOTATOR, MODEL_SELECTED
    # CSV presence validated by VariantAnnotator
    ANNOTATOR = VariantAnnotator(ENV["VARIANT_SCORES_PATH"])
    MODEL_SELECTED = choose_llm_model(ENV["LLM_CANDIDATES"], ENV["LLM_MIN_VRAM_GB"], ENV["LLM_VRAM_CATALOG"])
    if not MODEL_SELECTED:
        raise RuntimeError("No suitable LLM model found. Check MODEL_CANDIDATES and VRAM settings.")

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    annotated: List[VariantDetails] = [ANNOTATOR.annotate(v) for v in req.variants]
    prioritized: List[PrioritizedVariant] = []
    for v in annotated:
        score, label, rationale, evidence, rules = _score_variant(v)
        v.extra.evidence = evidence
        v.extra.audit.update({"scored_at": datetime.now(timezone.utc).isoformat(), "policy_version": POLICY_VERSION})
        prioritized.append(PrioritizedVariant(variant=v, priority_score=round(float(score), 6), priority_label=label, rationale=rationale))
    prioritized.sort(key=lambda x: x.priority_score, reverse=True)
    return AnalyzeResponse(patient_id=req.patient_id, policy_version=POLICY_VERSION, prioritized=prioritized)

@app.post("/llm_summary", response_model=LLMResponse)
def llm_summary(req: LLMRequest):
    if os.getenv("LLM_PROVIDER", "OLLAMA").upper() != "OLLAMA":
        raise HTTPException(status_code=400, detail="Only OLLAMA provider is supported in this build.")
    model = MODEL_SELECTED
    if not model:
        raise HTTPException(status_code=503, detail="No LLM model selected. Check /config.")
    ehr_for_prompt = req.ehr if ENV["SEND_EHR_TO_LLM"] else None
    prompt = build_llm_prompt(ehr_for_prompt, req.variants)
    try:
        text = _ollama_generate(model, SYSTEM_PROMPT, prompt, ENV["LLM_TIMEOUT_SECONDS"], ENV["LLM_TEMPERATURE"])
        # No trimming when SUMMARY_MAX_WORDS == 0
        if ENV["SUMMARY_MAX_WORDS"] and ENV["SUMMARY_MAX_WORDS"] > 0:
            words = text.split()
            if len(words) > ENV["SUMMARY_MAX_WORDS"]:
                text = " ".join(words[:ENV["SUMMARY_MAX_WORDS"]]) + "…"
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {e}")
    return LLMResponse(patient_id=req.patient_id, model=model, summary=text, generated_at=datetime.now(timezone.utc).isoformat())
