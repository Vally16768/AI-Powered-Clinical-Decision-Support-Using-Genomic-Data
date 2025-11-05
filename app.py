# app.py
# FastAPI service for rule-based genomic variant prioritization + optional LLM summary (Ollama)
# Implements:
# - Robust LLM model selection (choose the best that fits VRAM; fallback to smallest if none fit)
# - CSV validation at startup
# - Transparent evidence in responses (extra.evidence) + policy_version and audit info
# - New endpoints: GET /config and GET /audit/ping
#
# NOTE: Keeps the public contract of /analyze and /health/llm intact.

from __future__ import annotations

import os
import time
import json
import math
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

APP_VERSION = "2.0.0"
POLICY_VERSION = "1.0.0"

# ----------------------
# Environment & Defaults
# ----------------------

ENV = {
    "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "OLLAMA"),
    "LLM_MIN_VRAM_GB": float(os.getenv("LLM_MIN_VRAM_GB", "6.0")),
    "LLM_CANDIDATES": json.loads(os.getenv("LLM_CANDIDATES", '["qwen2.5:3b-instruct","phi3:mini","qwen2.5:7b-instruct","llama3:8b-instruct"]')),
    "LLM_VRAM_CATALOG": json.loads(os.getenv("LLM_VRAM_CATALOG", '{"qwen2.5:3b-instruct":6.0,"phi3:mini":4.0,"qwen2.5:7b-instruct":10.0,"llama3:8b-instruct":14.0}')),
    "OLLAMA_URL": os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
    "LLM_TIMEOUT_SECONDS": int(os.getenv("LLM_TIMEOUT_SECONDS", "10")),
    "VARIANT_SCORES_PATH": os.getenv("VARIANT_SCORES_PATH", "./variant_scores.csv"),
    "SEND_EHR_TO_LLM": os.getenv("SEND_EHR_TO_LLM", "true").lower() == "true",
    "CADD_MAX": float(os.getenv("CADD_MAX", "40")),
    # Scoring weights (can be tuned without touching code)
    "W_CADD": float(os.getenv("W_CADD", "0.45")),
    "W_POLYPHEN": float(os.getenv("W_POLYPHEN", "0.25")),
    "W_SIFT": float(os.getenv("W_SIFT", "0.10")),
    "W_CLINVAR": float(os.getenv("W_CLINVAR", "0.20")),
}

# ----------------------
# Utility
# ----------------------

def choose_llm_model(candidates: List[str], min_vram_gb: float, catalog: Dict[str, float]) -> Optional[str]:
    """
    Choose the most performant model (first in candidates) that fits VRAM.
    If none fits, return the smallest (lowest VRAM requirement).
    """
    if not candidates:
        return None
    # Filter those that fit
    fits = [m for m in candidates if catalog.get(m, float("inf")) <= min_vram_gb]
    if fits:
        return fits[0]  # "performant → small" order assumed
    # Fallback to smallest
    smallest = min(candidates, key=lambda m: catalog.get(m, float("inf")))
    return smallest

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()

def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()

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
        self.csv_path = csv_path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Variant score file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"variant_scores.csv is missing columns: {missing}")
        # Normalize column types roughly
        # Keep a dict for quick lookup by variant_id
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
            # Unknown variant; pass through gene if provided
            warnings.append("unknown_variant_in_scores_table")
            details = VariantDetails(
                variant_id=v.variant_id,
                gene=v.gene,
                cadd=None, polyphen=None, sift=None, clinvar=None,
                extra=VariantExtra(warnings=warnings, audit={"annotated_at": _ts()}),
            )
            return details
        # Merge requested gene if present (request overrides file if different)
        gene = v.gene or base.get("gene")
        details = VariantDetails(
            variant_id=base["variant_id"],
            gene=gene,
            cadd=base["cadd"],
            polyphen=base["polyphen"],
            sift=base["sift"],
            clinvar=base["clinvar"],
            extra=VariantExtra(warnings=warnings, audit={"annotated_at": _ts()}),
        )
        return details

# ----------------------
# Scoring & Rules (transparent, rule-based)
# ----------------------

def norm_cadd(cadd: Optional[float], max_cadd: float) -> Optional[float]:
    if cadd is None or math.isnan(cadd):
        return None
    return max(0.0, min(1.0, cadd / max_cadd))

_POLYPHEN_MAP = {
    "probably_damaging": 1.0,
    "possibly_damaging": 0.7,
    "benign": 0.0,
}

_SIFT_MAP = {
    "damaging": 1.0,
    "tolerated": 0.0,
}

_CLINVAR_MAP = {
    "pathogenic": 1.0,
    "likely_pathogenic": 0.8,
    "uncertain_significance": 0.3,
    "likely_benign": 0.1,
    "benign": 0.0,
}

def _score_variant(v: VariantDetails) -> Tuple[float, str, str, VariantEvidence, List[str]]:
    rules_fired: List[str] = []
    cadd_n = norm_cadd(v.cadd, ENV["CADD_MAX"])
    polyphen_score = _POLYPHEN_MAP.get((v.polyphen or "").lower())
    sift_score = _SIFT_MAP.get((v.sift or "").lower())
    clinvar_score = _CLINVAR_MAP.get((v.clinvar or "").lower())

    # Composite score (missing values treated as 0)
    score = 0.0
    if cadd_n is not None:
        score += ENV["W_CADD"] * cadd_n
        if v.cadd is not None and v.cadd >= 20:
            rules_fired.append("R1:CADD>=20:+{}".format(ENV["W_CADD"] * cadd_n))
    if polyphen_score is not None:
        score += ENV["W_POLYPHEN"] * polyphen_score
        if (v.polyphen or "").lower() in ("probably_damaging", "possibly_damaging"):
            rules_fired.append("R2:PolyPhen_damaging:+{}".format(ENV["W_POLYPHEN"] * polyphen_score))
    if sift_score is not None:
        score += ENV["W_SIFT"] * sift_score
        if (v.sift or "").lower() == "damaging":
            rules_fired.append("R3:SIFT_damaging:+{}".format(ENV["W_SIFT"] * sift_score))
    if clinvar_score is not None:
        score += ENV["W_CLINVAR"] * clinvar_score
        if (v.clinvar or "").lower() in ("pathogenic", "likely_pathogenic"):
            rules_fired.append("R4:ClinVar_pathogenic:+{}".format(ENV["W_CLINVAR"] * clinvar_score))

    # Priority label
    if score >= 0.75:
        label = "HIGH"
    elif score >= 0.4:
        label = "MEDIUM"
    else:
        label = "LOW"

    rationale_bits = []
    if v.cadd is not None:
        rationale_bits.append(f"CADD {v.cadd}")
    if v.polyphen:
        rationale_bits.append(f"PolyPhen {v.polyphen}")
    if v.sift:
        rationale_bits.append(f"SIFT {v.sift}")
    if v.clinvar:
        rationale_bits.append(f"ClinVar {v.clinvar}")
    rationale = f"Variant prioritized as {label} based on: " + ", ".join(rationale_bits) if rationale_bits else f"Variant prioritized as {label} with limited evidence."

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

def _ollama_generate(model: str, system: str, prompt: str, timeout_s: int) -> str:
    url = f"{ENV['OLLAMA_URL']}/api/generate"
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1}
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

SYSTEM_PROMPT_DEFAULT = (
    "Ești un asistent pentru interpretarea variantelor oncologice. "
    "Scrie rezumate scurte, cu limbaj neutru, fără recomandări terapeutice. "
    "Subliniază trasabilitatea și limitele datelor."
)

def build_llm_prompt(ehr: Optional[EHR], pv: List[PrioritizedVariant]) -> str:
    lines = []
    if ehr:
        safe_ehr = ehr.dict()
        lines.append(f"Pacient: {json.dumps(safe_ehr, ensure_ascii=False)}")
    lines.append("Variante prioritare:")
    for item in pv[:10]:
        v = item.variant
        lines.append(f"- {v.gene or 'NA'} {v.variant_id} | priority={item.priority_label} | score={round(item.priority_score,3)} | CADD={v.cadd} | PolyPhen={v.polyphen} | SIFT={v.sift} | ClinVar={v.clinvar}")
    lines.append("Scrie un scurt sumar clinic (3-6 propoziții), cu prudență și menționarea limitărilor.")
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

# Global annotator (load & validate at startup)
ANNOTATOR: VariantAnnotator
MODEL_SELECTED: Optional[str]
STARTUP_WARNINGS: List[str] = []

@app.on_event("startup")
def _startup() -> None:
    global ANNOTATOR, MODEL_SELECTED, STARTUP_WARNINGS
    # Load CSV and validate
    try:
        ANNOTATOR = VariantAnnotator(ENV["VARIANT_SCORES_PATH"])
    except Exception as e:
        # Fail early with clear message
        raise RuntimeError(f"Failed to initialize VariantAnnotator: {e}")
    # Choose LLM model
    MODEL_SELECTED = choose_llm_model(ENV["LLM_CANDIDATES"], ENV["LLM_MIN_VRAM_GB"], ENV["LLM_VRAM_CATALOG"])
    if MODEL_SELECTED is None:
        STARTUP_WARNINGS.append("no_llm_model_selected")

@app.get("/health/llm")
def health_llm():
    t0 = time.time()
    ok = True
    detail: Dict[str, Any] = {}
    # Try a very quick head call to Ollama
    try:
        r = requests.get(ENV["OLLAMA_URL"], timeout=3)
        ok = r.status_code < 500
    except Exception as e:
        ok = False
        detail["error"] = str(e)
    latency = int((time.time() - t0) * 1000)
    return {
        "ok": ok,
        "latency_ms": latency,
        "provider": ENV["LLM_PROVIDER"],
        "model": MODEL_SELECTED,
        "min_vram_gb": ENV["LLM_MIN_VRAM_GB"],
        "candidates": ENV["LLM_CANDIDATES"],
        **detail,
    }

@app.get("/config")
def get_config():
    return {
        "app_version": APP_VERSION,
        "policy_version": POLICY_VERSION,
        "provider": ENV["LLM_PROVIDER"],
        "model_selected": MODEL_SELECTED,
        "min_vram_gb": ENV["LLM_MIN_VRAM_GB"],
        "candidates": ENV["LLM_CANDIDATES"],
        "vram_catalog": ENV["LLM_VRAM_CATALOG"],
        "variant_scores_path": ENV["VARIANT_SCORES_PATH"],
        "send_ehr_to_llm": ENV["SEND_EHR_TO_LLM"],
    }

@app.get("/audit/ping")
def audit_ping():
    return {"version": APP_VERSION, "policy_version": POLICY_VERSION, "now": _ts()}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    # Annotate
    annotated: List[VariantDetails] = [ANNOTATOR.annotate(v) for v in req.variants]

    # Score and build response items
    prioritized: List[PrioritizedVariant] = []
    for v in annotated:
        score, label, rationale, evidence, rules = _score_variant(v)
        # Attach evidence & audit
        v.extra.evidence = evidence
        v.extra.audit.update({
            "scored_at": _ts(),
            "policy_version": POLICY_VERSION,
        })
        prioritized.append(
            PrioritizedVariant(variant=v, priority_score=round(float(score), 6), priority_label=label, rationale=rationale)
        )

    # Sort desc by score
    prioritized.sort(key=lambda x: x.priority_score, reverse=True)

    return AnalyzeResponse(
        patient_id=req.patient_id,
        policy_version=POLICY_VERSION,
        prioritized=prioritized,
    )

@app.post("/llm_summary", response_model=LLMResponse)
def llm_summary(req: LLMRequest):
    if ENV["LLM_PROVIDER"].upper() != "OLLAMA":
        raise HTTPException(status_code=400, detail="Only OLLAMA provider is supported in this build.")
    model = MODEL_SELECTED
    if not model:
        raise HTTPException(status_code=503, detail="No LLM model selected. Check /config.")
    # Build prompt
    ehr_for_prompt = req.ehr if ENV["SEND_EHR_TO_LLM"] else None
    prompt = build_llm_prompt(ehr_for_prompt, req.variants)
    # Generate
    try:
        text = _ollama_generate(model, SYSTEM_PROMPT_DEFAULT, prompt, ENV["LLM_TIMEOUT_SECONDS"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {e}")
    return LLMResponse(
        patient_id=req.patient_id,
        model=model,
        summary=text,
        generated_at=_ts(),
    )
