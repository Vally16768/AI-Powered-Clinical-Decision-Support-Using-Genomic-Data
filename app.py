#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import time
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# ----------------------
# ENV
# ----------------------
load_dotenv()

def _as_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")

def _parse_candidates(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    raw = raw.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    return [s.strip() for s in raw.split(",") if s.strip()]

def _parse_vram_catalog(raw: Optional[str]) -> Dict[str, float]:
    if not raw:
        return {}
    raw = raw.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items()}
    except Exception:
        pass
    cat: Dict[str, float] = {}
    for tok in raw.split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                cat[k.strip()] = float(v.strip())
            except Exception:
                continue
    return cat

ENV: Dict[str, Any] = {
    "VARIANT_SCORES_PATH": os.getenv("VARIANT_SCORES_PATH", os.getenv("SCORES_CSV", "./variant_scores.csv")),
    "POLICY_FILE": os.getenv("POLICY_FILE", "./policies/default.yaml"),
    "GENE_KNOWLEDGE_CSV": os.getenv("GENE_KNOWLEDGE_CSV", "./data/gene_knowledge.csv"),

    "MAX_VARIANTS_PER_PATIENT": int(os.getenv("MAX_VARIANTS_PER_PATIENT", "400")),

    "AUDIT_ENABLED": _as_bool(os.getenv("AUDIT_ENABLED", "true")),
    "AUDIT_DIR": os.getenv("AUDIT_DIR", "./audit"),

    "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "OLLAMA"),
    "OLLAMA_HOST": os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
    "LLM_CANDIDATES": _parse_candidates(os.getenv("LLM_CANDIDATES") or os.getenv("MODEL_CANDIDATES", "qwen2.5:7b-instruct, qwen2.5:3b-instruct, phi3:mini")),
    "LLM_MIN_VRAM_GB": float(os.getenv("LLM_MIN_VRAM_GB", "6.0")),
    "LLM_TIMEOUT_SECONDS": int(os.getenv("LLM_TIMEOUT_SECONDS", "25")),
    "LLM_TEMPERATURE": float(os.getenv("LLM_TEMPERATURE", "0.2")),
    "LLM_VRAM_CATALOG": (_parse_vram_catalog(os.getenv("LLM_VRAM_CATALOG")) or _parse_vram_catalog(os.getenv("MODEL_CATALOG")) or {
        "qwen2.5:7b-instruct": 8.0,
        "llama3:8b-instruct": 8.0,
        "qwen2.5:3b-instruct": 4.0,
        "phi3:mini": 3.0
    }),

    "SYSTEM_PROMPT_FILE": os.getenv("SYSTEM_PROMPT_FILE", "./prompts/system_prompt_en.txt"),
    "SUMMARY_INSTRUCTIONS_FILE": os.getenv("SUMMARY_INSTRUCTIONS_FILE", "./prompts/summary_instructions_en.txt"),
    "SEND_EHR_TO_LLM": _as_bool(os.getenv("SEND_EHR_TO_LLM", "true")),
    "SUMMARY_MAX_WORDS": int(os.getenv("SUMMARY_MAX_WORDS", "0")),

    "CADD_MAX": float(os.getenv("CADD_MAX", "40")),
    "W_CADD": float(os.getenv("W_CADD", "0.45")),
    "W_POLYPHEN": float(os.getenv("W_POLYPHEN", "0.25")),
    "W_SIFT": float(os.getenv("W_SIFT", "0.10")),
    "W_CLINVAR": float(os.getenv("W_CLINVAR", "0.20")),

    "REDACT_EHR_FIELDS": [x.strip() for x in os.getenv("REDACT_EHR_FIELDS", "").split(",") if x.strip()],

    # --- NEW: external knowledge connectors (API-only, no cache) ---
    "ENABLE_VEP": _as_bool(os.getenv("ENABLE_VEP", "true")),
    "ENABLE_OPENCRAVAT": _as_bool(os.getenv("ENABLE_OPENCRAVAT", "false")),
    "ENABLE_ONCOKB": _as_bool(os.getenv("ENABLE_ONCOKB", "true")),
    "ENABLE_CIVIC": _as_bool(os.getenv("ENABLE_CIVIC", "true")),
    "VEP_REST_BASE": os.getenv("VEP_REST_BASE", "https://rest.ensembl.org/vep/human/region"),
    "OPENCRAVAT_API_BASE": os.getenv("OPENCRAVAT_API_BASE", ""),  # optional if you have a CRAVAT API
    "ONCOKB_API_BASE": os.getenv("ONCOKB_API_BASE", "https://www.oncokb.org/api/v1"),
    "ONCOKB_API_TOKEN": os.getenv("ONCOKB_API_TOKEN", ""),
    "CIVIC_BASE_URL": os.getenv("CIVIC_BASE_URL", "https://civicdb.org/api/graphql"),
    "ANNOTATION_TIMEOUT_S": float(os.getenv("ANNOTATION_TIMEOUT_S", "8")),
}

Path(ENV["AUDIT_DIR"]).mkdir(parents=True, exist_ok=True)

# ----------------------
# Utils
# ----------------------
def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()

def error(code: str, message: str, details: Optional[List[Dict[str, Any]]] = None, status: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status, content={"code": code, "message": message, "details": details or []})

def redact_ehr(ehr: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not ehr or not ENV["REDACT_EHR_FIELDS"]:
        return ehr
    red = dict(ehr)
    for k in ENV["REDACT_EHR_FIELDS"]:
        if k in red:
            red[k] = "[REDACTED]"
    return red

# ----------------------
# Models
# ----------------------
class EHR(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    cancer_type: Optional[str] = None
    stage: Optional[str] = None
    treatment: Optional[str] = None

class VariantInput(BaseModel):
    variant_id: str
    gene: Optional[str] = None

class VariantEvidence(BaseModel):
    # internal normalized scores (rule engine)
    cadd_norm: Optional[float] = None
    polyphen_score: Optional[float] = None
    sift_score: Optional[float] = None
    clinvar_score: Optional[float] = None
    rules_fired: List[str] = []

class VariantExtra(BaseModel):
    warnings: List[str] = []
    evidence: VariantEvidence = VariantEvidence()              # existing score evidence
    api_evidence: List[Dict[str, Any]] = []                   # NEW: raw external evidence (VEP/CRAVAT/OncoKB/CIViC)
    knowledge: List[Dict[str, Any]] = []                      # clinician-facing statements (CSV + OncoKB + CIViC)
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

class AnalyzeRequest(BaseModel):
    patient_id: str
    variants: Optional[List[VariantInput]] = None
    ehr: Optional[EHR] = None

class AnalyzeResponse(BaseModel):
    patient_id: str
    policy_version: str
    prioritized: List[PrioritizedVariant]
    request_id: str
    duration_ms: int

class LLMRequest(BaseModel):
    patient_id: str
    variants: List[PrioritizedVariant]
    ehr: Optional[EHR] = None

class LLMResponse(BaseModel):
    patient_id: str
    model: Optional[str] = None
    summary: str
    generated_at: str
    request_id: str
    duration_ms: int

# ----------------------
# Upload store (demo)
# ----------------------
UPLOAD_STORE: Dict[str, Dict[str, Any]] = {}

# ----------------------
# Annotator (local CSV scorer, unchanged)
# ----------------------
class VariantAnnotator:
    def __init__(self, csv_path: str):
        self.df: Optional[pd.DataFrame] = None
        path = Path(csv_path)
        if not path.exists():
            print(f"[WARN] variant_scores.csv not found at {csv_path}.")
            self.map_by_vid = {}
            return
        self.df = pd.read_csv(path)
        if "variant_id" not in self.df.columns:
            raise RuntimeError("variant_scores.csv must contain 'variant_id'")
        for col in ["gene", "polyphen", "sift", "clinvar"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).replace("nan", "")
        if "cadd" in self.df.columns:
            try:
                self.df["cadd"] = pd.to_numeric(self.df["cadd"], errors="coerce")
            except Exception:
                pass
        self.map_by_vid: Dict[str, Dict[str, Any]] = {}
        for _, row in self.df.iterrows():
            vid = str(row["variant_id"]).strip()
            if not vid:
                continue
            self.map_by_vid[vid] = {
                "gene": (None if pd.isna(row.get("gene")) else str(row.get("gene")).strip() or None),
                "cadd": (None if pd.isna(row.get("cadd")) else float(row.get("cadd"))),
                "polyphen": (None if pd.isna(row.get("polyphen")) else str(row.get("polyphen")).strip() or None),
                "sift": (None if pd.isna(row.get("sift")) else str(row.get("sift")).strip() or None),
                "clinvar": (None if pd.isna(row.get("clinvar")) else str(row.get("clinvar")).strip() or None),
            }

    def annotate(self, v: VariantInput) -> VariantDetails:
        base = VariantDetails(variant_id=v.variant_id, gene=v.gene)
        row = self.map_by_vid.get(v.variant_id) if hasattr(self, "map_by_vid") else None
        if row:
            base.gene = base.gene or row.get("gene")
            base.cadd = row.get("cadd")
            base.polyphen = row.get("polyphen")
            base.sift = row.get("sift")
            base.clinvar = row.get("clinvar")
        return base

# ----------------------
# Policy & Gene-level knowledge (CSV fallback)
# ----------------------
POLICY: Dict[str, Any] = {}
POLICY_VERSION = "0.0.0"

def load_policy(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Policy file not found: {path}")
    if p.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("Policy file must contain a mapping/object")
    return data

class KnowledgeBase:
    def __init__(self, csv_path: str):
        self.by_gene: Dict[str, List[Dict[str, Any]]] = {}
        path = Path(csv_path)
        if not path.exists():
            print(f"[WARN] Knowledge CSV not found at {csv_path}.")
            return
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            g = str(row.get("gene", "")).strip().upper()
            if not g:
                continue
            self.by_gene.setdefault(g, []).append({
                "gene": g,
                "disease_area": None if pd.isna(row.get("disease_area")) else str(row.get("disease_area")),
                "evidence_note": None if pd.isna(row.get("evidence_note")) else str(row.get("evidence_note")),
                "url": None if pd.isna(row.get("url")) else str(row.get("url")),
                "source": "GENE_CSV",
            })

    def get(self, gene: Optional[str]) -> List[Dict[str, Any]]:
        if not gene:
            return []
        return self.by_gene.get(gene.upper(), [])

KNOWLEDGE: KnowledgeBase

# ----------------------
# Scoring (unchanged)
# ----------------------
def norm_cadd(x: Optional[float], cadd_max: float) -> Optional[float]:
    if x is None:
        return None
    if x <= 0:
        return 0.0
    return max(0.0, min(1.0, float(x) / float(cadd_max)))

def _score_variant(v: VariantDetails) -> Tuple[float, str, str, VariantEvidence, List[str]]:
    rules_fired: List[str] = []
    maps = POLICY.get("maps", {})
    weights = POLICY.get("weights", {})
    thresholds = POLICY.get("priority_thresholds", {})
    cadd_max = float(POLICY.get("cadd_max", ENV["CADD_MAX"]))

    poly_map = {str(k).lower(): float(val) for k, val in maps.get("polyphen", {}).items()}
    sift_map  = {str(k).lower(): float(val) for k, val in maps.get("sift", {}).items()}
    clin_map  = {str(k).lower(): float(val) for k, val in maps.get("clinvar", {}).items()}

    W_CADD = float(weights.get("cadd", ENV["W_CADD"]))
    W_POLY = float(weights.get("polyphen", ENV["W_POLYPHEN"]))
    W_SIFT = float(weights.get("sift", ENV["W_SIFT"]))
    W_CLIN = float(weights.get("clinvar", ENV["W_CLINVAR"]))

    cadd_n = norm_cadd(v.cadd, cadd_max)
    polyphen_score = poly_map.get((v.polyphen or "").lower())
    sift_score = sift_map.get((v.sift or "").lower())
    clinvar_score = clin_map.get((v.clinvar or "").lower())

    score = 0.0
    if cadd_n is not None:
        score += W_CADD * cadd_n
        if v.cadd is not None and v.cadd >= 20:
            rules_fired.append(f"R1:CADD>=20:+{W_CADD * cadd_n:.3f}")
    if polyphen_score is not None:
        score += W_POLY * polyphen_score
        if (v.polyphen or "").lower() in ("probably_damaging", "possibly_damaging"):
            rules_fired.append(f"R2:PolyPhen_damaging:+{W_POLY * polyphen_score:.3f}")
    if sift_score is not None:
        score += W_SIFT * sift_score
        if (v.sift or "").lower() == "damaging":
            rules_fired.append(f"R3:SIFT_damaging:+{W_SIFT * sift_score:.3f}")
    if clinvar_score is not None:
        score += W_CLIN * clinvar_score
        if (v.clinvar or "").lower() in ("pathogenic", "likely_pathogenic"):
            rules_fired.append(f"R4:ClinVar_pathogenic:+{W_CLIN * clinvar_score:.3f}")

    high_th = float(thresholds.get("high", 0.75))
    med_th  = float(thresholds.get("medium", 0.40))

    if score >= high_th: label = "HIGH"
    elif score >= med_th: label = "MEDIUM"
    else: label = "LOW"

    bits = []
    if v.cadd is not None: bits.append(f"CADD {v.cadd}")
    if v.polyphen: bits.append(f"PolyPhen {v.polyphen}")
    if v.sift: bits.append(f"SIFT {v.sift}")
    if v.clinvar: bits.append(f"ClinVar {v.clinvar}")
    rationale = "Variant prioritized as {} based on: {}".format(label, ", ".join(bits)) if bits else f"Variant prioritized as {label} with limited evidence."

    evidence = VariantEvidence(
        cadd_norm=cadd_n, polyphen_score=polyphen_score, sift_score=sift_score, clinvar_score=clinvar_score, rules_fired=rules_fired
    )
    return score, label, rationale, evidence, rules_fired

# ----------------------
# External connectors (imported)
# ----------------------
import asyncio
import httpx
from connectors.vep import annotate_vep
from connectors.opencravat import annotate_opencravat
from connectors.oncokb import annotate_oncokb
from connectors.civic import annotate_civic

# ----------------------
# LLM
# ----------------------
SYSTEM_PROMPT = ""
SUMMARY_INSTRUCTIONS = ""

def _read_file_or_default(path: str, default_text: str) -> str:
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return default_text

def load_prompts() -> None:
    global SYSTEM_PROMPT, SUMMARY_INSTRUCTIONS
    SYSTEM_PROMPT = _read_file_or_default(
        ENV["SYSTEM_PROMPT_FILE"],
        "You are a careful assistant that summarizes variant-level evidence for clinicians in neutral tone."
    )
    SUMMARY_INSTRUCTIONS = _read_file_or_default(
        ENV["SUMMARY_INSTRUCTIONS_FILE"],
        "Return 3–6 concise bullet points (no recommendations) and an optional final 1–2 sentence neutral recap."
    )

def choose_llm_model(candidates: List[str], min_vram_gb: float, catalog: Dict[str, float]) -> Optional[str]:
    for m in candidates:
        if catalog.get(m, 0) >= min_vram_gb:
            return m
    return candidates[-1] if candidates else None

def _ollama_generate(model: str, system_prompt: str, user_prompt: str, timeout_s: int, temperature: float) -> str:
    host = ENV["OLLAMA_HOST"].rstrip("/")
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}",
        "options": {"temperature": temperature},
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

def build_llm_prompt(ehr: Optional[Dict[str, Any]], pv: List[PrioritizedVariant]) -> str:
    lines = []
    if ehr:
        lines.append(f"EHR: {json.dumps(redact_ehr(ehr), ensure_ascii=False)}")
    lines.append("Prioritized variants:")
    for item in pv[:10]:
        v = item.variant
        lines.append(f"- {v.gene or 'NA'} {v.variant_id} | priority={item.priority_label} | score={round(item.priority_score,3)} | CADD={v.cadd} | PolyPhen={v.polyphen} | SIFT={v.sift} | ClinVar={v.clinvar}")
        # Inject top knowledge items (source + level + statement + url)
        kn = (v.extra.knowledge or [])[:3]
        if kn:
            for k in kn:
                src = k.get("source") or "SRC"
                lvl = k.get("evidence_level")
                stmt = k.get("statement") or (k.get("evidence_note") or "")
                url = k.get("url") or k.get("link") or ""
                lines.append(f"  • {src}{(' ' + str(lvl)) if lvl else ''}: {stmt}{(' [' + url + ']') if url else ''}")
    lines.append(SUMMARY_INSTRUCTIONS)
    return "\n".join(lines)

def _fallback_summary(patient_id: str, pv: List[PrioritizedVariant], ehr: Optional[Dict[str, Any]]) -> str:
    top = pv[:3]
    bullets = []
    for p in top:
        v = p.variant
        kb = None
        for k in (v.extra.knowledge or []):
            if k.get("url"):
                kb = k["url"]; break
        bullets.append(
            f"- {v.gene or 'NA'} {v.variant_id}: {p.priority_label} (score {round(p.priority_score,3)}). "
            f"Evidence: CADD={v.cadd}, PolyPhen={v.polyphen}, SIFT={v.sift}, ClinVar={v.clinvar}" + (f" [ref]({kb})" if kb else "")
        )
    tail = "Summary generated by rule-based fallback due to LLM unavailability."
    return "\n".join(bullets + [tail])

# ----------------------
# FastAPI + middleware
# ----------------------
app = FastAPI(title="Genomic CDSS API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception:
            duration_ms = int((time.perf_counter() - start) * 1000)
            request.state.duration_ms = duration_ms
            raise
        duration_ms = int((time.perf_counter() - start) * 1000)
        request.state.duration_ms = duration_ms
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-MS"] = str(duration_ms)
        return response

app.add_middleware(RequestContextMiddleware)

# ----------------------
# Validators & normalizers (unchanged)
# ----------------------
def normalize_vcf_line(line: str) -> Optional[Dict[str, str]]:
    parts = line.strip().split("\t")
    if len(parts) < 5:
        return None
    chrom, pos, _id, ref, alt = parts[0:5]
    chrom = str(chrom).lstrip("chr")
    vid = f"chr{chrom}:{pos}:{ref}:{alt}"
    return {"variant_id": vid, "gene": None}

def normalize_csv_row(row: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if "variant_id" in row:
        vid = str(row["variant_id"]).strip()
        gene = ("" if pd.isna(row.get("gene")) else str(row.get("gene"))).strip() or None
        return {"variant_id": vid, "gene": gene} if vid else None
    need = all(k in row for k in ("CHROM","POS","REF","ALT"))
    if need:
        vid = f"chr{row['CHROM']}:{row['POS']}:{row['REF']}:{row['ALT']}"
        gene = ("" if pd.isna(row.get("gene")) else str(row.get("gene"))).strip() or None
        return {"variant_id": vid, "gene": gene}
    return None

def parse_ehr_csv_text(txt: str) -> Dict[str, Any]:
    from io import StringIO
    df = pd.read_csv(StringIO(txt))
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    out = {}
    for k, v in row.items():
        if k == "age":
            try: out[k] = int(v)
            except Exception: out[k] = None
        else:
            out[k] = None if pd.isna(v) else str(v)
    return out

def validate_variants(rows: List[Dict[str, Optional[str]]], *, require_gene: bool) -> Optional[JSONResponse]:
    details = []
    seen = set()
    for idx, r in enumerate(rows):
        if not r.get("variant_id"):
            details.append({"row": idx+1, "field": "variant_id", "issue": "missing"})
            continue
        if r["variant_id"] in seen:
            details.append({"row": idx+1, "field": "variant_id", "issue": "duplicate"})
        seen.add(r["variant_id"])
        if require_gene and not r.get("gene"):
            details.append({"row": idx+1, "field": "gene", "issue": "missing"})
    if len(rows) > ENV["MAX_VARIANTS_PER_PATIENT"]:
        details.append({"limit": ENV["MAX_VARIANTS_PER_PATIENT"], "count": len(rows), "issue": "too_many_variants"})
    if details:
        return error("VALIDATION_ERROR", "Invalid variant upload", details, status=400)
    return None

def validate_ehr(ehr: Dict[str, Any]) -> Optional[JSONResponse]:
    allowed = {"age","sex","cancer_type","stage","treatment"}
    details = []
    for k in ehr.keys():
        if k not in allowed:
            details.append({"field": k, "issue": "unknown"})
    if "age" in ehr:
        try:
            if ehr["age"] is not None:
                ehr["age"] = int(ehr["age"])
        except Exception:
            details.append({"field":"age","issue":"not_integer"})
    if "sex" in ehr and ehr["sex"] not in (None,"M","F","Male","Female"):
        details.append({"field":"sex","issue":"invalid"})
    if details:
        return error("VALIDATION_ERROR", "Invalid EHR upload", details, status=400)
    return None

# ----------------------
# Startup
# ----------------------
ANNOTATOR: VariantAnnotator
MODEL_SELECTED: Optional[str]

@app.on_event("startup")
def _startup() -> None:
    global POLICY, POLICY_VERSION, KNOWLEDGE, ANNOTATOR, MODEL_SELECTED
    load_prompts()
    POLICY = load_policy(ENV["POLICY_FILE"])
    POLICY_VERSION = str(POLICY.get("version", "0.0.0"))
    KNOWLEDGE = KnowledgeBase(ENV["GENE_KNOWLEDGE_CSV"])
    ANNOTATOR = VariantAnnotator(ENV["VARIANT_SCORES_PATH"])
    MODEL_SELECTED = choose_llm_model(ENV["LLM_CANDIDATES"], ENV["LLM_MIN_VRAM_GB"], ENV["LLM_VRAM_CATALOG"])
    if ENV["LLM_PROVIDER"].upper() == "OLLAMA" and not MODEL_SELECTED:
        print("[WARN] No suitable LLM model fits VRAM; fallback may be used.")

# ----------------------
# Upload endpoints
# ----------------------
@app.post("/upload/variants")
async def upload_variants(patient_id: str = Form(...), file: UploadFile = File(...), request: Request = None):
    if file.content_type not in ("text/plain", "text/tab-separated-values", "text/csv", "application/octet-stream", "application/vnd.ms-excel"):
        return error("UNSUPPORTED_FORMAT", f"Unsupported content-type: {file.content_type}", status=415)

    content = (await file.read()).decode("utf-8", errors="replace")
    rows: List[Dict[str, Optional[str]]] = []

    lines = content.splitlines()
    header = lines[0] if lines else ""
    if header.startswith("##") or any(l.startswith("#CHROM") for l in lines[:5]):
        for line in lines:
            if not line or line.startswith("#"):
                continue
            if "\t" not in line and "  " in line:
                line = "\t".join([c for c in line.split() if c != ""])
            r = normalize_vcf_line(line)
            if r: rows.append(r)
    else:
        from io import StringIO
        df = pd.read_csv(StringIO(content))
        for _, row in df.iterrows():
            r = normalize_csv_row(row.to_dict())
            if r: rows.append(r)

    # Enrich missing gene via annotator before validation
    enriched = []
    for r in rows:
        if not r.get("gene"):
            ann = ANNOTATOR.annotate(VariantInput(variant_id=r["variant_id"], gene=None))
            if ann.gene:
                r["gene"] = ann.gene
        enriched.append(r)

    bad = validate_variants(enriched, require_gene=True)
    if bad:
        return bad

    UPLOAD_STORE.setdefault(patient_id, {})["variants"] = enriched
    return {"ok": True, "patient_id": patient_id, "count": len(enriched)}

@app.post("/upload/ehr")
async def upload_ehr(patient_id: str = Form(...), file: UploadFile = File(None), ehr_json: Optional[str] = Form(None)):
    if file is None and not ehr_json:
        return error("NO_INPUT", "Provide a JSON string in ehr_json or upload a CSV file.", status=400)
    if file:
        txt = (await file.read()).decode("utf-8", errors="replace")
        ehr = parse_ehr_csv_text(txt)
    else:
        try:
            ehr = json.loads(ehr_json)
        except Exception:
            return error("BAD_JSON", "ehr_json is not valid JSON", status=400)

    bad = validate_ehr(ehr)
    if bad:
        return bad

    UPLOAD_STORE.setdefault(patient_id, {})["ehr"] = ehr
    return {"ok": True, "patient_id": patient_id, "ehr_keys": list(ehr.keys())}

# ----------------------
# External annotation orchestrator (API-only, no cache)
# ----------------------
async def annotate_variant_all_external(variant: Dict[str, Any], ehr: Dict[str, Any]) -> Dict[str, Any]:
    """
    variant: VariantDetails.dict()
    Returns: merged extra.api_evidence and extra.knowledge (OncoKB/CIViC) without replacing local scores.
    """
    gene = variant.get("gene")
    disease = (ehr or {}).get("cancer_type")
    var_id = variant["variant_id"]

    api_evidence: List[Dict[str, Any]] = []
    ext_knowledge: List[Dict[str, Any]] = []

    async with httpx.AsyncClient() as session:
        tasks = []
        # VEP
        if ENV["ENABLE_VEP"]:
            tasks.append(annotate_vep(session, var_id))
        else:
            tasks.append(asyncio.sleep(0, result=[]))
        # OpenCRAVAT (optional API)
        if ENV["ENABLE_OPENCRAVAT"]:
            tasks.append(annotate_opencravat(session, var_id))
        else:
            tasks.append(asyncio.sleep(0, result=[]))
        # OncoKB
        if ENV["ENABLE_ONCOKB"] and gene:
            tasks.append(annotate_oncokb(session, var_id, gene, disease))
        else:
            tasks.append(asyncio.sleep(0, result=([], [])))
        # CIViC
        if ENV["ENABLE_CIVIC"] and gene:
            tasks.append(annotate_civic(session, gene, variant_label=var_id))
        else:
            tasks.append(asyncio.sleep(0, result=([], [])))

        vep_ev, cravat_ev, (onc_ev, onc_kn), (civ_ev, civ_kn) = await asyncio.gather(*tasks)

    api_evidence.extend(vep_ev); api_evidence.extend(cravat_ev); api_evidence.extend(onc_ev); api_evidence.extend(civ_ev)
    ext_knowledge.extend(onc_kn); ext_knowledge.extend(civ_kn)

    return {"api_evidence": api_evidence, "knowledge": ext_knowledge}

# ----------------------
# Analyze
# ----------------------
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, request: Request):
    _start = time.perf_counter()
    patient_id = req.patient_id
    stored = UPLOAD_STORE.get(patient_id, {})
    variants_in = req.variants or [VariantInput(**v) for v in stored.get("variants", [])]
    ehr_in = (req.ehr.dict() if req.ehr else stored.get("ehr")) if stored else (req.ehr.dict() if req.ehr else None)

    rows = [{"variant_id": v.variant_id, "gene": v.gene} for v in (variants_in or [])]
    if not rows:
        return error("NO_VARIANTS", "No variants provided (upload first or include in body).", status=400)

    bad = validate_variants(rows, require_gene=False)
    if bad:
        payload = bad.body
        payload["request_id"] = getattr(request.state, "request_id", str(uuid.uuid4()))
        payload["duration_ms"] = getattr(request.state, "duration_ms", 0)
        return JSONResponse(status_code=bad.status_code, content=payload)

    # local score annotation
    annotated: List[VariantDetails] = [ANNOTATOR.annotate(v) for v in variants_in]
    prioritized: List[PrioritizedVariant] = []
    for v in annotated:
        # Start knowledge with gene-level CSV fallback (if any)
        v.extra.knowledge = (KNOWLEDGE.get(v.gene) or [])
        score, label, rationale, evidence, rules = _score_variant(v)
        v.extra.evidence = evidence
        v.extra.audit.update({"scored_at": datetime.now(timezone.utc).isoformat(), "policy_version": POLICY_VERSION})
        prioritized.append(PrioritizedVariant(
            variant=v,
            priority_score=round(float(score), 6),
            priority_label=label,
            rationale=rationale
        ))
    prioritized.sort(key=lambda x: x.priority_score, reverse=True)

    # --- NEW: live external evidence/knowledge (API-only) ---
    ehr_dict = ehr_in or {}
    # Execute calls in parallel for all prioritized variants
    externals = await asyncio.gather(*[
        annotate_variant_all_external(item.variant.dict(), ehr_dict) for item in prioritized
    ])
    for item, ext in zip(prioritized, externals):
        # append external evidence
        item.variant.extra.api_evidence = (item.variant.extra.api_evidence or []) + (ext.get("api_evidence") or [])
        # merge knowledge: OncoKB/CIViC statements first, keep CSV after
        ext_kn = ext.get("knowledge") or []
        if ext_kn:
            # put external knowledge ahead of CSV entries
            item.variant.extra.knowledge = ext_kn + (item.variant.extra.knowledge or [])

        # audit mark
        item.variant.extra.audit["external_enrichment"] = {
            "sources": {
                "VEP": ENV["ENABLE_VEP"],
                "OpenCRAVAT": ENV["ENABLE_OPENCRAVAT"],
                "OncoKB": ENV["ENABLE_ONCOKB"],
                "CIViC": ENV["ENABLE_CIVIC"],
            },
            "completed_at": datetime.now(timezone.utc).isoformat()
        }

    _duration_ms = int((time.perf_counter() - _start) * 1000)

    if ENV["AUDIT_ENABLED"]:
        audit_artifact = {
            "ts": _ts(),
            "request_id": getattr(request.state, "request_id", ""),
            "endpoint": "/analyze",
            "input": {"patient_id": patient_id, "variants_count": len(variants_in)},
            "policy_version": POLICY_VERSION,
            "config": {
                "policy_file": ENV["POLICY_FILE"],
                "external": {
                    "vep": ENV["ENABLE_VEP"],
                    "opencravat": ENV["ENABLE_OPENCRAVAT"],
                    "oncokb": ENV["ENABLE_ONCOKB"],
                    "civic": ENV["ENABLE_CIVIC"],
                }
            },
            "result": [{"vid": p.variant.variant_id, "score": p.priority_score, "label": p.priority_label} for p in prioritized],
            "duration_ms": _duration_ms,
        }
        Path(ENV["AUDIT_DIR"], f"{getattr(request.state, 'request_id', str(uuid.uuid4()))}.analyze.json").write_text(
            json.dumps(audit_artifact, indent=2), encoding="utf-8"
        )

    return AnalyzeResponse(
        patient_id=patient_id,
        policy_version=POLICY_VERSION,
        prioritized=prioritized,
        request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
        duration_ms=_duration_ms,
    )

# ----------------------
# LLM summary (injects knowledge)
# ----------------------
@app.post("/llm_summary", response_model=LLMResponse)
def llm_summary(req: LLMRequest, request: Request):
    _start = time.perf_counter()
    model = choose_llm_model(ENV["LLM_CANDIDATES"], ENV["LLM_MIN_VRAM_GB"], ENV["LLM_VRAM_CATALOG"]) if ENV["LLM_PROVIDER"].upper() == "OLLAMA" else None

    ehr_for_prompt = req.ehr.dict() if (req.ehr and ENV["SEND_EHR_TO_LLM"]) else None
    prompt = build_llm_prompt(ehr_for_prompt, req.variants)

    used_fallback = False
    text = ""
    try:
        if model:
            text = _ollama_generate(model, SYSTEM_PROMPT, prompt, ENV["LLM_TIMEOUT_SECONDS"], ENV["LLM_TEMPERATURE"])
        else:
            raise RuntimeError("No LLM model selected")
        if ENV["SUMMARY_MAX_WORDS"] and ENV["SUMMARY_MAX_WORDS"] > 0:
            words = text.split()
            if len(words) > ENV["SUMMARY_MAX_WORDS"]:
                text = " ".join(words[:ENV["SUMMARY_MAX_WORDS"]]) + "…"
    except Exception:
        used_fallback = True
        text = _fallback_summary(req.patient_id, req.variants, ehr_for_prompt)

    _duration_ms = int((time.perf_counter() - _start) * 1000)

    if ENV["AUDIT_ENABLED"]:
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        audit_artifact = {
            "ts": _ts(),
            "request_id": getattr(request.state, "request_id", ""),
            "endpoint": "/llm_summary",
            "input": {"patient_id": req.patient_id, "variants_count": len(req.variants)},
            "llm": {"model": model if model else None, "used_fallback": used_fallback, "prompt_sha256": h},
            "duration_ms": _duration_ms,
        }
        Path(ENV["AUDIT_DIR"], f"{getattr(request.state, 'request_id', str(uuid.uuid4()))}.llm.json").write_text(
            json.dumps(audit_artifact, indent=2), encoding="utf-8"
        )

    return LLMResponse(
        patient_id=req.patient_id,
        model=(model if not used_fallback else "fallback-rule-based"),
        summary=text,
        generated_at=datetime.now(timezone.utc).isoformat(),
        request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
        duration_ms=_duration_ms,
    )

# ----------------------
# Health & config (minor bump)
# ----------------------
@app.get("/health/all")
def health_all():
    exists_policy = Path(ENV["POLICY_FILE"]).exists()
    exists_scores = Path(ENV["VARIANT_SCORES_PATH"]).exists()
    exists_knowledge = Path(ENV["GENE_KNOWLEDGE_CSV"]).exists()
    audit_writable = os.access(ENV["AUDIT_DIR"], os.W_OK)
    return {
        "ok": bool(exists_policy and exists_scores),
        "policy_file": {"path": ENV["POLICY_FILE"], "exists": exists_policy, "version": POLICY_VERSION},
        "scores_file": {"path": ENV["VARIANT_SCORES_PATH"], "exists": exists_scores},
        "knowledge_file": {"path": ENV["GENE_KNOWLEDGE_CSV"], "exists": exists_knowledge},
        "audit": {"path": ENV["AUDIT_DIR"], "writable": audit_writable, "enabled": ENV["AUDIT_ENABLED"]},
        "llm": {"provider": ENV["LLM_PROVIDER"], "selected_model": choose_llm_model(ENV["LLM_CANDIDATES"], ENV["LLM_MIN_VRAM_GB"], ENV["LLM_VRAM_CATALOG"])},
        "external": {
            "VEP": ENV["ENABLE_VEP"],
            "OpenCRAVAT": ENV["ENABLE_OPENCRAVAT"],
            "OncoKB": ENV["ENABLE_ONCOKB"],
            "CIViC": ENV["ENABLE_CIVIC"],
        },
        "service": {"name": "Genomic CDSS API", "version": "1.1.0"},
    }

@app.get("/config")
def get_config():
    return {
        "MAX_VARIANTS_PER_PATIENT": ENV["MAX_VARIANTS_PER_PATIENT"],
        "POLICY_FILE": ENV["POLICY_FILE"],
        "GENE_KNOWLEDGE_CSV": ENV["GENE_KNOWLEDGE_CSV"],
        "VARIANT_SCORES_PATH": ENV["VARIANT_SCORES_PATH"],
        "SYSTEM_PROMPT_FILE": ENV["SYSTEM_PROMPT_FILE"],
        "SUMMARY_INSTRUCTIONS_FILE": ENV["SUMMARY_INSTRUCTIONS_FILE"],
        "AUDIT_ENABLED": ENV["AUDIT_ENABLED"],
        "AUDIT_DIR": ENV["AUDIT_DIR"],
        "REDACT_EHR_FIELDS": ENV["REDACT_EHR_FIELDS"],
        "LLM": {
            "provider": ENV["LLM_PROVIDER"],
            "candidates": ENV["LLM_CANDIDATES"],
            "min_vram_gb": ENV["LLM_MIN_VRAM_GB"],
            "timeout_s": ENV["LLM_TIMEOUT_SECONDS"],
            "temperature": ENV["LLM_TEMPERATURE"],
            "vram_catalog": ENV["LLM_VRAM_CATALOG"],
        },
        "external": {
            "ENABLE_VEP": ENV["ENABLE_VEP"],
            "ENABLE_OPENCRAVAT": ENV["ENABLE_OPENCRAVAT"],
            "ENABLE_ONCOKB": ENV["ENABLE_ONCOKB"],
            "ENABLE_CIVIC": ENV["ENABLE_CIVIC"],
            "VEP_REST_BASE": ENV["VEP_REST_BASE"],
            "OPENCRAVAT_API_BASE": ENV["OPENCRAVAT_API_BASE"],
            "ONCOKB_API_BASE": ENV["ONCOKB_API_BASE"],
            "CIVIC_BASE_URL": ENV["CIVIC_BASE_URL"],
            "ANNOTATION_TIMEOUT_S": ENV["ANNOTATION_TIMEOUT_S"],
        },
        "policy_version": POLICY_VERSION,
    }

@app.get("/")
def root():
    return {"ok": True, "name": "Genomic CDSS API", "version": "1.1.0", "policy_version": POLICY_VERSION}
