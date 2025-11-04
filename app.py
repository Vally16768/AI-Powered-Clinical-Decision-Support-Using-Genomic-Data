import os
from typing import List, Optional, Literal, Dict, Any
import pandas as pd, requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

# -------- Config --------
load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL   = os.getenv("LLM_MODEL", "llama3:8b-instruct")
MAX_VARIANTS= int(os.getenv("MAX_VARIANTS_PER_PATIENT", "50"))
SCORES_CSV  = os.getenv("SCORES_CSV", "variant_scores.csv")

# -------- Schemas -------
class VariantIn(BaseModel):
    variant_id: str = Field(..., description="chr:pos:ref:alt")
    gene: Optional[str] = None

class PatientEHR(BaseModel):
    age: Optional[int] = None
    sex: Optional[Literal["M","F"]] = None
    cancer_type: Optional[str] = None
    stage: Optional[str] = None
    labs: Optional[Dict[str, float]] = None
    treatments: Optional[List[str]] = None

class AnalyzeRequest(BaseModel):
    patient_id: str
    variants: List[VariantIn]
    ehr: Optional[PatientEHR] = None

class VariantScore(BaseModel):
    variant_id: str
    gene: Optional[str] = None
    cadd: Optional[float] = None
    polyphen: Optional[str] = None
    sift: Optional[str] = None
    clinvar: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

class PrioritizedVariant(BaseModel):
    variant: VariantScore
    priority_score: float
    priority_label: Literal["HIGH","MEDIUM","LOW"]
    rationale: str

class AnalyzeResponse(BaseModel):
    patient_id: str
    prioritized: List[PrioritizedVariant]
    llm_summary: Optional[str] = None

# ----- Annotator (CSV) -----
class VariantAnnotator:
    def __init__(self, csv_path: str = SCORES_CSV):
        df = pd.read_csv(csv_path)
        df["variant_id"] = df["variant_id"].astype(str)
        self.idx = df.set_index("variant_id")

    def annotate(self, variants: List[VariantIn]) -> List[VariantScore]:
        out: List[VariantScore] = []
        for v in variants:
            if v.variant_id in self.idx.index:
                row = self.idx.loc[v.variant_id]
                out.append(VariantScore(
                    variant_id=v.variant_id,
                    gene=(row.get("gene") if isinstance(row.get("gene"), str) else v.gene),
                    cadd=float(row.get("cadd")) if pd.notna(row.get("cadd")) else None,
                    polyphen=row.get("polyphen") if isinstance(row.get("polyphen"), str) else None,
                    sift=row.get("sift") if isinstance(row.get("sift"), str) else None,
                    clinvar=row.get("clinvar") if isinstance(row.get("clinvar"), str) else None,
                ))
            else:
                out.append(VariantScore(variant_id=v.variant_id, gene=v.gene))
        return out

# -------- Rules ----------
POLYPHEN_MAP = {"benign":0.0, "possibly_damaging":0.5, "probably_damaging":1.0}
SIFT_MAP     = {"tolerated":0.0, "damaging":1.0}
CLINVAR_MAP  = {"benign":0.0, "likely_benign":0.25, "VUS":0.5, "likely_pathogenic":0.75, "pathogenic":1.0}
W_CADD=0.5; W_POLY=0.2; W_SIFT=0.15; W_CLIN=0.15

def _norm_cadd(c):
    if c is None: return 0.0
    c = min(max(c,0.0),40.0); return c/40.0

def _map(v,m): return float(m.get((v or "").lower(),0.0))

def _label(s): return "HIGH" if s>=0.65 else ("MEDIUM" if s>=0.35 else "LOW")

def _explain(v,s):
    parts=[]
    if v.cadd is not None: parts.append(f"CADD={v.cadd}")
    if v.polyphen: parts.append(f"PolyPhen={v.polyphen}")
    if v.sift: parts.append(f"SIFT={v.sift}")
    if v.clinvar: parts.append(f"ClinVar={v.clinvar}")
    detail = ", ".join(parts) or "no external scores available"
    gene = v.gene or "UnknownGene"
    return f"{gene} {v.variant_id} prioritized due to {detail}. Composite score={s:.2f}."

def prioritize(ann: List[VariantScore]) -> List[PrioritizedVariant]:
    out=[]
    for v in ann:
        s = (W_CADD*_norm_cadd(v.cadd) + W_POLY*_map(v.polyphen,POLYPHEN_MAP)
             + W_SIFT*_map(v.sift,SIFT_MAP) + W_CLIN*_map(v.clinvar,CLINVAR_MAP))
        out.append(PrioritizedVariant(variant=v, priority_score=s, priority_label=_label(s), rationale=_explain(v,s)))
    out.sort(key=lambda x: x.priority_score, reverse=True)
    return out

# -------- LLM (optional, Ollama) ----------
class LocalLLMClient:
    def __init__(self, host=OLLAMA_HOST, model=LLM_MODEL):
        self.url = f"{host}/api/generate"; self.model = model
    def generate(self, system_prompt: str, user_prompt: str, max_tokens=512, temperature=0.2) -> str:
        try:
            payload = {
                "model": self.model,
                "prompt": f"<s>[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n[USER]\n{user_prompt}\n[/USER]\n[ASSISTANT]",
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "stream": False
            }
            r = requests.post(self.url, json=payload, timeout=60)
            r.raise_for_status()
            return r.json().get("response","")
        except Exception:
            return ""

def system_prompt() -> str:
    return ("You are a clinical genomics assistant. Summarize prioritized variants for oncology. "
            "Use cautious language, avoid treatment advice, mention uncertainty and suggest validation.")

def user_prompt(pid: str, items: List[PrioritizedVariant], ehr: Optional[PatientEHR]) -> str:
    ehr_lines=[]
    if ehr:
        if ehr.age is not None: ehr_lines.append(f"Age: {ehr.age}")
        if ehr.sex: ehr_lines.append(f"Sex: {ehr.sex}")
        if ehr.cancer_type: ehr_lines.append(f"Cancer type: {ehr.cancer_type}")
        if ehr.stage: ehr_lines.append(f"Stage: {ehr.stage}")
    lines=[f"Patient: {pid}","","Prioritized variants:"]
    for pv in items[:10]:
        v=pv.variant
        lines.append(
            f"- {v.gene or ''} {v.variant_id}: {pv.priority_label} (score {pv.priority_score:.2f}); "
            f"CADD={v.cadd}, PolyPhen={v.polyphen}, SIFT={v.sift}, ClinVar={v.clinvar}. Rationale: {pv.rationale}"
        )
    guidance = ("\nPlease provide a short (<= 180 words) clinician-friendly summary "
                "that highlights key variants, states uncertainty, and avoids treatment advice.")
    return ("\n".join(ehr_lines) + "\n\n" + "\n".join(lines) + guidance)

# -------- FastAPI --------
app = FastAPI(title="CDSS Genomics – Orchestrator (no Docker)", version="0.1.0")
annotator = VariantAnnotator()
llm = LocalLLMClient()

@app.get("/")
def root(): return {"ok": True}

@app.post("/prioritize", response_model=AnalyzeResponse)
def prioritize_endpoint(req: AnalyzeRequest):
    if len(req.variants) > MAX_VARIANTS: raise HTTPException(400, f"Too many variants; limit={MAX_VARIANTS}")
    ann = annotator.annotate(req.variants); pr = prioritize(ann)
    return AnalyzeResponse(patient_id=req.patient_id, prioritized=pr)

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(req: AnalyzeRequest):
    base = prioritize_endpoint(req)
    try:
        sp = system_prompt(); up = user_prompt(base.patient_id, base.prioritized, req.ehr)
        summary = llm.generate(sp, up)
    except Exception: summary = ""
    base.llm_summary = summary or None
    return base

if __name__ == "__main__":
    demo = AnalyzeRequest(
        patient_id="P001",
        variants=[VariantIn(variant_id="chr17:7579472:C:T", gene="TP53"),
                  VariantIn(variant_id="chr12:25398284:G:A", gene="KRAS")],
        ehr=PatientEHR(age=71, sex="M", cancer_type="CRC", stage="III")
    )
    import json; print(json.dumps(analyze_endpoint(demo).model_dump(), indent=2))
