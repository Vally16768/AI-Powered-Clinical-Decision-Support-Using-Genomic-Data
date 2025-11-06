# connectors/civic.py
import os, httpx, asyncio
from typing import List, Dict, Any, Optional

BASE = os.getenv("CIVIC_BASE_URL", "https://civicdb.org/api/graphql")
TIMEOUT = float(os.getenv("ANNOTATION_TIMEOUT_S", "8"))

# Query EIs by Molecular Profile name, e.g. "TP53 R273H"
Q_BY_MP = """
query EIs($mp:String!, $first:Int = 20){
  evidenceItems(molecularProfileName:$mp, status:ACCEPTED, first:$first){
    nodes{
      id
      evidenceLevel
      evidenceType
      description
      disease { name }
      therapies { name }
      source { link }
    }
  }
}
"""

async def _post_graphql(session: httpx.AsyncClient, query: str, variables: dict, max_retries: int = 3):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "genomic-cdss/1.1"
    }
    delay = 0.75
    for attempt in range(1, max_retries + 1):
        try:
            r = await session.post(BASE, json={"query": query, "variables": variables},
                                   headers=headers, timeout=TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                print(f"[CIViC][WARN] HTTP {r.status_code} (attempt {attempt})")
                await asyncio.sleep(delay); delay *= 1.6; continue
            if r.status_code >= 400:
                print(f"[CIViC][ERR] HTTP {r.status_code} for vars={variables}")
                try:
                    print("[CIViC][ERR] Body:", (r.text or "")[:240])
                except Exception:
                    pass
            return r
        except Exception as e:
            print(f"[CIViC][WARN] Exception (attempt {attempt}): {e}")
            await asyncio.sleep(delay); delay *= 1.6
    return None

def _emit(nodes: List[Dict[str, Any]]):
    ev, kn = [], []
    for it in nodes or []:
        level = it.get("evidenceLevel")
        etype = it.get("evidenceType") or "evidence"
        desc  = it.get("description")
        disease = (it.get("disease") or {}).get("name")
        therapies = [x.get("name") for x in (it.get("therapies") or []) if x and x.get("name")]
        url = (it.get("source") or {}).get("link") or "https://civicdb.org"
        if desc:
            ev.append({"source":"CIViC","type": etype,
                       "key":"Description","value":desc,"url":url,"evidence_level":level})
            kn.append({"source":"CIViC","statement":desc,"url":url,
                       "evidence_level":level,"disease":disease,"drugs":therapies})
    return ev, kn

def _mp_candidates(gene: str, variant_label: Optional[str], hgvsp_short: Optional[str]) -> List[str]:
    cands = []
    # Prefer precise HGVSp short first (e.g., G12D, R273H)
    if hgvsp_short:
        cands.append(f"{gene} {hgvsp_short}")
        cands.append(f"{gene.upper()} {hgvsp_short.upper()}")
    # If the provided label already looks like short AA notation, try it too
    if variant_label:
        tok = variant_label.split()[-1].upper()
        # allow forms like R273H, G12D (letters+digits+letters)
        import re
        if re.match(r"^[A-Z]\d+[A-Z\*]$", tok):
            cands.append(f"{gene} {tok}")
            cands.append(f"{gene.upper()} {tok}")
    # Dedup preserving order
    seen = set(); out=[]
    for x in cands:
        if x not in seen:
            seen.add(x); out.append(x)
    return out or [f"{gene} {hgvsp_short}"] if hgvsp_short else []

async def annotate_civic(
    session: httpx.AsyncClient,
    gene: str,
    variant_label: str | None = None,
    hgvsp_short: str | None = None
):
    """
    CIViC v2 integration.
    Strategy:
      - Build Molecular Profile (MP) candidates like "TP53 R273H" from HGVSp short
        (preferred) and, if plausible, from `variant_label`.
      - Query evidenceItems(molecularProfileName: ...) for each candidate until we get hits.
    Returns: (evidence, knowledge)
    """
    mps = _mp_candidates(gene, variant_label, hgvsp_short)
    for mp in mps:
        r = await _post_graphql(session, Q_BY_MP, {"mp": mp, "first": 20})
        if not r:
            continue
        if r.status_code >= 400:
            continue
        data = r.json() or {}
        nodes = (data.get("data") or {}).get("evidenceItems", {}).get("nodes", []) or []
        if nodes:
            return _emit(nodes)

    print(f"[CIViC][WARN] No evidence for gene={gene} mp_candidates={mps or ['-']}")
    return [], []
