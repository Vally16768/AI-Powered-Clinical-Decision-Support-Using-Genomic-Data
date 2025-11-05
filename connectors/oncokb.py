import os, httpx

BASE = os.getenv("ONCOKB_API_BASE", "https://www.oncokb.org/api/v1")
TOKEN = os.getenv("ONCOKB_API_TOKEN", "")
TIMEOUT = float(os.getenv("ANNOTATION_TIMEOUT_S", "8"))

async def annotate_oncokb(session: httpx.AsyncClient, variant_id: str, gene: str, disease: str|None):
    """
    Returns (evidence:list[dict], knowledge:list[dict]).
    knowledge items include: source, statement, url, evidence_level, disease, drugs
    """
    if not TOKEN:
        return [], []
    hdrs = {"Authorization": f"Bearer {TOKEN}"}
    chrom, pos, ref, alt = variant_id.replace("chr","").split(":")
    params = {"genomicLocation": f"{chrom}:{pos}:{ref}:{alt}", "referenceGenome": "GRCh37"}
    r = await session.get(f"{BASE}/annotate/mutations/byGenomicChange", headers=hdrs, params=params, timeout=TIMEOUT)
    if r.status_code >= 400:
        return [], []
    d = r.json() or {}
    ev=[]; kn=[]
    me = d.get("mutationEffect") or {}
    if me.get("description"):
        ev.append({"source":"OncoKB","type":"effect","key":"MutationEffect","value":me["description"],"url":"https://www.oncokb.org/"})
    for tx in d.get("treatments", []) or []:
        level = tx.get("levelOfEvidence")
        drugs = [t.get("drugName") for t in (tx.get("drugs") or []) if t.get("drugName")]
        stmt = f"{gene} {ref}>{alt}: evidence {level}" + (f" for {', '.join(drugs)}" if drugs else "")
        url = tx.get("url") or "https://www.oncokb.org/"
        kn.append({"source":"OncoKB","statement":stmt,"url":url,"evidence_level":level,"disease":disease,"drugs":drugs})
    return ev, kn
