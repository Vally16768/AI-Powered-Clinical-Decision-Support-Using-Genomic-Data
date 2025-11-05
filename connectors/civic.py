import os, httpx

BASE = os.getenv("CIVIC_BASE_URL", "https://civicdb.org/api/graphql")
TIMEOUT = float(os.getenv("ANNOTATION_TIMEOUT_S", "8"))

QUERY = """
query VariantEvidence($gene:String!, $variant:String!){
  evidenceItems(geneSymbol:$gene, variantName:$variant, status:ACCEPTED){
    nodes { id, evidenceLevel, evidenceType, description, disease { name }, drugs { name }, source { url } }
  }
}
"""

async def annotate_civic(session: httpx.AsyncClient, gene: str, variant_label: str):
    """
    Returns (evidence:list[dict], knowledge:list[dict]).
    """
    r = await session.post(BASE, json={"query": QUERY, "variables": {"gene": gene, "variant": variant_label}}, timeout=TIMEOUT)
    if r.status_code >= 400:
        return [], []
    data = r.json() or {}
    nodes = (data.get("data") or {}).get("evidenceItems", {}).get("nodes", []) or []
    ev=[]; kn=[]
    for it in nodes:
        level = it.get("evidenceLevel")
        desc = it.get("description")
        disease = (it.get("disease") or {}).get("name")
        drugs = [x.get("name") for x in (it.get("drugs") or []) if x.get("name")]
        url = (it.get("source") or {}).get("url") or "https://civicdb.org"
        if desc:
            ev.append({"source":"CIViC","type": it.get("evidenceType") or "evidence",
                       "key":"Description","value":desc,"url":url,"evidence_level":level})
            kn.append({"source":"CIViC","statement":desc,"url":url,"evidence_level":level,"disease":disease,"drugs":drugs})
    return ev, kn
