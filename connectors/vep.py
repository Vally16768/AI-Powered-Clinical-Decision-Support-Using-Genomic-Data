import os, httpx

BASE = os.getenv("VEP_REST_BASE", "https://rest.ensembl.org/vep/human/region")
TIMEOUT = float(os.getenv("ANNOTATION_TIMEOUT_S", "8"))

def _fmt_variant(variant_id: str) -> str:
    # "chr17:7579472:C:T" -> "17:7579472/C/T"
    chrom, pos, ref, alt = variant_id.replace("chr", "").split(":")
    return f"{chrom}:{pos}/{ref}/{alt}"

async def annotate_vep(session: httpx.AsyncClient, variant_id: str):
    """
    Returns a list[dict] evidence items:
    {source:"VEP", type:"consequence|impact|transcript", key:"Consequence|Impact|Transcript", value:..., url:"https://www.ensembl.org/"}
    """
    region = _fmt_variant(variant_id)
    url = f"{BASE}/{region}"
    r = await session.get(url, params={"content-type":"application/json"}, timeout=TIMEOUT)
    if r.status_code >= 400:
        return []
    data = r.json() or []
    out=[]
    for rec in data:
        for tr in rec.get("transcript_consequences", []):
            if "consequence_terms" in tr:
                out.append({"source":"VEP","type":"consequence","key":"Consequence",
                            "value":", ".join(tr["consequence_terms"]), "url":"https://www.ensembl.org/"})
            if "impact" in tr:
                out.append({"source":"VEP","type":"impact","key":"Impact","value":tr["impact"], "url":"https://www.ensembl.org/"})
            if "transcript_id" in tr:
                out.append({"source":"VEP","type":"transcript","key":"Transcript","value":tr["transcript_id"], "url":"https://www.ensembl.org/"})
    # deduplicate
    seen=set(); dedup=[]
    for e in out:
        t=(e["type"], e["key"], str(e["value"]))
        if t in seen: continue
        seen.add(t); dedup.append(e)
    return dedup
