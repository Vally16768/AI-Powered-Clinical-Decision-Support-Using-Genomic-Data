# connectors/vep.py
import os
import re
from typing import List, Dict, Any, Optional
import httpx

# Correct VEP region endpoint (ALT only; add strand; ask for HGVS)
BASE = os.getenv("VEP_REST_BASE", "https://rest.ensembl.org/vep/human/region")
TIMEOUT = float(os.getenv("ANNOTATION_TIMEOUT_S", "8"))

AA3_TO_1 = {
    "Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G",
    "His":"H","Ile":"I","Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S",
    "Thr":"T","Trp":"W","Tyr":"Y","Val":"V","Ter":"*","Sec":"U"
}

def _to_hgvsp_short(hgvsp: str) -> Optional[str]:
    # Examples: "ENSP00000269305.4:p.Arg273His", "p.Gly12Asp"
    if not hgvsp:
        return None
    m = re.search(r":p\.([A-Za-z]{3})(\d+)([A-Za-z]{3}|\*)$", hgvsp) or \
        re.search(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3}|\*)$", hgvsp)
    if not m:
        return None
    a1, pos, a2 = m.groups()
    a1s = AA3_TO_1.get(a1[:3].title())
    a2s = "*" if a2 == "*" else AA3_TO_1.get(a2[:3].title())
    if not a1s or not a2s:
        return None
    return f"{a1s}{pos}{a2s}"

async def annotate_vep(session: httpx.AsyncClient, variant_id: str) -> List[Dict[str, Any]]:
    """
    Input variant_id: 'chr17:7579472:C:T'
      → Query GET {BASE}/{chrom}:{pos}-{pos}:1/{ALT}?hgvs=1
    Emits compact evidence and (if present) HGVSp_Short (e.g., R273H, G12D).
    """
    try:
        chrom, pos, ref, alt = variant_id.replace("chr", "").split(":")
        region = f"{chrom}:{pos}-{pos}:1"  # assume + strand; acceptable for SNVs
        url = f"{BASE}/{region}/{alt}?hgvs=1"
        r = await session.get(
            url, headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=TIMEOUT
        )
        if r.status_code >= 400:
            print(f"[VEP][WARN] HTTP {r.status_code} for {url}")
            return []
        data = r.json() or []
    except Exception as e:
        print(f"[VEP][WARN] Exception: {e}")
        return []

    out: List[Dict[str, Any]] = []
    for item in (data if isinstance(data, list) else [data]):
        transcripts = item.get("transcript_consequences") or []
        for t in transcripts:
            cons_terms = sorted(set(t.get("consequence_terms") or []))
            if cons_terms:
                out.append({"source":"VEP","type":"consequence","key":"Consequence","value":", ".join(cons_terms),"url":"https://www.ensembl.org/"})
            imp = t.get("impact")
            if imp:
                out.append({"source":"VEP","type":"impact","key":"Impact","value":imp,"url":"https://www.ensembl.org/"})
            tx = t.get("transcript_id")
            if tx:
                out.append({"source":"VEP","type":"transcript","key":"Transcript","value":tx,"url":"https://www.ensembl.org/"})

            # HGVS protein → HGVSp short (e.g., R273H) to feed CIViC
            hgvsp = t.get("hgvsp") or t.get("HGVSp")
            short = _to_hgvsp_short(hgvsp) if hgvsp else None
            if short:
                out.append({"source":"VEP","type":"hgvsp_short","key":"HGVSp_Short","value":short,"url":"https://www.ensembl.org/"})
    return out
