import os, httpx, asyncio
from typing import List, Dict, Any

BASE = os.getenv("ONCOKB_API_BASE", "https://www.oncokb.org/api/v1")
TOKEN = os.getenv("ONCOKB_API_TOKEN", "")
TIMEOUT = float(os.getenv("ANNOTATION_TIMEOUT_S", "8"))

async def _get_with_retries(session: httpx.AsyncClient, url: str, headers: dict, params: dict, max_retries: int = 3):
    delay = 0.75
    for attempt in range(1, max_retries + 1):
        try:
            r = await session.get(url, headers=headers, params=params, timeout=TIMEOUT)
            if r.status_code in (401, 403):
                print(f"[OncoKB][ERR] {r.status_code} unauthorized. Check ONCOKB_API_TOKEN.")
                try:
                    print("[OncoKB][ERR] Body:", (r.text or "")[:240])
                except Exception:
                    pass
                return r
            if r.status_code == 429 or r.status_code >= 500:
                print(f"[OncoKB][WARN] HTTP {r.status_code} (attempt {attempt})")
                await asyncio.sleep(delay); delay *= 1.6; continue
            return r
        except Exception as e:
            print(f"[OncoKB][WARN] Exception (attempt {attempt}): {e}")
            await asyncio.sleep(delay); delay *= 1.6
    return None

async def annotate_oncokb(session: httpx.AsyncClient, variant_id: str, gene: str, disease: str|None):
    """
    Returns (evidence:list[dict], knowledge:list[dict]).
    Tries GRCh37 first, then GRCh38. Requires token.
    """
    if not TOKEN:
        print("[OncoKB][ERR] ONCOKB_API_TOKEN is empty. Skipping.")
        return [], []
    hdrs = {"Authorization": f"Bearer {TOKEN}"}
    chrom, pos, ref, alt = variant_id.replace("chr","").split(":")
    url = f"{BASE}/annotate/mutations/byGenomicChange"

    async def _try_ref(refgen: str):
        params = {"genomicLocation": f"{chrom}:{pos}:{ref}:{alt}", "referenceGenome": refgen}
        r = await _get_with_retries(session, url, hdrs, params)
        if not r:
            return None
        if r.status_code >= 400:
            print(f"[OncoKB][WARN] HTTP {r.status_code} for {refgen} {variant_id}")
            try:
                print("[OncoKB][WARN] Body:", (r.text or "")[:240])
            except Exception:
                pass
            return None
        try:
            return r.json()
        except Exception as e:
            print(f"[OncoKB][WARN] JSON parse failed: {e}")
            return None

    d = await _try_ref("GRCh37") or await _try_ref("GRCh38")
    if not d:
        print(f"[OncoKB][WARN] No data for {variant_id}")
        return [], []

    ev=[]; kn=[]
    me = d.get("mutationEffect") or {}
    if me.get("description"):
        ev.append({"source":"OncoKB","type":"effect","key":"MutationEffect","value":me["description"],"url":"https://www.oncokb.org/"})
    for tx in d.get("treatments", []) or []:
        level = tx.get("levelOfEvidence")
        drugs = [t.get("drugName") for t in (tx.get("drugs") or []) if t.get("drugName")]
        stmt = f"{gene} {ref}>{alt}: evidence {level}" + (f" for {', '.join(drugs)}" if drugs else "")
        url2 = tx.get("url") or "https://www.oncokb.org/"
        kn.append({"source":"OncoKB","statement":stmt,"url":url2,"evidence_level":level,"disease":disease,"drugs":drugs})
    return ev, kn
