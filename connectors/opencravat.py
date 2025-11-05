import os, httpx

BASE = os.getenv("OPENCRAVAT_API_BASE", "")
TIMEOUT = float(os.getenv("ANNOTATION_TIMEOUT_S", "8"))

async def annotate_opencravat(session: httpx.AsyncClient, variant_id: str):
    """
    Generic example: expects a CRAVAT-like API endpoint. If none configured, returns [].
    """
    if not BASE:
        return []
    url = f"{BASE.rstrip('/')}/annotate"
    r = await session.get(url, params={"variant": variant_id}, timeout=TIMEOUT)
    if r.status_code >= 400:
        return []
    d = r.json() or {}
    out=[]
    for k,v in d.items():
        if v is None: continue
        out.append({"source":"OpenCRAVAT","type":"annotation","key":k,"value":v})
    return out
