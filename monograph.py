# monograph.py  — helper for Product Monographs (Health Canada)
from __future__ import annotations
import re, asyncio
from typing import Optional, Tuple, Dict, Iterable
from datetime import datetime, timezone
import httpx

UA = "PillScanUpdater/1.0 (+ops@pillscan.ca)"

def _parse_date(s: str) -> Optional[str]:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except Exception:
            pass
    m = re.search(r"(20\d{2}|19\d{2})", s)
    if m:
        return f"{m.group(1)}-01-01"
    return None

def _best_pdf_from_html(html: str) -> Optional[str]:
    pdfs = re.findall(r'href="([^"]+\.pdf[^"]*)"', html, flags=re.IGNORECASE)
    if not pdfs:
        return None
    def score(u: str) -> int:
        s = 0
        ul = u.lower()
        if "eng" in ul: s += 2
        if "fra" in ul: s -= 2
        if ul.startswith("https://health-products.canada.ca"): s += 1
        return s
    return sorted(pdfs, key=score, reverse=True)[0]

def _find_revision_date(html: str) -> Optional[str]:
    for lab in ("Revision Date", "Date", "Last updated", "Dernière mise"):
        m = re.search(rf"{lab}[^<:]*[:>]\s*([A-Za-z0-9/\- ]{{6,30}})", html, flags=re.IGNORECASE)
        if m:
            d = _parse_date(m.group(1))
            if d:
                return d
    return None

def pm_page_url_for_drug_code(drug_code: str) -> str:
    return f"https://health-products.canada.ca/dpd-bdpp/pm-mp.do?lang=en&code={drug_code}"

async def fetch_pm_for_drug_code(drug_code: str, client: httpx.AsyncClient) -> Tuple[Optional[str], Optional[str]]:
    url = pm_page_url_for_drug_code(drug_code)
    r = await client.get(url, headers={"User-Agent": UA}, timeout=20)
    if r.status_code != 200:
        return None, None
    html = r.text
    pdf = _best_pdf_from_html(html)
    rev = _find_revision_date(html)
    return pdf, rev

async def discover_monograph_for_din(din: str, din_to_drugcode: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    drug_code = din_to_drugcode.get(din)
    if not drug_code:
        return None, None
    async with httpx.AsyncClient(follow_redirects=True, headers={"User-Agent": UA}) as client:
        return await fetch_pm_for_drug_code(drug_code, client)

def head_conditional(url: str, etag: Optional[str], last_modified: Optional[str]) -> Tuple[int, Dict[str, str]]:
    """
    Cheap weekly check:
      304 => not modified (skip)
      200 => maybe changed; update validators (and optionally re-discover page)
    """
    headers = {"User-Agent": UA}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    with httpx.Client(follow_redirects=True, timeout=12) as s:
        r = s.head(url, headers=headers)
    return r.status_code, dict(r.headers)

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

async def batch_discover(dins: Iterable[str], din_to_drugcode: Dict[str, str], concurrency: int = 10):
    """
    Discover monograph PDFs for many DINs concurrently (polite limit).
    """
    sem = asyncio.Semaphore(concurrency)
    results: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

    async def one(d: str):
        async with sem:
            try:
                pdf, rev = await discover_monograph_for_din(d, din_to_drugcode)
                results[d] = (pdf, rev)
            except Exception:
                results[d] = (None, None)

    tasks = [one(d) for d in dins]
    await asyncio.gather(*tasks)
    return results
