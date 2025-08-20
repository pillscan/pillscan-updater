# run_pipeline.py — full, copy/paste version (with Fix B filtering added)
#
# What it does (fully automated):
# 1) Reads your current pills from Supabase (din + monograph fields)
# 2) Downloads Health Canada's DPD drug.zip from env(DPD_DRUG_URL)
#    to build DIN -> DRUG_CODE mapping
# 3) Filters to ONLY Oral Tablets/Capsules (Human + Approved/Marketed)
# 4) For rows missing monograph_url:
#      - discover real PM PDF + revision date (no PDF downloads)
# 5) For rows that already have monograph_url:
#      - send cheap HEAD requests to see if it changed (ETag/Last-Modified)
# 6) Upserts results back into public.pills on conflict(din)

import os
import io
import sys
import zipfile
from typing import Dict, List
import pandas as pd
import httpx
from tqdm import tqdm
from supabase import create_client, Client
from monograph import batch_discover, head_conditional, now_utc

# ---------------------------
# Config & helpers
# ---------------------------

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
TABLE_NAME = os.environ.get("PILLS_TABLE", "pills")

DPD_DRUG_URL = os.environ.get("DPD_DRUG_URL", "").strip()  # REQUIRED

PAGE_SIZE = 2000  # batch size when reading from Supabase

def die(msg: str):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)

def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        die("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY environment variables.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def read_whole_table(client: Client, table: str, columns: List[str]) -> pd.DataFrame:
    """Pull the whole table from Supabase in pages."""
    start = 0
    frames = []
    while True:
        end = start + PAGE_SIZE - 1
        resp = client.table(table).select(",".join(columns)).range(start, end).execute()
        rows = resp.data or []
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        if len(rows) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    if not frames:
        return pd.DataFrame(columns=columns)
    df = pd.concat(frames, ignore_index=True)
    if "din" in df.columns:
        df["din"] = df["din"].astype(str).str.strip().str.zfill(8)
    return df

def fetch_dpd_drug_csv(url: str) -> pd.DataFrame:
    """Download the DPD drug.zip, load the first CSV inside as a DataFrame."""
    if not url:
        die("DPD_DRUG_URL not set. Please add it in Render → Environment.")
    print(f"Downloading DPD drug.zip from: {url}")
    with httpx.Client(follow_redirects=True, timeout=60) as s:
        r = s.get(url)
        r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_members:
            die("No CSV found inside DPD drug.zip")
        with zf.open(csv_members[0]) as f:
            df = pd.read_csv(f, dtype=str, keep_default_na=False)
    df.columns = [c.strip().upper() for c in df.columns]
    if "DRUG_CODE" not in df.columns:
        die("DPD drug.csv is missing DRUG_CODE")
    return df

def build_din_to_drugcode(drug_df: pd.DataFrame) -> Dict[str, str]:
    """Build DIN (8-digit string) -> DRUG_CODE mapping."""
    mapping: Dict[str, str] = {}
    if "DRUG_IDENTIFICATION_NUMBER" in drug_df.columns:
        tmp = drug_df[["DRUG_CODE", "DRUG_IDENTIFICATION_NUMBER"]].copy()
        tmp["DIN8"] = tmp["DRUG_IDENTIFICATION_NUMBER"].astype(str).str.strip().str.zfill(8)
        mapping.update({row["DIN8"]: str(row["DRUG_CODE"]) for _, row in tmp.iterrows()})
    else:
        tmp = drug_df[["DRUG_CODE"]].copy()
        tmp["DIN8"] = tmp["DRUG_CODE"].astype(str).str.zfill(8)
        mapping.update({row["DIN8"]: str(row["DRUG_CODE"]) for _, row in tmp.iterrows()})
    return mapping

def upsert_rows(client: Client, table: str, rows: List[dict], on_conflict: str = "din"):
    if rows:
        client.table(table).upsert(rows, on_conflict=on_conflict).execute()

# ---------------------------
# Main
# ---------------------------

def main():
    print("▶ Starting PillScan monograph refresher …")
    client = get_supabase_client()

    # 1) Read existing rows from Supabase
    cols_to_read = [
        "din","monograph_url","monograph_revision_date",
        "pm_etag","pm_last_modified","pm_last_checked_at",
        "brand_name","active_ingredient","dosage_form","modified_release",
        "route","strength","multi_colour","coating","coated","scored",
        "manufacturer_name","status","drug_class","schedule",
    ]
    existing = read_whole_table(client, TABLE_NAME, cols_to_read)
    print(f"Loaded {len(existing):,} existing rows from Supabase.")
    if existing.empty:
        print("Nothing to process (table empty). Exiting.")
        return

    # 2) Build DIN -> DRUG_CODE mapping from DPD drug.zip
    drug_df = fetch_dpd_drug_csv(DPD_DRUG_URL)
    din_to_dc = build_din_to_drugcode(drug_df)
    print(f"Built DIN→DRUG_CODE map with {len(din_to_dc):,} entries.")

    # ---------------- FIX B: filter to Oral Tablets/Capsules only ----------------
    def _norm(v): return str(v or "").strip().lower()
    drug_df["ROUTE_N"]  = drug_df["ROUTE"].apply(_norm)
    drug_df["FORM_N"]   = drug_df["FORM"].apply(_norm)
    drug_df["CLASS_N"]  = drug_df["CLASS"].apply(_norm)
    drug_df["STATUS_N"] = drug_df["STATUS"].apply(_norm)

    TABLET_FORMS = {"tablet","tablet (extended-release)","tablet (sustained-release)",
                    "tablet (delayed-release)","tablet (chewable)",
                    "orally disintegrating tablet","orodispersible tablet"}
    CAPSULE_FORMS = {"capsule","capsule (extended-release)","capsule (sustained-release)",
                     "capsule (delayed-release)","soft capsule","hard capsule"}

    is_human   = drug_df["CLASS_N"].eq("human")
    is_status  = drug_df["STATUS_N"].isin({"marketed","approved"})
    is_oral    = drug_df["ROUTE_N"].str.contains(r"\boral\b", regex=True)
    is_tablet  = drug_df["FORM_N"].isin(TABLET_FORMS)
    is_capsule = drug_df["FORM_N"].isin(CAPSULE_FORMS)

    filtered = drug_df[is_human & is_status & is_oral & (is_tablet | is_capsule)].copy()
    def _to_dosage_form(form_n: str) -> str:
        return "Capsule" if form_n in CAPSULE_FORMS else "Tablet"
    filtered["dosage_form"] = filtered["FORM_N"].apply(_to_dosage_form)

    print(f"Rows before filter: {len(drug_df):,}")
    print("Top ROUTE:", drug_df["ROUTE_N"].value_counts().head(5).to_dict())
    print("Top FORM:", drug_df["FORM_N"].value_counts().head(5).to_dict())
    print("Top CLASS:", drug_df["CLASS_N"].value_counts().head(3).to_dict())
    print(f"After filter (oral tablets/capsules only): {len(filtered):,}")
    if len(filtered) == 0:
        raise RuntimeError("Filter returned 0 rows — check DPD column names or source data.")
    # --------------------------------------------------------------------------

    # 3) Ensure monograph-related columns exist in Supabase data
    for col in ["monograph_url","monograph_revision_date","pm_etag","pm_last_modified","pm_last_checked_at"]:
        if col not in existing.columns:
            existing[col] = None

    # 4) Discover monographs for missing rows
    need_mask = existing["monograph_url"].isna() | (existing["monograph_url"] == "")
    to_discover = list(existing.loc[need_mask,"din"].astype(str))
    if to_discover:
        print(f"▶ Discovering monographs for {len(to_discover):,} DINs …")
        import asyncio
        discovered = asyncio.run(batch_discover(to_discover, din_to_dc, concurrency=10))
        for din, (pdf, rev) in discovered.items():
            idxs = existing.index[existing["din"] == din]
            if not len(idxs): continue
            i = idxs[0]
            if pdf: existing.at[i,"monograph_url"] = pdf
            if rev: existing.at[i,"monograph_revision_date"] = rev
            existing.at[i,"pm_etag"] = None
            existing.at[i,"pm_last_modified"] = None
            existing.at[i,"pm_last_checked_at"] = now_utc()

    # 5) HEAD-check existing monograph URLs
    has_url = existing["monograph_url"].notna() & (existing["monograph_url"] != "")
    to_check = existing.loc[has_url, ["din","monograph_url","pm_etag","pm_last_modified"]].to_dict(orient="records")
    print(f"▶ HEAD-checking {len(to_check):,} existing monograph URLs …")
    changed = 0
    for r in tqdm(to_check, unit="row"):
        url = r["monograph_url"]
        if not url: continue
        sc, hdrs = head_conditional(url, r.get("pm_etag"), r.get("pm_last_modified"))
        i = existing.index[existing["din"] == r["din"]][0]
        existing.at[i,"pm_last_checked_at"] = now_utc()
        if sc == 304: continue
        if sc == 200:
            existing.at[i,"pm_etag"] = hdrs.get("ETag")
            existing.at[i,"pm_last_modified"] = hdrs.get("Last-Modified")
            changed += 1
    print(f"HEAD updated validators for ~{changed:,} rows.")

    # 6) Upsert back to Supabase
    upsert_cols = [
        "din","brand_name","active_ingredient","dosage_form","modified_release",
        "route","strength","multi_colour","coating","coated","scored",
        "manufacturer_name","status","drug_class","schedule",
        "monograph_url","monograph_revision_date",
        "pm_etag","pm_last_modified","pm_last_checked_at"
    ]
    present_cols = [c for c in upsert_cols if c in existing.columns]
    rows = existing[present_cols].fillna("").to_dict(orient="records")

    print(f"▶ Upserting {len(rows):,} rows into public.{TABLE_NAME} …")
    upsert_rows(client, TABLE_NAME, rows, on_conflict="din")
    print("✅ Done.")

if __name__ == "__main__":
    main()

