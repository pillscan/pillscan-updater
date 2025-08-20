# run_pipeline.py — full, copy/paste version
#
# What it does (fully automated):
# 1) Reads your current pills from Supabase (din + monograph fields)
# 2) Downloads Health Canada's DPD drug.zip from env(DPD_DRUG_URL)
#    to build DIN -> DRUG_CODE mapping
# 3) For rows missing monograph_url:
#      - discover real PM PDF + revision date (no PDF downloads)
# 4) For rows that already have monograph_url:
#      - send cheap HEAD requests to see if it changed (ETag/Last-Modified)
# 5) Upserts results back into public.pills on conflict(din)
#
# You do NOT have to manually check thousands each week.

import os
import io
import sys
import math
import zipfile
from typing import Dict, List, Tuple
from datetime import datetime, timezone

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
    """
    Pull the whole table from Supabase in pages. RLS should allow SELECT for service key.
    """
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
    # Ensure DIN is string zero-padded 8
    if "din" in df.columns:
        df["din"] = df["din"].astype(str).str.strip().str.zfill(8)
    return df

def fetch_dpd_drug_csv(url: str) -> pd.DataFrame:
    """
    Download the DPD drug.zip, load the first CSV inside as a DataFrame.
    """
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
        # Use the first CSV (DPD zips typically contain one)
        with zf.open(csv_members[0]) as f:
            # DPD CSVs are Latin-1 sometimes; let pandas guess
            df = pd.read_csv(f, dtype=str, keep_default_na=False)
    # Normalize columns
    df.columns = [c.strip().upper() for c in df.columns]
    # Ensure we have these columns
    if "DRUG_CODE" not in df.columns:
        die("DPD drug.csv is missing DRUG_CODE")
    # DIN may be in DRUG_IDENTIFICATION_NUMBER
    # Keep both for robustness
    return df

def build_din_to_drugcode(drug_df: pd.DataFrame) -> Dict[str, str]:
    """
    Build DIN (8-digit string) -> DRUG_CODE mapping.
    """
    mapping: Dict[str, str] = {}

    # Preferred: DRUG_IDENTIFICATION_NUMBER present
    if "DRUG_IDENTIFICATION_NUMBER" in drug_df.columns:
        tmp = drug_df[["DRUG_CODE", "DRUG_IDENTIFICATION_NUMBER"]].copy()
        tmp["DIN8"] = tmp["DRUG_IDENTIFICATION_NUMBER"].astype(str).str.strip().str.zfill(8)
        mapping.update({row["DIN8"]: str(row["DRUG_CODE"]) for _, row in tmp.iterrows()})
    else:
        # Fallback: map DRUG_CODE to itself zero-padded (not ideal, but prevents total failure)
        tmp = drug_df[["DRUG_CODE"]].copy()
        tmp["DIN8"] = tmp["DRUG_CODE"].astype(str).str.zfill(8)
        mapping.update({row["DIN8"]: str(row["DRUG_CODE"]) for _, row in tmp.iterrows()})

    return mapping

def upsert_rows(client: Client, table: str, rows: List[dict], on_conflict: str = "din"):
    if not rows:
        return
    # Supabase Python client upsert
    client.table(table).upsert(rows, on_conflict=on_conflict).execute()

# ---------------------------
# Main
# ---------------------------

def main():
    print("▶ Starting PillScan monograph refresher …")

    client = get_supabase_client()

    # 1) Read existing rows from Supabase (only fields we care about here)
    cols_to_read = [
        "din",
        "monograph_url",
        "monograph_revision_date",
        "pm_etag",
        "pm_last_modified",
        "pm_last_checked_at",
        # Optional: we pass through these on upsert if present
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

    # 3) Ensure columns exist
    for col in ["monograph_url","monograph_revision_date","pm_etag","pm_last_modified","pm_last_checked_at"]:
        if col not in existing.columns:
            existing[col] = None

    # 4) First-time discovery for rows missing monograph_url
    need_mask = existing["monograph_url"].isna() | (existing["monograph_url"] == "")
    to_discover = list(existing.loc[need_mask, "din"].astype(str))
    if to_discover:
        print(f"▶ Discovering monographs for {len(to_discover):,} DINs …")
        # Batch async discover
        discovered = asyncio.run(batch_discover(to_discover, din_to_dc, concurrency=10))
        updated_indices = []
        for din, (pdf, rev) in discovered.items():
            idxs = existing.index[existing["din"] == din]
            if len(idxs) == 0:
                continue
            i = idxs[0]
            if pdf:
                existing.at[i, "monograph_url"] = pdf
            if rev:
                existing.at[i, "monograph_revision_date"] = rev
            existing.at[i, "pm_etag"] = None
            existing.at[i, "pm_last_modified"] = None
            existing.at[i, "pm_last_checked_at"] = now_utc()
            updated_indices.append(i)
        print(f"Discovered {len(updated_indices):,} monograph URLs.")

    # 5) Cheap weekly HEAD checks for rows that already have monograph_url
    has_url = existing["monograph_url"].notna() & (existing["monograph_url"] != "")
    to_check = existing.loc[has_url, ["din","monograph_url","pm_etag","pm_last_modified"]].to_dict(orient="records")

    print(f"▶ HEAD-checking {len(to_check):,} existing monograph URLs …")
    changed = 0
    for r in tqdm(to_check, unit="row"):
        url = r["monograph_url"]
        if not url:
            continue
        sc, hdrs = head_conditional(url, r.get("pm_etag"), r.get("pm_last_modified"))
        i = existing.index[existing["din"] == r["din"]][0]
        existing.at[i, "pm_last_checked_at"] = now_utc()
        if sc == 304:
            continue  # not modified
        if sc == 200:
            # Update validators; OPTIONAL: re-discover page to refresh revision_date immediately
            existing.at[i, "pm_etag"] = hdrs.get("ETag")
            existing.at[i, "pm_last_modified"] = hdrs.get("Last-Modified")
            changed += 1
        # other status codes: ignore for now (could be intermittent)

    print(f"HEAD updated validators for ~{changed:,} rows.")

    # 6) Upsert back to Supabase (only columns we manage)
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
    # We import asyncio here to avoid top-level dependency if someone inspects this file.
    import asyncio
    main()
