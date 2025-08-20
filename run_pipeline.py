# run_pipeline.py — monograph-only refresher (safe selects, Fix B diagnostics)
#
# What it does:
# 1) Reads only the columns we need from Supabase: din + monograph fields
# 2) Downloads DPD drug.zip (env DPD_DRUG_URL) and builds DIN->DRUG_CODE
# 3) (Diagnostics) Prints Oral/Tablet/Capsule counts from DPD so you can see data health
# 4) For rows missing monograph_url: discover real PM PDF + revision date (no PDF downloads)
# 5) For rows already having monograph_url: cheap HEAD checks to update validators
# 6) Upserts back ONLY monograph fields (so no schema mismatch)

import os, io, sys, zipfile
from typing import List, Dict
import pandas as pd
import httpx
from supabase import create_client, Client
from monograph import batch_discover, head_conditional, now_utc

# ---------------- Config ----------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
TABLE_NAME = os.environ.get("PILLS_TABLE", "pills")
DPD_DRUG_URL = os.environ.get("DPD_DRUG_URL", "").strip()  # REQUIRED
PAGE_SIZE = 2000

def die(msg: str):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)

def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        die("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def read_table_monograph_view(client: Client, table: str) -> pd.DataFrame:
    """
    Read only monograph-related columns + din to avoid missing-column errors.
    """
    cols = ["din","monograph_url","monograph_revision_date","pm_etag","pm_last_modified","pm_last_checked_at"]
    start = 0
    frames = []
    while True:
        end = start + PAGE_SIZE - 1
        resp = client.table(table).select(",".join(cols)).range(start, end).execute()
        rows = resp.data or []
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        if len(rows) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    if not frames:
        return pd.DataFrame(columns=cols)
    df = pd.concat(frames, ignore_index=True)
    df["din"] = df["din"].astype(str).str.strip().str.zfill(8)
    return df

def fetch_dpd_drug_csv(url: str) -> pd.DataFrame:
    if not url:
        die("DPD_DRUG_URL not set in Render → Environment.")
    print(f"▶ Downloading DPD drug.zip …")
    with httpx.Client(follow_redirects=True, timeout=60) as s:
        r = s.get(url); r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_members:
            die("No CSV inside DPD drug.zip")
        with zf.open(csv_members[0]) as f:
            df = pd.read_csv(f, dtype=str, keep_default_na=False)
    df.columns = [c.strip().upper() for c in df.columns]
    if "DRUG_CODE" not in df.columns:
        die("DPD drug.csv missing DRUG_CODE")
    return df

def build_din_to_drugcode(drug_df: pd.DataFrame) -> Dict[str,str]:
    mapping: Dict[str,str] = {}
    if "DRUG_IDENTIFICATION_NUMBER" in drug_df.columns:
        tmp = drug_df[["DRUG_CODE","DRUG_IDENTIFICATION_NUMBER"]].copy()
        tmp["DIN8"] = tmp["DRUG_IDENTIFICATION_NUMBER"].astype(str).str.strip().str.zfill(8)
        mapping.update({row["DIN8"]: str(row["DRUG_CODE"]) for _, row in tmp.iterrows()})
    else:
        tmp = drug_df[["DRUG_CODE"]].copy()
        tmp["DIN8"] = tmp["DRUG_CODE"].astype(str).str.zfill(8)
        mapping.update({row["DIN8"]: str(row["DRUG_CODE"]) for _, row in tmp.iterrows()})
    return mapping

def upsert_monograph_fields(client: Client, table: str, df: pd.DataFrame):
    """
    Upsert only monograph-related fields so we never collide with missing columns.
    """
    cols = ["din","monograph_url","monograph_revision_date","pm_etag","pm_last_modified","pm_last_checked_at"]
    rows = df[cols].fillna("").to_dict(orient="records")
    if rows:
        client.table(table).upsert(rows, on_conflict="din").execute()

def main():
    print("▶ Starting PillScan monograph refresher …")
    client = get_supabase_client()

    # 1) Read current monograph view from Supabase
    existing = read_table_monograph_view(client, TABLE_NAME)
    print(f"Loaded {len(existing):,} rows from Supabase (monograph view).")
    if existing.empty:
        print("No rows found in pills. Nothing to do.")
        return

    # 2) Build DIN->DRUG_CODE map from DPD drug.zip
    drug_df = fetch_dpd_drug_csv(DPD_DRUG_URL)

    # ---- Fix B diagnostics: print Oral Tablet/Capsule counts (for your info) ----
    def _n(v): return str(v or "").strip().lower()
    for c in ("ROUTE","FORM","CLASS","STATUS"):
        if c not in drug_df.columns: drug_df[c] = ""
    drug_df["ROUTE_N"]  = drug_df["ROUTE"].apply(_n)
    drug_df["FORM_N"]   = drug_df["FORM"].apply(_n)
    drug_df["CLASS_N"]  = drug_df["CLASS"].apply(_n)
    drug_df["STATUS_N"] = drug_df["STATUS"].apply(_n)

    TABLET_FORMS = {"tablet","tablet (extended-release)","tablet (sustained-release)","tablet (delayed-release)",
                    "tablet (chewable)","orally disintegrating tablet","orodispersible tablet"}
    CAPSULE_FORMS = {"capsule","capsule (extended-release)","capsule (sustained-release)","capsule (delayed-release)",
                     "soft capsule","hard capsule"}

    is_human   = drug_df["CLASS_N"].eq("human")
    is_status  = drug_df["STATUS_N"].isin({"marketed","approved"})
    is_oral    = drug_df["ROUTE_N"].str.contains(r"\boral\b", regex=True)
    is_tab     = drug_df["FORM_N"].isin(TABLET_FORMS)
    is_cap     = drug_df["FORM_N"].isin(CAPSULE_FORMS)

    filtered_count = (drug_df[is_human & is_status & is_oral & (is_tab | is_cap)]).shape[0]
    print(f"DPD diagnostics — Oral Tablet/Capsule rows available: {filtered_count:,}")
    # ---------------------------------------------------------------------------

    din_to_dc = build_din_to_drugcode(drug_df)
    print(f"Built DIN→DRUG_CODE map with {len(din_to_dc):,} entries.")

    # 3) Ensure monograph columns exist locally
    for c in ["monograph_url","monograph_revision_date","pm_etag","pm_last_modified","pm_last_checked_at"]:
        if c not in existing.columns:
            existing[c] = None

    # 4) Discover for missing URLs
    need_mask = existing["monograph_url"].isna() | (existing["monograph_url"] == "")
    to_discover = list(existing.loc[need_mask, "din"].astype(str))
    if to_discover:
        print(f"▶ Discovering monographs for {len(to_discover):,} DINs …")
        import asyncio
        discovered = asyncio.run(batch_discover(to_discover, din_to_dc, concurrency=10))
        for din, (pdf, rev) in discovered.items():
            idxs = existing.index[existing["din"] == din]
            if not len(idxs): continue
            i = idxs[0]
            if pdf: existing.at[i, "monograph_url"] = pdf
            if rev: existing.at[i, "monograph_revision_date"] = rev
            existing.at[i, "pm_etag"] = None
            existing.at[i, "pm_last_modified"] = None
            existing.at[i, "pm_last_checked_at"] = now_utc()

    # 5) Cheap weekly HEAD checks
    has_url = existing["monograph_url"].notna() & (existing["monograph_url"] != "")
    to_check = existing.loc[has_url, ["din","monograph_url","pm_etag","pm_last_modified"]].to_dict(orient="records")
    print(f"▶ HEAD-checking {len(to_check):,} existing monograph URLs …")
    changed = 0
    for r in to_check:
        url = r["monograph_url"]
        if not url:
            continue
        sc, hdrs = head_conditional(url, r.get("pm_etag"), r.get("pm_last_modified"))
        i = existing.index[existing["din"] == r["din"]][0]
        existing.at[i, "pm_last_checked_at"] = now_utc()
        if sc == 304:
            continue
        if sc == 200:
            existing.at[i, "pm_etag"] = hdrs.get("ETag")
            existing.at[i, "pm_last_modified"] = hdrs.get("Last-Modified")
            changed += 1
    print(f"HEAD updated validators for ~{changed:,} rows.")

    # 6) Upsert ONLY monograph fields (safe even if your schema is missing other columns)
    print(f"▶ Upserting monograph fields back into public.{TABLE_NAME} …")
    upsert_monograph_fields(client, TABLE_NAME, existing)
    print("✅ Done.")

if __name__ == "__main__":
    main()
