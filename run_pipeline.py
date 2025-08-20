# run_pipeline.py — seed + monograph refresher (safe schema)
#
# What this does:
# 1) Downloads Health Canada DPD zips from env:
#    DPD_DRUG_URL, DPD_FORM_URL, DPD_ROUTE_URL, DPD_STATUS_URL, DPD_SCHEDULE_URL, DPD_COMP_URL
# 2) Joins them on DRUG_CODE, then filters to: Human + Oral + Tablet/Capsule + Marketed/Approved
# 3) Builds schema-safe rows for public.pills (only columns your table is guaranteed to have)
# 4) Discovers Product Monograph PDF URLs + (optional) revision dates without downloading PDFs
# 5) Upserts into Supabase by (din)
#
# Notes:
# - We DO NOT touch columns like 'coating'/'coated' etc to avoid "column does not exist" errors.
# - If pills is empty, this will seed it. If pills has data, this will refresh/enrich monographs.

import os, io, sys, zipfile
from typing import Dict, List, Optional
import pandas as pd
import httpx

from supabase import create_client, Client
from monograph import batch_discover, head_conditional, now_utc

# ---------------- Config ----------------
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
TABLE_NAME = os.environ.get("PILLS_TABLE", "pills")

DPD_DRUG_URL     = os.environ.get("DPD_DRUG_URL", "").strip()
DPD_FORM_URL     = os.environ.get("DPD_FORM_URL", "").strip()
DPD_ROUTE_URL    = os.environ.get("DPD_ROUTE_URL", "").strip()
DPD_STATUS_URL   = os.environ.get("DPD_STATUS_URL", "").strip()
DPD_SCHEDULE_URL = os.environ.get("DPD_SCHEDULE_URL", "").strip()
DPD_COMP_URL     = os.environ.get("DPD_COMP_URL", "").strip()  # company/manufacturer

PAGE_SIZE = 2000

# ---------------- Helpers ----------------
def die(msg: str):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)

def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        die("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def _download_zip_csv(url: str, label: str) -> pd.DataFrame:
    if not url:
        return pd.DataFrame()
    print(f"▶ Downloading {label} …")
    with httpx.Client(follow_redirects=True, timeout=60) as s:
        r = s.get(url); r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        csv_members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_members:
            print(f"⚠ No CSV in {label} zip")
            return pd.DataFrame()
        with zf.open(csv_members[0]) as f:
            df = pd.read_csv(f, dtype=str, keep_default_na=False)
    df.columns = [c.strip().upper() for c in df.columns]
    return df

def _norm(v): return str(v or "").strip().lower()

def _safe_col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series([""] * len(df))

def _zfill8(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.zfill(8)

def _build_din_to_drugcode(drug_df: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if "DRUG_CODE" not in drug_df.columns:
        return mapping
    if "DRUG_IDENTIFICATION_NUMBER" in drug_df.columns:
        tmp = drug_df[["DRUG_CODE","DRUG_IDENTIFICATION_NUMBER"]].copy()
        tmp["DIN8"] = _zfill8(tmp["DRUG_IDENTIFICATION_NUMBER"])
        mapping.update({row["DIN8"]: str(row["DRUG_CODE"]) for _, row in tmp.iterrows()})
    else:
        tmp = drug_df[["DRUG_CODE"]].copy()
        tmp["DIN8"] = tmp["DRUG_CODE"].astype(str).str.zfill(8)
        mapping.update({row["DIN8"]: str(row["DRUG_CODE"]) for _, row in tmp.iterrows()})
    return mapping

def _read_existing_mono(client: Client) -> pd.DataFrame:
    """
    Read only the safe set of columns from pills.
    """
    cols = [
        "din",
        "brand_name",
        "active_ingredient",
        "dosage_form",
        "modified_release",
        "route",
        "strength",
        "manufacturer_name",
        "status",
        "drug_class",
        "schedule",
        "monograph_url",
        "monograph_revision_date",
        "pm_etag",
        "pm_last_modified",
        "pm_last_checked_at",
    ]
    start = 0
    frames = []
    while True:
        end = start + PAGE_SIZE - 1
        try:
            resp = client.table(TABLE_NAME).select(",".join(cols)).range(start, end).execute()
        except Exception as e:
            # If some selected columns do not exist, progressively trim until it works
            # Start with minimal monograph set
            base_cols = ["din","monograph_url","monograph_revision_date","pm_etag","pm_last_modified","pm_last_checked_at"]
            resp = client.table(TABLE_NAME).select(",".join(base_cols)).range(start, end).execute()
            rows = resp.data or []
            if not rows:
                return pd.DataFrame(columns=base_cols)
            df = pd.DataFrame(rows)
            if "din" in df.columns:
                df["din"] = _zfill8(df["din"])
            # Ensure optional columns exist for downstream upsert
            for c in ["brand_name","active_ingredient","dosage_form","modified_release","route","strength",
                      "manufacturer_name","status","drug_class","schedule"]:
                if c not in df.columns:
                    df[c] = None
            return df

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
    if "din" in df.columns:
        df["din"] = _zfill8(df["din"])
    return df

def _upsert(client: Client, rows: List[dict]):
    if rows:
        client.table(TABLE_NAME).upsert(rows, on_conflict="din").execute()

# ---------------- Main ----------------
def main():
    print("▶ PillScan seeder + monograph refresher starting …")
    client = get_supabase_client()

    # 1) Pull existing pills (safe view)
    existing = _read_existing_mono(client)
    existing_rows = len(existing)
    print(f"Loaded {existing_rows:,} rows from Supabase.")

    # 2) Download DPD datasets
    drug_df     = _download_zip_csv(DPD_DRUG_URL,     "DPD drug.zip")
    form_df     = _download_zip_csv(DPD_FORM_URL,     "DPD form.zip")
    route_df    = _download_zip_csv(DPD_ROUTE_URL,    "DPD route.zip")
    status_df   = _download_zip_csv(DPD_STATUS_URL,   "DPD status.zip")
    schedule_df = _download_zip_csv(DPD_SCHEDULE_URL, "DPD schedule.zip")
    comp_df     = _download_zip_csv(DPD_COMP_URL,     "DPD company.zip")

    if drug_df.empty or "DRUG_CODE" not in drug_df.columns:
        die("drug.csv not available or missing DRUG_CODE — cannot proceed.")

    # 3) Join on DRUG_CODE (left joins)
    df = drug_df.copy()
    def _merge(left, right, suffix):
        if right.empty or "DRUG_CODE" not in right.columns:
            return left
        return left.merge(right, on="DRUG_CODE", how="left", suffixes=("", suffix))

    df = _merge(df, form_df, "_form")
    df = _merge(df, route_df, "_route")
    df = _merge(df, status_df, "_status")
    df = _merge(df, schedule_df, "_sched")
    df = _merge(df, comp_df, "_comp")

    # 4) Normalize and pick the most likely columns
    # Try to detect columns by common names across DPD files.
    col_brand = next((c for c in df.columns if c in {"BRAND_NAME","PRODUCT_NAME"}), None)
    col_class = "CLASS" if "CLASS" in df.columns else None
    col_status= "STATUS" if "STATUS" in df.columns else None
    col_route = "ROUTE" if "ROUTE" in df.columns else None
    col_form  = "FORM"  if "FORM"  in df.columns else None
    col_sched = "SCHEDULE" if "SCHEDULE" in df.columns else None
    col_comp  = next((c for c in df.columns if c in {"COMPANY_NAME","MANUFACTURER","MANUFACTURER_NAME"}), None)

    df["BRAND_OUT"] = df[col_brand] if col_brand else ""
    df["CLASS_OUT"] = df[col_class] if col_class else ""
    df["STATUS_OUT"]= df[col_status] if col_status else ""
    df["ROUTE_OUT"] = df[col_route] if col_route else ""
    df["FORM_OUT"]  = df[col_form]  if col_form  else ""
    df["SCHED_OUT"] = df[col_sched] if col_sched else ""
    df["MFR_OUT"]   = df[col_comp]  if col_comp  else ""

    # DIN
    if "DRUG_IDENTIFICATION_NUMBER" in df.columns:
        df["DIN"] = _zfill8(df["DRUG_IDENTIFICATION_NUMBER"])
    else:
        # as a last resort, derive a synthetic DIN from DRUG_CODE (still 8 chars)
        df["DIN"] = df["DRUG_CODE"].astype(str).str.zfill(8)

    # 5) Filter to Human + Oral + Tablet/Capsule + Approved/Marketed
    TABLET_FORMS = {
        "tablet","tablet (extended-release)","tablet (sustained-release)","tablet (delayed-release)",
        "tablet (chewable)","orally disintegrating tablet","orodispersible tablet","tablet (orally disintegrating)"
    }
    CAPSULE_FORMS = {
        "capsule","capsule (extended-release)","capsule (sustained-release)","capsule (delayed-release)",
        "soft capsule","hard capsule"
    }

    df["CLASS_N"]  = df["CLASS_OUT"].apply(_norm)
    df["STATUS_N"] = df["STATUS_OUT"].apply(_norm)
    df["ROUTE_N"]  = df["ROUTE_OUT"].apply(_norm)
    df["FORM_N"]   = df["FORM_OUT"].apply(_norm)

    is_human   = df["CLASS_N"].eq("human")
    is_status  = df["STATUS_N"].isin({"marketed","approved"})
    is_oral    = df["ROUTE_N"].str.contains(r"\boral\b", regex=True)
    is_tab     = df["FORM_N"].isin(TABLET_FORMS)
    is_cap     = df["FORM_N"].isin(CAPSULE_FORMS)

    filtered = df[is_human & is_status & is_oral & (is_tab | is_cap)].copy()

    def _to_dosage(form_n: str) -> str:
        return "Capsule" if form_n in CAPSULE_FORMS else "Tablet"

    filtered["dosage_form"] = filtered["FORM_N"].apply(_to_dosage)

    print(f"Rows before filter: {len(df):,}")
    print("Top ROUTE:", df["ROUTE_N"].value_counts().head(5).to_dict())
    print("Top FORM:", df["FORM_N"].value_counts().head(5).to_dict())
    print("Top CLASS:", df["CLASS_N"].value_counts().head(3).to_dict())
    print(f"After filter (oral tablets/capsules only): {len(filtered):,}")

    if len(filtered) == 0:
        die("Filter returned 0 rows. Check your DPD joins and column names.")

    # 6) Build output rows (schema-safe)
    out_cols = [
        "din","brand_name","active_ingredient","dosage_form","modified_release",
        "route","strength","manufacturer_name","status","drug_class","schedule",
        "monograph_url","monograph_revision_date","pm_etag","pm_last_modified","pm_last_checked_at",
    ]
    out = pd.DataFrame(columns=out_cols)
    out["din"]               = filtered["DIN"]
    out["brand_name"]        = filtered["BRAND_OUT"].replace({"": None})
    out["active_ingredient"] = None      # (optional later: build from INGREDIENT file)
    out["dosage_form"]       = filtered["dosage_form"]
    out["modified_release"]  = None
    out["route"]             = filtered["ROUTE_OUT"].replace({"": None})
    out["strength"]          = None      # (optional later)
    out["manufacturer_name"] = filtered["MFR_OUT"].replace({"": None})
    out["status"]            = filtered["STATUS_OUT"].replace({"": None})
    out["drug_class"]        = filtered["CLASS_OUT"].replace({"": None})
    out["schedule"]          = filtered["SCHED_OUT"].replace({"": None})

    # Ensure monograph columns exist
    for c in ["monograph_url","monograph_revision_date","pm_etag","pm_last_modified","pm_last_checked_at"]:
        if c not in out.columns:
            out[c] = None

    # 7) Monograph discovery for rows missing URL (first time) OR for all when seeding
    din_to_dc = _build_din_to_drugcode(drug_df)
    to_discover = list(out.loc[out["monograph_url"].isna() | (out["monograph_url"] == ""), "din"].astype(str))
    if to_discover:
        print(f"▶ Discovering monographs for {len(to_discover):,} DINs …")
        import asyncio
        discovered = asyncio.run(batch_discover(to_discover, din_to_dc, concurrency=10))
        for din, (pdf, rev) in discovered.items():
            idxs = out.index[out["din"] == din]
            if not len(idxs): continue
            i = idxs[0]
            if pdf: out.at[i, "monograph_url"] = pdf
            if rev: out.at[i, "monograph_revision_date"] = rev
            out.at[i, "pm_etag"] = None
            out.at[i, "pm_last_modified"] = None
            out.at[i, "pm_last_checked_at"] = now_utc()

    # 8) (For existing rows) Cheap weekly HEAD checks
    if existing_rows > 0:
        # Merge existing monograph validators where we already have them
        left = out.set_index("din")
        right = existing.set_index("din")[["monograph_url","pm_etag","pm_last_modified","pm_last_checked_at"]]
        left.update(right)
        out = left.reset_index()

        has_url = out["monograph_url"].notna() & (out["monograph_url"] != "")
        subset = out.loc[has_url, ["din","monograph_url","pm_etag","pm_last_modified"]].to_dict(orient="records")
        print(f"▶ HEAD-checking {len(subset):,} existing monograph URLs …")
        changed = 0
        for r in subset:
            url = r["monograph_url"]
            if not url: continue
            sc, hdrs = head_conditional(url, r.get("pm_etag"), r.get("pm_last_modified"))
            i = out.index[out["din"] == r["din"]][0]
            out.at[i, "pm_last_checked_at"] = now_utc()
            if sc == 304:
                continue
            if sc == 200:
                out.at[i, "pm_etag"] = hdrs.get("ETag")
                out.at[i, "pm_last_modified"] = hdrs.get("Last-Modified")
                changed += 1
        print(f"HEAD updated validators for ~{changed:,} rows.")

    # 9) Upsert into Supabase (on_conflict=din)
    rows = out[out_cols].fillna("").to_dict(orient="records")
    print(f"▶ Upserting {len(rows):,} rows into public.{TABLE_NAME} …")
    _upsert(client, rows)
    print("✅ Done.")

if __name__ == "__main__":
    main()
