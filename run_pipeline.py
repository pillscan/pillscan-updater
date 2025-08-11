# run_pipeline.py
# Ingest Health Canada DPD "Marketed" extracts into Supabase (links only)
# Filters: Human • Oral • Tablet/Capsule/Softgel

import os
import io
import re
import zipfile
import pandas as pd
import httpx
from supabase import create_client


# ==============
# Supabase setup
# ==============
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ============================
# DPD ZIP URLs from environment
# ============================
DRUG_ZIP     = os.environ["DPD_DRUG_URL"]      # drug.zip
FORM_ZIP     = os.environ["DPD_FORM_URL"]      # form.zip
ROUTE_ZIP    = os.environ["DPD_ROUTE_URL"]     # route.zip
INGRED_ZIP   = os.environ["DPD_INGRED_URL"]    # ingred.zip
COMPANY_ZIP  = os.environ["DPD_COMP_URL"]      # comp.zip
SCHEDULE_ZIP = os.environ["DPD_SCHEDULE_URL"]  # schedule.zip


# =========================
# Helpers: parse & cleaning
# =========================

# Official DPD "Marketed" layouts (no header in files) — column lists from Read Me (QRYM_* tables).
SCHEMAS = {
    "drug": [
        "DRUG_CODE","PRODUCT_CATEGORIZATION","CLASS","DRUG_IDENTIFICATION_NUMBER","BRAND_NAME",
        "DESCRIPTOR","PEDIATRIC_FLAG","ACCESSION_NUMBER","NUMBER_OF_AIS","LAST_UPDATE_DATE",
        "AI_GROUP_NO","CLASS_F","BRAND_NAME_F","DESCRIPTOR_F"
    ],
    "form": [
        "DRUG_CODE","PHARM_FORM_CODE","PHARMACEUTICAL_FORM","PHARMACEUTICAL_FORM_F"
    ],
    "route": [
        "DRUG_CODE","ROUTE_OF_ADMINISTRATION_CODE","ROUTE_OF_ADMINISTRATION","ROUTE_OF_ADMINISTRATION_F"
    ],
    "ingred": [
        "DRUG_CODE","ACTIVE_INGREDIENT_CODE","INGREDIENT","INGREDIENT_SUPPLIED_IND","STRENGTH",
        "STRENGTH_UNIT","STRENGTH_TYPE","DOSAGE_VALUE","BASE","DOSAGE_UNIT","NOTES",
        "INGREDIENT_F","STRENGTH_UNIT_F","STRENGTH_TYPE_F","DOSAGE_UNIT_F"
    ],
    "comp": [
        "DRUG_CODE","MFR_CODE","COMPANY_CODE","COMPANY_NAME","COMPANY_TYPE",
        "ADDRESS_MAILING_FLAG","ADDRESS_BILLING_FLAG","ADDRESS_NOTIFICATION_FLAG","ADDRESS_OTHER",
        "SUITE_NUMBER","STREET_NAME","CITY_NAME","PROVINCE","COUNTRY","POSTAL_CODE","POST_OFFICE_BOX",
        "PROVINCE_F","COUNTRY_F"
    ],
    "schedule": [
        "DRUG_CODE","SCHEDULE","SCHEDULE_F"
    ],
}


def read_csv_from_zip(url: str, kind: str) -> pd.DataFrame:
    """
    Download and read a DPD CSV/TXT from a ZIP.
    DPD 'Marketed' files have NO header row; we assign names from SCHEMAS above.
    Files are comma-separated; encoding tends to be UTF-8 with some latin-1 cases.
    """
    if kind not in SCHEMAS:
        raise RuntimeError(f"Unknown kind '{kind}' for {url}")

    schema = SCHEMAS[kind]

    r = httpx.get(url, timeout=120.0, follow_redirects=True)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        member = next((n for n in zf.namelist() if n.lower().endswith((".txt", ".csv"))), None)
        if not member:
            raise RuntimeError(f"No .txt/.csv inside ZIP: {url}")
        raw = zf.read(member)

    last_err = None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                header=None,              # << no header line in file
                names=schema,             # << assign proper names
                dtype=str,
                encoding=enc,
                sep=",",
                engine="python"
            ).fillna("")
            print(f"Loaded {kind} ({member}) with {len(df)} rows, columns: {df.columns.tolist()[:8]} ...")
            return df
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to parse {url} ({kind}): {last_err}")


def collapse(df: pd.DataFrame, key: str, col: str, out_col: str) -> pd.DataFrame:
    """Aggregate many-to-one as semicolon-separated unique values."""
    if df.empty or key not in df.columns or col not in df.columns:
        return pd.DataFrame({key: [], out_col: []})
    out = (
        df.groupby(key, as_index=False)[col]
          .apply(lambda s: "; ".join(sorted({str(x).strip() for x in s if str(x).strip()})))
    )
    return out.rename(columns={col: out_col})


# =====
# Main
# =====
def main():
    print("▶ Downloading DPD marketed files…")

    # Read all six tables with assigned headers
    drug   = read_csv_from_zip(DRUG_ZIP, "drug")
    form   = read_csv_from_zip(FORM_ZIP, "form")
    route  = read_csv_from_zip(ROUTE_ZIP, "route")
    ingred = read_csv_from_zip(INGRED_ZIP, "ingred")
    comp   = read_csv_from_zip(COMPANY_ZIP, "comp")
    sched  = read_csv_from_zip(SCHEDULE_ZIP, "schedule")

    # Minimal drug subset (only columns we actually use later)
    base_cols = [c for c in [
        "DRUG_CODE","DRUG_IDENTIFICATION_NUMBER","BRAND_NAME","PRODUCT_CATEGORIZATION","CLASS"
    ] if c in drug.columns]
    if not base_cols or "DRUG_CODE" not in base_cols:
        raise RuntimeError(f"drug.zip did not contain required fields. Present: {drug.columns.tolist()[:20]}")

    drug_small = drug[base_cols].drop_duplicates()

    # Build readable strength string on ingredients (if present)
    if not ingred.empty and "STRENGTH" in ingred.columns and "STRENGTH_UNIT" in ingred.columns:
        ingred["__STR"] = (ingred["STRENGTH"].astype(str).str.strip() + " " +
                           ingred["STRENGTH_UNIT"].astype(str).str.strip()).str.strip()
    else:
        ingred["__STR"] = ""

    # Collapse one-to-many tables to single row per DRUG_CODE
    form_c  = collapse(form,   "DRUG_CODE", "PHARMACEUTICAL_FORM", "DOSAGE_FORM")
    route_c = collapse(route,  "DRUG_CODE", "ROUTE_OF_ADMINISTRATION", "ROUTE")
    ingr_c  = collapse(ingred, "DRUG_CODE", "INGREDIENT", "ACTIVE_INGREDIENT")
    str_c   = collapse(ingred, "DRUG_CODE", "__STR", "STRENGTH")
    comp_c  = collapse(comp,   "DRUG_CODE", "COMPANY_NAME", "MANUFACTURER")
    sched_c = collapse(sched,  "DRUG_CODE", "SCHEDULE", "SCHEDULE")

    # Merge all onto drug_small by DRUG_CODE
    df = (
        drug_small
          .merge(form_c,  on="DRUG_CODE", how="left")
          .merge(route_c, on="DRUG_CODE", how="left")
          .merge(ingr_c,  on="DRUG_CODE", how="left")
          .merge(str_c,   on="DRUG_CODE", how="left")
          .merge(comp_c,  on="DRUG_CODE", how="left")
          .merge(sched_c, on="DRUG_CODE", how="left")
    )

    # -----------------------------
    # Filter: Human + Oral + Tablet/Capsule/Softgel
    # -----------------------------
    df["ROUTE_L"] = df.get("ROUTE", "").str.lower()
    df["FORM_L"]  = df.get("DOSAGE_FORM", "").str.lower()
    df["CAT_L"]   = df.get("PRODUCT_CATEGORIZATION", "").str.lower()

    is_human = df["CAT_L"].str.contains("human", na=False)
    is_oral  = df["ROUTE_L"].str.contains("oral", na=False)
    is_tabcap= df["FORM_L"].str.contains(r"tablet|capsule|softgel", na=False)

    df = df[is_human & is_oral & is_tabcap].copy()

    # DIN prefer DRUG_IDENTIFICATION_NUMBER; fallback to DRUG_CODE
    if "DRUG_IDENTIFICATION_NUMBER" in df.columns:
        df["din"] = df["DRUG_IDENTIFICATION_NUMBER"].where(
            df["DRUG_IDENTIFICATION_NUMBER"].astype(str).str.strip() != "",
            df["DRUG_CODE"].astype(str)
        ).astype(str).str.strip()
    else:
        df["din"] = df["DRUG_CODE"].astype(str).str.strip()

    # Map to your Supabase 'pills' columns
    df["brand_name"]        = df.get("BRAND_NAME","").astype(str).str.strip()
    df["dosage_form"]       = df.get("DOSAGE_FORM","").astype(str).str.strip()
    df["route"]             = df.get("ROUTE","").astype(str).str.strip()
    df["strength"]          = df.get("STRENGTH","").astype(str).str.strip()
    df["active_ingredient"] = df.get("ACTIVE_INGREDIENT","").astype(str).str.strip()
    df["manufacturer"]      = df.get("MANUFACTURER","").astype(str).str.strip()
    df["schedule"]          = df.get("SCHEDULE","").astype(str).str.strip()
    df["class"]             = df.get("CLASS","").astype(str).str.strip()
    df["status"]            = "Marketed"
    df["monograph_url"]     = "https://health-products.canada.ca/dpd-bdpp/info?lang=eng&code=" + df["DRUG_CODE"].astype(str)

    out = df[[
        "din","brand_name","dosage_form","route","strength","active_ingredient",
        "manufacturer","schedule","class","status","monograph_url"
    ]].fillna("")

    rows = out.to_dict(orient="records")
    print(f"▶ Upserting {len(rows)} rows into Supabase.pills …")
    if rows:
        client.table("pills").upsert(rows, on_conflict="din").execute()
    print("✅ Finished")


if __name__ == "__main__":
    main()

