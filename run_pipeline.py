# run_pipeline.py — Health Canada DPD (Marketed) → Supabase
# Filters to Human • Oral • Tablet/Capsule/Softgel (with diagnostics & safe fallback)

import os, io, zipfile, re
import pandas as pd
import httpx
from supabase import create_client

# ---- Supabase ----
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ---- DPD ZIP URLs (env) ----
DRUG_ZIP     = os.environ["DPD_DRUG_URL"]      # drug.zip
FORM_ZIP     = os.environ["DPD_FORM_URL"]      # form.zip
ROUTE_ZIP    = os.environ["DPD_ROUTE_URL"]     # route.zip
INGRED_ZIP   = os.environ["DPD_INGRED_URL"]    # ingred.zip
COMPANY_ZIP  = os.environ["DPD_COMP_URL"]      # comp.zip
SCHEDULE_ZIP = os.environ["DPD_SCHEDULE_URL"]  # schedule.zip

# ---- Official layouts (no header rows in files) ----
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
    """DPD files are comma-separated, UTF-8/latin-1, with NO header row."""
    names = SCHEMAS[kind]
    r = httpx.get(url, timeout=120.0, follow_redirects=True)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        member = next((n for n in zf.namelist() if n.lower().endswith((".txt",".csv"))), None)
        if not member:
            raise RuntimeError(f"No .txt/.csv inside ZIP: {url}")
        raw = zf.read(member)
    for enc in ("utf-8","latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), header=None, names=names, dtype=str, encoding=enc, sep=",", engine="python")
            return df.fillna("")
        except Exception:
            pass
    raise RuntimeError(f"Failed to parse {url}")

def collapse(df: pd.DataFrame, key: str, col: str, out_col: str) -> pd.DataFrame:
    if df.empty or key not in df.columns or col not in df.columns:
        return pd.DataFrame({key: [], out_col: []})
    agg = df.groupby(key, as_index=False)[col].apply(
        lambda s: "; ".join(sorted({str(x).strip() for x in s if str(x).strip()}))
    )
    return agg.rename(columns={col: out_col})

def main():
    print("▶ Downloading DPD marketed files…")
    drug   = read_csv_from_zip(DRUG_ZIP, "drug")
    form   = read_csv_from_zip(FORM_ZIP, "form")
    route  = read_csv_from_zip(ROUTE_ZIP, "route")
    ingred = read_csv_from_zip(INGRED_ZIP, "ingred")
    comp   = read_csv_from_zip(COMPANY_ZIP, "comp")
    sched  = read_csv_from_zip(SCHEDULE_ZIP, "schedule")

    # Keep only fields we use
    cols = [c for c in ["DRUG_CODE","DRUG_IDENTIFICATION_NUMBER","BRAND_NAME","PRODUCT_CATEGORIZATION","CLASS"] if c in drug.columns]
    if "DRUG_CODE" not in cols:
        raise RuntimeError(f"drug file missing DRUG_CODE. Present: {list(drug.columns)}")
    drug_small = drug[cols].drop_duplicates()

    # Strength string
    if {"STRENGTH","STRENGTH_UNIT"}.issubset(ingred.columns):
        ingred["__STR"] = (ingred["STRENGTH"].astype(str).str.strip() + " " + ingred["STRENGTH_UNIT"].astype(str).str.strip()).str.strip()
    else:
        ingred["__STR"] = ""

    # Collapse 1..N to 1
    form_c  = collapse(form,   "DRUG_CODE", "PHARMACEUTICAL_FORM", "DOSAGE_FORM")
    route_c = collapse(route,  "DRUG_CODE", "ROUTE_OF_ADMINISTRATION", "ROUTE")
    ingr_c  = collapse(ingred, "DRUG_CODE", "INGREDIENT", "ACTIVE_INGREDIENT")
    str_c   = collapse(ingred, "DRUG_CODE", "__STR", "STRENGTH")
    comp_c  = collapse(comp,   "DRUG_CODE", "COMPANY_NAME", "MANUFACTURER")
    sched_c = collapse(sched,  "DRUG_CODE", "SCHEDULE", "SCHEDULE")

    # Merge
    df = (drug_small
          .merge(form_c,  on="DRUG_CODE", how="left")
          .merge(route_c, on="DRUG_CODE", how="left")
          .merge(ingr_c,  on="DRUG_CODE", how="left")
          .merge(str_c,   on="DRUG_CODE", how="left")
          .merge(comp_c,  on="DRUG_CODE", how="left")
          .merge(sched_c, on="DRUG_CODE", how="left"))

    # -----------------------------
    # Diagnostics + forgiving filter
    # -----------------------------
    print("Rows before filter:", len(df))
    try:
        print("Top ROUTE:", df.get("ROUTE","").str.lower().value_counts().head(10).to_dict())
    except Exception:
        pass
    try:
        print("Top FORM:", df.get("DOSAGE_FORM","").str.lower().value_counts().head(10).to_dict())
    except Exception:
        pass
    try:
        print("Top CATEGORY:", df.get("PRODUCT_CATEGORIZATION","").str.lower().value_counts().head(10).to_dict())
    except Exception:
        pass

    is_human = df.get("PRODUCT_CATEGORIZATION","").str.contains(r"\bhuman\b|\bhumain\b", case=False, na=False)
    is_oral  = df.get("ROUTE","").str.contains(r"\boral\b|\borale\b|\bby mouth\b|\bper os\b", case=False, na=False)
    is_tabcap= df.get("DOSAGE_FORM","").str.contains(r"tablet|caplet|capsule|softgel|gélule|compr", case=False, na=False)

    filtered = df[is_human & is_oral & is_tabcap].copy()
    print("After filter:", len(filtered), "rows")

    if len(filtered) == 0:
        print("⚠️ Filter removed everything. Sending 200 preview rows so we can confirm Supabase upsert works.")
        filtered = df.head(200).copy()

    df = filtered

    # DIN
    if "DRUG_IDENTIFICATION_NUMBER" in df.columns:
        din = df["DRUG_IDENTIFICATION_NUMBER"].astype(str).str.strip()
        din = din.where(din != "", df["DRUG_CODE"].astype(str))
    else:
        din = df["DRUG_CODE"].astype(str)
    df["din"] = din

    # Map to Supabase columns
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
