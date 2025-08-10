import os, io, zipfile, pandas as pd, httpx
from supabase import create_client

# --- Supabase (already set in your Render env) ---
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# --- Health Canada DPD "Marketed" extract URLs (set these 6 in Render env) ---
DRUG_ZIP     = os.environ["DPD_DRUG_URL"]      # drug.zip
FORM_ZIP     = os.environ["DPD_FORM_URL"]      # form.zip
ROUTE_ZIP    = os.environ["DPD_ROUTE_URL"]     # route.zip
INGRED_ZIP   = os.environ["DPD_INGRED_URL"]    # ingred.zip
COMPANY_ZIP  = os.environ["DPD_COMP_URL"]      # comp.zip
SCHEDULE_ZIP = os.environ["DPD_SCHEDULE_URL"]  # schedule.zip

def read_csv_from_zip(url: str) -> pd.DataFrame:
    r = httpx.get(url, timeout=120.0)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        member = next((n for n in zf.namelist() if n.lower().endswith((".txt", ".csv"))), None)
        if member is None:
            raise RuntimeError(f"No .txt/.csv inside {url}")
        with zf.open(member) as f:
            # DPD extracts are standard CSV; UTF‑8 works for modern sets
            return pd.read_csv(f, dtype=str, encoding="utf-8", engine="python").fillna("")

def collapse(df: pd.DataFrame, key: str, col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame({key: [], col: []})
    return df.groupby(key, as_index=False)[col].apply(
        lambda s: "; ".join(sorted({str(x).strip() for x in s if str(x).strip()}))
    )

def main():
    print("▶ Downloading DPD marketed files…")
    drug   = read_csv_from_zip(DRUG_ZIP)      # DRUG_CODE, DRUG_IDENTIFICATION_NUMBER, BRAND_NAME, PRODUCT_CATEGORIZATION, CLASS
    form   = read_csv_from_zip(FORM_ZIP)      # DRUG_CODE, PHARMACEUTICAL_FORM
    route  = read_csv_from_zip(ROUTE_ZIP)     # DRUG_CODE, ROUTE_OF_ADMINISTRATION
    ingred = read_csv_from_zip(INGRED_ZIP)    # DRUG_CODE, INGREDIENT, STRENGTH, STRENGTH_UNIT
    comp   = read_csv_from_zip(COMPANY_ZIP)   # DRUG_CODE, COMPANY_NAME
    sched  = read_csv_from_zip(SCHEDULE_ZIP)  # DRUG_CODE, SCHEDULE

    # Keep a minimal drug table
    drug_small = drug[[
        "DRUG_CODE","DRUG_IDENTIFICATION_NUMBER","BRAND_NAME","PRODUCT_CATEGORIZATION","CLASS"
    ]].drop_duplicates()

    # Build a readable strength string
    if not ingred.empty:
        ingred["__STR"] = (ingred.get("STRENGTH","") + " " + ingred.get("STRENGTH_UNIT","")).str.strip()

    # Aggregate 1..N rows per DRUG_CODE into a single row
    ingr_c  = collapse(ingred, "DRUG_CODE", "INGREDIENT").rename(columns={"INGREDIENT":"ACTIVE_INGREDIENT"})
    str_c   = collapse(ingred, "DRUG_CODE", "__STR").rename(columns={"__STR":"STRENGTH"})
    form_c  = collapse(form,   "DRUG_CODE", "PHARMACEUTICAL_FORM").rename(columns={"PHARMACEUTICAL_FORM":"DOSAGE_FORM"})
    route_c = collapse(route,  "DRUG_CODE", "ROUTE_OF_ADMINISTRATION").rename(columns={"ROUTE_OF_ADMINISTRATION":"ROUTE"})
    comp_c  = collapse(comp,   "DRUG_CODE", "COMPANY_NAME").rename(columns={"COMPANY_NAME":"MANUFACTURER"})
    sched_c = collapse(sched,  "DRUG_CODE", "SCHEDULE")

    df = drug_small.merge(form_c,  on="DRUG_CODE", how="left") \
                   .merge(route_c, on="DRUG_CODE", how="left") \
                   .merge(ingr_c,  on="DRUG_CODE", how="left") \
                   .merge(str_c,   on="DRUG_CODE", how="left") \
                   .merge(comp_c,  on="DRUG_CODE", how="left") \
                   .merge(sched_c, on="DRUG_CODE", how="left")

    # --- Keep ONLY: Marketed + Human + Oral + Tablet/Capsule ---
    df["ROUTE_L"] = df["ROUTE"].str.lower()
    df["FORM_L"]  = df["DOSAGE_FORM"].str.lower()
    df["CAT_L"]   = df["PRODUCT_CATEGORIZATION"].str.lower()

    is_human = df["CAT_L"].str.contains("human", na=False)
    is_oral  = df["ROUTE_L"].str.contains("oral", na=False)
    is_tabcap= df["FORM_L"].str.contains(r"tablet|capsule|softgel", na=False)

    df = df[is_human & is_oral & is_tabcap].copy()

    # DIN: prefer DRUG_IDENTIFICATION_NUMBER; fallback to DRUG_CODE
    df["din"] = df["DRUG_IDENTIFICATION_NUMBER"].where(
        df["DRUG_IDENTIFICATION_NUMBER"].notna() & (df["DRUG_IDENTIFICATION_NUMBER"].str.strip()!=""),
        df["DRUG_CODE"].astype(str)
    ).str.strip()

    # Map to your Supabase columns
    df["brand_name"]        = df["BRAND_NAME"].str.strip()
    df["dosage_form"]       = df["DOSAGE_FORM"].str.strip()
    df["route"]             = df["ROUTE"].str.strip()
    df["strength"]          = df["STRENGTH"].str.strip()
    df["active_ingredient"] = df["ACTIVE_INGREDIENT"].str.strip()
    df["manufacturer"]      = df["MANUFACTURER"].str.strip()
    df["schedule"]          = df["SCHEDULE"].str.strip()
    df["class"]             = df["CLASS"].str.strip()
    df["status"]            = "Marketed"   # we’re reading the marketed set

    # Link to the DPD product page (often shows the Product Monograph link)
    df["monograph_url"] = "https://health-products.canada.ca/dpd-bdpp/info?lang=eng&code=" + df["DRUG_CODE"].astype(str)

    out = df[[
        "din","brand_name","dosage_form","route","strength","active_ingredient",
        "manufacturer","schedule","class","status","monograph_url"
    ]].fillna("")

    rows = out.to_dict(orient="records")
    print(f"▶ Upserting {len(rows)} rows into Supabase.pills …")
    client.table("pills").upsert(rows, on_conflict="din").execute()
    print("✅ Finished")

if __name__ == "__main__":
    main()
