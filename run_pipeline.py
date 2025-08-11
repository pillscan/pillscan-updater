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
# Health Canada DPD ZIP URLs
# (you already put these in Render Env)
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
def _detect_sep(sample: bytes) -> str:
    """Guess delimiter: DPD extracts often use '|'."""
    head = sample.decode("utf-8", errors="ignore")
    counts = { "|": head.count("|"), ",": head.count(","), "\t": head.count("\t") }
    # pick the most frequent among common delimiters
    sep = max(counts, key=counts.get)
    # basic sanity fallback
    return sep if counts[sep] > 0 else ","


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Uppercase + underscores for stable column names."""
    def norm(c: str) -> str:
        c = c.strip().upper()
        c = re.sub(r"[^\w]+", "_", c)
        return c
    df.columns = [norm(str(c)) for c in df.columns]
    return df


def read_csv_from_zip(url: str) -> pd.DataFrame:
    """Download and read the single CSV/TXT inside a DPD ZIP."""
    r = httpx.get(url, timeout=120.0, follow_redirects=True)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        member = next((n for n in zf.namelist() if n.lower().endswith((".txt", ".csv"))), None)
        if not member:
            raise RuntimeError(f"No .txt/.csv inside ZIP: {url}")
        with zf.open(member) as f:
            raw = f.read()

    sep = _detect_sep(raw[:4096])

    last_err = None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                io.BytesIO(raw),
                dtype=str,
                encoding=enc,
                sep=sep,
                engine="python"
            ).fillna("")
            df = _normalize_columns(df)
            print(f"Loaded {url.split('/')[-1]} with sep='{sep}', enc='{enc}': {list(df.columns)[:8]} ...")
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to parse {url} (sep='{sep}'): {last_err}")


def collapse(df: pd.DataFrame, key: str, col: str) -> pd.DataFrame:
    """Aggregate many-to-one as semicolon-separated unique values."""
    if df.empty:
        return pd.DataFrame({key: [], col: []})
    return (
        df.groupby(key, as_index=False)[col]
          .apply(lambda s: "; ".join(sorted({str(x).strip() for x in s if str(x).strip()})))
    )


def require_columns(df: pd.DataFrame, needed: list, label: str):
    """Friendly check to fail early with helpful message."""
    have = set(df.columns)
    missing = [c for c in needed if c not in have]
    if missing:
        raise RuntimeError(
            f"{label}: missing columns {missing}. Present columns: {list(df.columns)[:20]} ..."
        )


# =====
# Main
# =====
def main():
    print("▶ Downloading DPD marketed files…")
    drug   = read_csv_from_zip(DRUG_ZIP)      # DRUG_CODE, DRUG_IDENTIFICATION_NUMBER, BRAND_NAME, PRODUCT_CATEGORIZATION, CLASS
    form   = read_csv_from_zip(FORM_ZIP)      # DRUG_CODE, PHARMACEUTICAL_FORM
    route  = read_csv_from_zip(ROUTE_ZIP)     # DRUG_CODE, ROUTE_OF_ADMINISTRATION
    ingred = read_csv_from_zip(INGRED_ZIP)    # DRUG_CODE, INGREDIENT, STRENGTH, STRENGTH_UNIT
    comp   = read_csv_from_zip(COMPANY_ZIP)   # DRUG_CODE, COMPANY_NAME
    sched  = read_csv_from_zip(SCHEDULE_ZIP)  # DRUG_CODE, SCHEDULE

    # Sanity checks on required columns
    require_columns(drug,   ["DRUG_CODE","BRAND_NAME"], "drug.zip")
    # DIN + CLASS + PRODUCT_CATEGORIZATION are commonly present, but we'll be safe:
    # If DRUG_IDENTIFICATION_NUMBER missing for some rows, we'll fallback to DRUG_CODE later.
    # For joins:
    for_df_cols  = ["DRUG_CODE","PHARMACEUTICAL_FORM"]
    route_cols   = ["DRUG_CODE","ROUTE_OF_ADMINISTRATION"]
    ingred_cols  = ["DRUG_CODE","INGREDIENT","STRENGTH","STRENGTH_UNIT"]
    comp_cols    = ["DRUG_CODE","COMPANY_NAME"]
    sched_cols   = ["DRUG_CODE","SCHEDULE"]

    for label, df, cols in [
        ("form.zip",   form,   for_df_cols),
        ("route.zip",  route,  route_cols),
        ("ingred.zip", ingred, ingred_cols),
        ("comp.zip",   comp,   comp_cols),
        ("schedule.zip", sched, sched_cols),
    ]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            # Don’t hard-fail; just warn. We can still proceed with other tables.
            print(f"⚠️  {label}: missing expected columns {missing}. Present: {list(df.columns)[:12]}")

    # Build minimal drug table
    cols_present = [c for c in ["DRUG_CODE","DRUG_IDENTIFICATION_NUMBER","BRAND_NAME","PRODUCT_CATEGORIZATION","CLASS"] if c in drug.columns]
    drug_small = drug[cols_present].drop_duplicates()

    # Strength string on ingredients (if present)
    if not ingred.empty and "STRENGTH" in ingred.columns and "STRENGTH_UNIT" in ingred.columns:
        ingred["__STR"] = (ingred.get("STRENGTH","") + " " + ingred.get("STRENGTH_UNIT","")).str.strip()
    else:
        ingred["__STR"] = ""

    # Aggregate 1..N to 1
    def safe_collapse(df, key, col, rename_to):
        if key not in df.columns or col not in df.columns:
            return pd.DataFrame({key: [], rename_to: []})
        out = collapse(df, key, col)
        return out.rename(columns={col: rename_to})

    ingr_c  = safe_collapse(ingred, "DRUG_CODE", "INGREDIENT", "ACTIVE_INGREDIENT")
    str_c   = safe_collapse(ingred, "DRUG_CODE", "__STR", "STRENGTH")
    form_c  = safe_collapse(form,   "DRUG_CODE", "PHARMACEUTICAL_FORM", "DOSAGE_FORM")
    route_c = safe_collapse(route,  "DRUG_CODE", "ROUTE_OF_ADMINISTRATION", "ROUTE")
    comp_c  = safe_collapse(comp,   "DRUG_CODE", "COMPANY_NAME", "MANUFACTURER")
    sched_c = safe_collapse(sched,  "DRUG_CODE", "SCHEDULE", "SCHEDULE")

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
    df["ROUTE_L"] = df.get("ROUTE","").str.lower()
    df["FORM_L"]  = df.get("DOSAGE_FORM","").str.lower()
    df["CAT_L"]   = df.get("PRODUCT_CATEGORIZATION","").str.lower()

    is_human = df["CAT_L"].str.contains("human", na=False)
    is_oral  = df["ROUTE_L"].str.contains("oral", na=False)
    is_tabcap= df["FORM_L"].str.contains(r"tablet|capsule|softgel", na=False)

    df = df[is_human & is_oral & is_tabcap].copy()

    # DIN: prefer DRUG_IDENTIFICATION_NUMBER; fallback to DRUG_CODE
    if "DRUG_IDENTIFICATION_NUMBER" in df.columns:
        df["din"] = df["DRUG_IDENTIFICATION_NUMBER"].where(
            df["DRUG_IDENTIFICATION_NUMBER"].astype(str).str.strip() != "",
            df["DRUG_CODE"].astype(str)
        ).astype(str).str.strip()
    else:
        df["din"] = df["DRUG_CODE"].astype(str).str.strip()

    # Map output columns expected by your Supabase 'pills' table
    df["brand_name"]        = df.get("BRAND_NAME","").astype(str).str.strip()
    df["dosage_form"]       = df.get("DOSAGE_FORM","").astype(str).str.strip()
    df["route"]             = df.get("ROUTE","").astype(str).str.strip()
    df["strength"]          = df.get("STRENGTH","").astype(str).str.strip()
    df["active_ingredient"] = df.get("ACTIVE_INGREDIENT","").astype(str).str.strip()
    df["manufacturer"]      = df.get("MANUFACTURER","").astype(str).str.strip()
    df["schedule"]          = df.get("SCHEDULE","").astype(str).str.strip()
    df["class"]             = df.get("CLASS","").astype(str).str.strip()
    df["status"]            = "Marketed"  # using the marketed set
    df["monograph_url"]     = "https://health-products.canada.ca/dpd-bdpp/info?lang=eng&code=" + df["DRUG_CODE"].astype(str)

    out = df[[
        "din","brand_name","dosage_form","route","strength","active_ingredient",
        "manufacturer","schedule","class","status","monograph_url"
    ]].fillna("")

    # Upsert to Supabase
    rows = out.to_dict(orient="records")
    print(f"▶ Upserting {len(rows)} rows into Supabase.pills …")
    if rows:
        client.table("pills").upsert(rows, on_conflict="din").execute()
    print("✅ Finished")


if __name__ == "__main__":
    main()
