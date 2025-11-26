import os 
import re
import sys
import pandas as pd
from typing import List
from scrape_reviews import scrape_google_play, scrape_app_store, filter_reviews
import argparse
import csv
from decimal import Decimal, InvalidOperation

# Config
DEFAULT_EXCEL = "app_scraper/data/App_IDs_List.xlsx"
DEFAULT_SHEET = 0
DEFAULT_OUTDIR = "app_scraper/data"
DEFAULT_LANG = "en"
DEFAULT_MAX = 10000
DEFAULT_REGEX = ".*"
COUNTRY_TO_CODE = {
    "canada" : "ca", 
    "united states" : "us", 
    "usa" : "us", 
    "united kingdom" : "gb", 
    "uk" : "gb", 
    "india" : "in",
    "australia" : "au", 
}


# Utilities
def norm_country_to_code(val: str) -> str:
    if not val: 
        return "ca"
    s = str(val).strip().lower()
    if s in COUNTRY_TO_CODE:
        return COUNTRY_TO_CODE[s]
    if re.fullmatch(r"[a-z]{2}", s):
        return s 
    return s[:2]

def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d: 
        os.makedirs(d, exist_ok=True)

def parse_play_id(x) -> str:
    "Accepts Google Play package names or Play URLs"
    if x is None or (isinstance(x, float) and pd.isna(x)):
        raise ValueError("Missing Google Play app id")
    s = str(x).strip()
    if not s:
        raise ValueError("Missing Google Play app id")
    if "play.google.com" in s:
        m = re.search(r"[?&]id=([A-Za-z0-9._]+)", s)
        if not m:
            raise ValueError(f"Could not extract package id from Play URL: {s}")
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9._]+", s):
        return s
    raise ValueError(f"Google Play App ID must be a package name or Play Store URL. Got: {s}")


def parse_app_store_id(x) -> str:
    "Accepts numeric App Store IDs or App Store URLs"
    if x is None or (isinstance(x, float) and pd.isna(x)):
        raise ValueError("Missing App Store app id")
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return str(int(x))
    s = str(x).strip()
    if not s:
        raise ValueError("Missing App Store app id")
    if "apps.apple.com" in s:
        m = re.search(r"/id(\d+)", s)
        if not m:
            raise ValueError(f"Could not extract numeric id from App Store URL: {s}")
        return m.group(1)
    if re.fullmatch(r"\d{6,}", s):
        return s
    try:
        as_decimal = Decimal(s)
    except (InvalidOperation, ValueError):
        pass
    else:
        integral = as_decimal.to_integral_value()
        if as_decimal == integral:
            return str(int(integral))
    raise ValueError(f"App Store App ID must be numeric or an App Store URL. Got: {s}")


def main():
    p = argparse.ArgumentParser(description='Batch scrape from App IDs List')
    p.add_argument("--excel", default = DEFAULT_EXCEL)
    p.add_argument("--sheet", default=DEFAULT_SHEET)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--lang", default=DEFAULT_LANG)
    p.add_argument("--max", type=int ,default=DEFAULT_MAX)
    p.add_argument("--regex", default=DEFAULT_REGEX)
    args = p.parse_args()

    df = pd.read_excel(args.excel, sheet_name=args.sheet)

    required_cols = ["App Name", "Country", "Store", "App ID"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns.str.strip())}")
    
    os.makedirs(args.outdir, exist_ok=True)

    successes = 0
    failures = 0
    summary_rows: List[dict] = []

    total = len(df)
    print(f"Found {total} rows in '{args.excel}. Writing CSVs to '{args.outdir}'")

    for i, row in df.iterrows():
        app_name = str(row["App Name"])
        country_code = norm_country_to_code(row["Country"])
        store_raw = str(row["Store"]).strip().lower()
        app_raw = row["App ID"]

        store = "play" if store_raw in ("play store", "google play", "play") else \
        "appstore" if store_raw in ("app store", "appstore", "apple") else store_raw

        try:
            if store == "play":
                app_id = parse_play_id(app_raw)
            elif store == "appstore":
                app_id = parse_app_store_id(app_raw)
            else:
                raise ValueError(f"Unknown store value '{row['Store']}' (row {i+1})")
            
            safe_id = re.sub(r"[^A-Za-z0-9_]+", "_", str(app_id))
            outfile = os.path.join(args.outdir, f"{safe_id}_{store}.csv")
            ensure_dir(outfile)

            print(f"\n[{i+1}/{total}] {store.upper()} :: {app_name} ({app_id}) :: country={country_code} max={args.max}")

            if store == "play":
                rows = scrape_google_play(app_id, args.lang, country_code, args.max)
            else:
                rows = scrape_app_store(app_id, country_code, args.max)

            if args.regex.strip() != ".*":
                patt = re.compile(args.regex, flags = re.IGNORECASE)
                rows = filter_reviews(rows, patt)

            if not rows:
                headers = ["store","app_id","review_id","user_name","rating","title","text","version",
                           "thumbs_up","reply_text","reply_date","at","language","country"]
                with open(outfile, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=headers).writeheader() # Saving header-only file if empty 
                print(f"-> No rows to save. Wrote header-only file: {outfile}")
            else:
                pd.DataFrame(rows).drop_duplicates(subset=["store","review_id"]).to_csv(outfile, index=False, encoding="utf-8")
                print(f"-> Saved {len(rows)} rows to {outfile}")

            successes += 1
            summary_rows.append({
                "row_index": i+1,
                "app_name": app_name,
                "store": store,
                "app_id": app_id,
                "country": country_code,
                "lang": args.lang if store == "play" else "",
                "max": args.max,
                "regex": args.regex,
                "out": outfile,
                "status": "ok",
                "rows_saved": len(rows),
            })
        except Exception as e:
            failures += 1 
            print(f"!! Row {i+1} FAILED: {e}")
            summary_rows.append({
                "row_index": i+1,
                "app_name": app_name,
                "store": store_raw,
                "app_id": str(app_raw),
                "country": country_code,
                "status": f"error: {e}",
                "rows_saved": 0,
            })
    summary_path = os.path.join(args.outdir, "_batch_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nDone. Successes: {successes}, Failures: {failures}. Summary → {summary_path}")

if __name__ == "__main__": 
    main()
