import re 
import csv
import sys
import time
import argparse
from datetime import datetime 
from typing import List, Dict, Any

import pandas as pd 
from tqdm import tqdm 
from tenacity import retry, stop_after_attempt, wait_exponential

# Keywords for apps

DEFAULT_REGEX = r".*"
KEY_RE = re.compile(DEFAULT_REGEX, flags=re.IGNORECASE|re.VERBOSE)

# Google Play Scraping 
def scrape_google_play(app_id: str, lang: str, country: str , max_reviews: int) -> List[Dict[str, Any]]:
    from google_play_scraper import Sort, reviews

    all_rows: List[Dict[str, Any]] = []
    count_step = 200 #batch size per request
    fetched = 0
    token = None

    with tqdm(total = max_reviews, desc = f"Google Play {app_id}[{country}/{lang}]") as pbar:
        while fetched < max_reviews:
            batch_size = min(count_step, max_reviews-fetched)
            chunk, token = reviews(
                app_id, 
                lang = lang, 
                country = country, 
                sort = Sort.NEWEST, 
                count = batch_size, 
                continuation_token=token
            )
            if not chunk: 
                break
            for r in chunk: 
                row = {
                    "store" : "google_play",
                    "app_id" : app_id, 
                    "review_id" : r.get("reviewId"),
                    "user_name" : r.get("userName"),
                    "rating" : r.get("score"), 
                    "title" : None, 
                    "text" : r.get("contents"),
                    "version" : r.get("reviewCreatedVersion"), 
                    "thumbs_up" : r.get("thumbsUpCount"),
                    "reply_text" : (r.get("replyContent") or None),
                    "reply_date" : _to_iso(r.get("repliedAt")),
                    "at" : _to_iso(r.get("at")),
                    "language" : lang, 
                    "country" : country
                }
                all_rows.append(row)
            fetched += len(chunk)
            pbar.update(len(chunk))
            if token is None:
                break
            time.sleep(0.3)
    return all_rows

# App Store
@retry(stop = stop_after_attempt(3), wait = wait_exponential(multiplier=1, min =1 , max = 4))

def app_store_page(app_id: str, country: str, page: int):
    from app_store_scraper import AppStore
    app = AppStore(country=country, app_name = "", app_id=app_id)
    app.review(how_many=200, page = page)
    return app.reviews

def scrape_app_store(app_id: str, country: str, max_reviews: int) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    count_step = 200 #batch size per request
    page = 1

    with tqdm(total = max_reviews, desc = f"App Store {app_id}[{country}]")as pbar:
        while len(all_rows) < max_reviews:
            try:
                reviews = app_store_page(app_id, country, page)
            except Exception as e:
                break
            if not reviews:
                break
            for r in reviews: 
                row = {
                    "store" : "app_store",
                    "app_id" : app_id, 
                    "review_id" : r.get("id"),
                    "user_name" : r.get("userName") or r.get("user"),
                    "rating" : r.get("rating"), 
                    "title" : r.get("title"), 
                    "text" : r.get("review") or r.get("text"),
                    "version" : r.get("version"), 
                    "thumbs_up" : None, # No like count
                    "reply_text" : r.get("developerResponse"),
                    "reply_date" : None,
                    "at" : _normalize_apple_date(r.get("date")),
                    "language" : None, 
                    "country" : country
                }
                all_rows.append(row)
            page += 1
            pbar.update(len(reviews))
            time.sleep(0.5)
    return all_rows[:max_reviews]

# Helpers

def _to_iso(dt) -> str:
    if not dt:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    try:
        return pd.to_datetime(dt).isoformat()
    except Exception: 
        return None 

def _normalize_apple_date(d) -> str:
    if not d: 
        return None
    try: 
        return pd.to_datetime(d).isoformat()
    except Exception:
        return None 

def filter_reviews(rows: List[Dict[str, Any]], pattern: re.Pattern) -> List[Dict[str, Any]]:
    out = []
    for r in rows: 
        text = " ".join([str(r.get("title") or ""), str(r.get("text") or "")])
        if pattern.search(text):
            out.append(r)
    return out

def parse_args(): 
    p = argparse.ArgumentParser(description= " Scraper for 'must-have' app reviews")
    p.add_argument("--store", choices = ["play", "appstore"], required=True, help = "Which store to scrape: Google Play or Apple App Store")
    p.add_argument("--app", required=True,
                   help="Google Play package name (e.g., com.whatsapp) or Apple numeric app id (e.g., 310633997).")
    p.add_argument("--countries", nargs="+", default=["ca"],
                   help="List of country codes (e.g., us ca gb in).")
    p.add_argument("--lang", default="en",
                   help="Language for Google Play (e.g., en, en_US). Ignored for App Store.")
    p.add_argument("--max", type=int, default=500,
                   help="Max reviews to fetch per country.")
    p.add_argument("--regex", default=".*",
                   help="Custom regex for filtering (defaults to broad variants).")
    p.add_argument("--out", default="reviews.csv",
                   help="Output CSV file path.")
    return p.parse_args()

def main():
    args = parse_args()
    pattern = re.compile(args.regex, flags=re.IGNORECASE) if args.regex else KEY_RE

    all_hits: List[Dict[str, Any]] = []
    for ctry in args.countries:
        if args.store == "play":
            rows = scrape_google_play(args.app, args.lang, ctry, args.max)
        else:
            rows = scrape_app_store(args.app, ctry, args.max)
        hits = filter_reviews(rows, pattern)
        all_hits.extend(hits)

    if not all_hits:
        print("No reviews found with current settings.")
        headers = ["store","app_id","review_id","user_name","rating","title","text","version","thumbs_up",
                   "reply_text","reply_date","at","language","country"]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        sys.exit(0)

    df = pd.DataFrame(all_hits).drop_duplicates(subset=["store","review_id"]).sort_values("at", na_position="last")

    # Clean up whitespace in text fields
    for col in ["title","text","reply_text"]:
        if col in df.columns:
            df[col] = df[col].fillna("").str.replace(r"\s+", " ", regex=True).str.strip() 

    df.to_csv( args.out, index=False, encoding="utf-8")
    print(f"Saved {len(df)} filtered reviews -> {args.out}")

if __name__ == "__main__":
    main()
