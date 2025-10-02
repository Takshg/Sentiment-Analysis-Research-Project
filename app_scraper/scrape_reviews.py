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

DEFAULT_REGEX = r"""
\b(must[\s\-]?have|must\s*buy|must\s*download|essential\s+app|can'?t\s+live\s+without)\b
"""
KEY_RE = re.compile(DEFAULT_REGEX, flags=re.IGNORECASE|re.VERBOSE)

# Google Play Scraping 
def scrape_google_play(app_id: str, lang: str, country: str , max_reviews: int) -> List[Dict[str, Any]]:
    from google_play_scraper import Sort, reviews

    all_rows: List[Dict[str, Any]] = []
    count_step = 200 #batch size per request
    fetched = 0
    token = None

    with tqdm(total = max_reviews, desc = f"Google Play {app_id[{country}/{lang}]") as pbar:
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
                    

                }