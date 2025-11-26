import os 
import pandas as pd
import argparse
from tqdm import tqdm

DEFAULT_COLUMNS = ["store", "app_id", "review_id", "user_name", "rating", "title", "text", 
                   "version", "thumbs_up", "reply_text", "reply_date", "at", "language", "country"]

def main():
    parser = argparse.ArgumentParser(description="Merge all review CSVs into one master CSV.")
    parser.add_argument("--indir", default="data", help="Directory containing scraped CSVs.")
    parser.add_argument("--out", default="merged_reviews.csv", help="Output merged CSV.")
    parser.add_argument("--summary", default="merged_summary.csv", help="Summary of per-file row counts.")
    args = parser.parse_args()

    indir = args.indir
    out_path = args.out
    summary_path = args.summary

    print(f"Checking folder: {indir}")

    #List CSVs
    files = [
        os.path.join(indir, f) for f in os.listdir(indir)
        if f.endswith(".csv") and "_batch_summary" not in f
    ]

    if not files:
        print("No CSV files found.")
        return 
    
    print(f"Found {len(files)} CSV files.")

    merged_frames = []
    summary_rows = []

    for f in tqdm(files, desc="Loading CSVs"):
        try:
            df = pd.read_csv(f)
            for col in DEFAULT_COLUMNS:
                if col not in df.columns:
                    df[col] = None

            merged_frames.append(df[DEFAULT_COLUMNS])
            summary_rows.append({
                "file": os.path.basename(f),
                "rows": len(df)
            })
        except Exception as e:
            print(f"Could not load {f}: {e}")

    print("Concatenating")
    merged = pd.concat(merged_frames, ignore_index=True)

    print("Dropping Duplicates on (store, review_id)...")
    merged.drop_duplicates(subset=["store", "review_id"], inplace=True)

    print(f"Saving merged output: {out_path}")
    merged.to_csv(out_path, index=False, encoding="utf-8")

    print(f"💾 Saving summary: {summary_path}")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("Merge complete.")
    print(f"Total merged rows: {len(merged)}")

if __name__ == "__main__":
    main()