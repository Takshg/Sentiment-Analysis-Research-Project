"""
DEPRECATED: The main workflow now lives in notebooks/fabsa_mvp.ipynb.
"""
import sys
print("⚠️  Deprecated: use notebooks/fabsa_mvp.ipynb", file=sys.stderr)

import argparse
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import ASPECTS, MAX_LEN, MODEL_DIR, ID_COL, TEXT_COL
from utils.io_utils import save_csv

def build_pairs(df, text_col="text"):
    rows = []
    for _, r in df.iterrows():
        for asp in ASPECTS:
            rows.append({ID_COL: r[ID_COL], "text": r[TEXT_COL], "aspect": asp})
    return pd.DataFrame(rows)

def main(): 
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", default = "outputs/preds/review_preds.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    pairs = build_pairs(df)

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

    ds = Dataset.from_pandas(pairs)
    ds = ds.map(lambda ex: tok(ex["text"], ex["aspect"], truncation=True, max_length = MAX_LEN, batched=True))
    ds.set_format(type="torch", columns=["input_ids","attention_mask"]) 

    logits = model(**{k: ds[k] for k in ["input_ids", "attention_mask"]}).logits.detach().cpu().numpy()
    probs = (logits - logits.max(axis=1, keepdims = True))
    probs = np.exp(probs) / np.exp(probs).sum(axis=1,keepdims=True)
    pred_ids = logits.argmax(axis=1)

    out = pairs.copy()
    out["pred_label_id"] = pred_ids
    id2label = model.config.id2label
    out["pred_label_str"] = [id2label[int(i)] for i in pred_ids]
    out["prob"] = probs[np.arrange(len(probs)), pred_ids]

    # Keeping only non-absent rows 
    #out = out[out["pred_label_str"] != "absent"]
    
    save_csv(out, args.output_csv)
    print(f"Saved predictions to: {args.output_csv}")

if __name__ == "__main__": 
    main()
