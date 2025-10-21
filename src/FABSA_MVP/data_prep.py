"""
DEPRECATED: The main workflow now lives in notebooks/fabsa_mvp.ipynb.
"""
import sys
print("⚠️  Deprecated: use notebooks/fabsa_mvp.ipynb", file=sys.stderr)

import ast, json
import os
import pandas as pd
from typing import List,Tuple
from sklearn.model_selection import train_test_split
from config import ASPECTS, LABELS, LABEL2ID, FABSA_MASTER, FABSA_TRAIN, FABSA_DEV, FABSA_TEST, SEED, ID_COL, TEXT_COL, LABELS_COL, EXTRA_COLS

import ast, json
from typing import List, Tuple
from config import ASPECTS

SENT_MAP = {"-1":"negative", "0":"neutral", "1":"positive"}

def _parse_labels(cell) -> List[Tuple[str, str]]:
    """
    Accepts strings like: "['online-experience.app-website.-1', 'value.price-value-for-money.-1']"
    Returns: list of (aspect, sentiment_word) 
    """
    if cell is None or cell == "" or (isinstance(cell, float) and pd.isna(cell)):
        return []
    
    if isinstance(cell, list):
        items=cell
    else: 
        try: 
            items = ast.literal_eval(str(cell))
        except Exception:
            return []
    out = []
    for item in items: 
        if not isinstance(item,str): 
            continue
        parts = item.split(".")
        if len(parts) < 2:
            continue
        aspects = parts[-2].strip()
        sent_code = parts[-1].strip()
        sent = SENT_MAP.get(sent_code)
        if sent is None:
            continue
        out.append((aspects, sent))
    return out

def _primary_label(row) -> str:
    labs = _parse_labels(row.get(LABELS_COL, "")) or []
    if not labs:
        return "NONE"
    a, s = labs[0]
    return f"{a}|{s}"

def _validate_master_schema(df: pd.DataFrame):
    required = {ID_COL, TEXT_COL, LABELS_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Master FABSA file missing columns: {missing}. "
            f"Expected at least: {sorted(required)}. Got: {list(df.columns)}"
        )
    
def _ensure_splits_exist(
    train_path: os.PathLike,
    dev_path: os.PathLike,
    test_path: os.PathLike,
    master_path: os.PathLike,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if all(os.path.exists(p) for p in [train_path, dev_path, test_path]):
        return pd.read_csv(train_path), pd.read_csv(dev_path), pd.read_csv(test_path)

    if not os.path.exists(master_path):
        raise FileNotFoundError(
            f"FABSA master file not found at {master_path}. "
            f"Expected a CSV with at least: {ID_COL},{TEXT_COL},{LABELS_COL}"
        )

    master = pd.read_csv(master_path)
    _validate_master_schema(master)

    # Build key: primary label + industry
    master = master.copy()
    primary = master.apply(_primary_label, axis=1)
    if "industry" in master.columns:
        master["_strat"] = primary.astype(str) + "##" + master["industry"].astype(str)
    else:
        master["_strat"] = primary

    try:
        train, temp = train_test_split(
            master, test_size=0.30, random_state=seed, shuffle=True, stratify=master["_strat"]
        )
        dev, test = train_test_split(
            temp, test_size=(2/3), random_state=seed, shuffle=True, stratify=temp["_strat"]
        )
    except ValueError:
        train, temp = train_test_split(
            master, test_size=0.30, random_state=seed, shuffle=True
        )
        dev, test = train_test_split(
            temp, test_size=(2/3), random_state=seed, shuffle=True
        )

    for name, df in [("train", train), ("dev", dev), ("test", test)]:
        df.drop(columns=["_strat"], errors="ignore").to_csv(
            {"train": train_path, "dev": dev_path, "test": test_path}[name], index=False
        )

    return (
        train.drop(columns=["_strat"], errors="ignore"),
        dev.drop(columns=["_strat"], errors="ignore"),
        test.drop(columns=["_strat"], errors="ignore"),
    )

def load_fabsa_split(train_path=FABSA_TRAIN, dev_path=FABSA_DEV, test_path=FABSA_TEST):
    """Loads train/dev/test if present; else creates them from FABSA_MASTER (70/10/20)."""
    return _ensure_splits_exist(train_path, dev_path, test_path, FABSA_MASTER, seed=SEED)

def make_sentence_pairs(df_reviews: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_reviews.iterrows(): 
        gold_pairs = set(_parse_labels(r.get(LABELS_COL, ""))) # Format is (aspect, sentiment)

        for asp in ASPECTS: 
            sentiment = "absent" #Default 
            #If aspect is present in gold, use its sentiment
            for (ga,gs) in gold_pairs: 
                if ga == asp: 
                    sentiment = gs
                    break

            item = {
                ID_COL: r[ID_COL],
                "text": r[TEXT_COL], 
                "aspect": asp, 
                "target_label_str": sentiment, 
                "target_label_id": LABEL2ID[sentiment], 
            }
            for c in EXTRA_COLS: 
                if c in r: 
                    item[c] = r[c]
            rows.append(item)
    return pd.DataFrame(rows)