import ast, json
import os
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from config import ASPECTS, LABELS, LABEL2ID, FABSA_MASTER, FABSA_TRAIN, FABSA_DEV, FABSA_TEST, SEED, ID_COL, TEXT_COL, LABELS_COL, EXTRA_COLS

# data_prep.py (replace _parse_labels with this)
import ast, json
from typing import List, Tuple
from config import ASPECTS

SENTIMENTS = {"positive", "negative", "neutral"}

def _parse_labels(cell) -> List[Tuple[str, str]]:
    """
    Accepts:
      - stringified Python list (ast-literal), or JSON string
      - list of lists/tuples: [aspect, sentiment, ...]
      - list of dicts: {"aspect": "...", "sentiment": "..."} (also tries keys like "category","polarity")
      - list of triples/quads where aspect & sentiment are among the items
    Returns: list of (aspect, sentiment) tuples. Unknowns are skipped.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)) or cell == "":
        return []

    obj = cell
    if isinstance(cell, str):
        s = cell.strip()
        try:
            obj = ast.literal_eval(s)
        except Exception:
            try:
                obj = json.loads(s)
            except Exception:
                return []

    if not isinstance(obj, (list, tuple)):
        return []

    out = []
    for item in obj:
        aspect_val = None
        sent_val = None

        if isinstance(item, dict):
            for k in ["aspect", "category", "target", "aspect_category"]:
                if k in item and isinstance(item[k], str):
                    aspect_val = item[k].strip()
                    break
            for k in ["sentiment", "polarity", "opinion"]:
                if k in item and isinstance(item[k], str):
                    cand = item[k].strip().lower()
                    if cand in SENTIMENTS:
                        sent_val = cand
                        break

        elif isinstance(item, (list, tuple)):
            for x in item:
                if isinstance(x, str):
                    xs = x.strip()
                    if aspect_val is None and (xs in ASPECTS or " " in xs or "-" in xs):
                        aspect_val = xs
                    if sent_val is None and xs.lower() in SENTIMENTS:
                        sent_val = xs.lower()
        if aspect_val and sent_val:
            out.append((aspect_val, sent_val))

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
    """
    Input: df with columns including ID_COL, TEXT_COL, LABELS_COL, and EXTRA_COLS.
    Output: per-(id,aspect) rows with target labels, preserving EXTRA_COLS.
    """
    rows = []
    for _, r in df_reviews.iterrows():
        gold = set(_parse_labels(r.get(LABELS_COL, "")))
        for asp in ASPECTS:
            sentiment = "absent"
            for (ga, gs) in gold:
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