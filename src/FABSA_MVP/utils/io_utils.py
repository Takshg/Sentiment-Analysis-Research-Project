"""
DEPRECATED: The main workflow now lives in notebooks/fabsa_mvp.ipynb.
"""
import sys
print("⚠️  Deprecated: use notebooks/fabsa_mvp.ipynb", file=sys.stderr)

import pandas as pd 
from datasets import Dataset

def df_to_hf_dataset(df):
    return Dataset.from_pandas(df,preserve_index=False)

def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)