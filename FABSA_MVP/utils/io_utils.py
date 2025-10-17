import pandas as pd 
from datasets import Dataset

def df_to_hf_dataset(df):
    return Dataset.from_pandas(df,preserve_index=False)

def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)