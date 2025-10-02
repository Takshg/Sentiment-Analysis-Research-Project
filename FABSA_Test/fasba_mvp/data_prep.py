import ast
import pandas as pd
from FABSA_Test.fasba_mvp.config import ASPECTS, LABELS, LABEL2ID

def _parse_labels(cell): 
    if pd.isna(cell) or cell=="":
        return []
    v = cell if isinstance(cell, list) else ast.literal_eval(cell)
    return [(a,s) for a,s in v]

def make_sentence_pairs(df_reviews, text_col = "text", labels_col = "labels"): 
    rows = []
    for _,r in df_reviews.iterrows(): 
        for asp in ASPECTS:
            sentiment = "absent"
            for (ga, gs) in gold: 
                if ga == asp:
                    sentiment = gs
                    break
            rows.append({
                "review_id" : r["review_id"], 
                "text" : r[text_col], 
                "aspect" : asp, 
                "target_label_str" : sentiment, 
                "target_label_id" : LABEL2ID[sentiment]
            })
    return pd.DataFrame(rows)

def load_fabsa_split(train_path, dev_path, test_path):
    train = pd.read_csv(train_path)
    dev = pd.read_csv(dev_path)
    test = pd.read_csv(test_path)
    return train,dev,test