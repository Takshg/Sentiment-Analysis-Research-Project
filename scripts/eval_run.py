"""
DEPRECATED: The main workflow now lives in notebooks/fabsa_mvp.ipynb.
"""
import sys
print("⚠️  Deprecated: use notebooks/fabsa_mvp.ipynb", file=sys.stderr)


import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from config import (FABSA_TEST, FABSA_TEST_PAIRS, MODEL_DIR, MAX_LEN)
from data_prep import make_sentence_pairs, load_fabsa_split
from utils.metrics import hf_classification_metrics
from utils.io_utils import save_csv

def main(): 

    # (1) Load test & Expand to pairs
    _,_, test_df = load_fabsa_split(FABSA_TEST, FABSA_TEST,FABSA_TEST)
    test_pairs = make_sentence_pairs(test_df)
    save_csv(test_df)

    # (2) Load Model/Tokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

    def tokenize(ex):
        return tok(ex["text"], ex["aspect"], truncation = True, max_length = MAX_LEN)
    
    hf_test = Dataset.from_pandas(test_pairs)
    hf_test = hf_test.map(tokenize, batched=True)
    hf_test = hf_test.rename_column("target_label_id", "labels")
    hf_test.set_format(type = "torch", columns = ["input_ids", "attention_mask", "labels"])

    # (3) User Trainer only for evaluation util
    args = TrainingArguments(output_dir="tmp_eval", per_device_eval_batch_size=64)
    trainer = Trainer(model=model, args=args, tokenizer=tok, compute_metrics=hf_classification_metrics)

    metrics = trainer.evaluate(hf_test)
    print(metrics)

if __name__ == "__main__": 
    main()
    