import pandas as pd 
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from config import(FABSA_TRAIN, FABSA_DEV, FABSA_TRAIN_PAIRS, FABSA_DEV_PAIRS, 
                   MAX_LEN, LR, EPOCHS, TRAIN_BS, EVAL_BS, MODEL_DIR, SEED)
from data_prep import load_fasba_split, make_sentence_pairs
from models.pair_classifier import load_model_and_tokenizer
from utils.metrics import hf_classification_metrics
from utils.io_utils import save_csv

def main(): 
    # (1) Loading FASBA splits 
    train_df, dev_df, _ = load_fasba_split(FABSA_TRAIN, FABSA_DEV, FABSA_TRAIN)

    # (2) Expanding to sentence-pair rows
    train_pairs = make_sentence_pairs(train_df)
    dev_pairs = make_sentence_pairs(dev_df)
    save_csv(train_pairs, FABSA_TRAIN_PAIRS)
    save_csv(dev_pairs, FABSA_DEV_PAIRS)

    #(3) Tokenizer & tokenization 
    model, tok = load_model_and_tokenizer()

    def tokenize(ex):
        return tok(ex["text"], ex["aspect"], truncation=True, max_length=MAX_LEN)

    hf_train = Dataset.from_pandas(train_pairs)
    hf_dev = Dataset.from_pandas(dev_pairs)
    hf_train = hf_train.map(tokenize, batched=True)
    hf_train = hf_train.rename_column("target_label_id", "labels")
    hf_dev = hf_dev.rename_column("target_label_id", "labels")
    hf_train.set_format(type='torch', columns=["input_ids", "attention_mask", "labels"])
    hf_dev.set_format(type='torch', columns=["input_ids","attention_mask"])