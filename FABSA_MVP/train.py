import traceback
import pandas as pd 
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from config import(FABSA_TRAIN, FABSA_DEV, FABSA_TEST, FABSA_TRAIN_PAIRS, FABSA_DEV_PAIRS,
                    MAX_LEN, LR, EPOCHS, TRAIN_BS, EVAL_BS, MODEL_DIR, SEED, ID_COL, TEXT_COL)
from data_prep import load_fabsa_split, make_sentence_pairs
from models.pair_classifier import load_model_and_tokenizer
from utils.metrics import hf_classification_metrics
from utils.io_utils import save_csv

def main(): 
    print("[START] train.py")
    print("[PATHS]")
    print("  FABSA_TRAIN:", FABSA_TRAIN)
    print("  FABSA_DEV  :", FABSA_DEV)
    print("  FABSA_TEST :", FABSA_TEST)
    
    
    # (1) Loading FASBA splits 
    print("[STEP] Loading/creating FABSA splits (70/10/20 if missing)...")
    train_df, dev_df, test_df = load_fabsa_split(FABSA_TRAIN, FABSA_DEV, FABSA_TEST)
    print(f"[INFO] Split sizes -> train: {len(train_df)}, dev: {len(dev_df)}, test: {len(test_df)}")

    # (2) Expanding to sentence-pair rows
    print("[STEP] Building sentence–aspect pairs...")
    train_pairs = make_sentence_pairs(train_df)
    dev_pairs = make_sentence_pairs(dev_df)

    print(f"[INFO] Pair sizes -> train_pairs: {len(train_pairs)}, dev_pairs: {len(dev_pairs)}")
    if len(train_pairs) == 0:
        print("[ERROR] train_pairs is EMPTY. Check your 'labels' column format in data/fabsa/*.csv")
        return
    
    print("[STEP] Saving pair CSVs...")
    save_csv(train_pairs, FABSA_TRAIN_PAIRS)
    save_csv(dev_pairs, FABSA_DEV_PAIRS)
    print("  wrote:", FABSA_TRAIN_PAIRS)
    print("  wrote:", FABSA_DEV_PAIRS)


    # (3) Tokenizer & tokenization
    print("[STEP] Loading tokenizer & model...")
    model, tok = load_model_and_tokenizer()

    def tokenize_fn(batch):
        # batched=True -> batch["text"]/batch["aspect"] are lists
        return tok(
            batch["text"], 
            batch["aspect"], 
            truncation=True, 
            max_length=MAX_LEN,
            return_overflowing_tokens=False)

    print("[STEP] Creating HF datasets...")
    hf_train = Dataset.from_pandas(train_pairs)
    hf_dev   = Dataset.from_pandas(dev_pairs)

    # Rename target label column to 'labels' first
    hf_train = hf_train.rename_column("target_label_id", "labels")
    hf_dev   = hf_dev.rename_column("target_label_id", "labels")

    # Keep only tokenizer outputs + 'labels'
    cols_to_remove_train = [c for c in hf_train.column_names if c not in ("labels",)]
    cols_to_remove_dev   = [c for c in hf_dev.column_names   if c not in ("labels",)]

    hf_train = hf_train.map(tokenize_fn, batched=True, remove_columns=cols_to_remove_train)
    hf_dev   = hf_dev.map(tokenize_fn,   batched=True, remove_columns=cols_to_remove_dev)

    print("[DEBUG] train columns after map:", hf_train.column_names)
    print("[DEBUG] dev   columns after map:", hf_dev.column_names)

    hf_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    hf_dev.set_format(type="torch",   columns=["input_ids", "attention_mask", "labels"])
        
    # (4) Trainer
    print("[STEP] Configuring Trainer...")
    args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        gradient_accumulation_steps=8, # Effective batch size without RAM spiking
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="f1_macro",
        seed=SEED,
        dataloader_num_workers=0,        # lower loader memory/threads
        dataloader_pin_memory=False,     # MPS ignores pinning; keep False
        logging_steps=10,
        disable_tqdm=False,
        report_to="none",
        use_mps_device=True, 
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=hf_train,
        eval_dataset=hf_dev,
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tokenizer=tok),
        compute_metrics=hf_classification_metrics
    )

    # (5) Train
    print("[STEP] Starting training...")
    trainer.train()
    print("[STEP] Training complete.")

    # (6)) Save
    print("[STEP] Saving model & tokenizer ...")
    trainer.save_model(str(MODEL_DIR))
    tok.save_pretrained(str(MODEL_DIR))
    print("[DONE] Saved to:", MODEL_DIR)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL] Uncaught exception in train.py")
        traceback.print_exc()