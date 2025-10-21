"""
DEPRECATED: The main workflow now lives in notebooks/fabsa_mvp.ipynb.
"""
import sys
print("⚠️  Deprecated: use notebooks/fabsa_mvp.ipynb", file=sys.stderr)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_NAME, LABELS

def load_model_and_tokenizer(num_labels=len(LABELS)):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)  # lighter on memory
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label={i:l for i,l in enumerate(LABELS)},
        label2id={l:i for i,l in enumerate(LABELS)},
    )

    model.gradient_checkpointing_enable()  # reduces activation memory
    model.config.use_cache = False         # required with gradient checkpointing
    return model, tok
