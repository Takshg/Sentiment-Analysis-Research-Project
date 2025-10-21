"""
DEPRECATED: The main workflow now lives in notebooks/fabsa_mvp.ipynb.
"""
import sys
print("⚠️  Deprecated: use notebooks/fabsa_mvp.ipynb", file=sys.stderr)


from pathlib import Path 

DATA_DIR = Path("data")
FABSA_MASTER = DATA_DIR /"fabsa" /"fabsa_dataset.csv"
REVIEWS_DIR = DATA_DIR / "Raw_Reviews"
OUT_DIR = Path("outputs")
MODEL_DIR = OUT_DIR  / "Models" / "deberta_pair"
PREDS_DIR = OUT_DIR  / "Preds"
REPORTS_DIR = OUT_DIR  / "Reports"
for x in [DATA_DIR, REVIEWS_DIR, MODEL_DIR, PREDS_DIR, REPORTS_DIR]: 
    x.mkdir(parents= True, exist_ok=True)

#FABSA Dataset 
ID_COL = "id"
TEXT_COL = "text"
LABELS_COL = "labels"
EXTRA_COLS = ["org_index", "industry"]

# Task Space
ASPECTS = [
    "app-website",
    "general-satisfaction",
    "ease-of-use",
    "attitude-of-staff",
    "price-value-for-money",
    "speed",
    "competitor",
    "account-access",
    "discounts-promotions",
    "phone",
    "reviews",
    "email",
]

LABELS = ['negative', 'neutral', 'positive', 'absent']
LABEL2ID = {l:i for i,l in enumerate(LABELS)}
ID2LABEL = {i:l for l,i in LABEL2ID.items()}

# Model & Training 

#------ Different models for try to to avoid runtime error due to high RAM usage (ordered highest to lowest) ---------
#MODEL_NAME = "microsoft/deberta-v3-base"
#MODEL_NAME = "microsoft/deberta-v3-small"     
MODEL_NAME = "distilroberta-base"
# MODEL_NAME = "distilbert-base-uncased"

MAX_LEN = 64
LR = 3e-5
EPOCHS = 2 # Prevoiusly 10
TRAIN_BS = 1 # Prevoiusly 16
EVAL_BS = 2 # Prevoiusly 32
SEED = 42

FABSA_TRAIN = DATA_DIR / "train.csv"
FABSA_DEV = DATA_DIR / "dev.csv"
FABSA_TEST = DATA_DIR/ "test.csv"

FABSA_TRAIN_PAIRS = DATA_DIR / "train_pairs.csv"
FABSA_DEV_PAIRS = DATA_DIR / "dev_pairs.csv"
FABSA_TEST_PAIRS = DATA_DIR/ "test_pairs.csv"