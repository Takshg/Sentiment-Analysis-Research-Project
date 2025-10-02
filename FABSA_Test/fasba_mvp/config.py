from pathlib import Path 

DATA_DIR = Path("data")
FABSA_DIR = DATA_DIR / "fabsa"
REVIEWS_DIR = DATA_DIR / "Raw_Reviews"
OUT_DIR = Path("outputs")
MODEL_DIR = OUT_DIR / "Models" / "deberta_pair"
PREDS_DIR = OUT_DIR / "Preds"
REPORTS_DIR = OUT_DIR / "Reports"
for x in [FABSA_DIR, REVIEWS_DIR, MODEL_DIR, PREDS_DIR, REPORTS_DIR]: 
    x.mkdir(parents= True, exist_ok=True)

# Task Space
ASPECTS = [
    'general-satisfaction', 'product-quality', 'price-value-for-money', 'delivery', 
    'returns-refunds', 'customer-service', 'attitude-of-staff', 'account-access', 
    'app-website', 'communications', 'competitor', 'phone'
]

LABELS = ['absent', 'positive', 'negative', 'neutral']
LABEL2ID = {l:i for i,l in enumerate(LABELS)}
ID2LABEL = {i:l for l,i in LABEL2ID.items()}

# Model & Training 
MODEL_NAME = 'microsoft/deberta-v3-base'
MAX_LEN = 100
LR = 3e-5
EPOCHS = 10
TRAIN_BS = 16
EVAL_BS = 32
SEED = 42

FABSA_TRAIN = FABSA_DIR / "train.csv"
FABSA_DEV = FABSA_DIR / "dev.csv"
FABSA_TEST = FABSA_DIR/ "test.csv"

FABSA_TRAIN_PAIRS = FABSA_DIR / "train_pairs.csv"
FABSA_DEV_PAIRS = FABSA_DIR / "dev_pairs.csv"
FABSA_TEST_PAIRS = FABSA_DIR/ "test_pairs.csv"