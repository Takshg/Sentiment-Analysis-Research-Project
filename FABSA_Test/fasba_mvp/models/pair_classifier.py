from transformers import AutoTokenizer, AutoModelForSequenceClassification
from FABSA_Test.fasba_mvp.config import MODEL_NAME, LABELS

def load_model_and_tokenizer(num_labels = len(LABELS)):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels = num_labels, 
        id2label = {i:l for i,l in enumerate(LABELS)},
        label2id = {l:i for i,l in enumerate(LABELS)}
    )
    return model, tok
