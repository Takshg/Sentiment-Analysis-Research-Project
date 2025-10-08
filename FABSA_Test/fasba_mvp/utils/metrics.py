import numpy as np
import evaluate
_f1 = evaluate.load("f1")
_acc = evaluate.load("accuracy")
_prec = evaluate.load("precision")
_rec = evaluate.load("recall")

def hf_classification_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis = -1)
    return{
        "accuracy": _acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1_micro": _f1.compute(predictions=preds, references=labels, average = "micro")["f1"], 
        "f1_macro": _f1.compute(predictions=preds, references=labels, average = "macro")["f1"],
        "precision_macro": _prec.compute(predictions=preds, references=labels, average="macro")["precision"],
        "recall_macro": _rec.compute(predictions=preds, references=labels, average="macro")["recall"],
    }
