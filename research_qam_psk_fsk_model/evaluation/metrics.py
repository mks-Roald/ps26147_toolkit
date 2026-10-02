"""Seven-class evaluation helpers for future TRAIN/VALIDATION reports."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from ..labels import CLASS_NAMES


def summarize_predictions(y_true: list[int], y_pred: list[int]) -> dict:
    report = classification_report(y_true,y_pred,labels=list(range(7)),target_names=CLASS_NAMES,
                                   output_dict=True,zero_division=0)
    matrix=confusion_matrix(y_true,y_pred,labels=list(range(7)))
    return {"accuracy":float(accuracy_score(y_true,y_pred)),
            "macro_f1":float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
            "macro_precision":float(precision_score(y_true,y_pred,average="macro",zero_division=0)),
            "macro_recall":float(recall_score(y_true,y_pred,average="macro",zero_division=0)),
            "per_class":report,"confusion_matrix":matrix.tolist(),
            "cross_confusions":{
                "16QAM_to_64QAM":int(matrix[3,4]),"64QAM_to_16QAM":int(matrix[4,3]),
                "2FSK_to_4FSK":int(matrix[5,6]),"4FSK_to_2FSK":int(matrix[6,5])}}
