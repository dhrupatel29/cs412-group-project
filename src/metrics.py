"""Metrics and validation-selected threshold."""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, roc_auc_score)


def best_f1_threshold(y_val, p_val):
    """Threshold that maximizes F1 on clean validation data. Kept fixed for all test conditions."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


def evaluate(y, p, threshold):
    return {
        "auroc": roc_auc_score(y, p),
        "ap": average_precision_score(y, p),  # AP, not trapezoidal AUPRC
        "f1": f1_score(y, (p >= threshold).astype(int)),
        "brier": brier_score_loss(y, p),
    }
