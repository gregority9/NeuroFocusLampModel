from sklearn.metrics import (balanced_accuracy_score,
                             confusion_matrix,
                             f1_score,
                             precision_score,
                             recall_score,
                             roc_auc_score)


def compute_metrics(y_true, y_pred, y_score=None):
    """
    Compute basic metrics for binary classification.
    :param y_true:contains true labels.
    :param y_pred: contains predicted labels.
    :param y_score: contains probabilities or decision scores for class 1.
    """

    metrics = {
        "balanced_accurracy": balanced_accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }

    if y_score is not None and len(set(y_true)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_true, y_score)
    else:
        metrics['roc_auc'] = None

    return metrics


