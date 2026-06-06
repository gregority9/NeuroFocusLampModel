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
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
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

def compute_metrics_by_column(predictions_df, column, target_column):
    """ Compute metrics separately for each value in a selected column. """
    rows = []
    for value in predictions_df[column].unique():
        subset = predictions_df[predictions_df[column] == value]

        y_true = subset[target_column]
        y_pred = subset["prediction"]

        if "prediction_score" in subset.columns:
            y_score = subset["prediction_score"]
        else:
            y_score = None

        metrics = compute_metrics(y_true, y_pred, y_score)
        metrics[column] = value
        metrics["n_windows"] = len(subset)

        rows.append(metrics)
    return rows

def compute_metrics_per_task(predictions_df, target_column):
    """ Task1 / Task2 / Rest """
    rows = []

    for task in predictions_df["task"].unique():
        subset = predictions_df[predictions_df["task"] == task]

        # Usually each task has only one true label:
        # R has label 0, while TASK1 and TASK2 have label 1.
        labels = set(subset[target_column].unique())

        # Row is one output record for one task.
        # - which task is summarized,
        # - how many windows belong to this task,
        # - what labels are present in this task subset,
        # - what the average predicted class is.
        row = {
            "task": task,
            "n_windows": len(subset),
            "true_labels": str(sorted(labels)),
            "mean_prediction": subset["prediction"].mean(),
        }

        # If the subset contains only true rest windows,
        # then the correct behavior is predicting class 0.
        if labels == {0}:
            row["correct_detection_rate"] = (subset["prediction"] == 0).mean()
            row["expected_class"] = 0

        # If the subset contains only true task windows,
        # then the correct behavior is predicting class 1.
        elif labels == {1}:
            row["correct_detection_rate"] = (subset["prediction"] == 1).mean()
            row["expected_class"] = 1

        # This branch is for unusual cases where one task subset contains both labels
        else:
            y_true = subset[target_column]
            y_pred = subset["prediction"]

            if "prediction_score" in subset.columns:
                y_score = subset["prediction_score"]
            else:
                y_score = None

            metrics = compute_metrics(y_true, y_pred, y_score)
            row.update(metrics)
            row["expected_class"] = "mixed"

        rows.append(row)

    return rows



def compute_metrics_per_group(predictions_df, target_column):
    """ Group: ADHD / Control """
    rows = []

    for group in predictions_df["group"].unique():
        subset = predictions_df[predictions_df["group"] == group]

        y_true = subset[target_column]
        y_pred = subset["prediction"]

        if "prediction_score" in subset.columns:
            y_score = subset["prediction_score"]
        else:
            y_score = None

        metrics = compute_metrics(y_true, y_pred, y_score)
        metrics["group"] = group
        metrics["n_windows"] = len(subset)
        rows.append(metrics)
    return rows
