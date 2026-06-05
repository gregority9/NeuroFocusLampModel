import os
import sys

import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.evaluation.reports import save_results
from src.evaluation.reports import summarize_results
from src.training.train import load_config


def load_predictions(predictions_path):
    """Load saved experiment predictions."""
    return pd.read_csv(predictions_path)


def rebuild_results_from_predictions(predictions_df, config):
    """ Recompute per-fold metrics from an existing predictions table. """
    target_column = config["data"]["target_column"]

    if "fold_id" not in predictions_df.columns:
        predictions_df["fold_id"] = 0

    rows = []

    for fold_id in predictions_df["fold_id"].unique():
        fold_df = predictions_df[predictions_df["fold_id"] == fold_id]

        y_true = fold_df[target_column]
        y_pred = fold_df["prediction"]

        if "prediction_score" in fold_df.columns:
            y_score = fold_df["prediction_score"]
        else:
            y_score = None

        result = {
            "fold_id": fold_id,
            "n_test": len(fold_df),
        }

        if "test_subject" in fold_df.columns:
            result["test_subject"] = fold_df["test_subject"].iloc[0]
        elif "subject_id" in fold_df.columns and fold_df["subject_id"].nunique() == 1:
            result["test_subject"] = fold_df["subject_id"].iloc[0]
        else:
            result["test_subject"] = "unknown"

        metrics = compute_metrics(y_true, y_pred, y_score)
        result.update(metrics)

        rows.append(result)

    return pd.DataFrame(rows)


def load_feature_importance(experiment_dir):
    """Load feature importance if it exists for this experiment."""
    feature_importance_path = os.path.join(experiment_dir, "feature_importance.csv")

    if os.path.exists(feature_importance_path):
        return pd.read_csv(feature_importance_path)

    return pd.DataFrame()


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("python -m src.evaluation.evaluate configs/model_bandpower_logreg.yaml reports/experiments/bandpower_logreg")
        return

    config_path = sys.argv[1]
    experiment_dir = sys.argv[2]
    predictions_path = os.path.join(experiment_dir, "predictions.csv")

    config = load_config(config_path)
    config["experiment"]["name"] = os.path.basename(os.path.normpath(experiment_dir))

    predictions_df = load_predictions(predictions_path)
    results_df = rebuild_results_from_predictions(predictions_df, config)
    feature_importance_df = load_feature_importance(experiment_dir)

    summary = summarize_results(results_df, predictions_df, config)
    save_results(results_df, predictions_df, feature_importance_df, summary, config)

    print("Recomputed reports for:", config["experiment"]["name"])
    print("Mean balanced accuracy:", summary["mean_balanced_accuracy"])
    print("Mean F1:", summary["mean_f1"])


if __name__ == "__main__":
    main()
