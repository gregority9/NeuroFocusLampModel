import os
import sys

import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.evaluation.reports import save_results
from src.evaluation.reports import summarize_results
from src.training.train import load_config


class ExistingExperimentEvaluator:
    """Recompute reports from an existing predictions.csv file."""

    def __init__(self, config, experiment_dir):
        self.config = config
        self.experiment_dir = experiment_dir
        self.predictions_path = os.path.join(experiment_dir, "predictions.csv")

    def load_predictions(self):
        return pd.read_csv(self.predictions_path)

    def rebuild_results_from_predictions(self, predictions_df):
        target_column = self.config["data"]["target_column"]

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

            result = {}
            result["fold_id"] = fold_id
            result["n_test"] = len(fold_df)

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

    def load_feature_importance(self):
        feature_importance_path = os.path.join(self.experiment_dir, "feature_importance.csv")

        if os.path.exists(feature_importance_path):
            return pd.read_csv(feature_importance_path)

        return pd.DataFrame()

    def run(self):
        predictions_df = self.load_predictions()
        results_df = self.rebuild_results_from_predictions(predictions_df)
        feature_importance_df = self.load_feature_importance()

        summary = summarize_results(results_df, predictions_df, self.config)
        save_results(results_df, predictions_df, feature_importance_df, summary, self.config)

        return summary


def load_predictions(predictions_path):
    return pd.read_csv(predictions_path)


def rebuild_results_from_predictions(predictions_df, config):
    evaluator = ExistingExperimentEvaluator(config, "")
    return evaluator.rebuild_results_from_predictions(predictions_df)


def load_feature_importance(experiment_dir):
    evaluator = ExistingExperimentEvaluator({}, experiment_dir)
    return evaluator.load_feature_importance()


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("python -m src.evaluation.evaluate configs/model_bandpower_logreg.yaml reports/experiments/bandpower_logreg")
        return

    config_path = sys.argv[1]
    experiment_dir = sys.argv[2]

    config = load_config(config_path)
    config["experiment"]["name"] = os.path.basename(os.path.normpath(experiment_dir))

    evaluator = ExistingExperimentEvaluator(config, experiment_dir)
    summary = evaluator.run()

    print("Recomputed reports for:", config["experiment"]["name"])
    print("Mean balanced accuracy:", summary["mean_balanced_accuracy"])
    print("Mean F1:", summary["mean_f1"])


if __name__ == "__main__":
    main()
