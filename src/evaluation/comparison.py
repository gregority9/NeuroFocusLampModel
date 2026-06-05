import json
import os
import pandas as pd

def build_experiment_comparison(experiments_dir):
    rows = []

    for experiment_name in os.listdir(experiments_dir):
        experiment_dir = os.path.join(experiments_dir, experiment_name)
        metrics_path = os.path.join(experiment_dir, "metrics.json")

        if not os.path.isdir(experiment_dir):
            continue

        if not os.path.exists(metrics_path):
            continue

        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        row = {
            "experiment": metrics.get("experiment", experiment_name),
            "mean_balanced_accuracy": metrics.get("mean_balanced_accuracy"),
            "std_balanced_accuracy": metrics.get("std_balanced_accuracy"),
            "mean_f1": metrics.get("mean_f1"),
            "mean_precision": metrics.get("mean_precision"),
            "mean_recall": metrics.get("mean_recall"),
            "mean_roc_auc": metrics.get("mean_roc_auc"),
        }

        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    if len(comparison_df) > 0:
        comparison_df = comparison_df.sort_values(
            "mean_balanced_accuracy",
            ascending=False
        )

    return comparison_df


def save_experiment_comparison(experiments_dir):
    comparison_df = build_experiment_comparison(experiments_dir)

    output_path = os.path.join(experiments_dir, "comparison.csv")
    comparison_df.to_csv(output_path, index=False)

    return comparison_df
