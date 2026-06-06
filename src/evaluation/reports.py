import json
import os

import pandas as pd
import yaml

from sklearn.metrics import confusion_matrix

from src.evaluation.metrics import compute_metrics_per_group
from src.evaluation.metrics import compute_metrics_per_task
from src.evaluation.plots import save_confusion_matrix_plot
from src.evaluation.plots import save_roc_curve_plot


def aggregate_feature_importance(feature_importance_df):
    """
    Aggregate feature coefficients across LOSO folds.

    The per-fold coefficients show what happened in each held-out subject split.
    This summary shows which features were important more consistently across folds.
    """
    if len(feature_importance_df) == 0:
        return pd.DataFrame()

    summary_df = feature_importance_df.groupby("feature").agg(
        mean_coefficient=("coefficient", "mean"),
        std_coefficient=("coefficient", "std"),
        mean_abs_coefficient=("abs_coefficient", "mean"),
        max_abs_coefficient=("abs_coefficient", "max"),
        n_folds=("fold_id", "nunique"),
    ).reset_index()

    summary_df = summary_df.sort_values("mean_abs_coefficient", ascending=False)

    return summary_df


def summarize_results(results_df, predictions_df, config):
    """Aggregate fold-level results into one experiment summary."""
    target_column = config["data"]["target_column"]

    y_true = predictions_df[target_column]
    y_pred = predictions_df["prediction"]

    summary = {
        "experiment": config["experiment"]["name"],
        "mean_balanced_accuracy": results_df["balanced_accuracy"].mean(),
        "std_balanced_accuracy": results_df["balanced_accuracy"].std(),
        "mean_f1": results_df["f1"].mean(),
        "mean_precision": results_df["precision"].mean(),
        "mean_recall": results_df["recall"].mean(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if "roc_auc" in results_df.columns:
        summary["mean_roc_auc"] = results_df["roc_auc"].dropna().mean()

    return summary


def compute_artifact_prediction_correlation(predictions_df, config):
    """
    Compute correlation between artifact score and model outputs.

    High correlation may suggest that the model is partially driven by artifacts
    instead of task-related EEG patterns.
    """
    artifact_column = config["data"].get("artifact_score_column", "artifact_score")

    if artifact_column not in predictions_df.columns:
        return pd.DataFrame()

    rows = []
    prediction_columns = ["prediction", "prediction_score"]

    for prediction_column in prediction_columns:
        if prediction_column not in predictions_df.columns:
            continue

        values = predictions_df[[artifact_column, prediction_column]].copy()
        values[artifact_column] = pd.to_numeric(values[artifact_column], errors="coerce")
        values[prediction_column] = pd.to_numeric(values[prediction_column], errors="coerce")
        values = values.dropna()

        if len(values) > 1:
            correlation = values[artifact_column].corr(values[prediction_column])
        else:
            correlation = None

        rows.append({
            "artifact_column": artifact_column,
            "prediction_column": prediction_column,
            "correlation": correlation,
            "n_windows": len(values),
        })

    return pd.DataFrame(rows)


def save_results(results_df, predictions_df, feature_importance_df, summary, config):
    """Save metrics, predictions, plots, and report files for the experiment."""
    experiment_name = config["experiment"]["name"]
    output_dir = os.path.join("reports", "experiments", experiment_name)

    os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(os.path.join(output_dir, "metrics_per_subject.csv"), index=False)
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    if len(feature_importance_df) > 0:
        feature_importance_df.to_csv(
            os.path.join(output_dir, "feature_importance.csv"),
            index=False
        )

        feature_importance_summary_df = aggregate_feature_importance(feature_importance_df)
        feature_importance_summary_df.to_csv(
            os.path.join(output_dir, "feature_importance_summary.csv"),
            index=False
        )

    target_column = config["data"]["target_column"]

    save_confusion_matrix_plot(
        predictions_df[target_column],
        predictions_df["prediction"],
        os.path.join(output_dir, "confusion_matrix.png"),
    )

    if "prediction_score" in predictions_df.columns:
        save_roc_curve_plot(
            predictions_df[target_column],
            predictions_df["prediction_score"],
            os.path.join(output_dir, "roc_curve.png"),
        )

    task_metrics = compute_metrics_per_task(
        predictions_df,
        target_column
    )

    group_metrics = compute_metrics_per_group(
        predictions_df,
        target_column
    )

    pd.DataFrame(task_metrics).to_csv(
        os.path.join(output_dir, "metrics_per_task.csv"),
        index=False
    )

    pd.DataFrame(group_metrics).to_csv(
        os.path.join(output_dir, "metrics_per_group.csv"),
        index=False
    )

    artifact_correlation_df = compute_artifact_prediction_correlation(predictions_df, config)
    if len(artifact_correlation_df) > 0:
        artifact_correlation_df.to_csv(
            os.path.join(output_dir, "artifact_prediction_correlation.csv"),
            index=False
        )

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with open(os.path.join(output_dir, "config_used.yaml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return output_dir
