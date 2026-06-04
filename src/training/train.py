"""Train EEG focus classifiers with Leave-One-Subject-Out validation."""

import json
import os
import sys

import pandas as pd
import yaml

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

from src.models.model_pipeline import build_model_pipeline


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_feature_table(config):
    input_file = config["data"]["input_file"]
    df = pd.read_csv(input_file)

    # Optioanlly remove rejected windows
    rejected_column = config["data"].get("rejected_column")
    if rejected_column in df.columns:
        df = df[df[rejected_column] == False].copy()

    return df


def get_feature_columns(df, config):
    """Select model input columns by excluding metadata columns."""
    metadata_columns = config["data"].get("metadata_columns", [])
    feature_columns = []

    for column in df.columns:
        if column not in metadata_columns:
            feature_columns.append(column)

    return feature_columns


def get_prediction_scores(model, X_test):
    """Return probability or decision scores for the positive class."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)

    return None


def run_loso_training(df, config):
    """
    Train and evaluate the configured model using LOSO validation.
    Cross validation / Leave-One-Subject-Out training is used in this project because EEG data is dependent on person
    This is important because in our case, samples from the same person are similar.
    """
    target_column = config["data"]["target_column"]
    group_column = config["validation"]["group_column"]
    feature_columns = get_feature_columns(df, config)

    X = df[feature_columns]
    y = df[target_column]
    groups = df[group_column]

    logo = LeaveOneGroupOut()
    fold_results = []
    all_predictions = []

    for fold_id, split in enumerate(logo.split(X, y, groups)):
        train_index, test_index = split

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        train_subjects = set(groups.iloc[train_index])
        test_subjects = set(groups.iloc[test_index])

        if len(train_subjects.intersection(test_subjects)) > 0:
            raise ValueError("This person is in both train i test")

        test_subject = list(test_subjects)[0]

        model = build_model_pipeline(config)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_score = get_prediction_scores(model, X_test)

        result = {
            "fold_id": fold_id,
            "test_subject": test_subject,
            "n_train": len(train_index),
            "n_test": len(test_index),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
        }

        if y_score is not None and len(set(y_test)) == 2:
            result["roc_auc"] = roc_auc_score(y_test, y_score)
        else:
            result["roc_auc"] = None

        fold_results.append(result)

        fold_predictions = df.iloc[test_index].copy()
        fold_predictions["fold_id"] = fold_id
        fold_predictions["prediction"] = y_pred

        if y_score is not None:
            fold_predictions["prediction_score"] = y_score

        all_predictions.append(fold_predictions)

    results_df = pd.DataFrame(fold_results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    return results_df, predictions_df


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


def save_results(results_df, predictions_df, summary, config):
    """Save metrics and predictions for the finished experiment."""
    experiment_name = config["experiment"]["name"]
    output_dir = os.path.join("reports", "experiments", experiment_name)

    os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(os.path.join(output_dir, "metrics_per_subject.csv"), index=False)
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def main():
    """Run training from the CLI"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("python -m src.training.train configs/model_bandpower_logreg.yaml")
        return

    config_path = sys.argv[1]

    config = load_config(config_path)
    df = load_feature_table(config)

    results_df, predictions_df = run_loso_training(df, config)
    summary = summarize_results(results_df, predictions_df, config)
    save_results(results_df, predictions_df, summary, config)

    print("Mean balanced accuracy:", summary["mean_balanced_accuracy"])
    print("Mean F1:", summary["mean_f1"])


if __name__ == "__main__":
    main()
