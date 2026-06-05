import sys

import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.evaluation.reports import save_results
from src.evaluation.reports import summarize_results
from src.models.model_pipeline import build_model_pipeline
from src.training.train import extract_feature_importance
from src.training.train import get_feature_columns
from src.training.train import get_prediction_scores
from src.training.train import load_config
from src.training.train import load_feature_table
from src.training.train import validate_feature_table


def select_transfer_data(df, train_task, test_task):
    """
    Select rows for task-transfer validation.

    Train set uses rest and one cognitive task.
    Test set uses rest and the other cognitive task.
    """
    train_df = df[df["task"].isin(["R", train_task])].copy()
    test_df = df[df["task"].isin(["R", test_task])].copy()

    return train_df, test_df


def validate_task_transfer_data(train_df, test_df, target_column):
    """Check if train and test subsets are usable for binary classification."""
    if len(train_df) == 0:
        raise ValueError("Task-transfer train set is empty.")

    if len(test_df) == 0:
        raise ValueError("Task-transfer test set is empty.")

    if train_df[target_column].nunique() < 2:
        raise ValueError("Task-transfer train set needs both rest and task labels.")

    if test_df[target_column].nunique() < 2:
        raise ValueError("Task-transfer test set needs both rest and task labels.")


def run_task_transfer(df, config, train_task, test_task):
    """
    Train on rest vs one task and test on rest vs another task.

    This checks whether the model learns general cognitive engagement
    or only task-specific patterns.
    """
    target_column = config["data"]["target_column"]
    feature_columns = get_feature_columns(df, config)

    train_df, test_df = select_transfer_data(df, train_task, test_task)
    validate_task_transfer_data(train_df, test_df, target_column)

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    model = build_model_pipeline(config)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = get_prediction_scores(model, X_test)

    metrics = compute_metrics(y_test, y_pred, y_score)

    result = {
        "fold_id": 0,
        "test_subject": "task_transfer",
        "train_task": train_task,
        "test_task": test_task,
        "n_train": len(train_df),
        "n_test": len(test_df),
    }
    result.update(metrics)

    predictions_df = test_df.copy()
    predictions_df["fold_id"] = 0
    predictions_df["prediction"] = y_pred

    if y_score is not None:
        predictions_df["prediction_score"] = y_score

    feature_importance_df = extract_feature_importance(
        model,
        feature_columns,
        0,
        "task_transfer"
    )

    if feature_importance_df is None:
        feature_importance_df = pd.DataFrame()

    results_df = pd.DataFrame([result])

    return results_df, predictions_df, feature_importance_df


def make_task_transfer_config(config, train_task, test_task):
    """Create a copy of config with an experiment name for task transfer."""
    transfer_config = config.copy()
    transfer_config["experiment"] = config.get("experiment", {}).copy()
    transfer_config["experiment"]["name"] = (
        config["experiment"]["name"]
        + "_train_"
        + train_task.lower()
        + "_test_"
        + test_task.lower()
    )

    return transfer_config


def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print("python -m src.evaluation.task_transfer configs/model_bandpower_logreg.yaml TASK1 TASK2")
        return

    config_path = sys.argv[1]
    train_task = sys.argv[2]
    test_task = sys.argv[3]

    config = load_config(config_path)
    df = load_feature_table(config)
    validate_feature_table(df, config)

    transfer_config = make_task_transfer_config(config, train_task, test_task)

    results_df, predictions_df, feature_importance_df = run_task_transfer(
        df,
        transfer_config,
        train_task,
        test_task
    )

    summary = summarize_results(results_df, predictions_df, transfer_config)
    save_results(results_df, predictions_df, feature_importance_df, summary, transfer_config)

    print("Task-transfer experiment:", transfer_config["experiment"]["name"])
    print("Mean balanced accuracy:", summary["mean_balanced_accuracy"])
    print("Mean F1:", summary["mean_f1"])


if __name__ == "__main__":
    main()
