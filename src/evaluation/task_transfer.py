import sys

import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.evaluation.reports import save_results
from src.evaluation.reports import summarize_results
from src.models.model_pipeline import build_model_pipeline
from src.training.train import BandpowerTrainingPipeline
from src.training.train import extract_feature_importance
from src.training.train import get_prediction_scores
from src.training.train import load_config


class TaskTransferExperiment:
    """Train on rest vs one task and test on rest vs another task."""

    def __init__(self, config, train_task, test_task):
        self.config = config
        self.train_task = train_task
        self.test_task = test_task
        self.target_column = config["data"]["target_column"]
        self.feature_columns = []

    def load_data(self):
        training_pipeline = BandpowerTrainingPipeline(self.config)
        df = training_pipeline.load_feature_table()
        training_pipeline.validate_feature_table(df)

        return df

    def select_transfer_data(self, df):
        train_df = df[df["task"].isin(["R", self.train_task])].copy()
        test_df = df[df["task"].isin(["R", self.test_task])].copy()

        return train_df, test_df

    def validate_transfer_data(self, train_df, test_df):
        if len(train_df) == 0:
            raise ValueError("Task-transfer train set is empty.")

        if len(test_df) == 0:
            raise ValueError("Task-transfer test set is empty.")

        if train_df[self.target_column].nunique() < 2:
            raise ValueError("Task-transfer train set needs both rest and task labels.")

        if test_df[self.target_column].nunique() < 2:
            raise ValueError("Task-transfer test set needs both rest and task labels.")

    def make_transfer_config(self):
        transfer_config = self.config.copy()
        transfer_config["experiment"] = self.config.get("experiment", {}).copy()
        transfer_config["experiment"]["name"] = (
            self.config["experiment"]["name"]
            + "_train_"
            + self.train_task.lower()
            + "_test_"
            + self.test_task.lower()
        )

        return transfer_config

    def prepare_data(self, df):
        training_pipeline = BandpowerTrainingPipeline(self.config)
        feature_columns = training_pipeline.get_feature_columns(df)
        df = training_pipeline.apply_subject_relative_to_rest(df, feature_columns)
        self.feature_columns = feature_columns

        return df

    def run_model(self, train_df, test_df):
        X_train = train_df[self.feature_columns]
        y_train = train_df[self.target_column]
        X_test = test_df[self.feature_columns]
        y_test = test_df[self.target_column]

        model = build_model_pipeline(self.config)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_score = get_prediction_scores(model, X_test)

        return model, y_test, y_pred, y_score

    def build_outputs(self, model, train_df, test_df, y_test, y_pred, y_score):
        metrics = compute_metrics(y_test, y_pred, y_score)

        result = {
            "fold_id": 0,
            "test_subject": "task_transfer",
            "train_task": self.train_task,
            "test_task": self.test_task,
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
            self.feature_columns,
            0,
            "task_transfer"
        )

        if feature_importance_df is None:
            feature_importance_df = pd.DataFrame()

        results_df = pd.DataFrame([result])

        return results_df, predictions_df, feature_importance_df

    def run(self):
        df = self.load_data()
        df = self.prepare_data(df)

        train_df, test_df = self.select_transfer_data(df)
        self.validate_transfer_data(train_df, test_df)

        model, y_test, y_pred, y_score = self.run_model(train_df, test_df)
        results_df, predictions_df, feature_importance_df = self.build_outputs(
            model,
            train_df,
            test_df,
            y_test,
            y_pred,
            y_score
        )

        transfer_config = self.make_transfer_config()
        summary = summarize_results(results_df, predictions_df, transfer_config)
        save_results(results_df, predictions_df, feature_importance_df, summary, transfer_config)

        return summary, transfer_config


def select_transfer_data(df, train_task, test_task):
    train_df = df[df["task"].isin(["R", train_task])].copy()
    test_df = df[df["task"].isin(["R", test_task])].copy()

    return train_df, test_df


def validate_task_transfer_data(train_df, test_df, target_column):
    if len(train_df) == 0:
        raise ValueError("Task-transfer train set is empty.")

    if len(test_df) == 0:
        raise ValueError("Task-transfer test set is empty.")

    if train_df[target_column].nunique() < 2:
        raise ValueError("Task-transfer train set needs both rest and task labels.")

    if test_df[target_column].nunique() < 2:
        raise ValueError("Task-transfer test set needs both rest and task labels.")


def run_task_transfer(df, config, train_task, test_task):
    experiment = TaskTransferExperiment(config, train_task, test_task)
    df = experiment.prepare_data(df)
    train_df, test_df = experiment.select_transfer_data(df)
    experiment.validate_transfer_data(train_df, test_df)

    model, y_test, y_pred, y_score = experiment.run_model(train_df, test_df)
    return experiment.build_outputs(model, train_df, test_df, y_test, y_pred, y_score)


def make_task_transfer_config(config, train_task, test_task):
    experiment = TaskTransferExperiment(config, train_task, test_task)
    return experiment.make_transfer_config()


def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print("python -m src.evaluation.task_transfer configs/model_bandpower_logreg.yaml TASK1 TASK2")
        return

    config_path = sys.argv[1]
    train_task = sys.argv[2]
    test_task = sys.argv[3]

    config = load_config(config_path)
    experiment = TaskTransferExperiment(config, train_task, test_task)
    summary, transfer_config = experiment.run()

    print("Task-transfer experiment:", transfer_config["experiment"]["name"])
    print("Mean balanced accuracy:", summary["mean_balanced_accuracy"])
    print("Mean F1:", summary["mean_f1"])


if __name__ == "__main__":
    main()
