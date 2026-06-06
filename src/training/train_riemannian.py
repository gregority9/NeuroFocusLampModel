import sys

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import LeaveOneGroupOut

from src.evaluation.metrics import compute_metrics
from src.evaluation.reports import save_results
from src.evaluation.reports import summarize_results
from src.models.riemannian import build_riemannian_logreg_pipeline
from src.training.train import get_prediction_scores
from src.training.train import save_model_artifacts


class RiemannianTrainingPipeline:
    """Training pipeline for covariance matrices."""

    def __init__(self, config):
        self.config = config

    def load_data(self):
        covariance_file = self.config["data"]["covariance_file"]
        metadata_file = self.config["data"]["metadata_file"]

        X = np.load(covariance_file)
        metadata_df = pd.read_csv(metadata_file)

        rejected_column = self.config["data"].get("rejected_column")
        if rejected_column in metadata_df.columns:
            if metadata_df[rejected_column].dtype == "object":
                rejected = metadata_df[rejected_column].astype(str).str.lower()
                keep_mask = rejected != "true"
            else:
                keep_mask = metadata_df[rejected_column] == False

            metadata_df = metadata_df[keep_mask].copy()
            X = X[keep_mask.to_numpy()]

        return X, metadata_df

    def validate_data(self, X, metadata_df):
        target_column = self.config["data"]["target_column"]
        group_column = self.config["validation"]["group_column"]

        required_columns = self.config["data"].get("required_columns", []).copy()

        if target_column not in required_columns:
            required_columns.append(target_column)

        if group_column not in required_columns:
            required_columns.append(group_column)

        missing_columns = []
        for column in required_columns:
            if column not in metadata_df.columns:
                missing_columns.append(column)

        if len(missing_columns) > 0:
            raise ValueError("Missing columns in metadata table: " + str(missing_columns))

        if len(metadata_df) == 0:
            raise ValueError("Metadata table is empty after filtering rejected windows.")

        if len(X) != len(metadata_df):
            raise ValueError("Number of covariance matrices does not match metadata rows.")

        if len(X.shape) != 3:
            raise ValueError("Covariance matrices should have shape n_windows x n_channels x n_channels.")

        if X.shape[1] != X.shape[2]:
            raise ValueError("Covariance matrices must be square.")

        if metadata_df[group_column].nunique() < 2:
            raise ValueError("LOSO validation needs at least two subjects.")

        if metadata_df[target_column].nunique() < 2:
            raise ValueError("Classification needs at least two labels.")

    def run_loso_training(self, X, metadata_df):
        target_column = self.config["data"]["target_column"]
        group_column = self.config["validation"]["group_column"]

        y = metadata_df[target_column]
        groups = metadata_df[group_column]

        logo = LeaveOneGroupOut()
        fold_results = []
        all_predictions = []

        for fold_id, split in enumerate(logo.split(X, y, groups)):
            train_index, test_index = split

            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            train_subjects = set(groups.iloc[train_index])
            test_subjects = set(groups.iloc[test_index])

            if len(train_subjects.intersection(test_subjects)) > 0:
                raise ValueError("This person is in both train and test.")

            test_subject = list(test_subjects)[0]

            model = build_riemannian_logreg_pipeline(self.config)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_score = get_prediction_scores(model, X_test)

            result = {
                "fold_id": fold_id,
                "test_subject": test_subject,
                "n_train": len(train_index),
                "n_test": len(test_index),
            }

            metrics = compute_metrics(y_test, y_pred, y_score)
            result.update(metrics)

            fold_results.append(result)

            fold_predictions = metadata_df.iloc[test_index].copy()
            fold_predictions["fold_id"] = fold_id
            fold_predictions["prediction"] = y_pred

            if y_score is not None:
                fold_predictions["prediction_score"] = y_score

            all_predictions.append(fold_predictions)

        results_df = pd.DataFrame(fold_results)
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        feature_importance_df = pd.DataFrame()

        return results_df, predictions_df, feature_importance_df

    def train_final_model(self, X, metadata_df):
        target_column = self.config["data"]["target_column"]
        y = metadata_df[target_column]

        model = build_riemannian_logreg_pipeline(self.config)
        model.fit(X, y)

        return model

    def run(self):
        X, metadata_df = self.load_data()
        self.validate_data(X, metadata_df)

        results_df, predictions_df, feature_importance_df = self.run_loso_training(
            X,
            metadata_df
        )

        summary = summarize_results(results_df, predictions_df, self.config)
        output_dir = save_results(results_df, predictions_df, feature_importance_df, summary, self.config)

        final_model = self.train_final_model(X, metadata_df)
        save_model_artifacts(final_model, [], self.config, output_dir, input_format="covariance_matrices")

        return summary


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_riemannian_data(config):
    pipeline = RiemannianTrainingPipeline(config)
    return pipeline.load_data()


def validate_riemannian_data(X, metadata_df, config):
    pipeline = RiemannianTrainingPipeline(config)
    pipeline.validate_data(X, metadata_df)


def run_riemannian_loso_training(X, metadata_df, config):
    pipeline = RiemannianTrainingPipeline(config)
    return pipeline.run_loso_training(X, metadata_df)


def train_final_riemannian_model(X, metadata_df, config):
    pipeline = RiemannianTrainingPipeline(config)
    return pipeline.train_final_model(X, metadata_df)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python -m src.training.train_riemannian configs/model_riemannian_logreg.yaml")
        return

    config_path = sys.argv[1]
    config = load_config(config_path)

    pipeline = RiemannianTrainingPipeline(config)
    summary = pipeline.run()

    print("Mean balanced accuracy:", summary["mean_balanced_accuracy"])
    print("Mean F1:", summary["mean_f1"])


if __name__ == "__main__":
    main()
