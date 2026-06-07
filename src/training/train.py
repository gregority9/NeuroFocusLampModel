import json
import os
import sys

import joblib
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from src.evaluation.metrics import compute_metrics
from src.evaluation.reports import save_results
from src.evaluation.reports import summarize_results
from src.models.model_pipeline import build_model_pipeline
from src.training.config_loader import load_config as load_yaml_config
from src.training.feature_selection import FeatureGroupSelector


class BandpowerTrainingPipeline:
    """Training pipeline for bandpower feature tables."""

    def __init__(self, config):
        self.config = config
        self.feature_columns = []

    def load_feature_table(self):
        input_file = self.config["data"]["input_file"]
        df = pd.read_csv(input_file)
        df = self.filter_rejected_windows(df)

        return df

    def should_reject_rejected_windows(self):
        artifacts_config = self.config.get("artifacts", {})

        return artifacts_config.get("reject_rejected_windows", True)

    def filter_rejected_windows(self, df):
        """Remove rejected windows when this is enabled in the config."""
        if self.should_reject_rejected_windows() == False:
            return df

        rejected_column = self.config["data"].get("rejected_column")

        if rejected_column is None:
            return df

        if rejected_column not in df.columns:
            return df

        if df[rejected_column].dtype == "object":
            rejected = df[rejected_column].astype(str).str.lower()
            return df[~rejected.isin(["true", "1", "yes"])].copy()

        return df[df[rejected_column] == False].copy()

    def get_feature_columns(self, df):
        """Select model input columns by excluding metadata columns."""
        metadata_columns = self.config["data"].get("metadata_columns", [])
        feature_columns = []

        for column in df.columns:
            if column not in metadata_columns:
                feature_columns.append(column)

        feature_columns = self.apply_feature_filters(feature_columns)

        return feature_columns

    def feature_contains_channel(self, feature, channels):
        feature_tokens = feature.replace("-", "_").split("_")

        for channel in channels:
            if channel in feature_tokens:
                return True

        return False

    def feature_contains_band(self, feature, bands):
        feature_name = feature.lower()

        for band in bands:
            if band.lower() in feature_name:
                return True

        return False

    def apply_feature_filters(self, feature_columns):
        """
        Apply artifact-control feature filters from config.

        This is used for experiments such as:
        - no AF3/AF4,
        - no high beta,
        - only occipital channels,
        - only frontal channels.
        """
        feature_config = self.config.get("features", {})
        feature_group_selector = FeatureGroupSelector(
            feature_config.get("feature_groups", [])
        )
        feature_columns = feature_group_selector.select(feature_columns)

        include_channels = feature_config.get("include_channels", [])
        exclude_channels = feature_config.get("exclude_channels", [])
        include_bands = feature_config.get("include_bands", [])
        exclude_bands = feature_config.get("exclude_bands", [])

        filtered_columns = []

        for feature in feature_columns:
            if len(include_channels) > 0 and not self.feature_contains_channel(feature, include_channels):
                continue

            if len(exclude_channels) > 0 and self.feature_contains_channel(feature, exclude_channels):
                continue

            if len(include_bands) > 0 and not self.feature_contains_band(feature, include_bands):
                continue

            if len(exclude_bands) > 0 and self.feature_contains_band(feature, exclude_bands):
                continue

            filtered_columns.append(feature)

        return filtered_columns

    def apply_subject_relative_to_rest(self, df, feature_columns):
        """
        Normalize features relative to each subject's own rest baseline.

        This assumes that a rest recording is available for every subject.
        For a new subject, this corresponds to a calibration recording.
        """
        normalization_config = self.config.get("normalization", {})

        if not normalization_config.get("subject_relative_to_rest", False):
            return df

        subject_column = normalization_config.get(
            "subject_column",
            self.config["validation"].get("group_column", "subject_id")
        )
        task_column = normalization_config.get("task_column", "task")
        rest_task = normalization_config.get("rest_task", "R")
        epsilon = normalization_config.get("epsilon", 1e-8)

        normalized_df = df.copy()
        required_columns = [subject_column, task_column]
        missing_columns = []

        for column in required_columns:
            if column not in normalized_df.columns:
                missing_columns.append(column)

        if len(missing_columns) > 0:
            raise ValueError(
                "Missing columns for subject-relative normalization: "
                + str(missing_columns)
            )

        for subject in normalized_df[subject_column].unique():
            subject_rows = normalized_df[subject_column] == subject
            rest_rows = subject_rows & (normalized_df[task_column] == rest_task)

            if rest_rows.sum() == 0:
                raise ValueError("No rest baseline found for subject: " + str(subject))

            rest_mean = normalized_df.loc[rest_rows, feature_columns].mean()
            rest_std = normalized_df.loc[rest_rows, feature_columns].std()
            rest_std = rest_std.replace(0, epsilon).fillna(epsilon)

            normalized_df.loc[subject_rows, feature_columns] = (
                normalized_df.loc[subject_rows, feature_columns] - rest_mean
            ) / rest_std

        return normalized_df

    def validate_feature_table(self, df):
        """
        Are there any required columns?
        Is there any data left after the is_rejected rejection?
        Are there at least two people for RANDOM?
        Are there at least two classes?
        Are the labels 0 and 1?
        Are there feature columns?
        Do the features have no NaNs?
        """
        required_columns = self.config["data"].get("required_columns", []).copy()
        target_column = self.config["data"]["target_column"]
        group_column = self.config["validation"]["group_column"]

        if target_column not in required_columns:
            required_columns.append(target_column)

        if group_column not in required_columns:
            required_columns.append(group_column)

        missing_columns = []
        for column in required_columns:
            if column not in df.columns:
                missing_columns.append(column)

        if len(missing_columns) > 0:
            raise ValueError("Missing columns in feature table: " + str(missing_columns))

        if len(df) == 0:
            raise ValueError("Feature table is empty after filtering rejected windows.")

        if df[group_column].nunique() < 2:
            raise ValueError("LOSO validation needs at least two subjects.")

        if df[target_column].nunique() < 2:
            raise ValueError("Classification needs at least two labels.")

        allowed_labels = self.config["data"].get("allowed_labels")
        if allowed_labels is not None:
            labels = set(df[target_column].dropna().unique())
            allowed_labels = set(allowed_labels)

            if labels != allowed_labels:
                raise ValueError("Unexpected labels: " + str(labels))

        feature_columns = self.get_feature_columns(df)
        if len(feature_columns) == 0:
            raise ValueError("No feature columns found.")

        if df[feature_columns].isnull().any().any():
            raise ValueError("Feature columns contain NaN values.")

    def get_prediction_scores(self, model, X_test):
        """Return probability or decision scores for the positive class."""
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_test)[:, 1]

        if hasattr(model, "decision_function"):
            return model.decision_function(X_test)

        return None

    def prepare_model_data(self, df):
        """Apply feature selection and model-level normalization."""
        feature_columns = self.get_feature_columns(df)
        df = self.apply_subject_relative_to_rest(df, feature_columns)
        self.feature_columns = feature_columns

        return df, feature_columns

    def extract_feature_importance(self, model, fold_id, test_subject):
        """
       Extract feature coefficients from Logistic Regression.
       Positive coefficient means stronger support for class 1.
       Negative coefficient means stronger support for class 0.
       """
        classifier = model.named_steps["classifier"]

        if not hasattr(classifier, "coef_"):
            return None

        coefficients = classifier.coef_[0]
        rows = []

        for feature, coefficient in zip(self.feature_columns, coefficients):
            rows.append({
                "fold_id": fold_id,
                "test_subject": test_subject,
                "feature": feature,
                "coefficient": coefficient,
                "abs_coefficient": abs(coefficient),
            })

        return pd.DataFrame(rows)

    def run_loso_training(self, df):
        """
        Train and evaluate the configured model using LOSO validation.
        Cross validation / Leave-One-Subject-Out training is used in this project because EEG data is dependent on person
        This is important because in our case, samples from the same person are similar.
        """
        target_column = self.config["data"]["target_column"]
        group_column = self.config["validation"]["group_column"]
        df, feature_columns = self.prepare_model_data(df)

        X = df[feature_columns]
        y = df[target_column]
        groups = df[group_column]

        logo = LeaveOneGroupOut()
        fold_results = []
        all_predictions = []
        all_feature_importance = []

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

            model = build_model_pipeline(self.config)
            model.fit(X_train, y_train)

            feature_importance = self.extract_feature_importance(
                model,
                fold_id,
                test_subject
            )

            if feature_importance is not None:
                all_feature_importance.append(feature_importance)

            y_pred = model.predict(X_test)
            y_score = self.get_prediction_scores(model, X_test)

            result = {
                "fold_id": fold_id,
                "test_subject": test_subject,
                "n_train": len(train_index),
                "n_test": len(test_index),
            }

            metrics = compute_metrics(y_test, y_pred, y_score)
            result.update(metrics)

            fold_results.append(result)

            fold_predictions = df.iloc[test_index].copy()
            fold_predictions["fold_id"] = fold_id
            fold_predictions["prediction"] = y_pred

            if y_score is not None:
                fold_predictions["prediction_score"] = y_score

            all_predictions.append(fold_predictions)

        results_df = pd.DataFrame(fold_results)
        predictions_df = pd.concat(all_predictions, ignore_index=True)

        if len(all_feature_importance) > 0:
            feature_importance_df = pd.concat(all_feature_importance, ignore_index=True)
        else:
            feature_importance_df = pd.DataFrame()

        return results_df, predictions_df, feature_importance_df

    def train_final_model(self, df):
        """Train one final model on all available rows after LOSO evaluation."""
        target_column = self.config["data"]["target_column"]
        df, feature_columns = self.prepare_model_data(df)

        X = df[feature_columns]
        y = df[target_column]

        model = build_model_pipeline(self.config)
        model.fit(X, y)

        return model, feature_columns

    def save_model_artifacts(self, model, feature_columns, output_dir, input_format="feature_table"):
        """Save the trained model and metadata needed for later prediction."""
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "model.joblib")
        feature_columns_path = os.path.join(output_dir, "feature_columns.json")
        model_metadata_path = os.path.join(output_dir, "model_metadata.json")

        joblib.dump(model, model_path)

        with open(feature_columns_path, "w", encoding="utf-8") as file:
            json.dump(feature_columns, file, indent=2)

        model_metadata = {}
        model_metadata["experiment"] = self.config["experiment"]["name"]
        model_metadata["input_format"] = input_format
        model_metadata["n_features"] = len(feature_columns)

        if "model" in self.config:
            model_metadata["model_type"] = self.config["model"].get("type")
        else:
            model_metadata["model_type"] = None

        if "normalization" in self.config:
            model_metadata["subject_relative_to_rest"] = self.config["normalization"].get(
                "subject_relative_to_rest",
                False
            )
        else:
            model_metadata["subject_relative_to_rest"] = False

        model_metadata["reject_rejected_windows"] = self.should_reject_rejected_windows()

        with open(model_metadata_path, "w", encoding="utf-8") as file:
            json.dump(model_metadata, file, indent=2)

    def run(self):
        df = self.load_feature_table()
        self.validate_feature_table(df)

        results_df, predictions_df, feature_importance_df = self.run_loso_training(df)
        summary = summarize_results(results_df, predictions_df, self.config)
        output_dir = save_results(results_df, predictions_df, feature_importance_df, summary, self.config)

        final_model, feature_columns = self.train_final_model(df)
        self.save_model_artifacts(final_model, feature_columns, output_dir)

        return summary


def load_config(config_path):
    return load_yaml_config(config_path)


def load_feature_table(config):
    pipeline = BandpowerTrainingPipeline(config)
    return pipeline.load_feature_table()


def should_reject_rejected_windows(config):
    pipeline = BandpowerTrainingPipeline(config)
    return pipeline.should_reject_rejected_windows()


def filter_rejected_windows(df, config):
    pipeline = BandpowerTrainingPipeline(config)
    return pipeline.filter_rejected_windows(df)


def get_feature_columns(df, config):
    pipeline = BandpowerTrainingPipeline(config)
    return pipeline.get_feature_columns(df)


def apply_subject_relative_to_rest(df, feature_columns, config):
    pipeline = BandpowerTrainingPipeline(config)
    return pipeline.apply_subject_relative_to_rest(df, feature_columns)


def validate_feature_table(df, config):
    pipeline = BandpowerTrainingPipeline(config)
    pipeline.validate_feature_table(df)


def get_prediction_scores(model, X_test):
    pipeline = BandpowerTrainingPipeline({})
    return pipeline.get_prediction_scores(model, X_test)


def run_loso_training(df, config):
    pipeline = BandpowerTrainingPipeline(config)
    return pipeline.run_loso_training(df)


def prepare_model_data(df, config):
    pipeline = BandpowerTrainingPipeline(config)
    return pipeline.prepare_model_data(df)


def train_final_model(df, config):
    pipeline = BandpowerTrainingPipeline(config)
    return pipeline.train_final_model(df)


def save_model_artifacts(model, feature_columns, config, output_dir, input_format="feature_table"):
    pipeline = BandpowerTrainingPipeline(config)
    pipeline.save_model_artifacts(model, feature_columns, output_dir, input_format)


def extract_feature_importance(model, feature_columns, fold_id, test_subject):
    pipeline = BandpowerTrainingPipeline({})
    pipeline.feature_columns = feature_columns
    return pipeline.extract_feature_importance(model, fold_id, test_subject)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python -m src.training.train configs/model_bandpower_logreg.yaml")
        return

    config_path = sys.argv[1]
    config = load_config(config_path)

    pipeline = BandpowerTrainingPipeline(config)
    summary = pipeline.run()

    print("Mean balanced accuracy:", summary["mean_balanced_accuracy"])
    print("Mean F1:", summary["mean_f1"])


if __name__ == "__main__":
    main()
