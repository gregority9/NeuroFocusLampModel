import json
import os
import sys

import joblib
import pandas as pd
import yaml

from sklearn.model_selection import LeaveOneGroupOut

from src.evaluation.metrics import compute_metrics
from src.evaluation.reports import save_results
from src.evaluation.reports import summarize_results
from src.models.model_pipeline import build_model_pipeline


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_feature_table(config):
    input_file = config["data"]["input_file"]
    df = pd.read_csv(input_file)

    df = filter_rejected_windows(df, config)

    return df


def should_reject_rejected_windows(config):
    """Decide if rejected windows should be removed before modeling."""
    artifacts_config = config.get("artifacts", {})

    return artifacts_config.get("reject_rejected_windows", True)


def filter_rejected_windows(df, config):
    """Remove rejected windows when this is enabled in the config."""
    if not should_reject_rejected_windows(config):
        return df

    rejected_column = config["data"].get("rejected_column")

    if rejected_column is None:
        return df

    if rejected_column not in df.columns:
        return df

    if df[rejected_column].dtype == "object":
        rejected = df[rejected_column].astype(str).str.lower()
        return df[~rejected.isin(["true", "1", "yes"])].copy()

    return df[df[rejected_column] == False].copy()


def get_feature_columns(df, config):
    """Select model input columns by excluding metadata columns."""
    metadata_columns = config["data"].get("metadata_columns", [])
    feature_columns = []

    for column in df.columns:
        if column not in metadata_columns:
            feature_columns.append(column)

    feature_columns = apply_feature_filters(feature_columns, config)

    return feature_columns


def feature_contains_channel(feature, channels):
    feature_tokens = feature.replace("-", "_").split("_")

    for channel in channels:
        if channel in feature_tokens:
            return True

    return False


def feature_contains_band(feature, bands):
    feature_name = feature.lower()

    for band in bands:
        if band.lower() in feature_name:
            return True

    return False


def apply_feature_filters(feature_columns, config):
    """
    Apply artifact-control feature filters from config.

    This is used for experiments such as:
    - no AF3/AF4,
    - no high beta,
    - only occipital channels,
    - only frontal channels.
    """
    feature_config = config.get("features", {})

    include_channels = feature_config.get("include_channels", [])
    exclude_channels = feature_config.get("exclude_channels", [])
    include_bands = feature_config.get("include_bands", [])
    exclude_bands = feature_config.get("exclude_bands", [])

    filtered_columns = []

    for feature in feature_columns:
        if len(include_channels) > 0 and not feature_contains_channel(feature, include_channels):
            continue

        if len(exclude_channels) > 0 and feature_contains_channel(feature, exclude_channels):
            continue

        if len(include_bands) > 0 and not feature_contains_band(feature, include_bands):
            continue

        if len(exclude_bands) > 0 and feature_contains_band(feature, exclude_bands):
            continue

        filtered_columns.append(feature)

    return filtered_columns


def apply_subject_relative_to_rest(df, feature_columns, config):
    """
    Normalize features relative to each subject's own rest baseline.

    This assumes that a rest recording is available for every subject.
    For a new subject, this corresponds to a calibration recording.
    """
    normalization_config = config.get("normalization", {})

    if not normalization_config.get("subject_relative_to_rest", False):
        return df

    subject_column = normalization_config.get(
        "subject_column",
        config["validation"].get("group_column", "subject_id")
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


def validate_feature_table(df, config):
    """
    Are there any required columns?
    Is there any data left after the is_rejected rejection?
    Are there at least two people for RANDOM?
    Are there at least two classes?
    Are the labels 0 and 1?
    Are there feature columns?
    Do the features have no NaNs?
    """
    required_columns = config["data"].get("required_columns", []).copy()
    target_column = config["data"]["target_column"]
    group_column = config["validation"]["group_column"]

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

    allowed_labels = config["data"].get("allowed_labels")
    if allowed_labels is not None:
        labels = set(df[target_column].dropna().unique())
        allowed_labels = set(allowed_labels)

        if labels != allowed_labels:
            raise ValueError("Unexpected labels: " + str(labels))

    feature_columns = get_feature_columns(df, config)
    if len(feature_columns) == 0:
        raise ValueError("No feature columns found.")

    if df[feature_columns].isnull().any().any():
        raise ValueError("Feature columns contain NaN values.")


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
    df, feature_columns = prepare_model_data(df, config)

    X = df[feature_columns]
    y = df[target_column]
    groups = df[group_column]

    # Fold = one experiment, one training test in which one person is placed for testing, others to train
    logo = LeaveOneGroupOut()
    fold_results = []
    all_predictions = []
    all_feature_importance = []


    # Calculating result for every fold separately
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

        feature_importance = extract_feature_importance(
            model,
            feature_columns,
            fold_id,
            test_subject
        )

        if feature_importance is not None:
            all_feature_importance.append(feature_importance)

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


def prepare_model_data(df, config):
    """Apply feature selection and model-level normalization."""
    feature_columns = get_feature_columns(df, config)
    df = apply_subject_relative_to_rest(df, feature_columns, config)

    return df, feature_columns


def train_final_model(df, config):
    """Train one final model on all available rows after LOSO evaluation."""
    target_column = config["data"]["target_column"]
    df, feature_columns = prepare_model_data(df, config)

    X = df[feature_columns]
    y = df[target_column]

    model = build_model_pipeline(config)
    model.fit(X, y)

    return model, feature_columns


def save_model_artifacts(model, feature_columns, config, output_dir, input_format="feature_table"):
    """Save the trained model and metadata needed for later prediction."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.joblib")
    feature_columns_path = os.path.join(output_dir, "feature_columns.json")
    model_metadata_path = os.path.join(output_dir, "model_metadata.json")

    joblib.dump(model, model_path)

    with open(feature_columns_path, "w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=2)

    model_metadata = {
        "experiment": config["experiment"]["name"],
        "model_type": config.get("model", {}).get("type"),
        "input_format": input_format,
        "n_features": len(feature_columns),
        "subject_relative_to_rest": config.get("normalization", {}).get(
            "subject_relative_to_rest",
            False
        ),
        "reject_rejected_windows": should_reject_rejected_windows(config),
    }

    with open(model_metadata_path, "w", encoding="utf-8") as file:
        json.dump(model_metadata, file, indent=2)


def extract_feature_importance(model, feature_columns, fold_id, test_subject):
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

    for feature, coefficient in zip(feature_columns, coefficients):
        rows.append({
            "fold_id": fold_id,
            "test_subject": test_subject,
            "feature": feature,
            "coefficient": coefficient,
            "abs_coefficient": abs(coefficient),
        })

    return pd.DataFrame(rows)



def main():
    """Run training from the CLI"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("python -m src.training.train configs/model_bandpower_logreg.yaml")
        return

    config_path = sys.argv[1]

    config = load_config(config_path)
    df = load_feature_table(config)
    validate_feature_table(df, config)

    results_df, predictions_df, feature_importance_df = run_loso_training(df, config)
    summary = summarize_results(results_df, predictions_df, config)
    output_dir = save_results(results_df, predictions_df, feature_importance_df, summary, config)

    final_model, feature_columns = train_final_model(df, config)
    save_model_artifacts(final_model, feature_columns, config, output_dir)

    print("Mean balanced accuracy:", summary["mean_balanced_accuracy"])
    print("Mean F1:", summary["mean_f1"])


if __name__ == "__main__":
    main()
