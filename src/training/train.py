import sys

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

    # Optionally remove rejected windows
    rejected_column = config["data"].get("rejected_column")
    if rejected_column in df.columns:
        if df[rejected_column].dtype == "object":
            rejected = df[rejected_column].astype(str).str.lower()
            df = df[rejected != "true"].copy()
        else:
            df = df[df[rejected_column] == False].copy()

    return df


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
    feature_columns = get_feature_columns(df, config)

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
    save_results(results_df, predictions_df, feature_importance_df, summary, config)

    print("Mean balanced accuracy:", summary["mean_balanced_accuracy"])
    print("Mean F1:", summary["mean_f1"])


if __name__ == "__main__":
    main()
