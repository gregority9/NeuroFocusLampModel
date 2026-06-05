import json
import os
import sys

import joblib
import pandas as pd

from src.training.train import apply_subject_relative_to_rest
from src.training.train import filter_rejected_windows
from src.training.train import get_prediction_scores
from src.training.train import load_config


def load_prediction_artifacts(experiment_dir):
    """Load model, feature list, config, and metadata from an experiment directory."""
    model_path = os.path.join(experiment_dir, "model.joblib")
    feature_columns_path = os.path.join(experiment_dir, "feature_columns.json")
    config_path = os.path.join(experiment_dir, "config_used.yaml")
    model_metadata_path = os.path.join(experiment_dir, "model_metadata.json")

    missing_files = []

    for path in [model_path, feature_columns_path, config_path, model_metadata_path]:
        if not os.path.exists(path):
            missing_files.append(path)

    if len(missing_files) > 0:
        raise ValueError("Missing prediction artifacts: " + str(missing_files))

    model = joblib.load(model_path)
    config = load_config(config_path)

    with open(feature_columns_path, "r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    with open(model_metadata_path, "r", encoding="utf-8") as file:
        model_metadata = json.load(file)

    if model_metadata.get("input_format") != "feature_table":
        raise ValueError(
            "This predict.py expects a feature table model. "
            "The selected model uses input format: "
            + str(model_metadata.get("input_format"))
        )

    return model, feature_columns, config


def validate_prediction_features(df, feature_columns):
    """Check whether all training features are present in the prediction table."""
    missing_features = []

    for feature in feature_columns:
        if feature not in df.columns:
            missing_features.append(feature)

    if len(missing_features) > 0:
        raise ValueError("Missing feature columns for prediction: " + str(missing_features))

    if df[feature_columns].isnull().any().any():
        raise ValueError("Prediction feature columns contain NaN values.")


def prepare_prediction_data(input_file, feature_columns, config):
    """Load and prepare a feature table for prediction."""
    df = pd.read_csv(input_file)
    df = filter_rejected_windows(df, config)

    if len(df) == 0:
        raise ValueError("Prediction table is empty after filtering rejected windows.")

    validate_prediction_features(df, feature_columns)
    df = apply_subject_relative_to_rest(df, feature_columns, config)

    return df


def resolve_output_path(experiment_dir, output_argument):
    """Resolve CLI output argument into a CSV path."""
    if output_argument is None:
        return os.path.join(experiment_dir, "new_predictions.csv")

    if output_argument.endswith(".csv"):
        return output_argument

    return os.path.join(output_argument, "new_predictions.csv")


def save_predictions(predictions_df, output_path):
    """Save predictions to a CSV file."""
    output_dir = os.path.dirname(output_path)

    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    predictions_df.to_csv(output_path, index=False)


def run_prediction(experiment_dir, input_file, output_path):
    """Run prediction using a model saved by the training pipeline."""
    model, feature_columns, config = load_prediction_artifacts(experiment_dir)
    prediction_df = prepare_prediction_data(input_file, feature_columns, config)

    X = prediction_df[feature_columns]
    y_pred = model.predict(X)
    y_score = get_prediction_scores(model, X)

    prediction_df["prediction"] = y_pred

    if y_score is not None:
        prediction_df["prediction_score"] = y_score

    save_predictions(prediction_df, output_path)

    return prediction_df


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("python -m src.evaluation.predict reports/experiments/bandpower_logreg data/processed/new_subject_features.csv")
        print("python -m src.evaluation.predict reports/experiments/bandpower_logreg data/processed/new_subject_features.csv reports/predictions/new_subject.csv")
        return

    experiment_dir = sys.argv[1]
    input_file = sys.argv[2]

    if len(sys.argv) >= 4:
        output_argument = sys.argv[3]
    else:
        output_argument = None

    output_path = resolve_output_path(experiment_dir, output_argument)
    predictions_df = run_prediction(experiment_dir, input_file, output_path)

    print("Saved predictions:", output_path)
    print("Predicted windows:", len(predictions_df))


if __name__ == "__main__":
    main()
