import json
import os
import sys

import joblib
import pandas as pd

from src.training.train import apply_subject_relative_to_rest
from src.training.train import filter_rejected_windows
from src.training.train import get_prediction_scores
from src.training.train import load_config


def get_missing_features(df, feature_columns):
    """Return feature columns which are not present in the input table."""
    missing_features = []

    for feature in feature_columns:
        if feature not in df.columns:
            missing_features.append(feature)

    return missing_features


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("python -m src.evaluation.predict reports/experiments/bandpower_logreg data/processed/new_subject_features.csv")
        print("python -m src.evaluation.predict reports/experiments/bandpower_logreg data/processed/new_subject_features.csv reports/predictions/new_subject.csv")
        return

    experiment_dir = sys.argv[1]
    input_file = sys.argv[2]

    if len(sys.argv) >= 4:
        output_path = sys.argv[3]
    else:
        output_path = os.path.join(experiment_dir, "new_predictions.csv")

    model_path = os.path.join(experiment_dir, "model.joblib")
    feature_columns_path = os.path.join(experiment_dir, "feature_columns.json")
    config_path = os.path.join(experiment_dir, "config_used.yaml")
    model_metadata_path = os.path.join(experiment_dir, "model_metadata.json")

    if not os.path.exists(model_path):
        print("No saved model:", model_path)
        return

    if not os.path.exists(feature_columns_path):
        print("No feature column file:", feature_columns_path)
        return

    if not os.path.exists(config_path):
        print("No config file:", config_path)
        return

    model = joblib.load(model_path)
    config = load_config(config_path)

    with open(feature_columns_path, "r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    if os.path.exists(model_metadata_path):
        with open(model_metadata_path, "r", encoding="utf-8") as file:
            model_metadata = json.load(file)

        if model_metadata.get("input_format") != "feature_table":
            print("This script expects feature table input.")
            print("Model input format:", model_metadata.get("input_format"))
            return

    df = pd.read_csv(input_file)
    df = filter_rejected_windows(df, config)

    if len(df) == 0:
        print("No rows left after artifact rejection.")
        return

    missing_features = get_missing_features(df, feature_columns)

    if len(missing_features) > 0:
        print("Missing features:")
        print(missing_features)
        return

    if df[feature_columns].isnull().any().any():
        print("There are NaN values in feature columns.")
        return

    df = apply_subject_relative_to_rest(df, feature_columns, config)

    X = df[feature_columns]
    prediction = model.predict(X)
    score = get_prediction_scores(model, X)

    df["prediction"] = prediction

    if score is not None:
        df["prediction_score"] = score

    output_dir = os.path.dirname(output_path)

    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("Saved predictions:", output_path)
    print("Predicted windows:", len(df))


if __name__ == "__main__":
    main()
