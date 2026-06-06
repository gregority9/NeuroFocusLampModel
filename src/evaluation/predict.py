import json
import os
import sys

import joblib
import pandas as pd

from src.training.train import BandpowerTrainingPipeline
from src.training.train import get_prediction_scores
from src.training.train import load_config


class FeatureTablePredictor:
    """Predict labels for a new feature table using a saved experiment."""

    def __init__(self, experiment_dir, input_file, output_path):
        self.experiment_dir = experiment_dir
        self.input_file = input_file
        self.output_path = output_path
        self.model = None
        self.config = None
        self.feature_columns = []

    def get_paths(self):
        paths = {}
        paths["model"] = os.path.join(self.experiment_dir, "model.joblib")
        paths["features"] = os.path.join(self.experiment_dir, "feature_columns.json")
        paths["config"] = os.path.join(self.experiment_dir, "config_used.yaml")
        paths["metadata"] = os.path.join(self.experiment_dir, "model_metadata.json")

        return paths

    def load_artifacts(self):
        paths = self.get_paths()

        if not os.path.exists(paths["model"]):
            print("No saved model:", paths["model"])
            return False

        if not os.path.exists(paths["features"]):
            print("No feature column file:", paths["features"])
            return False

        if not os.path.exists(paths["config"]):
            print("No config file:", paths["config"])
            return False

        self.model = joblib.load(paths["model"])
        self.config = load_config(paths["config"])

        with open(paths["features"], "r", encoding="utf-8") as file:
            self.feature_columns = json.load(file)

        if os.path.exists(paths["metadata"]):
            with open(paths["metadata"], "r", encoding="utf-8") as file:
                model_metadata = json.load(file)

            if model_metadata.get("input_format") != "feature_table":
                print("This script expects feature table input.")
                print("Model input format:", model_metadata.get("input_format"))
                return False

        return True

    def get_missing_features(self, df):
        missing_features = []

        for feature in self.feature_columns:
            if feature not in df.columns:
                missing_features.append(feature)

        return missing_features

    def load_prediction_table(self):
        df = pd.read_csv(self.input_file)

        training_pipeline = BandpowerTrainingPipeline(self.config)
        df = training_pipeline.filter_rejected_windows(df)

        if len(df) == 0:
            print("No rows left after artifact rejection.")
            return None

        missing_features = self.get_missing_features(df)

        if len(missing_features) > 0:
            print("Missing features:")
            print(missing_features)
            return None

        if df[self.feature_columns].isnull().any().any():
            print("There are NaN values in feature columns.")
            return None

        df = training_pipeline.apply_subject_relative_to_rest(df, self.feature_columns)

        return df

    def save_predictions(self, df):
        output_dir = os.path.dirname(self.output_path)

        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)

        df.to_csv(self.output_path, index=False)

    def run(self):
        artifacts_loaded = self.load_artifacts()

        if artifacts_loaded == False:
            return None

        df = self.load_prediction_table()

        if df is None:
            return None

        X = df[self.feature_columns]
        prediction = self.model.predict(X)
        score = get_prediction_scores(self.model, X)

        df["prediction"] = prediction

        if score is not None:
            df["prediction_score"] = score

        self.save_predictions(df)

        return df


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

    predictor = FeatureTablePredictor(experiment_dir, input_file, output_path)
    predictions_df = predictor.run()

    if predictions_df is None:
        return

    print("Saved predictions:", output_path)
    print("Predicted windows:", len(predictions_df))


if __name__ == "__main__":
    main()
