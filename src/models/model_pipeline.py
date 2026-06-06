from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class ModelPipelineBuilder:
    """Build sklearn pipelines from experiment config."""

    def __init__(self, config):
        self.config = config

    def build_scaler(self):
        """
        2 modes:
        ---
        a) robust scaler: x' = (x-median) / IQR
        where IQR = Q3 - Q1

        b) Standard Scaler: x' = (x - mu) / std_dev
        where mu = mean and std_dev = standard deviation.
        """
        scaler_name = self.config.get("normalization", {}).get("scaler", "robust")

        if scaler_name == "robust":
            return RobustScaler()

        if scaler_name == "standard":
            return StandardScaler()

        if scaler_name == "none":
            return None

        raise ValueError("Unknown scaler: " + str(scaler_name))

    def build_classifier(self):
        model_config = self.config.get("model", {})
        model_type = model_config.get("type", "logistic_regression")

        if model_type == "logistic_regression":
            return LogisticRegression(
                class_weight=model_config.get("class_weight", "balanced"),
                C=model_config.get("C", 1.0),
                max_iter=model_config.get("max_iter", 1000),
            )

        if model_type == "linear_svm":
            return LinearSVC(
                class_weight=model_config.get("class_weight", "balanced"),
                C=model_config.get("C", 1.0),
                max_iter=model_config.get("max_iter", 1000),
            )

        raise ValueError("Unknown model: " + str(model_type))

    def build(self):
        steps = []

        scaler = self.build_scaler()
        if scaler is not None:
            steps.append(("scaler", scaler))

        classifier = self.build_classifier()
        steps.append(("classifier", classifier))

        return Pipeline(steps)


def build_scaler(config):
    builder = ModelPipelineBuilder(config)
    return builder.build_scaler()


def build_classifier(config):
    builder = ModelPipelineBuilder(config)
    return builder.build_classifier()


def build_model_pipeline(config):
    builder = ModelPipelineBuilder(config)
    return builder.build()
