from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


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

        if model_type == "rbf_svm":
            return SVC(
                kernel="rbf",
                class_weight=model_config.get("class_weight", "balanced"),
                C=model_config.get("C", 1.0),
                gamma=model_config.get("gamma", "scale"),
                max_iter=model_config.get("max_iter", -1),
            )
        
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=model_config.get("n_estimators", 500),
                max_depth=model_config.get("max_depth"),
                min_samples_leaf=model_config.get("min_samples_leaf", 1),
                max_features=model_config.get("max_features", "sqrt"),
                class_weight=model_config.get("class_weight", "balanced"),
                random_state=model_config.get("random_state", 42),
                n_jobs=model_config.get("n_jobs", -1),
            )
        
        if model_type == "extra_trees":
            return ExtraTreesClassifier(
                n_estimators=model_config.get("n_estimators", 500),
                max_depth=model_config.get("max_depth"),
                min_samples_leaf=model_config.get("min_samples_leaf", 1),
                max_features=model_config.get("max_features", "sqrt"),
                class_weight=model_config.get("class_weight", "balanced"),
                random_state=model_config.get("random_state", 42),
                n_jobs=model_config.get("n_jobs", -1),
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
