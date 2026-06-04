from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_scaler(config):
    """
    2 modes:
    ---
    a) robust scaler: x' = (x-median) / IQR
    where IQR = Q3 - Q1
    Robust Scaler is better when data has outliers -> The median and quartiles are much less
    susceptible to extreme values thab the mean and standard deviation.

    b) Standard Scaler: x' = (x - mu) / std_dev
    where mu = mean and std_dev = standard deviation.
    StandardScaler is good when the data is relatively normal, with no large outliers.
    Byt it's sensitive to outliers, as the mean and standard deviation are significantly affected by them
    """
    scaler_name = config.get("normalization", {}).get("scaler", "robust")

    if scaler_name == "robust":
        return RobustScaler()

    if scaler_name == "standard":
        return StandardScaler()

    if scaler_name == "none":
        return None

    raise ValueError("Unknown scaler: " + str(scaler_name))


def build_classifier(config):
    model_config = config.get("model", {})
    model_type = model_config.get("type", "logistic_regression")

    # Bandpower + LR
    if model_type == "logistic_regression":
        return LogisticRegression(
            class_weight=model_config.get("class_weight", "balanced"),
            C=model_config.get("C", 1.0),
            max_iter=model_config.get("max_iter", 1000),
        )

    # Bandpower + SVM
    if model_type == "linear_svm":
        return LinearSVC(
            class_weight=model_config.get("class_weight", "balanced"),
            C=model_config.get("C", 1.0),
            max_iter=model_config.get("max_iter", 1000),
        )

    raise ValueError("Unknown model: " + str(model_type))


def build_model_pipeline(config):
    """
    Pipeline:
    Scaler -> Classifier -> ...
    """
    steps = []

    scaler = build_scaler(config)
    if scaler is not None:
        steps.append(("scaler", scaler))

    classifier = build_classifier(config)
    steps.append(("classifier", classifier))

    return Pipeline(steps)
