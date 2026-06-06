from pyriemann.tangentspace import TangentSpace

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class RiemannianPipelineBuilder:
    """Build a Riemannian Tangent Space + Logistic Regression pipeline."""

    def __init__(self, config):
        self.config = config

    def build(self):
        model_config = self.config.get("model", {})
        tangent_config = self.config.get("riemannian", {})

        return Pipeline([
            ("tangent_space", TangentSpace(
                metric=tangent_config.get("metric", "riemann")
            )),
            ("classifier", LogisticRegression(
                class_weight=model_config.get("class_weight", "balanced"),
                C=model_config.get("C", 1.0),
                max_iter=model_config.get("max_iter", 1000),
            )),
        ])


def build_riemannian_logreg_pipeline(config):
    builder = RiemannianPipelineBuilder(config)
    return builder.build()
