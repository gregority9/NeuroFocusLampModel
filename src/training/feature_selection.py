class FeatureGroupSelector:
    """Select feature columns by semantic feature groups."""

    RAW_BANDS = ["delta", "theta", "alpha", "beta"]
    RAW_CHANNELS = ["AF3", "AF4", "F3", "F4", "FC5", "FC6", "O2", "O1"]

    def __init__(self, feature_groups=None):
        self.feature_groups = feature_groups or []

    def select(self, feature_columns):
        if len(self.feature_groups) == 0:
            return feature_columns

        selected_columns = []

        for feature in feature_columns:
            if self.matches_any_group(feature):
                selected_columns.append(feature)

        return selected_columns

    def matches_any_group(self, feature):
        for group in self.feature_groups:
            if self.matches_group(feature, group):
                return True

        return False

    def matches_group(self, feature, group):
        if group == "raw_bandpower":
            return self.is_raw_bandpower(feature)

        if group == "log_bandpower":
            return self.is_log_bandpower(feature)

        if group == "ratios":
            return self.is_ratio(feature)

        if group == "regional":
            return self.is_regional(feature)

        if group == "asymmetry":
            return self.is_asymmetry(feature)

        raise ValueError("Unknown feature group: " + str(group))

    def is_raw_bandpower(self, feature):
        for band in self.RAW_BANDS:
            for channel in self.RAW_CHANNELS:
                if feature == f"{band}_{channel}":
                    return True

        return False

    def is_log_bandpower(self, feature):
        if not feature.startswith("log_"):
            return False

        raw_feature = feature.replace("log_", "", 1)
        return self.is_raw_bandpower(raw_feature)

    def is_ratio(self, feature):
        return feature.startswith("ratio_")

    def is_regional(self, feature):
        region_tokens = [
            "_frontal_mean",
            "_frontocentral_mean",
            "_occipital_mean",
            "_left_mean",
            "_right_mean",
        ]

        for token in region_tokens:
            if token in feature:
                return True

        return False

    def is_asymmetry(self, feature):
        return feature.startswith("asym_")
