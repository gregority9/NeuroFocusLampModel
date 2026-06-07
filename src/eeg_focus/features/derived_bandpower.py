import math


class DerivedBandpowerFeatureExtractor:
    """Create derived features from per-channel bandpower values."""

    EPSILON = 1e-12

    REGIONS = {
        "frontal": ["AF3", "AF4", "F3", "F4"],
        "frontocentral": ["FC5", "FC6"],
        "occipital": ["O1", "O2"],
        "left": ["AF3", "F3", "FC5", "O1"],
        "right": ["AF4", "F4", "FC6", "O2"],
    }

    ASYMMETRY_PAIRS = [
        ("AF3", "AF4"),
        ("F3", "F4"),
        ("FC5", "FC6"),
        ("O1", "O2"),
    ]

    def __init__(self, bands, channels):
        self.bands = list(bands)
        self.channels = list(channels)

    def transform(self, features):
        derived_features = features.copy()

        self.add_log_bandpower_features(derived_features)
        self.add_ratio_features(derived_features)
        self.add_regional_mean_features(derived_features)
        self.add_asymmetry_features(derived_features)

        return derived_features

    def add_log_bandpower_features(self, features):
        for band in self.bands:
            for channel in self.channels:
                feature_name = f"{band}_{channel}"

                if feature_name not in features:
                    continue

                value = max(features[feature_name], self.EPSILON)
                features[f"log_{feature_name}"] = math.log(value)

    def add_ratio_features(self, features):
        ratio_pairs = [
            ("theta", "alpha"),
            ("beta", "alpha"),
            ("theta", "beta"),
            ("alpha", "beta"),
        ]

        for channel in self.channels:
            for numerator, denominator in ratio_pairs:
                numerator_name = f"{numerator}_{channel}"
                denominator_name = f"{denominator}_{channel}"

                if numerator_name not in features or denominator_name not in features:
                    continue

                ratio_name = f"ratio_{numerator}_over_{denominator}_{channel}"
                features[ratio_name] = (
                    features[numerator_name]
                    / (features[denominator_name] + self.EPSILON)
                )

    def add_regional_mean_features(self, features):
        for band in self.bands:
            for region_name, region_channels in self.REGIONS.items():
                values = []

                for channel in region_channels:
                    feature_name = f"{band}_{channel}"

                    if feature_name in features:
                        values.append(features[feature_name])

                if len(values) == 0:
                    continue

                mean_value = sum(values) / len(values)
                features[f"{band}_{region_name}_mean"] = mean_value
                features[f"log_{band}_{region_name}_mean"] = math.log(
                    max(mean_value, self.EPSILON)
                )

    def add_asymmetry_features(self, features):
        for band in self.bands:
            for left_channel, right_channel in self.ASYMMETRY_PAIRS:
                left_name = f"{band}_{left_channel}"
                right_name = f"{band}_{right_channel}"

                if left_name not in features or right_name not in features:
                    continue

                left_value = features[left_name]
                right_value = features[right_name]

                features[f"asym_{band}_{left_channel}_minus_{right_channel}"] = (
                    left_value - right_value
                )

                features[f"asym_log_{band}_{left_channel}_minus_{right_channel}"] = (
                    math.log(max(left_value, self.EPSILON))
                    - math.log(max(right_value, self.EPSILON))
                )
