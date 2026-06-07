from src.eeg_focus.io.config import ConfigLoader

import numpy as np


class Accelerometer:
    def __init__(self, config=None):
        self.config = config or ConfigLoader().get()

    def run(self, accel_df):
        accel_channels = self.config["data"]["accel"]
        sampling_rate = self.config["data"]["sampling_rate"]
        sample_column = self.config["data"]["sample_column"]

        acc_cfg = self.config["accelerometer"]
        motion_cfg = acc_cfg["motion_score"]
        artifact_cfg = acc_cfg["artifact_detection"]

        method = artifact_cfg.get("threshold_method", "median_mad")
        window_sec = motion_cfg["window_sec"]
        mad_multiplier = artifact_cfg["mad_multiplier"]
        min_threshold = artifact_cfg.get("min_threshold", None)
        max_threshold = artifact_cfg.get("max_threshold", None)

        window_samples = int(window_sec * sampling_rate)

        accel_only = accel_df[accel_channels].copy()

        accel_only["acc_norm"] = np.sqrt(
            accel_only["Accel_x"] ** 2
            + accel_only["Accel_y"] ** 2
            + accel_only["Accel_z"] ** 2
        )

        accel_only["acc_delta"] = accel_only["acc_norm"].diff().abs().fillna(0)

        accel_only["motion_score"] = (
            accel_only["acc_delta"]
            .rolling(window=window_samples, min_periods=1)
            .mean()
        )

        threshold = self._calculate_threshold(
            accel_only["motion_score"],
            method,
            mad_multiplier,
        )

        if min_threshold is not None:
            threshold = max(threshold, min_threshold)

        if max_threshold is not None:
            threshold = min(threshold, max_threshold)

        accel_only["motion_threshold"] = threshold

        accel_only["is_motion_artifact"] = (
            accel_only["motion_score"] > threshold
        )

        result_df = accel_df[[sample_column]].copy()
        result_df[accel_channels] = accel_only[accel_channels]
        result_df["acc_norm"] = accel_only["acc_norm"]
        result_df["acc_delta"] = accel_only["acc_delta"]
        result_df["motion_score"] = accel_only["motion_score"]
        result_df["motion_threshold"] = accel_only["motion_threshold"]
        result_df["is_motion_artifact"] = accel_only["is_motion_artifact"]

        return result_df

    def _calculate_threshold(self, motion_score, method, mad_multiplier):
        match method:
            case "median_mad":
                median_motion = motion_score.median()
                mad_motion = motion_score.sub(median_motion).abs().median()
                return median_motion + mad_multiplier * mad_motion
            case _:
                raise ValueError(f"Unsupported accelerometer threshold method: {method}")
