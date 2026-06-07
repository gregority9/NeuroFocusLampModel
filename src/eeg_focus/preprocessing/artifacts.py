import numpy as np
import pandas as pd

from src.eeg_focus.io.config import ConfigLoader


class Artifacts:
    def __init__(self, config=None):
        self.config = config or ConfigLoader().get()

    def run(self, filtered_eeg_df, processed_accel_df):
        sample_column = self.config["data"]["sample_column"]
        eeg_channels = self.config["data"]["channels"]
        columns_cfg = self.config["artifacts"]["columns"]
        detection_cfg = self.config["artifacts"]["detection"]

        eeg_only = filtered_eeg_df[eeg_channels].copy()
        artifact_df = filtered_eeg_df[[sample_column]].copy()

        artifact_df[columns_cfg["eeg_amplitude_raw"]] = eeg_only.abs().max(axis=1)
        artifact_df[columns_cfg["eeg_amplitude_score"]] = self._robust_zscore(
            artifact_df[columns_cfg["eeg_amplitude_raw"]]
        )

        eeg_diff = eeg_only.diff().abs().fillna(0)

        artifact_df[columns_cfg["eeg_jump_raw"]] = eeg_diff.max(axis=1)
        artifact_df[columns_cfg["eeg_jump_score"]] = self._robust_zscore(
            artifact_df[columns_cfg["eeg_jump_raw"]]
        )

        artifact_df = artifact_df.merge(
            processed_accel_df[
                [sample_column, "motion_score", "is_motion_artifact"]
            ],
            on=sample_column,
            how="left",
        )

        artifact_df["motion_score"] = artifact_df["motion_score"].fillna(0)
        artifact_df["is_motion_artifact"] = (
            artifact_df["is_motion_artifact"].fillna(False).astype(bool)
        )

        artifact_df[columns_cfg["motion_artifact_score"]] = self._robust_zscore(
            artifact_df["motion_score"]
        )

        score_columns = [
            columns_cfg["eeg_amplitude_score"],
            columns_cfg["eeg_jump_score"],
            columns_cfg["motion_artifact_score"],
        ]

        artifact_df[columns_cfg["artifact_score"]] = self._combine_scores(
            artifact_df[score_columns],
            detection_cfg["combine_method"],
        )

        artifact_df[columns_cfg["is_artifact"]] = (
            artifact_df[columns_cfg["artifact_score"]] > detection_cfg["threshold"]
        )

        return artifact_df

    def _robust_zscore(self, series):
        median = series.median()
        mad = (series - median).abs().median()

        if mad == 0:
            zero_score = self.config["artifacts"]["robust_zscore"]["mad_zero_score"]
            return pd.Series(np.full(len(series), zero_score), index=series.index)

        return (series - median).abs() / mad

    def _combine_scores(self, scores_df, method):
        match method:
            case "max":
                return scores_df.max(axis=1)
            case "mean":
                return scores_df.mean(axis=1)
            case _:
                raise ValueError(f"Unsupported artifact combine method: {method}")
