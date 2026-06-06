from eeg_focus.io.config import ConfigLoader

import numpy as np
import pandas as pd


class QualityControl:
    def __init__(self, config=None):
        self.config = config or ConfigLoader().get()

    def run(self, df, subject=None, datatype=None):
        eeg_channels = self.config["data"]["channels"]
        accel_channels = self.config["data"]["accel"]

        report = {
            "subject": subject,
            "datatype": datatype,
            "n_samples": int(len(df)),
            "eeg_channel_stats": self._channel_stats(df, eeg_channels),
            "accel_channel_stats": self._channel_stats(df, accel_channels),
            "bad_channel_candidates": {},
            "motion_summary": {},
            "warnings": [],
        }

        report["bad_channel_candidates"] = self._detect_bad_channel_candidates(
            report["eeg_channel_stats"]
        )

        if self._has_motion_columns(df):
            report["motion_summary"] = self._motion_summary(df)

        self._add_warnings(report)

        return report

    def _channel_stats(self, df, channels):
        stats = {}

        for channel in channels:
            if channel not in df.columns:
                continue

            series = pd.to_numeric(df[channel], errors="coerce")
            values = series.dropna()

            if len(values) == 0:
                stats[channel] = {
                    "valid_samples": 0,
                    "nan_count": int(series.isna().sum()),
                    "inf_count": 0,
                    "status": "empty",
                }
                continue

            values_np = values.to_numpy()

            inf_count = int(np.isinf(values_np).sum())
            finite_values = values_np[np.isfinite(values_np)]

            if len(finite_values) == 0:
                stats[channel] = {
                    "valid_samples": 0,
                    "nan_count": int(series.isna().sum()),
                    "inf_count": inf_count,
                    "status": "non_finite",
                }
                continue

            stats[channel] = {
                "valid_samples": int(len(finite_values)),
                "nan_count": int(series.isna().sum()),
                "inf_count": inf_count,
                "mean": float(np.mean(finite_values)),
                "std": float(np.std(finite_values)),
                "median": float(np.median(finite_values)),
                "min": float(np.min(finite_values)),
                "max": float(np.max(finite_values)),
                "peak_to_peak": float(np.max(finite_values) - np.min(finite_values)),
                "rms": float(np.sqrt(np.mean(finite_values ** 2))),
                "variance": float(np.var(finite_values)),
                "status": "ok",
            }

        return stats

    def _detect_bad_channel_candidates(self, channel_stats):
        if not channel_stats:
            return {
                "flat_channels": [],
                "low_variance_channels": [],
                "high_variance_channels": [],
                "high_nan_channels": [],
                "non_finite_channels": [],
            }

        variances = {
            channel: stats["variance"]
            for channel, stats in channel_stats.items()
            if stats.get("status") == "ok"
        }

        if len(variances) == 0:
            return {
                "flat_channels": [],
                "low_variance_channels": [],
                "high_variance_channels": [],
                "high_nan_channels": [],
                "non_finite_channels": list(channel_stats.keys()),
            }

        variance_values = np.array(list(variances.values()))
        median_variance = float(np.median(variance_values))

        low_variance_threshold = median_variance * 0.05
        high_variance_threshold = median_variance * 20.0

        flat_channels = []
        low_variance_channels = []
        high_variance_channels = []
        high_nan_channels = []
        non_finite_channels = []

        for channel, stats in channel_stats.items():
            if stats.get("status") != "ok":
                non_finite_channels.append(channel)
                continue

            variance = stats["variance"]
            nan_count = stats["nan_count"]
            valid_samples = stats["valid_samples"]

            if variance == 0:
                flat_channels.append(channel)

            if variance < low_variance_threshold:
                low_variance_channels.append(channel)

            if variance > high_variance_threshold:
                high_variance_channels.append(channel)

            if valid_samples > 0:
                nan_ratio = nan_count / (nan_count + valid_samples)
                if nan_ratio > 0.01:
                    high_nan_channels.append(channel)

        return {
            "flat_channels": flat_channels,
            "low_variance_channels": low_variance_channels,
            "high_variance_channels": high_variance_channels,
            "high_nan_channels": high_nan_channels,
            "non_finite_channels": non_finite_channels,
            "median_variance": median_variance,
            "low_variance_threshold": low_variance_threshold,
            "high_variance_threshold": high_variance_threshold,
        }

    def _has_motion_columns(self, df):
        return (
            "motion_score" in df.columns
            and "is_motion_artifact" in df.columns
        )

    def _motion_summary(self, df):
        motion_score = pd.to_numeric(df["motion_score"], errors="coerce").dropna()

        if len(motion_score) == 0:
            return {
                "status": "empty_motion_score",
            }

        is_motion_artifact = df["is_motion_artifact"].fillna(False).astype(bool)

        return {
            "motion_score_mean": float(motion_score.mean()),
            "motion_score_std": float(motion_score.std()),
            "motion_score_median": float(motion_score.median()),
            "motion_score_min": float(motion_score.min()),
            "motion_score_max": float(motion_score.max()),
            "motion_artifact_samples": int(is_motion_artifact.sum()),
            "motion_artifact_percentage": float(is_motion_artifact.mean() * 100),
        }

    def _add_warnings(self, report):
        bad = report["bad_channel_candidates"]

        if bad.get("flat_channels"):
            report["warnings"].append({
                "type": "flat_channels",
                "channels": bad["flat_channels"],
            })

        if bad.get("low_variance_channels"):
            report["warnings"].append({
                "type": "low_variance_channels",
                "channels": bad["low_variance_channels"],
            })

        if bad.get("high_variance_channels"):
            report["warnings"].append({
                "type": "high_variance_channels",
                "channels": bad["high_variance_channels"],
            })

        if bad.get("high_nan_channels"):
            report["warnings"].append({
                "type": "high_nan_channels",
                "channels": bad["high_nan_channels"],
            })

        if bad.get("non_finite_channels"):
            report["warnings"].append({
                "type": "non_finite_channels",
                "channels": bad["non_finite_channels"],
            })

        motion = report.get("motion_summary", {})

        if motion.get("motion_artifact_percentage", 0) > 10:
            report["warnings"].append({
                "type": "high_motion_artifact_percentage",
                "percentage": motion["motion_artifact_percentage"],
            })
