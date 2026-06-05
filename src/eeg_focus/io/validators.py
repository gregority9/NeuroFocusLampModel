from .config import ConfigLoader

import pandas as pd

class Validator:
    def __init__(self, config=None):
        self.config = config or ConfigLoader().get()

    def ValidateData(self, df, datatype):
        eeg_channels = self.config["data"]["channels"]
        accel_channels = self.config["data"]["accel"]
        sample_column = self.config["data"]["sample_column"]

        all_channels = eeg_channels + accel_channels + [sample_column]

        missing = [ch for ch in all_channels if ch not in df.columns]

        if missing:
            raise ValueError(f"There are missing channels in CSV: {missing}")
        
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        delta = max(df["timestamp_utc"]) - min(df["timestamp_utc"])

        expected = self.config["validation"]["time"]["expected"][datatype]

        duration_cfg = self.config["validation"]["time"]["tolerance"][datatype]
        ok_min, ok_max = duration_cfg["ok_min"], duration_cfg["ok_max"]
        warning_min, warning_max = duration_cfg["warning_min"], duration_cfg["warning_max"]

        if ok_min <= delta.total_seconds() <= ok_max:
            print("Time OK")
        elif warning_min <= delta.total_seconds() <= warning_max:
            print("Time warning")
        else:
            raise ValueError(f"Wrong time in dataframe, expected: {expected}, actual: {delta.total_seconds()}")

        n_samples = len(df)

        samples_per_second = n_samples / delta.total_seconds()

        samples_cfg = self.config["validation"]["samples"]["tolerance"]
        samples_ok_min, samples_ok_max = samples_cfg["ok_min"], samples_cfg["ok_max"]
        
        if samples_ok_min <= samples_per_second <= samples_ok_max:
            print("Samples OK")
        else:
            expected_samples = self.config["validation"]["samples"]["expected"]
            raise ValueError(
                "Wrong number of samples per second, "
                f"expected: {expected_samples}, actual: {samples_per_second}"
            )
