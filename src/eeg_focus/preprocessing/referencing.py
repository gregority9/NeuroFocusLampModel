from src.eeg_focus.io.config import ConfigLoader


class Referencing:
    def __init__(self, config=None):
        self.config = config or ConfigLoader().get()

    def applyReferencing(self, eeg_df, bad_channels=None):
        return self.apply_referencing(eeg_df, bad_channels)

    def apply_referencing(self, eeg_df, bad_channels=None):
        method = self.config["reference"]["method"]
        eeg_channels = self.config["data"]["channels"]
        sample_column = self.config["data"]["sample_column"]

        eeg_signals = eeg_df[eeg_channels]
        bad_channels = set(bad_channels or [])
        good_channels = [ch for ch in eeg_channels if ch not in bad_channels]

        match method:
            case "none":
                referenced = eeg_signals.copy()
            case "common_average":
                row_mean = eeg_signals.mean(axis=1)
                referenced = eeg_signals.sub(row_mean, axis=0)
            case "average_good_channels":
                if not good_channels:
                    raise ValueError("Cannot apply average_good_channels without good channels.")
                row_mean = eeg_signals[good_channels].mean(axis=1)
                referenced = eeg_signals.sub(row_mean, axis=0)
            case _:
                raise ValueError(f"Unsupported reference method: {method}")

        eeg_referenced = eeg_df[[sample_column]].copy()
        eeg_referenced[eeg_channels] = referenced

        return eeg_referenced
