from eeg_focus.io.config import ConfigLoader


class Referencing:
    def __init__(self, config=None):
        self.config = config or ConfigLoader().get()

    def applyReferencing(self, eeg_df):
        method = self.config["reference"]["method"]
        eeg_channels = self.config["data"]["channels"]
        sample_column = self.config["data"]["sample_column"]

        eeg_signals = eeg_df[eeg_channels]

        match method:
            case "none":
                referenced = eeg_signals.copy()
            case "common_average":
                row_mean = eeg_signals.mean(axis=1)
                referenced = eeg_signals.sub(row_mean, axis=0)
            case _:
                raise ValueError(f"Unsupported reference method: {method}")

        eeg_referenced = eeg_df[[sample_column]].copy()
        eeg_referenced[eeg_channels] = referenced

        return eeg_referenced
