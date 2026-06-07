from src.eeg_focus.io.config import ConfigLoader

class Modalities:
    def __init__(self, config=None):
        self.config = config or ConfigLoader().get()

    def separate_eeg_accelerometer(self, df):
        sample_column = self.config["data"]["sample_column"]

        eeg_channels = self.config["data"]["channels"] + [sample_column]
        accel_channels = self.config["data"]["accel"] + [sample_column]

        df_eeg = df[eeg_channels]
        df_accel = df[accel_channels]

        return df_eeg, df_accel
        

        
