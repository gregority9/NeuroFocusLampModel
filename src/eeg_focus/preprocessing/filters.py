from eeg_focus.io.config import ConfigLoader
from .mne_utils import MneUtils

class Filters:
    def __init__(self, config=None):
        self.config = config or ConfigLoader().get()
        self.mneUtils = MneUtils()

    def apply_filters(self, eeg_df):
        high_pass = self.config["filters"]["high_pass"]
        low_pass = self.config["filters"]["low_pass"]
        notch = self.config["filters"]["notch"]
        mne_verbose = self.config["filters"]["mne_verbose"]
        eeg_channels = self.config["data"]["channels"]
        sample_column = self.config["data"]["sample_column"]
        sampling_rate = self.config["data"]["sampling_rate"]

        eeg_signals = eeg_df[eeg_channels] * 1e-6

        ch_types = ["eeg"] * len(eeg_channels)
        eeg_signals = eeg_signals.T.to_numpy(copy=True)

        raw = self.mneUtils.create_mne(
            eeg_channels, 
            ch_types, 
            eeg_signals,
            sampling_rate,
            mne_verbose,
            )

        raw.notch_filter(freqs=notch, verbose=mne_verbose)
        raw.filter(l_freq=high_pass, h_freq=low_pass, verbose=mne_verbose)

        filtered = raw.get_data().T

        filtered_df = eeg_df[[sample_column]].copy()
        filtered_df[eeg_channels] = filtered

        return filtered_df






        
