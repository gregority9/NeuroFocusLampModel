import numpy as np
from scipy.signal import welch

class BandpowerExtractor:
    def __init__(self, sampling_rate, bands=None):
        self.sfreq = sampling_rate
        if bands is None:
            self.bands = {
                "delta": (1.0, 4.0),
                "theta": (4.0, 8.0),
                "alpha": (8.0, 12.0),
                "beta": (12.0, 30.0)
            }
        else:
            self.bands = bands

    def extract_window(self, window_data, channels):
        """
        Extracts bandpower features for all requested channels in a given window.
        
        Args:
            window_data: pd.DataFrame containing the epoch data samples.
                         Must have shape (n_samples, n_channels).
            channels: list of strings, names of the EEG channels to process.
        
        Returns:
            dict of { "band_channel": power_value }
        """
        features = {}
        for ch in channels:
            sig = window_data[ch].values
            
            freqs, psd = welch(sig, self.sfreq, nperseg=len(sig))
            
            for band_name, (low, high) in self.bands.items():
                idx_band = np.logical_and(freqs >= low, freqs <= high)
                
                # integrade psd
                if np.sum(idx_band) > 0:
                    try:
                        band_power = np.trapezoid(psd[idx_band], freqs[idx_band])
                    except AttributeError:
                        band_power = np.trapz(psd[idx_band], freqs[idx_band]) 
                else:
                    band_power = 0.0
                    
                features[f"{band_name}_{ch}"] = float(band_power)
                
        return features