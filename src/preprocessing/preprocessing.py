import pandas as pd
import mne
import yaml
import numpy as np
from pathlib import Path

df = pd.read_csv("data/raw/Subject203/S203_R_ALL_DATA.csv")
with open("configs/preprocessing.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

sfreq = config["data"]["sampling_rate"]
eeg_channels = config["data"]["channels"]
accel_channels = config["data"]["accel"]
subjects = config["data"]["subjects"]
datatypes = config["data"]["data_types"]

label_map = {
    "R": 0,
    "TASK1": 1,
    "TASK2": 1,
}

all_channels = eeg_channels + accel_channels


for subject in subjects:
    for datatype in datatypes:
        df = pd.read_csv("data/raw/Subject" + str(subject) + "/S" + str(subject) + "_" + str(datatype) + "_ALL_DATA.csv")
        
        missing = [ch for ch in eeg_channels if ch not in df.columns]

        if missing:
            raise ValueError(f"Brakuje kanałów w CSV: {missing}")

        eeg = df[eeg_channels].copy()
        accel = df[accel_channels].copy()

        eeg = eeg.sub(eeg.mean(axis=1), axis=0)
        eeg = eeg - eeg.median()

        data_eeg = eeg.T.to_numpy(copy=True) * 1e-6
        data_accel = accel.T.to_numpy(copy=True)

        data = np.vstack([data_eeg, data_accel])

        ch_types = ["eeg"] * len(eeg_channels) + ["misc"] * len(accel_channels)

        info = mne.create_info(
            ch_names=all_channels,
            sfreq=sfreq,
            ch_types=ch_types
        )

        raw = mne.io.RawArray(data, info)

        

        
    

