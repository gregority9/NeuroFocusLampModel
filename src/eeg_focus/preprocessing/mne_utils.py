import mne

class MneUtils:
    def create_mne(self, ch_names, ch_types, df, sfreq=250, verbose=None):
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=ch_types
        )

        return mne.io.RawArray(df, info, verbose=verbose)
