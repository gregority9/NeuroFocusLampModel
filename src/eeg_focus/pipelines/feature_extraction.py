from pathlib import Path
import pandas as pd
import numpy as np

from eeg_focus.features.bandpower import BandpowerExtractor


class FeatureExtractionPipeline:
    def __init__(self, config):
        self.config = config
        self.sfreq = self.config["data"]["sampling_rate"]
        self.channels = self.config["data"]["channels"]
        self.bandpower = BandpowerExtractor(self.sfreq)
        self.window_size_sec = 2.0 
        self.window_samples = int(self.window_size_sec * self.sfreq)

    def run(self, preprocessed_df_path):
        print(f"Loading preprocessed data from {preprocessed_df_path}...")
        df = pd.read_csv(preprocessed_df_path)
        
        grouped = df.groupby(["subject", "datatype"])
        
        epoch_records = []
        
        for (subject, datatype), group_df in grouped:
            sample_col = self.config["data"].get("sample_column", "sample_number")
            if sample_col in group_df.columns:
                group_df = group_df.sort_values(sample_col)
            
            n_samples = len(group_df)
            n_epochs = n_samples // self.window_samples
            
            for i in range(n_epochs):
                start_idx = i * self.window_samples
                end_idx = start_idx + self.window_samples
                
                window_df = group_df.iloc[start_idx:end_idx]
                
                subject_id = f"Subject{subject}"
                session_id = "ses-001" 
                
                group_name = "ADHD" if str(subject).startswith("2") else "control"
                task = datatype
                label = window_df["label"].iloc[0]
                
                artifact_threshold = self.config.get("artifacts", {}).get("detection", {}).get("threshold", 5.0)
                artifact_score = float(window_df["artifact_score"].mean())
                is_rejected = artifact_score > artifact_threshold
                
                features = self.bandpower.extract_window(window_df, self.channels)
                
                record = {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "group": group_name,
                    "task": task,
                    "label": int(label),
                    "artifact_score": artifact_score,
                    "is_rejected": is_rejected
                }
                
                # append extracted features
                record.update(features)
                epoch_records.append(record)
                
        features_df = pd.DataFrame(epoch_records)
        
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "features_bandpower.csv"
        
        features_df.to_csv(output_path, index=False)
        print(f"Feature extraction completed successfully! Saved {len(features_df)} windows to {output_path}")
        
        return {
            "features_path": str(output_path),
            "n_windows": len(features_df),
            "n_rejected": features_df["is_rejected"].sum(),
            "n_accepted": (~features_df["is_rejected"]).sum(),
            "features": list(self.bandpower.bands.keys())
        }