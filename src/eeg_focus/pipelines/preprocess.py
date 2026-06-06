import json
from pathlib import Path

import pandas as pd

from eeg_focus.io.config import ConfigLoader
from eeg_focus.io.loaders import FileLoader
from eeg_focus.io.validators import Validator
from eeg_focus.preprocessing.accelerometer import Accelerometer
from eeg_focus.preprocessing.artifacts import Artifacts
from eeg_focus.preprocessing.filters import Filters
from eeg_focus.preprocessing.modalities import Modalities
from eeg_focus.preprocessing.qc import QualityControl
from eeg_focus.preprocessing.referencing import Referencing


class PreprocessingPipeline:
    def __init__(self, config_path="configs/preprocessing.yaml"):
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.get()

        self.file_loader = FileLoader()
        self.validator = Validator(self.config)
        self.modalities = Modalities(self.config)
        self.referencing = Referencing(self.config)
        self.filters = Filters(self.config)
        self.accelerometer = Accelerometer(self.config)
        self.artifacts = Artifacts(self.config)
        self.quality_control = QualityControl(self.config)

    def run(self):
        subjects = self.config["data"]["subjects"]
        datatypes = self.config["data"]["data_types"]
        sample_column = self.config["data"]["sample_column"]
        label_map = self.config["labels"]

        output_dir = Path(self.config["output"]["base_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        processed_frames = []
        reports = []
        failures = []

        for subject in subjects:
            for datatype in datatypes:
                try:
                    result_df, report = self._process_one(
                        subject,
                        datatype,
                        label_map[datatype],
                        sample_column,
                    )
                    processed_frames.append(result_df)
                    reports.append(report)
                except Exception as exc:
                    failures.append({
                        "subject": subject,
                        "datatype": datatype,
                        "error": str(exc),
                    })

        if not processed_frames:
            raise RuntimeError(
                "Preprocessing did not produce any rows. "
                f"Failures: {failures}"
            )

        combined_df = pd.concat(processed_frames, ignore_index=True)

        combined_path = output_dir / self.config["output"]["combined_filename"]
        manifest_path = output_dir / self.config["output"]["manifest_filename"]

        combined_df.to_csv(combined_path, index=False)

        manifest = {
            "combined_path": str(combined_path),
            "manifest_path": str(manifest_path),
            "n_rows": int(len(combined_df)),
            "n_subjects": int(combined_df["subject"].nunique()),
            "data_types": sorted(combined_df["datatype"].unique().tolist()),
            "reports": reports,
            "failures": failures,
        }

        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        return manifest

    def _process_one(self, subject, datatype, label, sample_column):
        df = self.file_loader.get_file(subject, datatype)
        self.validator.ValidateData(df, datatype)

        df_eeg, df_accel = self.modalities.separate_eeg_accelerometer(df)

        referenced_eeg_df = self.referencing.applyReferencing(df_eeg)
        filtered_eeg_df = self.filters.apply_filters(referenced_eeg_df)
        processed_accel_df = self.accelerometer.run(df_accel)
        artifact_df = self.artifacts.run(filtered_eeg_df, processed_accel_df)

        result_df = filtered_eeg_df.merge(
            processed_accel_df,
            on=sample_column,
            how="left",
        ).merge(
            artifact_df,
            on=[sample_column, "motion_score", "is_motion_artifact"],
            how="left",
        )

        result_df.insert(0, "subject", subject)
        result_df.insert(1, "datatype", datatype)
        result_df.insert(2, "label", label)

        report = self.quality_control.run(result_df, subject, datatype)

        return result_df, report
