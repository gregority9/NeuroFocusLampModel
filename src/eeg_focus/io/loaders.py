from pathlib import Path

import pandas as pd

class FileLoader:
    def __init__(self, config=None, base_path=None):
        self.config = config or {}
        data_config = self.config.get("data", {})
        self.base_path = Path(base_path or data_config.get("raw_base_path", "data/raw"))
        self.file_pattern = data_config.get(
            "raw_file_pattern",
            "Subject{subject}/S{subject}_{datatype}_ALL_DATA.csv",
        )

    def get_file(self, subject, datatype):
        relative_path = self.file_pattern.format(subject=subject, datatype=datatype)
        file_path = self.base_path / relative_path

        return pd.read_csv(file_path)
