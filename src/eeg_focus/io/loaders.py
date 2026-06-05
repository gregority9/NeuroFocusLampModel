import pandas as pd

class FileLoader:
    def __init__(self, base_path="data/raw"):
        self.base_path = base_path

    def get_file(self, subject, datatype):
        file_path = (
            self.base_path
            + f"/Subject{subject}"
            + f"/S{subject}_{datatype}_ALL_DATA.csv"
        ) 
        
        
        return pd.read_csv(file_path)