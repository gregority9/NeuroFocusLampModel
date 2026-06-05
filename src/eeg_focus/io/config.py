import yaml

class ConfigLoader:
    def __init__(self, path="configs/preprocessing.yaml"):
        self.config = self.open(path)

    def open(self, path):
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
        
    def get(self):
        return self.config
