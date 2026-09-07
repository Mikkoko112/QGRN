import yaml
from pathlib import Path

class ConfigParser:
    def __init__(self, load_from_file : Path):
        self.features = self.load_features(load_from_file)

    def load_features(self, load_from_file):
        #load yaml file
        with open(load_from_file) as f:
            config = yaml.safe_load(f)
        
        return config
    
    def save_yaml(self, path : Path):
        with open(path, "w") as f:
            yaml.safe_dump(self.features, f, sort_keys=False)

    def get_features(self):
        return self.features
    
    def get_value(self, feature_path : str, default=None):
        keys = feature_path.split(".")
        features = self.features
        for key in keys:
            if key in features:
                features = features[key]
            else:
                raise ValueError(
                    f"feature_path: '{feature_path}' is not valid, failed at key: '{key}'"
                )
        if features is None:
            return default
        return features
    
    def set_value(self, feature_path : str, value, create_missing: bool = False):
        keys = feature_path.split(".")
        features = self.features
        for key in keys[:-1]:
            if key in features:
                features = features[key]
                if not isinstance(features, dict):
                    raise ValueError(
                        f"feature_path: '{feature_path}' is not valid, key: '{key}' does not point to a nested mapping"
                    )
            elif create_missing:
                features[key] = {}
                features = features[key]
            else:
                raise ValueError(
                    f"feature_path: '{feature_path}' is not valid, failed at key: '{key}'"
                )
        features[keys[-1]] = value
