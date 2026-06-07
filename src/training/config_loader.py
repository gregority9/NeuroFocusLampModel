from pathlib import Path

import yaml


class ConfigLoader:
    """Load YAML configs with optional inheritance through an extends key."""

    def load(self, config_path):
        return self._load_path(Path(config_path), [])

    def _load_path(self, config_path, stack):
        config_path = config_path.expanduser()

        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path

        config_path = config_path.resolve()

        if config_path in stack:
            cycle = stack + [config_path]
            cycle_text = " -> ".join(str(path) for path in cycle)
            raise ValueError("Config inheritance cycle detected: " + cycle_text)

        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        parent_path = config.pop("extends", None)
        if parent_path is None:
            return config

        parent_path = self._resolve_parent_path(parent_path, config_path.parent)
        parent_config = self._load_path(parent_path, stack + [config_path])

        return self._deep_merge(parent_config, config)

    def _resolve_parent_path(self, parent_path, child_dir):
        parent_path = Path(parent_path).expanduser()

        if parent_path.is_absolute():
            return parent_path

        if parent_path.exists():
            return parent_path

        return child_dir / parent_path

    def _deep_merge(self, base, override):
        merged = base.copy()

        for key, value in override.items():
            base_value = merged.get(key)

            if isinstance(base_value, dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(base_value, value)
            else:
                merged[key] = value

        return merged


def load_config(config_path):
    loader = ConfigLoader()
    return loader.load(config_path)
