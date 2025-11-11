"""Configuration management for GenTbD.

This module handles loading and merging YAML configuration files with
command-line arguments.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration container for GenTbD settings."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary.

        Args:
            config_dict: Dictionary of configuration values
        """
        self._config = config_dict

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML file is invalid
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(config_dict)

    @classmethod
    def from_default(cls) -> 'Config':
        """Load default configuration.

        Returns:
            Config instance with default settings
        """
        default_path = Path(__file__).parent.parent / "config" / "default.yaml"
        return cls.from_yaml(str(default_path))

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path (e.g., 'detection.model')
            default: Default value if key doesn't exist

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation.

        Args:
            key_path: Dot-separated path (e.g., 'detection.model')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config

        # Navigate to the parent dict
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set the value
        config[keys[-1]] = value

    def merge_args(self, args):
        """Merge command-line arguments into config.

        Command-line arguments take precedence over config file values.

        Args:
            args: argparse.Namespace with command-line arguments
        """
        # Video source
        if hasattr(args, 'webcam') and args.webcam is not None:
            self.set('video.source', args.webcam)
        elif hasattr(args, 'video') and args.video:
            self.set('video.source', args.video)

        # Video processing
        if hasattr(args, 'max_dimension') and args.max_dimension:
            self.set('video.max_dimension', args.max_dimension)
        if hasattr(args, 'skip_frames') and args.skip_frames:
            self.set('video.skip_frames', args.skip_frames)
        if hasattr(args, 'save_output') and args.save_output:
            self.set('video.save_output', args.save_output)
        if hasattr(args, 'output') and args.output:
            self.set('video.output_path', args.output)

        # Detection
        if hasattr(args, 'detector_type') and args.detector_type:
            self.set('detection.type', args.detector_type)
        if hasattr(args, 'model') and args.model:
            self.set('detection.model', args.model)
        if hasattr(args, 'device') and args.device:
            self.set('detection.device', args.device)
        if hasattr(args, 'conf') and args.conf:
            self.set('detection.conf_threshold', args.conf)
        if hasattr(args, 'classes') and args.classes:
            self.set('detection.classes', args.classes)

        # Logging
        if hasattr(args, 'verbose') and args.verbose:
            self.set('logging.verbose', args.verbose)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation of config
        """
        return self._config.copy()

    def __repr__(self) -> str:
        """String representation of config."""
        return f"Config({self._config})"
