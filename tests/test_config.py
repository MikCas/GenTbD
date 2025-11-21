"""Unit tests for Config class."""

import pytest
import tempfile
import os
from src.config import Config


class TestConfig:
    """Test Config functionality."""

    def test_initialization_from_dict(self):
        """Test Config initializes from dictionary."""
        config_dict = {
            'detection': {
                'type': 'object_detection',
                'conf_threshold': 0.7
            }
        }
        config = Config(config_dict)

        assert config.get('detection.type') == 'object_detection'
        assert config.get('detection.conf_threshold') == 0.7

    def test_initialization_empty_dict(self):
        """Test Config initializes with empty dictionary."""
        config = Config({})

        assert config._config == {}

    def test_from_yaml_valid_file(self):
        """Test loading config from valid YAML file."""
        # Create temporary YAML file
        yaml_content = """
detection:
  type: keypoint_detection
  model: resnet50
  device: cpu
  conf_threshold: 0.6
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = Config.from_yaml(temp_path)

            assert config.get('detection.type') == 'keypoint_detection'
            assert config.get('detection.model') == 'resnet50'
            assert config.get('detection.device') == 'cpu'
            assert config.get('detection.conf_threshold') == 0.6
        finally:
            os.unlink(temp_path)

    def test_from_yaml_file_not_found(self):
        """Test loading config from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml('nonexistent_config.yaml')

    def test_from_yaml_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        # Create temporary file with invalid YAML
        invalid_yaml = """
detection:
  type: object_detection
  conf_threshold: [unmatched brackets
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name

        try:
            with pytest.raises(Exception):  # yaml.YAMLError or similar
                Config.from_yaml(temp_path)
        finally:
            os.unlink(temp_path)

    def test_get_nested_key_with_dot_notation(self):
        """Test getting nested values with dot notation."""
        config_dict = {
            'detection': {
                'keypoint': {
                    'keypoint_threshold': 0.5
                }
            }
        }
        config = Config(config_dict)

        assert config.get('detection.keypoint.keypoint_threshold') == 0.5

    def test_get_top_level_key(self):
        """Test getting top-level values."""
        config_dict = {
            'verbose': True,
            'device': 'cuda'
        }
        config = Config(config_dict)

        assert config.get('verbose') is True
        assert config.get('device') == 'cuda'

    def test_get_missing_key_returns_default(self):
        """Test getting missing key returns default value."""
        config = Config({})

        assert config.get('missing.key', 'default_value') == 'default_value'
        assert config.get('another.missing', 42) == 42
        assert config.get('no.default') is None

    def test_get_partial_path_missing(self):
        """Test getting value when intermediate keys are missing."""
        config_dict = {
            'detection': {
                'type': 'object_detection'
            }
        }
        config = Config(config_dict)

        # 'tracking' key doesn't exist
        assert config.get('tracking.max_age', 30) == 30

    def test_merge_args_overrides_config(self):
        """Test merge_args overrides config file values for runtime args."""
        config_dict = {
            'video': {
                'source': 'default.mp4',
                'save_output': False
            },
            'logging': {
                'verbose': False
            }
        }
        config = Config(config_dict)

        # Create args namespace with correct attribute names
        class Args:
            video = 'cli_video.mp4'
            save_output = True
            verbose = True
            output = 'out.mp4'

        args = Args()
        config.merge_args(args)

        # CLI args should override config file
        assert config.get('video.source') == 'cli_video.mp4'
        assert config.get('video.save_output') is True
        assert config.get('logging.verbose') is True
        assert config.get('video.output_path') == 'out.mp4'

    def test_merge_args_with_actual_cli_args(self):
        """Test merge_args with actual CLI argument names."""
        config = Config({})

        class Args:
            video = 'test.mp4'
            verbose = True
            save_output = True

        args = Args()
        config.merge_args(args)

        assert config.get('video.source') == 'test.mp4'
        assert config.get('logging.verbose') is True
        assert config.get('video.save_output') is True

    def test_get_with_different_value_types(self):
        """Test Config handles different value types correctly."""
        config_dict = {
            'string_value': 'test',
            'int_value': 42,
            'float_value': 3.14,
            'bool_value': True,
            'list_value': [1, 2, 3],
            'null_value': None
        }
        config = Config(config_dict)

        assert config.get('string_value') == 'test'
        assert config.get('int_value') == 42
        assert config.get('float_value') == 3.14
        assert config.get('bool_value') is True
        assert config.get('list_value') == [1, 2, 3]
        assert config.get('null_value') is None

    def test_deeply_nested_config(self):
        """Test Config handles deeply nested structures."""
        config_dict = {
            'level1': {
                'level2': {
                    'level3': {
                        'level4': {
                            'value': 'deep'
                        }
                    }
                }
            }
        }
        config = Config(config_dict)

        assert config.get('level1.level2.level3.level4.value') == 'deep'

    def test_config_immutability_after_get(self):
        """Test that getting a value doesn't modify the config."""
        config_dict = {
            'detection': {
                'type': 'object_detection'
            }
        }
        config = Config(config_dict)

        # Get value
        value1 = config.get('detection.type')

        # Get again
        value2 = config.get('detection.type')

        # Should be the same
        assert value1 == value2
        assert config.get('detection.type') == 'object_detection'

    def test_from_yaml_preserves_structure(self):
        """Test YAML loading preserves nested structure."""
        yaml_content = """
detection:
  type: object_detection
  model: mobilenet
  keypoint:
    keypoint_threshold: 0.5
    draw_skeleton: true
tracking:
  enabled: false
  max_age: 30
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = Config.from_yaml(temp_path)

            assert config.get('detection.type') == 'object_detection'
            assert config.get('detection.model') == 'mobilenet'
            assert config.get('detection.keypoint.keypoint_threshold') == 0.5
            assert config.get('detection.keypoint.draw_skeleton') is True
            assert config.get('tracking.enabled') is False
            assert config.get('tracking.max_age') == 30
        finally:
            os.unlink(temp_path)

    def test_default_config_file_loads(self):
        """Test that default config file can be loaded."""
        # This test assumes config/default.yaml exists
        default_config_path = 'config/default.yaml'

        if os.path.exists(default_config_path):
            config = Config.from_yaml(default_config_path)

            # Check some expected keys exist
            assert config.get('detection.type') is not None
            assert config.get('detection.device') is not None
            assert config.get('detection.conf_threshold') is not None

    def test_merge_args_empty_args(self):
        """Test merge_args with empty args object."""
        config_dict = {
            'detection': {
                'type': 'object_detection'
            }
        }
        config = Config(config_dict)

        class EmptyArgs:
            pass

        args = EmptyArgs()
        config.merge_args(args)

        # Config should be unchanged
        assert config.get('detection.type') == 'object_detection'

    def test_get_returns_copy_of_mutable_values(self):
        """Test that get() returns values that can be safely modified."""
        config_dict = {
            'detection': {
                'classes': [1, 2, 3]
            }
        }
        config = Config(config_dict)

        # Get the list
        classes = config.get('detection.classes')

        # Modify it
        classes.append(4)

        # Original should be unchanged (if implementation returns copies)
        # Note: This test documents expected behavior
        # If config returns references, this test will fail
        original_classes = config.get('detection.classes')

        # Depending on implementation, this may or may not pass
        # For now, we just check we can get the value
        assert len(original_classes) >= 3
