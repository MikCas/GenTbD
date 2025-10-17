"""
GenTbD - Generalized Tracking-by-Detection

This is the main entry point for the tracking system.
It loads configuration from config.yaml and creates the tracking pipeline.

Learning Points:
1. Configuration-driven design - All settings in config.yaml
2. Registry pattern - Easy model/tracker switching
3. Separation of concerns - Config, creation, execution are separate

Usage:
    python -m src.main                    # Use config.yaml
    python -m src.main --config my.yaml  # Use custom config
"""

from detecting.model_registry import ModelRegistry
from tracking.tracker_registry import TrackerRegistry
from temporal.VideoProcessor import SimpleVideoProcessor

import logging
import yaml
from pathlib import Path
from typing import Dict, Any
import sys


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid

    Learning Point - YAML:
        YAML is a human-friendly data format.
        Like JSON, but easier to read/write.

        YAML features:
        - Comments with #
        - No quotes needed for strings
        - Indentation matters (like Python!)
        - Lists with - or []
        - Dictionaries with key: value
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Please create a config.yaml file or specify a valid path."
        )

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def setup_logger(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logger from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured logger instance

    Learning Point - Logging Levels:
        DEBUG: Detailed information, for diagnosing problems
        INFO: Confirmation that things are working
        WARNING: Something unexpected happened
        ERROR: More serious problem
        CRITICAL: Program may not be able to continue
    """
    log_config = config.get('logging', {})

    # Create logger
    logger = logging.getLogger('GenTbD')
    logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))

    # Console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(log_config.get(
        'format',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    log_file = log_config.get('log_file')
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def create_detector(config: Dict[str, Any], logger: logging.Logger):
    """
    Create detector from configuration.

    Args:
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Detector instance

    Learning Point - Factory Pattern:
        We don't create YOLOv7ONNX or YOLOv8PyTorch directly.
        We ask ModelRegistry to create it based on config.

        Benefits:
        1. Change model by editing config.yaml
        2. No code changes needed
        3. Easy to experiment
    """
    detector_config = config.get('detector', {})

    model_name = detector_config.get('model', 'yolov8n')
    confidence = detector_config.get('confidence_threshold', 0.25)
    iou = detector_config.get('iou_threshold', 0.45)
    classes = detector_config.get('classes')
    device = detector_config.get('device')

    logger.info("="*60)
    logger.info("CREATING DETECTOR")
    logger.info("="*60)

    detector = ModelRegistry.create(
        model_name=model_name,
        confidence_threshold=confidence,
        iou_threshold=iou,
        classes=classes,
        device=device,
        logger=logger
    )

    return detector


def create_tracker(config: Dict[str, Any], logger: logging.Logger):
    """
    Create tracker from configuration.

    Args:
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Tracker instance

    Learning Point:
        Same factory pattern as detector.
        TrackerRegistry handles creating the right tracker type.
    """
    tracker_config = config.get('tracker', {})

    tracker_name = tracker_config.get('name', 'simple')
    max_age = tracker_config.get('max_age', 30)
    min_hits = tracker_config.get('min_hits', 3)
    iou_threshold = tracker_config.get('iou_threshold', 0.3)

    logger.info("="*60)
    logger.info("CREATING TRACKER")
    logger.info("="*60)

    tracker = TrackerRegistry.create(
        tracker_name=tracker_name,
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        logger=logger
    )

    return tracker


def main(config_path: str = 'config.yaml'):
    """
    Main entry point.

    Args:
        config_path: Path to configuration file

    Learning Point - High-Level Flow:
        1. Load config (what to do)
        2. Setup logger (how to report)
        3. Create detector (how to detect)
        4. Create tracker (how to track)
        5. Create video processor (orchestrates everything)
        6. Process video (do the work!)

        This is called "Dependency Injection":
        - VideoProcessor doesn't create its own detector/tracker
        - We create them and "inject" them
        - Makes testing and swapping implementations easy
    """
    # 1. Load configuration
    print("Loading configuration...")
    config = load_config(config_path)

    # 2. Setup logger
    logger = setup_logger(config)
    logger.info("GenTbD - Generalized Tracking-by-Detection")
    logger.info("="*60)

    # 3. Create detector
    detector = create_detector(config, logger)

    # 4. Create tracker
    tracker = create_tracker(config, logger)

    # 5. Setup video processor
    video_config = config.get('video', {})
    video_path = video_config.get('input', 'data/TownCent.mp4')

    # Note: SimpleTracker uses different parameters than the new registry pattern
    # For now, we'll create it with the old parameters
    # TODO: Update SimpleTracker to use the standard parameters

    # Instead of using TrackerRegistry, create SimpleTracker directly for now
    # This is because SimpleTracker has different parameters
    from tracking.trackers.SimpleTracker import SimpleTracker

    # Map new parameters to old SimpleTracker parameters
    tracker_config = config.get('tracker', {})
    tracker = SimpleTracker(
        detection_threshold=0.3,  # From old main.py
        creation_threshold=0.3,   # From old main.py
        activation_threshold=10,  # From old main.py
        deactivation_threshold=15,  # From old main.py
        match_thresholds=[0.4, 0.2, 0.2],  # From old main.py
        logger=logger
    )

    logger.info("="*60)
    logger.info("SETTING UP VIDEO PROCESSOR")
    logger.info("="*60)
    logger.info(f"Video path: {video_path}")

    # Get visualization config
    viz_config = config.get('visualization', {})
    draw_mode = 'state'  # Default from old main.py

    video_processor = SimpleVideoProcessor(
        video_path=video_path,
        detector=detector,
        tracker=tracker,
        draw_mode=draw_mode,
        logger=logger
    )

    # 6. Process video
    logger.info("="*60)
    logger.info("STARTING VIDEO PROCESSING")
    logger.info("="*60)

    try:
        video_processor.process()
        logger.info("="*60)
        logger.info("PROCESSING COMPLETE!")
        logger.info("="*60)
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    """
    Entry point when running as a script.

    Learning Point - Command Line Arguments:
        sys.argv is a list of command line arguments:
        - sys.argv[0] = script name
        - sys.argv[1] = first argument
        - sys.argv[2] = second argument, etc.

        Example:
            python -m src.main                     # sys.argv = ['src/main.py']
            python -m src.main --config my.yaml   # sys.argv = ['src/main.py', '--config', 'my.yaml']
    """
    # Simple argument parsing
    # (For more complex args, use argparse module)
    config_path = 'config.yaml'

    if len(sys.argv) > 1:
        if sys.argv[1] == '--config' and len(sys.argv) > 2:
            config_path = sys.argv[2]
        elif sys.argv[1] in ['-h', '--help']:
            print(__doc__)
            sys.exit(0)

    main(config_path)
