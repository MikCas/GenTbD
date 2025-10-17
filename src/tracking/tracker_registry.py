"""
Tracker Registry for easy tracker creation.

This module provides a factory pattern for creating trackers from configuration.
Similar to ModelRegistry, but for tracking algorithms.

Why?
- Switch between tracking algorithms easily
- Experiment with different trackers via config
- Add new trackers without modifying main code

Learning Point:
    Different trackers have different strengths:
    - SimpleTracker: Fast, good for simple scenes
    - DeepSORT: Better for crowded scenes (uses appearance features)
    - ByteTrack: State-of-the-art, handles occlusions well
"""

from typing import Optional, Dict, Any
import logging

from tracking.trackers.Tracker import Tracker
from tracking.trackers.SimpleTracker import SimpleTracker


class TrackerRegistry:
    """
    Factory for creating trackers from tracker names.

    Example:
        >>> tracker = TrackerRegistry.create('simple', max_age=30)
        >>> trackers = TrackerRegistry.list_trackers()
    """

    # Tracker definitions
    _TRACKERS = {
        'simple': {
            'class': SimpleTracker,
            'description': 'Simple IOU-based tracker (fast, good for simple scenes)',
            'default_max_age': 30,
            'default_min_hits': 3,
            'default_iou_threshold': 0.3,
        },
        # Future trackers can be added here:
        # 'deepsort': {
        #     'class': DeepSORTTracker,
        #     'description': 'DeepSORT with appearance features',
        # },
        # 'bytetrack': {
        #     'class': ByteTrackTracker,
        #     'description': 'ByteTrack (state-of-the-art)',
        # },
    }

    @classmethod
    def create(cls,
               tracker_name: str,
               max_age: Optional[int] = None,
               min_hits: Optional[int] = None,
               iou_threshold: Optional[float] = None,
               logger: Optional[logging.Logger] = None) -> Tracker:
        """
        Create a tracker from tracker name.

        Args:
            tracker_name: Tracker identifier (e.g., 'simple', 'deepsort')
            max_age: Maximum frames to keep track without detections
            min_hits: Minimum detections before track is confirmed
            iou_threshold: IoU threshold for matching detections to tracks
            logger: Logger instance

        Returns:
            Tracker instance ready to use

        Raises:
            ValueError: If tracker_name not found

        Learning Point - Tracker Parameters:
            max_age: How long to remember lost tracks
                - Too low: Tracks lost during brief occlusions
                - Too high: Ghost tracks persist too long
                - Good range: 20-50 frames

            min_hits: Anti-noise filter
                - Too low: False positives become tracks
                - Too high: Real objects take long to track
                - Good range: 2-5 detections

            iou_threshold: How much overlap to consider "same object"
                - Too low: Same object gets multiple IDs
                - Too high: Different objects get same ID
                - Good range: 0.2-0.5

        Example:
            >>> # Use defaults
            >>> tracker = TrackerRegistry.create('simple')
            >>>
            >>> # Conservative tracking (fewer false positives)
            >>> tracker = TrackerRegistry.create('simple', min_hits=5, iou_threshold=0.4)
            >>>
            >>> # Aggressive tracking (handles occlusions better)
            >>> tracker = TrackerRegistry.create('simple', max_age=50, min_hits=2)
        """
        if tracker_name not in cls._TRACKERS:
            available = ', '.join(cls._TRACKERS.keys())
            raise ValueError(
                f"Unknown tracker: '{tracker_name}'. "
                f"Available trackers: {available}"
            )

        # Get tracker metadata
        tracker_info = cls._TRACKERS[tracker_name]
        tracker_class = tracker_info['class']

        # Use defaults if not overridden
        max_age = max_age or tracker_info.get('default_max_age', 30)
        min_hits = min_hits or tracker_info.get('default_min_hits', 3)
        iou_threshold = iou_threshold or tracker_info.get('default_iou_threshold', 0.3)

        # Build kwargs
        kwargs = {
            'max_age': max_age,
            'min_hits': min_hits,
            'iou_threshold': iou_threshold,
            'logger': logger,
        }

        # Create tracker
        if logger:
            logger.info(f"Creating tracker: {tracker_name} ({tracker_info['description']})")

        return tracker_class(**kwargs)

    @classmethod
    def list_trackers(cls) -> Dict[str, str]:
        """
        List all available trackers.

        Returns:
            Dictionary mapping tracker_name -> description

        Example:
            >>> trackers = TrackerRegistry.list_trackers()
            >>> for name, desc in trackers.items():
            >>>     print(f"{name}: {desc}")
        """
        return {
            name: info['description']
            for name, info in cls._TRACKERS.items()
        }

    @classmethod
    def get_tracker_info(cls, tracker_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a tracker.

        Args:
            tracker_name: Tracker identifier

        Returns:
            Dictionary with tracker metadata

        Raises:
            ValueError: If tracker not found
        """
        if tracker_name not in cls._TRACKERS:
            raise ValueError(f"Unknown tracker: '{tracker_name}'")
        return cls._TRACKERS[tracker_name].copy()

    @classmethod
    def register_tracker(cls,
                        name: str,
                        tracker_class: type,
                        description: str,
                        default_max_age: int = 30,
                        default_min_hits: int = 3,
                        default_iou_threshold: float = 0.3,
                        **kwargs) -> None:
        """
        Register a new tracker.

        This allows users to add custom trackers without modifying this file.

        Args:
            name: Tracker identifier
            tracker_class: Tracker class to instantiate
            description: Human-readable description
            default_max_age: Default max_age parameter
            default_min_hits: Default min_hits parameter
            default_iou_threshold: Default iou_threshold parameter
            **kwargs: Additional tracker-specific parameters

        Example:
            >>> # Register custom tracker
            >>> TrackerRegistry.register_tracker(
            >>>     name='my-tracker',
            >>>     tracker_class=MyCustomTracker,
            >>>     description='My awesome tracker',
            >>>     default_max_age=40
            >>> )
            >>>
            >>> # Use it
            >>> tracker = TrackerRegistry.create('my-tracker')
        """
        cls._TRACKERS[name] = {
            'class': tracker_class,
            'description': description,
            'default_max_age': default_max_age,
            'default_min_hits': default_min_hits,
            'default_iou_threshold': default_iou_threshold,
            **kwargs
        }
