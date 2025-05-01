from Detecting.Detections.Detection import Detection
from collections import OrderedDict
from typing import Optional, Any

class Trajectory(OrderedDict):
    """
    A class to represent a trajectory of detections over time.

    This class extends `OrderedDict` to store detections indexed by timesteps.
    It supports a maximum size to limit the number of stored detections and provides
    utility methods to query and manage the trajectory.

    Attributes:
        max_size (int): The maximum number of detections to store in the trajectory.
                        If set to 0, the trajectory has no size limit.
    """
    ##### PROPERTIES #####
    __slots__ = ['_max_size']

    @property
    def max_size(self) -> int:
        """
        Get the maximum size of the trajectory.

        Returns:
            int: The maximum number of detections the trajectory can store.
        """
        return self._max_size

    ##### SETUP #####
    def __init__(self, *args, max_size: int = 0, **kwargs):
        """
        Initialize a Trajectory instance.

        Args:
            max_size (int, optional): The maximum number of detections to store. Defaults to 0 (no limit).
            *args: Additional positional arguments for `OrderedDict`.
            **kwargs: Additional keyword arguments for `OrderedDict`.

        Raises:
            ValueError: If `max_size` is less than 0.
        """
        if max_size < 0:
            raise ValueError("Maximum size must be 0 or greater")
        self._max_size = max_size
        super().__init__(*args, **kwargs)

    ##### FUNCTIONS #####
    def __setitem__(self, timestep: int, detection: Detection) -> None:
        """
        Add a detection to the trajectory.

        If the maximum size is exceeded, the oldest detection is removed.

        Args:
            timestep (int): The timestep when the detection was added to the trajectory.
            detection (Any): The detection data.
        """
        super().__setitem__(timestep, detection)

        # Remove the oldest item if the maximum size is exceeded
        if self._max_size > 0 and len(self) > self._max_size:
            self.popitem(last=False)

    def has_timestep(self, timestep: int) -> bool:
        """
        Check if a specific timestep exists in the trajectory.

        Args:
            timestep (int): The timestep to check.

        Returns:
            bool: True if the timestep exists, False otherwise.
        """
        return timestep in self

    def get_all_detections(self) -> list[Detection]:
        """
        Get all detections in the trajectory.

        Returns:
            list[Detection]: A list of all detections in the trajectory.
        """
        return list(self.values())

    def get_detection_at_timestep(self, timestep: int) -> Optional[Detection]:
        """
        Get the detection at a specific timestep.

        Args:
            timestep (int): The timestep to retrieve the detection for.

        Returns:
            Optional[Any]: The detection data at the specified timestep, or None if no detection exists.
        """
        return self.get(timestep)

    def get_most_recent_timestep(self) -> Optional[int]:
        """
        Get the most recent timestep in the trajectory.

        Returns:
            Optional[int]: The most recent timestep, or None if the trajectory is empty.
        """
        return max(self.keys()) if self else None

    def get_most_recent_detection(self) -> Optional[Detection]:
        """
        Get the most recent detection in the trajectory.

        Returns:
            Optional[Any]: The detection data for the most recent timestep, or None if the trajectory is empty.
        """
        most_recent_timestep = self.get_most_recent_timestep()
        return self.get(most_recent_timestep)

    ##### DISPLAY #####
    def __str__(self) -> str:
        """
        String representation of the trajectory.

        Returns:
            str: A formatted string representing the timesteps in the trajectory.
        """
        timesteps = list(self.keys())
        return f"TRAJ={timesteps}"

