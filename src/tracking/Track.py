from Properties.BoundingBox import BoundingBox
from Detecting.Detections.ObjectDetection import ObjectDetection as Detection
from Tracking.Trajectory import Trajectory
from Tracking.KalmanFilter import KalmanFilter
from enum import Enum
from typing import Any, Optional
import cv2

class TrackState(Enum):
    """
    Enum to represent the state of a track.
    """
    NEW = 0
    MATCHED = 1
    LOST = 2
    RESERVED = 3

    def __str__(self):
        return self.name
    
class Track:
    """
    A class to represent an object identity in the tracking system.

    Attributes:
        id (int): Unique identifier for the track.
        track_state (TrackState): Current state of the track.
        lost_count (int): Number of consecutive frames the track has been unmatched. This reflects a track's temporal consistency relative to the current timestep. 
        trajectory (Trajectory): Trajectory of detections associated with the track.
        kalman_filter (KalmanFilter): Kalman filter for state prediction and updates.
    """

    __slots__ = ['_id', '_track_state', '_lost_count', '_trajectory', '_kalman_filter']

    ##### CONSTANTS #####
    TRAJECTORY_MAX_SIZE = 50  # Maximum number of detections in the trajectory

    ##### PROPERTIES #####
    @property
    def id(self) -> int: return self._id

    @property
    def track_state(self) -> TrackState: return self._track_state
    @track_state.setter
    def track_state(self, value: TrackState) -> None: self._track_state = value

    @property
    def lost_count(self) -> int: return self._lost_count
    @lost_count.setter
    def lost_count(self, value: int) -> None: self._lost_count = value

    @property
    def trajectory(self) -> Trajectory: return self._trajectory

    @property
    def kalman_filter(self) -> KalmanFilter: return self._kalman_filter

    ##### SETUP #####
    def __init__(self, id: int, track_state: TrackState = TrackState.RESERVED,):
        """
        Args:
            id (int): Unique identifier for the track.
            track_state (TrackState, optional): Initial state of the track. Defaults to RESERVED.
        """
        self._id = id
        self._track_state = track_state
        self._lost_count = 0
        self._trajectory = Trajectory(max_size=Track.TRAJECTORY_MAX_SIZE)
        self._kalman_filter = KalmanFilter()

    ##### UTILITIES #####
    def reset_lost_count(self) -> None: self._lost_count = 0
    def increment_lost_count(self) -> None: self._lost_count += 1
    
    def initialise_kalman_filter(self, detection: Detection) -> None:
        """
        Initialise the Kalman filter with the detection's bounding box.

        Args:
            detection (Detection): Detection data to initialise the Kalman filter.
        """
        self._kalman_filter.initialise(detection.bounding_box)
        self._kalman_filter.predict()
    def update_kalman_filter(self, detection: Detection) -> None:
        """
        Update the Kalman filter with the detection's bounding box.

        Args:
            detection (Detection): Detection data to update the Kalman filter.
        """
        self._kalman_filter.update(detection.bounding_box)
        self._kalman_filter.predict()
    def get_kalman_filter_state(self) -> Detection:
        """
        Get the predicted state from the Kalman filter.

        Returns:
            Detection: detection with a predicted bounding box.
        """
        prediction = self._kalman_filter.get_state()
        width, height = prediction[2], prediction[3]
        predicted_bounding_box =  BoundingBox.from_corners(
            prediction[0], prediction[1], prediction[0] + width, prediction[1] + height
        )
        return Detection(
            class_id=0,
            bounding_box=predicted_bounding_box,
            confidence_score=1.0
        )

    def update_trajectory(self, timestep: int, detection: Detection) -> None:
        """
        Update the trajectory with a new detection.

        Args:
            timestep (int): Current frame count.
            detection (Detection): Detection data to add to the trajectory.
        """
        self._trajectory[timestep] = detection

    def match_update(self, timestep: int, detection: Detection, track_state: TrackState = None) -> None:
        """
        Update the track with a new detection.

        Args:
            timestep (int): Current timestep.
            detection (Detection): Detection data to update the track.
            track_state (TrackState, optional): New state for the track. Defaults to None.
        """
        
        if track_state is not None:
            self._track_state = track_state

        self.update_kalman_filter(detection)
        self.update_trajectory(timestep, detection)
    def unmatch_update(self, track_state: TrackState = None) -> None:
        """
        Update the track when it is unmatched in the current frame.

        Args:
            track_state (TrackState, optional): New state for the track.
        """
        
        if track_state is not None:
            self._track_state = track_state
        self.increment_lost_count()
        # self._kalman_filter.predict()
    def reset(self, track_state: TrackState = TrackState.RESERVED, timestep: int = None, detection: Detection = None) -> None:
        """
        Reset the track, including its state, trajectory, lost count, and Kalman filter.
        Args:
            track_state (TrackState): New state for the track.
            timestep (int): Current frame count.
            detection (Detection): Detection data to initialize the Kalman filter.
        """
        self._track_state = track_state
        self.reset_lost_count()

        # If the detection is reset with a detection, then re-initialise the Kalman Filter and update the trajectory
        if detection:
            self._kalman_filter.initialise(detection.bounding_box)
        
        if timestep is not None and detection is not None:
            self.update_trajectory(timestep, detection)
        else:
            self._trajectory.clear()

    ##### TRACKING #####
    def calculate_cost(self, detection: Detection) -> float:
        """
        Calculate the cost of associating a detection with the track.
        The cost is calculated as 1 - similarity, where similarity is the similarity score between the predicted bounding box and the detection's bounding box.

        Args:
            detection (Detection): Detection to calculate the cost.

        Returns:
            float: Cost value (cost = 1 - similarity).
        """
        predicted_detection = self.get_kalman_filter_state()
        similarity = detection.calculate_similarity(predicted_detection)
        return 1 - similarity

    ##### LIFECYCLE #####   
    def creation(self, timestep: int, detection: Detection) -> None:
        """
        RESERVED -> NEW
        The track is created when it is first associated with a detection, marking the beginning of its identity.
        
        Args:
            timestep (int): Current timestep.
            detection (Detection): Detection used to create a new track.
        Raises:
            ValueError: If the track is not in the RESERVED state.
        """

        if self._track_state != TrackState.RESERVED:
            raise ValueError("Track must be in RESERVED state to create.")
        
        self._track_state = TrackState.NEW
        self.initialise_kalman_filter(detection)
        self.update_trajectory(timestep, detection)
    def activation(self, timestep: int, detection: Detection) -> None:
        """
        NEW -> MATCHED
        The track is activated when it is confirmed to be a consistent object identity.
        Args:
            timestep (int): Current timestep.
            detection (Detection): Detection used to activate a new track.
        Raises:
            ValueError: If the track is not in the NEW state.
        """
        if self._track_state != TrackState.NEW:
            raise ValueError("Track must be in NEW state to activate.")
        self.match_update(timestep, detection, track_state=TrackState.MATCHED)
    def reactivation(self, timestep: int, detection: Detection) -> None:
        """
        RESERVED -> MATCHED
        The track is reactivated if the object identity is successfully matched with a detection after deactivation

        Note that this should mainly be used when using a re-id model, as re-id models provide a more accurate representation of identity. 

        Args:
            timestep (int): Current timestep.
            detection (Detection): Detection used to reactivate the track.
        Raises:
            ValueError: If the track is not in the RESERVED state.
        """
        if self._track_state != TrackState.RESERVED:
            raise ValueError("Track must be in RESERVED state to reactivate.")
        self.reset(track_state=TrackState.MATCHED, timestep=timestep, detection=detection)
    def deactivation(self) -> None:
        """
        LOST -> RESERVED
        The track is deactivated when it has been LOST for a significant amount of time (e.g., due to occlusion or missed detections), indicating that the object identity is not temporally consistent anymore.

        Raises:
            ValueError: If the track is not in the LOST state.
        """
        if self._track_state != TrackState.LOST:
            raise ValueError("Track must be in LOST state to be deactivated.")
        self._track_state = TrackState.RESERVED
        self.reset_lost_count()
    
    ##### DISPLAY #####
    def draw_state(self, image: cv2.Mat, detection: Detection) -> None:
        """
        Draw a detection coloured based on its state.

        Args:
            image (cv2.Mat): The frame to display the track on.
            detection (Detection): The most recent detection of the track.
        """
        label = f"{self._id} - {self._track_state}"
        colour = {
            TrackState.NEW: (255, 0, 0),        # Blue
            TrackState.MATCHED: (0, 255, 0),    # Green
            TrackState.LOST: (0, 0, 255),       # Red
            TrackState.RESERVED: (255, 255, 0)  # Cyan
        }.get(self._track_state, (0, 0, 0))      # Default to black
        detection.draw(image, label=label, colour=colour)
    def draw_id(self, image: cv2.Mat, detection: Detection) -> None:
        """
        Draw a detection coloured uniquely based on its ID.

        Args:
            image (cv2.Mat): The frame to display the track on.
            detection (Detection): The most recent detection of the track.
        """
        label = f"{self._id}"
        detection.draw(image, label=label, seed=self._id)
    def draw(self, image: cv2.Mat, mode: str = 'state') -> None:
        """
        Draw the most recent detection of a track on a frame.

        Args:
            image (cv2.Mat): The frame to display the track on.
            mode (str, optional): Display mode ('state' or 'id'). Defaults to 'state'.
        Raises:
            ValueError: If the mode is not 'state' or 'id'.
        """

        # Validate mode
        if mode not in {'state', 'id'}:
            raise ValueError("Invalid mode. Use 'state' or 'id'.")
        
        # Get the most recent detection
        detection = self.trajectory.get_most_recent_detection()
        if mode == 'state':
            self.draw_state(image, detection)
        elif mode == 'id':
            self.draw_id(image, detection)
    def __str__(self) -> str:
        return (
            f"Track(ID={self._id}, STATE={self._track_state}, LOSTCNT={self.lost_count}, TRAJSIZE={self._trajectory}, MOSTRECENTDET={self._trajectory.get_most_recent_detection()})"
        )
    