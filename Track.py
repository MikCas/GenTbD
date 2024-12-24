from enum import Enum
import numpy as np

from properties.BoundingBox import BoundingBox
from Detection import Detection 
from Trajectory import Trajectory
from KalmanFilter import KalmanFilter

# TODO: I WANT TO GO OVER THE FUNCTIONALITY OF THIS CLASS AGAIN, (TOGETHER WITH THE TRAJECOTR CLASS) TO SEE IF THERE NEEDS TO BE ANY CHANGES FOR BETTER FUNCTIONALITY
class TrackState(Enum):
    NEW = 0     
    MATCHED = 1 
    LOST = 2
    RESERVED = 3

class Track:

    __slots__ = ['_id', '_track_state', '_lost_counter', '_trajectory', '_kalman_filter']

    _MAX_LOST_COUNT = 10  # Max number of frames the track can be lost
    _TRAJECTORY_MAX_SIZE = 10  # Max number of detections in the trajectory

    def __init__(self, track_id, track_state=TrackState.RESERVED):
        self._id = track_id                                             # Unique track ID
        self._track_state = track_state                                 # Track state
        self._lost_counter = 0                                          # Number of consecutive frames the track has been lost
        self._trajectory = Trajectory(max_size=Track._TRAJECTORY_MAX_SIZE)   # Trajectory acts like a LIFO queue
        self._kalman_filter = KalmanFilter()                            # Kalman filter instance for tracking object state

    @property
    def id(self):
        return self._id

    @property
    def track_state(self):
        return self._track_state

    @property
    def lost_count(self):
        return self._lost_counter

    @lost_count.setter
    def lost_count(self, value):
        self._lost_counter = value

    # STATE_HANDLING
    def reset_lost_count(self):
        self._lost_counter = 0
    
    def increment_lost_count(self):
        self._lost_counter += 1

    # ACTIVATE TRACK - RESERVED -> NEW
    def activate(self, frame_count, detection):
        self._track_state = TrackState.NEW
        self.initialise_kalman_filter(detection)
        self.update_trajectory(frame_count, detection)

    # UPDATE TRACK STATE IN THE CASE OF A MATCH
    def update_matched(self, frame_count, detection):
        # NEW -> MATCHED
        if self._track_state == TrackState.NEW:
            self._track_state = TrackState.MATCHED
            self.update_kalman_filter(detection)
            self.update_trajectory(frame_count, detection)

        # MATCHED -> MATCHED
        elif self._track_state == TrackState.MATCHED:
            self.update_kalman_filter(detection)
            self.update_trajectory(frame_count, detection)

        # LOST -> MATCHED 
        elif self._track_state == TrackState.LOST:
            self._track_state = TrackState.MATCHED
            self.reset_lost_count()
            self.update_kalman_filter(detection)
            self.update_trajectory(frame_count, detection)

    # Update track state in the case of no match
    def update_unmatched(self):
        # NEW -> RESERVED
        if self._track_state == TrackState.NEW:
            self._track_state = TrackState.RESERVED
            self.reset_lost_count()

        # MATCHED -> LOST
        elif self._track_state == TrackState.MATCHED:
            self._track_state = TrackState.LOST
            self.increment_lost_count()

        # TODO: DEACTIVATION CONDITION USED HERE (MAYBE DEFINE INTERNALLY)
        # LOST ->
        elif self._track_state == TrackState.LOST:
            # LOST -> LOST (LOST COUNTER < MAX)
            if self._lost_counter < Track._MAX_LOST_COUNT:
                self.increment_lost_count()
            # LOST -> RESERVED (LOST COUNTER > MAX)
            else:
                self._track_state = TrackState.RESERVED
                self.reset_lost_count()

    # TRACK HISTORY
    def get_trajectory(self):
        return self._trajectory

    def get_most_recent_detection(self):
        return self._trajectory.get_most_recent_detection()
    
    def update_trajectory(self, frame_count, detection):
        self._trajectory[frame_count] = detection

    def initialise_kalman_filter(self, detection):
        self._kalman_filter.initialise(detection._bounding_box)
        self._kalman_filter.predict()

    def update_kalman_filter(self, detection):
        self._kalman_filter.update(detection._bounding_box)
        self._kalman_filter.predict()
    
    def get_predicted_state(self):
        # TODO: CHANEG THE TRAJCEOTRY AND KALMAN FILRER METHODS TO SNAKE CASE
        prediction = self._kalman_filter.getState()
        width = prediction[2]
        height = prediction[3]

        predicted_state = BoundingBox.from_corners(prediction[0], prediction[1], prediction[0] + width, prediction[1] + height)
        return predicted_state
    
    # TRACK COST
    # Calculate the similarity between the detection and the predicted state of the track/the most recent detection in the trajctory. Then combine the similarity scores (also using NIPP features) to calculate the cost.
    # TODO: Update the cost by incorporating different similarity features
    # TODO: Can also use statistics on the trajectory to calculate cost
    # TODO: Can also use other NIPP features to calculate cost
    def calculate_cost(self, detection):
        predicted_state = self.get_predicted_state()
        similarity = detection.calculate_similarity(predicted_state)
        cost = 1 - similarity
        return cost
    
    # TODO: ADD ANOTEHR PARAMETER TO THIS SO THAT I CAN CHOOSE IF I WANT CERTAIN TRACKS TO BE A CERTAIN COLOR - LIEK NEW TRCAKS
    def display(self, frame):
        self._trajectory.get_most_recent_detection().display(frame, self.id)

    def __repr__(self):
        return f"Track(ID={self.id}, State={self.track_state}, LostCount={self.lost_count}, TrajectoryLength={len(self.get_trajectory())})"