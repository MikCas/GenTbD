from properties.BoundingBox import BoundingBox
from Detection import Detection 
from Trajectory import Trajectory
from KalmanFilter import KalmanFilter

from enum import Enum
import numpy as np

# TRACK STATE
class TrackState(Enum):
    NEW = 0     
    MATCHED = 1 
    LOST = 2
    RESERVED = 3

    def __str__(self):
        return self.name
    
class Track:

    __slots__ = ['_id', '_track_state', '_lost_counter', '_trajectory', '_kalman_filter']

    # TODO: ADD AS PARAMETERS IN TRACKER
    # STATIC VARIABLES
    _MAX_LOST_COUNT = 10       # MAX NUMBER OF FRAMES A TRACK CAN BE LOST
    _TRAJECTORY_MAX_SIZE = 10  # MAX NUMBER OF DETECTIONS IN TRAJECTORY

    # CONSTRUCTOR
    def __init__(self, track_id, track_state=TrackState.RESERVED):
        self._id = track_id                                                  # ID
        self._track_state = track_state                                      # STATE
        self._lost_counter = 0                                               # LOST COUNTER
        self._trajectory = Trajectory(max_size=Track._TRAJECTORY_MAX_SIZE)   # TRAJECTORY 
        self._kalman_filter = KalmanFilter()                                 # KALMAN FILTER

    # TRACK STATE 
    @property
    def id(self): return self._id
    @property
    def track_state(self): return self._track_state
    @property
    def lost_counter(self): return self._lost_counter
    
    @track_state.setter
    def track_state(self, value): self._track_state = value
    @lost_counter.setter
    def lost_counter(self, value): self._lost_counter = value

    def reset_lost_count(self): self.lost_counter = 0
    def increment_lost_count(self): self.lost_counter += 1

    def activate(self, frame_count, detection):            # ACTIVATE TRACK - RESERVED -> NEW
        self.track_state = TrackState.NEW
        self.initialise_kalman_filter(detection)
        self.update_trajectory(frame_count, detection)
    
    def update_matched(self, frame_count, detection):       # UPDATE TRACK STATE - MATCHED
        if self.track_state == TrackState.NEW:              # NEW -> MATCHED
            self.track_state = TrackState.MATCHED
            self.update_kalman_filter(detection)
            self.update_trajectory(frame_count, detection)

        elif self.track_state == TrackState.MATCHED:        # MATCHED -> MATCHED
            self.update_kalman_filter(detection)
            self.update_trajectory(frame_count, detection)

        elif self.track_state == TrackState.LOST:           # LOST -> MATCHED 
            self.track_state = TrackState.MATCHED
            self.reset_lost_count()
            self.update_kalman_filter(detection)
            self.update_trajectory(frame_count, detection)

    def update_unmatched(self):                             # UPDATE TRACK STATE - UNMATCHED
        if self.track_state == TrackState.NEW:              # NEW -> RESERVED
            self.track_state = TrackState.RESERVED
            self.reset_lost_count()

        elif self.track_state == TrackState.MATCHED:        # MATCHED -> LOST
            self.track_state = TrackState.LOST
            self.increment_lost_count()

        # TODO: DEACTIVATION CONDITION USED HERE (MAYBE DEFINE INTERNALLY)
        elif self.track_state == TrackState.LOST:           # LOST ->
            if self.lost_counter < Track._MAX_LOST_COUNT:   # LOST -> LOST (LOST COUNTER < MAX) 
                self.increment_lost_count()
            else:                                           # LOST -> RESERVED (LOST COUNTER > MAX)
                self.track_state = TrackState.RESERVED
                self.reset_lost_count()

    # TRAJECTORY
    @property
    def trajectory(self):
        return self._trajectory
    
    def update_trajectory(self, frame_count, detection):
        self.trajectory[frame_count] = detection

    def get_most_recent_detection(self):
        return self.trajectory.get_most_recent_detection()
    
    # KALMAN FILTER
    @property
    def kalman_filter(self):
        return self._kalman_filter
    
    def initialise_kalman_filter(self, detection):
        self.kalman_filter.initialise(detection.bounding_box)
        self.kalman_filter.predict()

    def update_kalman_filter(self, detection):
        self.kalman_filter.update(detection.bounding_box)
        self.kalman_filter.predict()
    
    def get_predicted_state(self):
        # TODO: CHANEG THE TRAJCEOTRY AND KALMAN FILRER METHODS TO SNAKE CASE
        prediction = self.kalman_filter.getState()
        width = prediction[2]
        height = prediction[3]

        predicted_state = BoundingBox.from_corners(prediction[0], prediction[1], prediction[0] + width, prediction[1] + height)
        return predicted_state

    # COST 
    # Calculate the similarity between the detection and the predicted state of the track/the most recent detection in the trajctory. Then combine the similarity scores (also using NIPP features) to calculate the cost.
    # TODO: Update the cost by incorporating different similarity features
    # TODO: Can also use statistics on the trajectory to calculate cost
    # TODO: Can also use other NIPP features to calculate cost
    def calculate_cost(self, detection):
        predicted_state = self.get_predicted_state()
        similarity = detection.calculate_similarity(predicted_state)
        cost = 1 - similarity
        return cost
    
    def display(self, frame, mode='state'):
        label = f"{self.id} - {self.track_state}"
        
        if mode == 'state':
            # USE FIXED COLORS BASED ON TRACK STATE
            match self.track_state:
                case TrackState.NEW:
                    colour = (255, 0, 0)      # Blue
                case TrackState.MATCHED:
                    colour = (0, 255, 0)      # Green  
                case TrackState.LOST:
                    colour = (0, 0, 255)      # Red
                case TrackState.RESERVED:
                    colour = (255, 255, 0)    # Cyan
                case _:
                    colour = (0, 0, 0)        # Black
            
            self.trajectory.get_most_recent_detection().display(frame, label=label, colour=colour)
            
        elif mode == 'id':
            # GENERATE UNIQUE COLOR BASED ON TRACK ID
            self.trajectory.get_most_recent_detection().display(frame, label=label, seed=self.id)

    def __repr__(self):
        return f"TRK(ID={self.id}, STATE={self.track_state}, LC={self.lost_count}, TL={len(self.get_trajectory())}, MRD={self.get_most_recent_detection()})"