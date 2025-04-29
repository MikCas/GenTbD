from Properties.BoundingBox import BoundingBox
from Tracking.Trajectory import Trajectory
from Tracking.KalmanFilter import KalmanFilter

from enum import Enum

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

    # STATIC
    _MAX_LOST_COUNT = 10       # MAX NUMBER OF FRAMES A TRACK CAN BE LOST
    @classmethod
    def set_max_lost_count(cls, value): cls._MAX_LOST_COUNT = value

    _TRAJECTORY_MAX_SIZE = 10  # MAX NUMBER OF DETECTIONS IN TRAJECTORY
    @classmethod
    def set_trajectory_max_size(cls, value): cls._TRAJECTORY_MAX_SIZE = value

    # CONSTRUCTOR
    def __init__(self, track_id, track_state=TrackState.RESERVED):
        self._id = track_id                                                  # ID
        self._track_state = track_state                                      # STATE
        self._lost_counter = 0                                               # LOST COUNTER
        self._trajectory = Trajectory(max_size=Track._TRAJECTORY_MAX_SIZE)   # TRAJECTORY 
        self._kalman_filter = KalmanFilter()                                 # KALMAN FILTER

    ####### TRACK STATE #######
    @property
    def id(self): return self._id

    @property
    def track_state(self): return self._track_state
    @track_state.setter
    def track_state(self, value): self._track_state = value

    @property
    def lost_counter(self): return self._lost_counter
    @lost_counter.setter
    def lost_counter(self, value): self._lost_counter = value
    def reset_lost_count(self): self.lost_counter = 0
    def increment_lost_count(self): self.lost_counter += 1

    # ACTIVATE TRACK - RESERVED -> NEW
    def activate(self, frame_count, detection):            
        self.track_state = TrackState.NEW
        self.initialise_kalman_filter(detection)
        self.update_trajectory(frame_count, detection)
    
    # UPDATE TRACK STATE - MATCHED
    def update_matched(self, frame_count, detection): 
        # NEW -> MATCHED      
        if self.track_state == TrackState.NEW:             
            self.track_state = TrackState.MATCHED
            self.update_kalman_filter(detection)
            self.update_trajectory(frame_count, detection)

        # MATCHED -> MATCHED
        elif self.track_state == TrackState.MATCHED:        
            self.update_kalman_filter(detection)
            self.update_trajectory(frame_count, detection)

        # LOST -> MATCHED 
        elif self.track_state == TrackState.LOST:           
            self.track_state = TrackState.MATCHED
            self.reset_lost_count()
            self.update_kalman_filter(detection)
            self.update_trajectory(frame_count, detection)

    # UPDATE TRACK STATE - UNMATCHED
    def update_unmatched(self):   
        # NEW -> RESERVED                          
        if self.track_state == TrackState.NEW:              
            self.track_state = TrackState.RESERVED
            self.reset_lost_count()

        # MATCHED -> LOST
        elif self.track_state == TrackState.MATCHED:        
            self.track_state = TrackState.LOST
            self.increment_lost_count()

        # LOST ->
        elif self.track_state == TrackState.LOST:  
            # LOST -> LOST (LOST COUNTER < MAX)          
            if self.lost_counter < Track._MAX_LOST_COUNT:   
                self.increment_lost_count()
            else:           
            # LOST -> RESERVED (LOST COUNTER > MAX)                                
                self.track_state = TrackState.RESERVED
                self.reset_lost_count()

    ####### TRAJECTORY #######
    @property
    def trajectory(self): return self._trajectory

    def update_trajectory(self, frame_count, detection):
        self.trajectory[frame_count] = detection

    def get_most_recent_detection(self):
        return self.trajectory.get_most_recent_detection()
    
    ####### KALMAN FILTER #######
    @property
    def kalman_filter(self): return self._kalman_filter
    
    def initialise_kalman_filter(self, detection):
        self.kalman_filter.initialise(detection.bounding_box)
        self.kalman_filter.predict()

    def update_kalman_filter(self, detection):
        self.kalman_filter.update(detection.bounding_box)
        self.kalman_filter.predict()
    
    def get_predicted_state(self):
        # TODO: CHANEG THE TRAJCEOTRY AND KALMAN FILRER METHODS TO SNAKE CASE
        prediction = self.kalman_filter.get_state()
        width = prediction[2]
        height = prediction[3]

        predicted_bounding_box = BoundingBox.from_corners(prediction[0], prediction[1], prediction[0] + width, prediction[1] + height)
        return predicted_bounding_box

    ####### COST #######
    # CALCULATE SIMILARITY BETWEEN DETECTION AND PREDICTED STATE
    # TODO: ADD DIFFERENT METHODS TO COMBINE THE SIMILARITY SCORES - WEIGHTED SUMS, GATING, ETC..
    # TODO: CAN ALSO USE NIPP FEATURES TO CALCULATE COST (MULTIPLE WITH CONFIDENCE SCORE)
    def calculate_cost(self, detection):
        predicted_bounding_box = self.get_predicted_state()
        similarity = detection.calculate_similarity(predicted_bounding_box)
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