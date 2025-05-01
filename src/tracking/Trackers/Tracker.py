from Tracking.Track import Track, TrackState
from Tracking.Partition import Partition
from Detecting.Detections.Detection import Detection

from abc import ABC, abstractmethod
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import List, Optional
import logging
import cv2

class TrackList:
    """
    Class to manage a list of tracks.
    
    Attributes:
        tracks (list[Track]): List of tracks.
        track_states (list[TrackState]): List of track states associated with this TrackList.
    """

    ##### ATTRIBUTES #####
    @property 
    def tracks(self) -> List[Track]: return self._tracks
    @property
    def track_states(self) -> List[TrackState]: return self._track_states

    ##### SETUP #####
    def __init__(self, track_states: TrackState | List[TrackState], tracks: Optional[List[Track]] = None):
        """
        Initialize a TrackList instance.

        Args:
            track_states (TrackState | List[TrackState]): A single track state or a list of track states.
        """
        if isinstance(track_states, TrackState):
            self._track_states = [track_states]
        else:
            self._track_states = track_states
        self._tracks = tracks or []

    ##### UTILITIES #####
    def add(self, tracks: Track | List[Track]) -> None:
        if isinstance(tracks, Track):
            self._tracks.append(tracks)
        else:
            self._tracks.extend(tracks)

    def remove(self, track: Track) -> None:
        self._tracks.remove(track)

    def length(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks = []

    @staticmethod
    def combine(*track_lists: 'TrackList') -> 'TrackList':
        """
        Combine the tracks from multiple TrackLists into a single TrackList.

        Args:
            *track_lists (TrackList): TrackLists to combine.

        Returns:
            TrackList: A new TrackList containing tracks from all input TrackLists.
        """
        combined_tracks = []
        combined_states = []
        for track_list in track_lists:
            combined_tracks.extend(track_list.tracks)
            combined_states.extend(track_list.track_states)
        combined_track_list = TrackList(combined_states, tracks = combined_tracks)
        return combined_track_list

    ##### DISPLAY #####
    def draw(self, image: cv2.Mat, mode: str) -> None:
        """
        Draw the tracks in the list.
        """
        for track in self._tracks:
            track.draw(image, mode)

    def __str__(self):
        """
        String representation of the TrackList.
        """
        return f"TRACKS[{', '.join(state.name for state in self._track_states)}]={[track.id for track in self._tracks]}"

class Tracker(ABC):
    """
    Abstract class for trackers, a tracker can be implemented using different assignment algorithms, based on a different number of detectors.
    This class is responsible for managing the state of tracks, including activation, deactivation, and assignment of detections to tracks. 
    It uses a linear assignment algorithm to match tracks with detections based on a cost matrix.
    
    Attributes:
        match_threshold (float): Threshold to determine if the cost of a match is valid.
        logger (Optional[logging.Logger]): Logger instance for logging messages (optional).
        id_count (int): Counter for unique track IDs.
        new_tracks (TrackList): List of new tracks.
        matched_tracks (TrackList): List of matched tracks.
        lost_tracks (TrackList): List of lost tracks.
        reserved_tracks (TrackList): List of removed tracks.
    """

    ##### PROPERTIES #####
    
    ##### SETUP ##### 
    def __init__(self, 
                 match_threshold: float = 0.05,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            match_threshold (float): Threshold to determine if the cost of a match is valid
            logger (Optional[logging.Logger], optional): Logger instance for logging messages (optional).
        """
        self._match_threshold: float = match_threshold
        self._logger: logging.Logger = logger 
        self._id_count: int = 0
        self._new_tracks: TrackList = TrackList(TrackState.NEW)
        self._matched_tracks: TrackList = TrackList(TrackState.MATCHED)
        self._lost_tracks: TrackList = TrackList(TrackState.LOST)
        self._reserved_tracks: TrackList = TrackList(TrackState.RESERVED)
    
    ##### UTILITIES #####
    def reset(self) -> None:
        """
        Reset the tracker, clear all track lists and reset ID count.
        """
        self._new_tracks.reset()
        self._matched_tracks.reset()
        self._lost_tracks.reset()
        self._reserved_tracks.reset()
        self._id_count = 0

        self.log(logging.INFO, "TRACKER RESET - ID COUNT: {}".format(self._id_count))

    def increment_id_count(self) -> int:
        """
        Increment the ID count and return the new ID.
        Returns:
            int: The new unique ID.
        """
        self._id_count += 1
        return self._id_count + 1
    
    ##### TRACKING #####
    @abstractmethod
    def match_condition(self, cost: float) -> bool:
        """
        Determines whether a match is valid based on the cost.
        Args:
            cost (float): The cost of the match.
        Returns:
            bool: True if match is valid (passes the condition), otherwise False. 
        """
        pass
    
    def create_cost_matrix(self, xs: List[Detection], ys: List[Detection]) -> np.ndarray:
        """
        Create a cost matrix for matching detections in two sets.
        Args:
            xs (list): The first set of detections.
            ys (list): The second set of detections.
        Returns:
            np.ndarray: A 2D cost matrix where each entry represents the cost of matching a detection from `xs` to `ys`.
        """
        
        num_xs = len(xs)
        num_ys = len(ys)

        # Initialize the cost matrix with ones
        cost_matrix = np.ones((num_xs, num_ys))

        # Populate the cost matrix with calculated costs
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                cost_matrix[i, j] = x.calculate_cost(y)

        return cost_matrix
    
    def linear_assignment(self, timestep: int, xs: List[Track], ys: List[Detection]) -> Partition:
        """
        Perform linear assignment to match detections in two sets based on a cost matrix.
        Args:
            timestep (int): The current timestep.
            xs (list[Track]): The tracks to match
            ys (list[Detection]): The detections to match
        Returns:
            Partition: A Partition object containing matched pairs and unmatched detections.
        """
        # Handle empty input sets
        if len(xs) == 0 or len(ys) == 0:
            return Partition(unmatched_x=xs, unmatched_y=ys)

        # Calculate the cost matrix
        cost_matrix = self.create_cost_matrix(xs, ys)

        # Perform linear sum assignment
        matched_xs_indexes, matched_ys_indexes = linear_sum_assignment(cost_matrix)

        # Process unmatched - all elements that are not in the matched indexes
        unmatched_xs = [xs[i] for i in range(len(xs)) if i not in matched_xs_indexes]
        unmatched_ys = [ys[i] for i in range(len(ys)) if i not in matched_ys_indexes]

        # Process matched - all elements that are in the matched indexes
        matched = []
        for x_index, y_index in zip(matched_xs_indexes, matched_ys_indexes):
            cost = cost_matrix[x_index][y_index]

            # Check if the match satisfies the condition
            if self.match_condition(cost):
                matched.append((xs[x_index], ys[y_index]))
            else:
                unmatched_xs.append(xs[x_index])
                unmatched_ys.append(ys[y_index])

        partition = Partition(matched=matched, unmatched_x=unmatched_xs, unmatched_y=unmatched_ys)
        return partition

    def assignment(self, timestep: int, track_list: TrackList, detections: List[Detection]) -> Partition:
        """
        Naive assignment procedure to match tracks with detections.
        Args:
            timestep (int): The current timestep.
            tracks (TrackList): The list of tracks to match.
            detections (list[Detection]): The list of detections to track.
        Returns:
            Partition: A Partition object containing matched pairs and unmatched detections.
        """

        # Perform linear assignment
        partition = self.linear_assignment(timestep, track_list.tracks, detections)
        return partition

    ##### LIFECYCLE #####

    #     \begin{itemize}
#     \item \textbf{Creation}: The track is created when it is first associated with a detection, marking the beginning of its identity.
#     \item \textbf{Activation}: A track transitions from \textit{NEW} to \textit{MATCHED} when it is confirmed to be a consistent object identity, likely to be matched again.
#     \item \textbf{Deactivation}: If a track fails to reappear after being temporarily lost (e.g., due to occlusion or missed detections), it enters the \textit{RESERVED} state, ending its active tracking.
#     \item \textbf{Reactivation}: A track in the \textit{RESERVED} state can be reactivated if it reappears and is successfully matched with a detection, moving back to the \textit{MATCHED} state.
# \end{itemize}

    # @abstractmethod
    # def create(self, detection: Detection) -> Track:

    #     pass
    
    # @abstractmethod
    # def activate(self, track: Track) -> None:
    #     """
    #     Activate a track, marking it as matched.
    #     Args:
    #         track (Track): The track to activate.
    #     """
    #     # track.activate()
    #     # self._matched_tracks.add(track)
    #     # self._new_tracks.remove(track)
    #     # self.log(logging.INFO, "TRACK {} ACTIVATED".format(track.id))
    #     pass

    # @abstractmethod
    # def reactivate(self, track: Track) -> None:
    #     """
    #     Deactivate a track, marking it as reserved.
    #     Args:
    #         track (Track): The track to deactivate.
    #     """
    #     # track.state = TrackState.RESERVED
    #     # self._reserved_tracks.add(track)
    #     # self._matched_tracks.remove(track)
    #     # self.log(logging.INFO, "TRACK {} DEACTIVATED".format(track.id))

    # @abstractmethod
    # def reactivate(self, track: Track) -> None:
    #     pass


    @abstractmethod
    def update(self, timestep: int, detections: List[Detection]) -> None:
        """
        Update the tracker with new detections at a given timestep.
        Args:
            timestep (int): The current timestep.
            detections (list[Detection]): The list of detections to update the tracker with.
        """
        pass
 
    ##### DISPLAY #####
    def log(self, level: int, message: str) -> None:
        """
        Log a message at the specified logging level.
        Args:
            level (int): Logging level (e.g., logging.INFO, logging.ERROR).
            message (str): Message to log.
        """
        if self._logger:
            self._logger.log(level, message)

    def draw_tracks(self, image: cv2.Mat, mode: str):
        """
        Draws the tracks on the image.
        Args:
            image (cv2.Mat): The image to display the tracks on.
            mode (str): The mode to display the tracks in. Options are 'id', 'score'.
        Raises: 
            ValueError: If the mode is not 'id', 'state'
        """
        if mode not in ['id', 'state']:
            raise ValueError("Invalid mode. Options are 'id', 'bbox', 'state'")
        self._new_tracks.draw(image, mode)
        self._matched_tracks.draw(image, mode)
        self._lost_tracks.draw(image, mode)
