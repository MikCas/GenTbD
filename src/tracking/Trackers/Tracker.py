from Tracking.Track import Track, TrackState
from Tracking.Partition import Partition
from Detecting.Detections.Detection import Detection

from abc import ABC, abstractmethod
from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import List, Optional, Tuple
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
        logger (Optional[logging.Logger]): Logger instance for logging messages (optional).
        id_count (int): Counter for unique track IDs.
        new_tracks (TrackList): List of new tracks.
        matched_tracks (TrackList): List of matched tracks.
        lost_tracks (TrackList): List of lost tracks.
        reserved_tracks (TrackList): List of removed tracks.
    """

    ##### SETUP ##### 
    def __init__(self, 
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            logger (Optional[logging.Logger], optional): Logger instance for logging messages (optional).
        """
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
    
    def linear_assignment(self, timestep: int, xs: List[Track], ys: List[Detection], match_threshold: float) -> Partition:
        """
        Perform linear assignment to match detections in two sets based on a cost matrix.
        Args:
            timestep (int): The current timestep.
            xs (list[Track]): The tracks to match
            ys (list[Detection]): The detections to match
            match_threshold (float): Threshold to determine if the cost of a match is valid.
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

            # If the cost is less than the threshold, consider it a match
            if cost <= match_threshold:
                matched.append((xs[x_index], ys[y_index]))
            else:
                unmatched_xs.append(xs[x_index])
                unmatched_ys.append(ys[y_index])

        partition = Partition(matched=matched, unmatched_x=unmatched_xs, unmatched_y=unmatched_ys)
        return partition

    def assignment(self, timestep: int, track_list: TrackList, detections: List[Detection], match_threshold: float) -> Partition:
        """
        Naive assignment procedure to match tracks with detections.
        Args:
            timestep (int): The current timestep.
            tracks (TrackList): The list of tracks to match.
            detections (list[Detection]): The list of detections to track.
            match_threshold (float): Threshold to determine if the cost of a match is valid.
        Returns:
            Partition: A Partition object containing matched pairs and unmatched detections.
        """

        # Perform linear assignment
        partition = self.linear_assignment(timestep, track_list.tracks, detections, match_threshold)
        return partition
    
    # @abstractmethod
    # def cascaded_assignment(self, timestep: int, track_list: TrackList, detections: List[Detection]) -> Partition:
    #     """
    #     Cascaded assignment procedure to match tracks with detections.
    #     Args:
    #         timestep (int): The current timestep.
    #         track_list (TrackList): The list of tracks to match.
    #         detections (list[Detection]): The list of detections to track.
    #     Returns:
    #         Partition: A Partition object containing matched pairs and unmatched detections.
    #     """
    #     pass

        ##### LIFECYCLE #####
        @abstractmethod
        def create(self, timestep: int, detection: Detection) -> Optional[Track]:
            """
            Abstract method to create a new track identity given a detection.

            Args:
                timestep (int): The current timestep.
                detection (Detection): The detection initializing the track identity.

            Returns:
                Optional[Track]: The created track or None if the creation condition is not passed.
            """
            pass

        @abstractmethod
        def activate(self, timestep: int, track: Track, detection: Detection) -> None:
            """
            Abstract method to activate a track identity given a detection.

            Args:
                timestep (int): The current timestep.
                track (Track): The track to be activated.
                detection (Detection): The detection activating the track identity.
            """
            pass

        @abstractmethod
        def deactivate(self, track: Track) -> None:
            """
            Abstract method to deactivate a track.

            Args:
                track (Track): The track to be deactivated.
            """
            pass

        ##### TRACK MANAGEMENT #####
        @abstractmethod
        def process_matches(self, timestep: int, matches: List) -> None:
            """
            Abstract method to process the matches between tracks and detections.

            Args:
                timestep (int): The current timestep.
                matches (List): List of matched track-detection pairs.
            """
            pass

        @abstractmethod
        def process_unmatched_tracks(self, tracks: List[Track]) -> None:
            """
            Abstract method to process unmatched tracks and update their state.

            Args:
                tracks (List[Track]): List of unmatched tracks to process.
            """
            pass

        @abstractmethod
        def process_unmatched_detections(self, timestep: int, detections: List[Detection]) -> None:
            """
            Abstract method to process unmatched detections and update their state.

            Args:
                timestep (int): The current timestep.
                detections (List[Detection]): List of unmatched detections to process.
            """
            pass

        @abstractmethod
        def track_management(self, timestep: int, partition: Partition) -> None:
            """
            Abstract method to manage the state of tracks based on the partition of detections.

            Args:
                timestep (int): The current timestep.
                partition (Partition): The partition of detections to manage.
            """
            pass
        
    def cascaded_assignment(
        self, 
        timestep: int,
        tracks: TrackList, 
        detections_list: List[List[Detection]], 
        match_thresholds: List[float]
    ) -> List[Partition]:
        """
        Perform cascaded assignment to perform an iterative assignment of unmatched tracks to a set of detections in multiple cascading stages 
        At each stage, unmatched tracks are matched with a subset of detections using a specific match threshold.

        Args:
            timestep (int): The current timestep.
            tracks (TrackList): List of tracks to be assigned.
            detections_list (List[List[Detection]]): List of detection subsets for each stage.
            match_thresholds (List[float]): List of match thresholds for each stage.

        Returns:
            Partition: A list of Partition objects, one for each stage of the assignment.
                            The last Partition contains the unmatched tracks and detections.
        """
        self.log(logging.INFO, "\t||CASCADED ASSIGNMENT")

        # Cascade results
        matched = []  # List to store matched tracks and detections at each stage
        unmatched_detections = []  # List to store unmatched detections at each stage
        curr_unmatched_tracks = tracks  # Tracks remaining unmatched for further assignment

        # Iterate through each stage of assignment
        for detections, match_threshold in zip(detections_list, match_thresholds):

            # Perform linear assignment for the current stage
            curr_partition = self.assignment(
                timestep=timestep,  
                track_list=curr_unmatched_tracks,
                detections=detections,
                match_threshold=match_threshold
            )

            curr_unmatched_tracks = TrackList(
                track_states=curr_unmatched_tracks.track_states, 
                tracks=curr_partition.unmatched_x
            )

            # Add the partition to the results
            matched.extend(curr_partition.matched)
            unmatched_detections.extend(curr_partition.unmatched_y)

        partition = Partition(
            matched=matched,
            unmatched_x=curr_unmatched_tracks.tracks,
            unmatched_y=unmatched_detections
        )

        return partition
    
    @abstractmethod
    def update(self, timestep: int, detections: List[Detection]) -> None:
        """
        Update the tracker with new detections at a given timestep.
        Args:
            timestep (int): The current timestep.
            detections (list[Detection]): The list of detections to update the tracker with.
        """
        pass
        
    ##### LIFECYCLE #####
    @abstractmethod
    def create(self, timestep: int, detection: Detection) -> Optional[Track]:
        """
        Create a new track identity given a detection.
        Args:
            timestep (int): The current timestep.
            detection (Detection): The detection initializing the track identity.
        Returns:
            Optional[Track]: The created track or None if the creation condition is not passed.
        """
        pass

    @abstractmethod
    def activate(self, timestep: int, track: Track, detection: Detection) -> None:
        """
        Activate a track identity given a detection.
        Args:
            timestep (int): The current timestep.
            track (Track): The track to be activated.
            detection (Detection): The detection activating the track identity.
        """
        pass

    @abstractmethod
    def deactivate(self, track: Track) -> None:
        """
        Deactivate a track.
        Args:
            track (Track): The track to be deactivated.
        """
        pass

    ##### TRACK MANAGEMENT #####
    @abstractmethod
    def process_matches(self, timestep: int, matches: List) -> None:
        """
        Process the matches between tracks and detections.
        Args:
            timestep (int): The current timestep.
            matches (List): List of matched track-detection pairs.
        """
        pass

    @abstractmethod
    def process_unmatched_tracks(self, tracks: List[Track]) -> None:
        """
        Process unmatched tracks and update their state.
        Args:
            tracks (List[Track]): List of unmatched tracks to process.
        """
        pass

    @abstractmethod
    def process_unmatched_detections(self, timestep: int, detections: List[Detection]) -> None:
        """
        Process unmatched detections and update their state.
        Args:
            timestep (int): The current timestep.
            detections (List[Detection]): List of unmatched detections to process.
        """
        pass

    def track_management(self, timestep: int, partition: Partition) -> None:
        """
        Manages the state of tracks based on the partition of detections.
        This includes updating matched tracks, handling unmatched tracks, and activating new tracks.

        Args:
            timestep (int): The current timestep.
            partition (Partition): The partition of detections to manage.
        """
        self.log(logging.INFO, "\t||TRACK MANAGEMENT")
        
        # Reset NEW, MATCHED, and LOST tracks
        self._new_tracks.reset()
        self._matched_tracks.reset()
        self._lost_tracks.reset()

        self.process_matches(timestep, partition.matched)
        self.process_unmatched_tracks(partition.unmatched_x)
        self.process_unmatched_detections(timestep, partition.unmatched_y)
    
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
