from detecting.detections.ObjectDetection import ObjectDetection as Detection
from tracking.Track import Track, TrackState
from tracking.Partition import Partition
from tracking.trackers.Tracker import TrackList, Tracker

import logging
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
import cv2

class SimpleTracker(Tracker):
    """
    Inherits from the Tracker class and implements the tracking logic.

    Attributes:
        match_thresholds (List[float]): List of thresholds for matching tracks and detections.
        detection_threshold (float): Threshold to determine if a detection is valid.
        creation_threshold (float): Threshold to determine if a detection can create a new track, based on its confidence score.
        activation_threshold (float): Threshold to determine if a track can be activated.
        deactivation_threshold (int): Threshold to determine if a track has been lost for too long.
    """

    def __init__(self, 
                 *args, 
                 match_thresholds = [0.2],
                 detection_threshold=0.3, 
                 creation_threshold=0.4,
                 activation_threshold = 5, 
                 deactivation_threshold = 10, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._match_thresholds = match_thresholds
        self._detection_threshold = detection_threshold         
        self._creation_threshold = creation_threshold
        self._activation_threshold = activation_threshold
        self._deactivation_threshold = deactivation_threshold

        self.log(logging.INFO,
            "|| TRACKER INITIALISED\n"
            "\t\t\t\t    - MATCH THRESHOLDS: {}\n"
            "\t\t\t\t    - DETECTION THRESHOLD: {}\n"
            "\t\t\t\t    - CREATION THRESHOLD: {}".format(
            self._match_thresholds, self._detection_threshold, self._creation_threshold
            )
        )
    ####### TRACKING #######
    def match_condition(self, cost):
        """
        A match is valid if the cost is below the match threshold.

        Args:
            cost (float): The cost of the match.

        Returns:
            bool: True if the match is valid, False otherwise.
        """
        return cost <= self._match_threshold
    def partition_detections(self, detections: List[Detection]) -> Tuple[List[Detection], List[Detection]]:
        """
        Partitions the detections into two groups based on the detection condition:
        1. High-confidence detections 
        2. Low-confidence detections

        Args:
            detections (List[Detection]): List of detections to partition.

        Returns:
            Tuple: A tuple containing the two detection lists
        """

        high_confidence_detections = []
        low_confidence_detections = []

        for detection in detections:
            if detection.confidence_score >= self._detection_threshold:
                high_confidence_detections.append(detection)
            else:
                low_confidence_detections.append(detection)

        return high_confidence_detections, low_confidence_detections

    ####### LIFECYCLE #######
    def create(self, timestep: int, detection: Detection) -> Optional[Track]:
        """
        Creates a new track identity given a detection

        Args:
            timestep (int): The current timestep
            detection (Detection): The detection initialising the track identity

        Returns:
            Optional[Track]: The created track or None if the creation condition is not passed
        """

        # Creation condition - if the detection has a relatively high confidence score
        if (detection.confidence_score < self._creation_threshold):
            return
        
        # Create a new track with the detection
        self.increment_id_count()
        track = Track(self._id_count) # Initialise a RESERVED track with given ID
        track.creation(timestep, detection) # Track creation 
        self._new_tracks.add(track)
        self.log(logging.DEBUG, "\t//CREATED TRACK {}".format(track.id))

    def activate(self, timestep: int, track: Track, detection: Detection) -> None:
        """
        Activates a track identity given a detection

        Args:
            timestep (int): The current timestep
            track (Track): The track to be activated
            detection (Detection): The detection activating the track identity
        """

        # Activation condition - if the detection has matched three times
        if (track._trajectory.size > self._activation_threshold):
            track.activation(timestep, detection)
            self._matched_tracks.add(track)
            return
        else:
            # If the NEW track has not yet activated, but still matched, update the track
            track.match_update(timestep, detection)
            self._new_tracks.add(track)

    def deactivate(self, track: Track) -> None:
        # Deactivation condition - if the track has been lost for more than 10 timesteps
        if track.lost_count > self._deactivation_threshold:
            track.deactivation() # Deactivate the track
            self._reserved_tracks.add(track)
            self.log(logging.DEBUG, "\t//DEACTIVATED TRACK {}".format(track.id))
        else: 
            # If the track has not yet been lost for 10 timesteps, update the track
            track.unmatch_update()
            self._lost_tracks.add(track)
            self.log(logging.DEBUG, "\t//LOST TRACK {}".format(track.id))

    ##### TRACK MANAGEMENT #####
    def process_matches(self, timestep: int, matches: List) -> None:
        """
        Process the matches between tracks and detections.

        Args:
            tracks (List[Track]): List of tracks to process.
            detections (List[Detection]): List of detections to process.
        Raises: 
            ValueError: If the track state is invalid.
        """
        self.log(logging.INFO, "\t||PROCESS MATCHES")
        
        for track, detection in matches:
            # Process NEW matched tracks
            if track.track_state == TrackState.NEW:
                self.activate(timestep, track, detection) # Activate the track

            # Process LOST matched tracks
            elif track.track_state == TrackState.LOST:
                track.reset_lost_count()
                track.match_update(timestep, detection, TrackState.MATCHED)
                self._matched_tracks.add(track)

            # Process MATCHED matched tracks
            elif track.track_state == TrackState.MATCHED:
                track.match_update(timestep, detection)
                self._matched_tracks.add(track)
            else:
                raise ValueError(f"Invalid state for matched track: {track.state}")

    def process_unmatched_tracks(self, tracks: List[Track]) -> None: 
        """
        Process unmatched tracks and update their state.

        Args:
            timestep (int): The current timestep.
        """
        self.log(logging.INFO, "\t||PROCESS UNMATCHED TRACKS")
        
        for track in tracks:
            if track.track_state == TrackState.NEW:
                track.reset()
                self._reserved_tracks.add(track)
            elif track.track_state == TrackState.MATCHED:
                track.unmatch_update(track_state=TrackState.LOST)
                self._lost_tracks.add(track)
            elif track.track_state == TrackState.LOST:
                self.deactivate(track)
            else:
                raise ValueError(f"Invalid state for unmatched track: {track.track_state}")

    def process_unmatched_detections(self, timestep: int, detections: List[Detection]) -> None:
        """
        Process unmatched detections and update their state.

        Args:
            timestep (int): The current timestep.
            detections (List[Detection]): List of unmatched detections to process.
        """
        self.log(logging.INFO, "\t||PROCESS UNMATCHED DETECTIONS")
        
        for detection in detections:
            self.create(timestep, detection)

    ##### ASSOCIATION #####
    def bytetrack_association(self, timestep: int, detections: List[Detection]) -> Partition: 
        """

        Bytetrack association implementation 

        This procedure handles the assignment of detections to tracks in multiple stages:
        1. High-confidence detections are matched with matched and lost tracks.
        2. Remaining unmatched tracks are matched with low-confidence detections.
        3. New tracks are matched with remaining unmatched high-confidence detections.
 
        Args:
            timestep (int): The current timestep.
            detections (List[Detection]): List of detections to assign to tracks.
        Returns:
            Partition: A Partition object containing matched tracks, unmatched tracks, and unmatched detections.
        """
        self.log(logging.INFO, "\t||CASCADED ASSOCIATION")

        # Step 1: Partition detections into high and low confidence
        detections_high, detections_low = self.partition_detections(detections)

        # Step 2: Combine matched and lost tracks
        matched_lost_tracks = TrackList.combine(self._matched_tracks, self._lost_tracks)

        # Step 3: Assign high-confidence detections to matched and lost tracks
        partition1 = self.association(timestep, matched_lost_tracks, detections_high, self._match_thresholds[0])

        # Step 4: Assign low-confidence detections to unmatched tracks from partition1
        unmatched_tracks_partition1 = TrackList(
            track_states=matched_lost_tracks.track_states, 
            tracks=partition1.unmatched_x
        )
        partition2 = self.association(timestep, unmatched_tracks_partition1, detections_low, self._match_thresholds[1])

        # Step 5: Assign unmatched high-confidence detections to new tracks
        partition3 = self.association(timestep, self._new_tracks, partition1.unmatched_y, self._match_thresholds[2])

        # Step 6: Combine results from all partitions
        matched_tracks_detections = partition1.matched + partition2.matched + partition3.matched
        unmatched_tracks = partition2.unmatched_x + partition3.unmatched_x
        unmatched_detections = partition3.unmatched_y

        # Create final partition
        final_partition = Partition(
            matched=matched_tracks_detections, 
            unmatched_x=unmatched_tracks, 
            unmatched_y=unmatched_detections
        )

        return final_partition

    def update(self, timestep: int, detections: List[Detection]) -> None:
        """
        Update the tracker with new detections at a given timestep.
        Args:
            timestep (int): The current timestep.
            detections (list[Detection]): The list of detections to update the tracker with.
        """

        track_counts = {
            "new": self._new_tracks.length(),
            "matched": self._matched_tracks.length(),
            "lost": self._lost_tracks.length(),
        }
        self.log(logging.INFO, f"\t||UPDATE  MATCHED:{track_counts['matched']} - LOST:{track_counts['lost']} - NEW:{track_counts['new']}")

        # If not matched or lost tracks, create new tracks from detections
        if track_counts["matched"] == 0 and track_counts["lost"] == 0:
            # If no new tracks, create tracks from detections
            if track_counts["new"] == 0:
                for detection in detections:
                    self.create(timestep, detection)
            # If new tracks, perform simple procedure
            else:
                partition = self.association(timestep, self._new_tracks, detections, self._match_thresholds[0])
                self.track_management(timestep, partition)
        else:
            # If matched or lost tracks, perform subsequent tracking procedure
            partition = self.bytetrack_association(timestep, detections)
            self.track_management(timestep, partition)

        self.log(logging.INFO, "\t||{}".format(self._new_tracks))
        self.log(logging.INFO, "\t||{}".format(self._matched_tracks))
        self.log(logging.INFO, "\t||{}".format(self._lost_tracks))
        