from Detection.Detection import Detection
from Tracking.Track import Track, TrackState
from Tracking.Partition import Partition
from Tracking.Assignment import linear_assignment
from Tracking.Trackers.Tracker import TrackList, Tracker

import logging
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
import cv2

class SimpleTracker(Tracker):
    """
    Simple tracker class for managing object tracking. 
    Inherits from the Tracker class and implements the tracking logic.
    This class is responsible for managing the state of tracks, including activation, deactivation, and assignment of detections to tracks.
    It uses a linear assignment algorithm to match tracks with detections based on a cost matrix.

    Attributes:
        detection_threshold (float): Threshold to determine if a detection is valid.
        activation_threshold (float): Threshold to determine if a detection can activate a new track.
    """

    def __init__(self, 
                 *args, 
                 detection_threshold=0.3, 
                 creation_threshold=0.4, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._detection_threshold = detection_threshold         # USED TO PARTITION DETECTIONS IN SUBSEQUENT TRACKING PROCEDURE
        self._creation_threshold = creation_threshold      

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



    def detection_condition(self, detection: Detection) -> bool:
        """
        A detection is valid if the confidence is above the detection threshold.

        Args:
            detection: The detection to check.

        Returns:
            bool: True if the detection is valid, False otherwise.
        """
        return detection.confidence_score >= self._detection_threshold

    def partition_detections(self, detections: List[Detection]) -> Tuple[List[Detection], List[Detection]]:
        """
        Partitions the detections into two groups based on the detection condition:
        1. High-confidence detections (valid detections)
        2. Low-confidence detections (invalid detections)

        Args:
            detections (List[Detection]): List of detections to partition.

        Returns:
            Tuple: A tuple containing the two detection lists
        """
        valid_detections = [detection for detection in detections if self.detection_condition(detection)]
        invalid_detections = [detection for detection in detections if not self.detection_condition(detection)]
        return valid_detections, invalid_detections
    

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
        track = Track(self._id_count)
        track.creation(timestep, detection)
        self._new_tracks.add(track)
        self.log(logging.INFO, "CREATED - {}".format(track))

    def deactivate(self, track: Track) -> None:
        self._reserved_tracks.add(track)
        self.log(logging.INFO, "TRACK {} DEACTIVATED".format(track))

    def track_management(self, timestep: int, partition: Partition) -> None:
        """
        Manages the state of tracks based on the partition of detections.
        This includes updating matched tracks, handling unmatched tracks, and activating new tracks.

        Args:
            timestep (int): The current timestep.
            partition (Partition): The partition of detections to manage.
        """
        self.log(logging.INFO, "TRACK MANAGEMENT")
        
        # Reset NEW, MATCHED, and LOST tracks
        self._new_tracks.reset()
        self._matched_tracks.reset()
        self._lost_tracks.reset()

        # Process matched tracks
        for track, detection in partition.matched:
            track.update_matched(timestep, detection)
            self._matched_tracks.add(track)

        # Process unmatched tracks
        for track in partition.unmatched_x:
            track.update_unmatched()
            if track.track_state == TrackState.LOST:
                self._lost_tracks.add(track)
            elif track.track_state == TrackState.RESERVED:
                self.deactivate(track)
            else:
                raise ValueError(f"Invalid state for unmatched track: {track.state}")

        # Create new tracks from unmatched detections depending on creation condition
        for detection in partition.unmatched_y:
            self.create(timestep, detection)

        self.log(logging.INFO, "TRACK MANAGEMENT - END")

    def cascaded_assignment(self, timestep: int, detections: List[Detection]) -> Partition: 
        """
        Perform cascaded assignment for iterative tracking.

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
        self.log(logging.INFO, "CASCADED ASSIGNMENT STARTED")

        # Step 1: Partition detections into high and low confidence
        detections_high, detections_low = self.partition_detections(detections)

        # Step 2: Combine matched and lost tracks
        matched_lost_tracks = TrackList.combine(self._matched_tracks, self._lost_tracks)
        self.log(logging.INFO, f"Matched and Lost Tracks: {matched_lost_tracks}")

        # Step 3: Assign high-confidence detections to matched and lost tracks
        partition1 = self.assignment(timestep, matched_lost_tracks, detections_high)

        # Step 4: Assign low-confidence detections to unmatched tracks from partition1
        unmatched_tracks_partition1 = TrackList(
            track_states=matched_lost_tracks.track_states, 
            tracks=partition1.unmatched_x
        )
        partition2 = self.assignment(timestep, unmatched_tracks_partition1, detections_low)

        # Step 5: Assign unmatched high-confidence detections to new tracks
        partition3 = self.assignment(timestep, self._new_tracks, partition1.unmatched_y)

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
        self.log(logging.INFO, f"UPDATE - (MATCHED:{track_counts['matched']}, LOST:{track_counts['lost']}, NEW:{track_counts['new']})")

        # If not matched or lost tracks, create new tracks from detections
        if track_counts["matched"] == 0 and track_counts["lost"] == 0:
            # If no new tracks, create tracks from detections
            if track_counts["new"] == 0:
                self.log(logging.INFO, "NO TRACKS - CREATING NEW TRACKS")
                for detection in detections:
                    self.create(timestep, detection)
            # If new tracks, perform simple procedure
            else:
                self.log(logging.INFO, "NEW TRACKS - ASSIGNMENT PROCEDURE")
                partition = self.assignment(timestep, self._new_tracks, detections)
                self.track_management(timestep, partition)
        else:
            # If matched or lost tracks, perform subsequent tracking procedure
            self.log(logging.INFO, "MATCHED OR LOST TRACKS - COMPLEX ASSIGNMENT PROCEDURE")
            partition = self.cascaded_assignment(timestep, detections)
            self.track_management(timestep, partition)

        self.log(logging.INFO, "{}".format(self._new_tracks))
        self.log(logging.INFO, "{}".format(self._matched_tracks))
        self.log(logging.INFO, "{}".format(self._lost_tracks))

        # self._new_tracks.output_tracks()
        # self._matched_tracks.output_tracks()
        # self._lost_tracks.output_tracks()
        
    ####### OUTPUT #######
    def display_tracks(self, image: cv2.Mat, mode: str):
        """
        Displays the tracks on the image.
        Args:
            image (cv2.Mat): The image to display the tracks on.
            mode (str): The mode to display the tracks in. Options are 'id', 'bbox', 'score'.
        Raises: 
            ValueError: If the mode is not 'id', 'bbox', 'state'
        """
        if mode not in ['id', 'state']:
            raise ValueError("Invalid mode. Options are 'id', 'bbox', 'state'")
        self._new_tracks.display(image, mode)
        self._matched_tracks.display(image, mode)
        self._lost_tracks.display(image, mode)
