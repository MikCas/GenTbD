import logging
from collections import deque
from Track import Track, TrackState
from Assignment import linear_assignment

class Tracker:
    def __init__(self, max_tracks=100, detection_threshold=0.3, new_track_threshold=0.3, match_threshold=0.1, logger=None):
        
        # INITIALISE LOGGER, OTHERWISE USE DEFAULT LOGGER
        self._logger = logger if logger else logging.getLogger(__name__)

        # TRACKER PARAMETERS
        self._frame_count = 0
        self._max_tracks = max_tracks
        self._detection_threshold = detection_threshold
        self._new_track_threshold = new_track_threshold
        self._match_threshold = match_threshold

        # TRACK LISTS
        self._new_tracks = []
        self._matched_tracks = []
        self._lost_tracks = []
        self._reserved_tracks = deque()

        # INITIALISE TRACKER
        self.initialize()

    def initialize(self):
        self._frame_count = 0
        self._new_tracks.clear()
        self._matched_tracks.clear()
        self._lost_tracks.clear()
        self._reserved_tracks.clear()

        for i in range(self._max_tracks):
            self._reserved_tracks.append(Track(i + 1))
        
        self._logger.info(f"----TRACKER INITIALISED - {self._max_tracks} RESERVED TRACKS")

    def get_new_tracks(self):
        return self._new_tracks
    
    def get_matched_tracks(self):
        return self._matched_tracks

    def get_lost_tracks(self):
        return self._lost_tracks
    
    def get_reserved_tracks(self):
        return self._reserved_tracks

    def get_num_tracks(self, tracks):
        return len(tracks)
    
    # TRACK ACTIVATION
    # ACTIVATION CONDITION FOR DETECTIONS TO BECOME NEW TRACKS
    def activation_condition(self, detection):
        return detection.confidence_score > self._new_track_threshold
    
    def activate_track(self, detection, tracks):
        if self.activation_condition(detection):
            activated_track = self._reserved_tracks.popleft()
            activated_track.activate(self._frame_count, detection)
            tracks.append(activated_track)
            # self.logger.info(f"Activated track {activated_track.id} for detection with score {detection.confidence_score}.")

    def activate_tracks(self, detections):
        self._logger.info("----ACTIVATING TRACKS")
        for detection in detections:
            self.activate_track(detection, self._new_tracks)

    # PARTITIONING
    def handle_matches(self, tracks, detections, matched_indexes, matched_tracks):
        for track_index, detection_index in matched_indexes:
            track = tracks[track_index]
            detection = detections[detection_index]
            track.update_matched(self._frame_count, detection)
            matched_tracks.append(track)
            # self._logger.info(f"Track {track.get_id()} matched with detection {detection.get_id()}.")

    # def handle_unmatched_tracks(self, tracks, unmatched_tracks_indexes, unmatched_tracks, reserved_tracks):
    #     for track_index in unmatched_tracks_indexes:
    #         track = tracks[track_index]
    #         track.update_unmatched()
    #         if track.get_track_state() == TrackState.LOST:
    #             unmatched_tracks.append(track)
    #         elif track.get_track_state() == TrackState.RESERVED:
    #             reserved_tracks.append(track)

    def handle_unmatched_tracks(self, unmatched_tracks, lost_tracks, reserved_tracks):
        for track in unmatched_tracks:
            track.update_unmatched()
            if track.track_state == TrackState.LOST:
                lost_tracks.append(track)
            elif track.track_state == TrackState.RESERVED:
                reserved_tracks.append(track)

    def handle_unmatched_detections(self, detections, unmatched_detections_indexes, unmatched_detections):
        for detection_index in unmatched_detections_indexes:
            detection = detections[detection_index]
            unmatched_detections.append(detection)

        # Process unmatched detections
    def process_new_tracks(self, unmatched_detections, new_tracks):
        for detection in unmatched_detections: 
            self.activate_track(detection, new_tracks)

    def partition(self, tracks, detections, matched_indexes, unmatched_tracks_indexes, unmatched_detections_indexes, matched_tracks, unmatched_tracks, unmatched_detections):
        self._logger.info(f"----PERFORMING PARTITIONING")
        self.handle_matches(tracks, detections, matched_indexes, matched_tracks)
        unmatched_tracks = [tracks[i] for i in unmatched_tracks_indexes]
        unmatched_detections = [detections[i] for i in unmatched_detections_indexes]
        return unmatched_tracks, unmatched_detections

    # ASSIGNMENT
    def assignment(self, tracks, detections_list, match_thresholds):
        self._logger.info("--- Assignment")
        matched_tracks, unmatched_detections_list = [], []
        curr_unmatched_tracks = tracks

        for i, detections in enumerate(detections_list):
            match_threshold = match_thresholds[i]
            matched_indexes, unmatched_tracks_indexes, unmatched_detections_indexes = linear_assignment(curr_unmatched_tracks, detections, match_threshold=match_threshold)
            unmatched_tracks, unmatched_detections = self.partition(curr_unmatched_tracks, detections, matched_indexes, unmatched_tracks_indexes, unmatched_detections_indexes, matched_tracks, [], [])
            curr_unmatched_tracks = unmatched_tracks
            unmatched_detections_list.append(unmatched_detections)
        
        return matched_tracks, curr_unmatched_tracks, unmatched_detections_list

    # Tracking
    def tracking(self, detections):
        self._logger.info(f"----TRACKING")

        matched_tracks_buffer, lost_tracks_buffer, new_tracks_buffer = [], [], []
        match_thresholds = [0.3, 0.1]
        
        high_detections, low_detections = [], []
        for detection in detections:
            if detection.confidence_score >= self._detection_threshold:
                high_detections.append(detection)
            else:
                low_detections.append(detection)
        
        detections_list = [high_detections, low_detections]
        
        matched_lost_tracks = self._matched_tracks + self._lost_tracks
        matched_tracks, unmatched_tracks, unmatched_detections_list = self.assignment(matched_lost_tracks, detections_list, match_thresholds)

        matched_tracks_buffer.extend(matched_tracks)
        self.handle_unmatched_tracks(unmatched_tracks, lost_tracks_buffer, self._reserved_tracks)
        unmatched_detections1 = unmatched_detections_list[0]

        matched_tracks1, unmatched_tracks1, unmatched_detections_list1 = self.assignment(self._new_tracks, [unmatched_detections1], [0.3])
        matched_tracks_buffer.extend(matched_tracks1)
        self.handle_unmatched_tracks(unmatched_tracks1, lost_tracks_buffer, self._reserved_tracks)

        # Handle new tracks
        unmatched_detections2 = unmatched_detections_list[1]
        self.process_new_tracks(unmatched_detections2, new_tracks_buffer)

        self._new_tracks = new_tracks_buffer
        self._matched_tracks = matched_tracks_buffer
        self._lost_tracks = lost_tracks_buffer

        print(self._new_tracks)
        print(self._matched_tracks)
        print(self._lost_tracks)

    def simple_tracking(self, tracks, detections):
        self._logger.info(f"----SIMPLE TRACKING")
        matched_tracks_buffer, lost_tracks_buffer, new_tracks_buffer = [], [], []

        matched_tracks, unmatched_tracks, unmatched_detections_list = self.assignment(tracks, [detections], [0.3])

        matched_tracks_buffer.extend(matched_tracks)
        self.handle_unmatched_tracks(unmatched_tracks, lost_tracks_buffer, self._reserved_tracks)
        unmatched_detections = unmatched_detections_list[0]
        self.process_new_tracks(unmatched_detections, new_tracks_buffer)

        self._new_tracks = new_tracks_buffer
        self._matched_tracks = matched_tracks_buffer
        self._lost_tracks = lost_tracks_buffer

    # UPDATE
    # TODO: UDPAT EHT ELOGGING STATEMENTs
    def update(self, frame_count, detections):
        self._frame_count = frame_count
        self._logger.info(f"----Updating for frame {self._frame_count}")

        num_matched = self.get_num_tracks(self._matched_tracks)
        num_lost = self.get_num_tracks(self._lost_tracks)
        num_new = self.get_num_tracks(self._new_tracks)

        # print("UPDATE")
        

        # If there are no matched or lost tracks, handle new tracks
        if num_matched == 0 and num_lost == 0:
            if num_new == 0:
                self.activate_tracks(detections)
            else:
                self._logger.info("Performing simple tracking with existing new tracks.")
                self.simple_tracking(self._new_tracks, detections)
        else:
            self._logger.info("Performing normal tracking.")
            self.tracking(detections)
            # Call normal tracking procedure (optional based on your implementation)

    # Display
    def display_tracks(self, frame):
        for track in self._new_tracks + self._matched_tracks + self._lost_tracks:
            track.display(frame)