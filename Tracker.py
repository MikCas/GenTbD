import logging
from pprint import pprint
from collections import deque
from Track import Track, TrackState
from Assignment import linear_assignment

class Tracker:
    def __init__(self, max_tracks=100, detection_threshold=0.3, activation_threshold=0.4, match_threshold=0.05, logger=None):
        
        # INITIALISE LOGGER, OTHERWISE USE DEFAULT LOGGER
        self._logger = logger if logger else logging.getLogger(__name__)

        # TRACKER PARAMETERS
        self._frame_count = 0
        self._max_tracks = max_tracks
        self._detection_threshold = detection_threshold
        self._activation_threshold = activation_threshold
        self._match_threshold = match_threshold

        # TRACK LISTS
        self._new_tracks = []
        self._matched_tracks = []
        self._lost_tracks = []
        self._reserved_tracks = deque()

        # INITIALISE TRACKER
        self.initialize()

    def initialize(self):
        self._logger.info(f"----INITIALISING TRACKER - {self._max_tracks} TRACKS - {self.get_detection_threshold()} DETECTION THRESHOLD - {self.get_activation_threshold()} ACTIVATION THRESHOLD - {self.get_match_threshold()} MATCH THRESHOLD")
        self._frame_count = 0
        self._new_tracks.clear()
        self._matched_tracks.clear()
        self._lost_tracks.clear()
        self._reserved_tracks.clear()

        for i in range(self._max_tracks):
            self.return_reserved_track(Track(i + 1))

    def reset(self):
        self.initialize()
        self._logger.info("----RESET TRACKER")

   # TAKE FROM RESERVED TRACKS LIST
    def take_reserved_track(self):
        if self._reserved_tracks:
            track = self._reserved_tracks.popleft()
            self._logger.info(f"----TRACK {track.id} TAKEN FROM RESERVED TRACKS.")
            self.output_tracks(self._reserved_tracks, "RESERVED TRACKS")
            return track
        else:
            self._logger.warning("----NO RESERVED TRACKS AVAILABLE")
            return None

    # RETURN TO RESERVED TRACKS LIST
    # TODO: MAYBE ADD A CHECK TO SEE IF THE TRACK IS NOT ALREADY IN THE RESERVED TRACKS LIST
    # TODO: WHEN DEACTIVATING A TRACK, SHIULD I ALSO WIPE TRAJECOTRY?
    def return_reserved_track(self, track):
        self._reserved_tracks.append(track)
        self._logger.info(f"----TRACK {track.id} RETURNED TO RESERVED TRACKSS.")
        self.output_tracks(self._reserved_tracks, "RESERVED TRACKS")

    # GETTERS
    def get_detection_threshold(self):
        return self._detection_threshold
    
    def get_activation_threshold(self):
        return self._activation_threshold
    
    def get_match_threshold(self):
        return self._match_threshold

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
    
    def update_frame_count(self, frame_count): 
        self._frame_count = frame_count

    # CONDITIONS
    # DETECTION CONDITION TO PARTITION DETECTIONS 
    # TODO: CURRENTLY BASED ON CONFIDENCE SCORE BUT CAN BE CHANGED 
    def detection_condition(self, detection):
        return detection.confidence_score >= self.get_detection_threshold()
    
    # ACTIVATE CONDITION FOR DETECTIONS TO BECOME NEW TRACKS
    def activate_condition(self, detection):
        return detection.confidence_score >= self.get_activation_threshold()
    
    # PARTITION DETECTIONS BASED ON DETECTION THRESHOLD
    def partition_detections(self, detections):
        detections_high, detections_low = [], []

        # for detection in detections:
        #     if self.detection_condition(detection):
        #         detections_high.append(detection)
        #     else:
        #         detections_low.append(detection)
        
        detections_high = [detection for detection in detections if self.detection_condition(detection)]
        detections_low = [detection for detection in detections if not self.detection_condition(detection)]

        return [detections_high, detections_low]

    # TRACK MANAGEMENT
    # CHECKS IF A DETECTION PASSES THE ACTIVATION CONDITION AND ACTIVATES A NEW TRACK FROM THE RESERVED TRACKS
    # TODO: NEED TO DOUBLE CHECK WHAT THIS FUNCTION OTUPUTS WHEN ACTIVATE TRACK IS PASSED AND THERE ARE NO RESERVED TRACKS LEFT
    def activate_tracks(self, detections, tracks):
        # self._logger.info("----ACTIVATING")
        for detection in detections:
            if self.activate_condition(detection):
                activated_track = self.take_reserved_track()
                if activated_track is not None:
                    activated_track.activate(self._frame_count, detection)
                    tracks.append(activated_track)
        
    # UPDATED MATCHED TRACKS WITH NEW DETECTIONS
    def process_matched_tracks(self, tracks, detections, matched_tracks):
        for track, detection in zip(tracks, detections):
            track.update_matched(self._frame_count, detection)
            matched_tracks.append(track)

    # UPDATES UNMATCHED TRACKS AND DEPENDING ON CONDITION PLACES THEM IN LOST AND RESERVED TRACKS
    def process_unmatched_tracks(self, tracks, lost_tracks):
        for track in tracks:
            track.update_unmatched()
            if track.track_state == TrackState.LOST:
                lost_tracks.append(track)
            elif track.track_state == TrackState.RESERVED:
                self.return_reserved_track(track)
    
    # TODO: THIS IS BASED ON THE TREE DIAGRAM I WAS DRAWING, SO MAYBE DRAW A TREE DIAGRAM TO EXPLAIN THIS
    # PERFORM A CASCADED ASSIGNMENT TO CONTINUOUSLY MATCH TRACKS IN A NUMBER OF STAGES - GIVEN A LIST OF DETECTIONS (AND MATCH THRESHOLDS)
    def assignment(self, tracks, detections_list, match_thresholds):
        # self._logger.info("----ASSIGNMENT")
        matched_tracks_all = [] # TRACKS MATCHED ACROSS ALL ASSIGNMENT STAGES
        unmatched_detections_list = [] # UNMATCHED DETECTIONS LIST AT EACH ASSIGNMENT STAGE 
        curr_unmatched_tracks = tracks # CURRENT UNMATCHED TRACKS FOR FURTHER ASSIGNMENT

        for detections, match_threshold in zip(detections_list, match_thresholds):

            # LINEAR ASSIGNMENT
            matched_tracks, matched_detections, unmatched_tracks, unmatched_detections = linear_assignment(curr_unmatched_tracks, detections, match_threshold=match_threshold)

            # UPDATE MATCHED TRACKS WITH MATCHED DETECTIONS AND PLACE IN THE MATCHED TRACKS LIST
            self.process_matched_tracks(matched_tracks, matched_detections, matched_tracks_all)
            curr_unmatched_tracks = unmatched_tracks
            unmatched_detections_list.append(unmatched_detections)
        
        return matched_tracks_all, curr_unmatched_tracks, unmatched_detections_list
    
    # UPDATE TRACKS
    def track_management(self, matched_tracks_list, unmatched_tracks_list, unmatched_detections_list):
        matched_tracks, lost_tracks, new_tracks = [], [], []

        # MATCHED TRACKS 
        # TODO: NOT DOIGN PROCESS_MATCHED_TRACKS HERE SINCE THAT IS PERFORMED IN THE ASSIGNMENT FUNCTION 
        for tracks in matched_tracks_list:
            matched_tracks.extend(tracks)

        # UNMATCHED TRACKS
        for tracks in unmatched_tracks_list:
            self.process_unmatched_tracks(tracks, lost_tracks)
        
        # UNMATCHED DETECTIONS - NEW TRACKS
        for detections in unmatched_detections_list:
            self.activate_tracks(detections, new_tracks)

        self._new_tracks = new_tracks
        self._matched_tracks = matched_tracks
        self._lost_tracks = lost_tracks

    # INITIAL TRACKING PROCEDURE
    def initial_tracking(self, detections):
        self._logger.info(f"----INITIAL TRACKING")

        new_tracks = self._new_tracks
        
        detections_list = [detections]
        match_thresholds = [self.get_match_threshold()]

        matched_tracks, unmatched_tracks, unmatched_detections_list = self.assignment(new_tracks, detections_list, match_thresholds)
        matched_tracks_list = [matched_tracks]
        unmatched_tracks_list = [unmatched_tracks]

        self.track_management(matched_tracks_list, unmatched_tracks_list, unmatched_detections_list)

    # GENERAL TRACKING PROCEDURE
    # TODO: CUSTOMISABLE ASSIGNMENT CASCADES 
    def subsequent_tracking(self, detections):
        self._logger.info(f"----SUBSEQUENT TRACKING")
        
        matched_lost_tracks = self.get_matched_tracks() + self.get_lost_tracks()
        new_tracks = self.get_new_tracks()
        
        detections_list = self.partition_detections(detections)
        match_thresholds1 = [self.get_match_threshold(), 0.1]
        match_thresholds2 = [self.get_match_threshold()]

        # ASSIGNMENT CASCADE 1 - MATCHED AND LOST TRACKS
        matched_tracks1, unmatched_tracks1, unmatched_detections_list1 = self.assignment(matched_lost_tracks, detections_list, match_thresholds1)
        unmatched_detections11 = unmatched_detections_list1[0] # HIGH SCORING UNMATCHED DETECTIONS

        # ASSIGNMENT CASCADE 2 -  NEW TRACKS
        matched_tracks2, unmatched_tracks2, unmatched_detections_list2 = self.assignment(new_tracks, [unmatched_detections11], match_thresholds2)
        unmatched_detections21 = unmatched_detections_list2[0]  # UNMATCHED HIGH SCORING UNMATCHED DETECTIONS 

        matched_tracks_list = [matched_tracks1, matched_tracks2]
        unmatched_tracks_list = [unmatched_tracks1, unmatched_tracks2]
        unmatched_detections_list = [unmatched_detections21]
        self.track_management(matched_tracks_list, unmatched_tracks_list, unmatched_detections_list)

    # UPDATE TRACKS
    def update(self, frame_count, detections):
        self.update_frame_count(frame_count)
        self._logger.info(f"----UPDATE TRACKER")

        num_new = self.get_num_tracks(self._new_tracks)  
        num_matched = self.get_num_tracks(self._matched_tracks)
        num_lost = self.get_num_tracks(self._lost_tracks)
              
        # IF THERE ARE NO TRACKS WHICH MATCHED (MATCHED OR LOST), THEN FOCUS ON NEW TRACKS
        if num_matched == 0 and num_lost == 0:
            # IF THERE ARE NO NEW TRACKS, ACTIVATE NEW TRACKS FROM DETECTIONS
            if num_new == 0:
                self.activate_tracks(detections, self._new_tracks)
            else:
                self.initial_tracking(detections)
        else:
            self.subsequent_tracking(detections)

        self._logger.info(f"--------TRACK LISTS")
        self.output_tracks(self._new_tracks, "NEW TRACKS")
        self.output_tracks(self._matched_tracks, "MATCHED TRACKS")
        self.output_tracks(self._lost_tracks, "LOST TRACKS")

    # DISPLAY
    def display_tracks(self, frame):
        for track in self.get_new_tracks():
            track.display(frame)

        for track in self.get_matched_tracks():
            track.display(frame)

        for track in self.get_lost_tracks():
            track.display(frame) 
    
    # OUTPUT TRACKS
    def output_tracks(self, tracks, tracks_name):
       track_ids = [track.id for track in tracks]
       self._logger.info(f"--------{tracks_name} - {track_ids}")


