import logging

from Track import Track, TrackState
from Partition import Partition
from Assignment import linear_assignment

class Tracker:

    def __init__(self, detection_threshold=0.3, activation_threshold=0.4, match_threshold=0.05, logger=None):
        
        self._logger = logger if logger else None
        
        # TRACKER PARAMETERS
        self._detection_threshold = detection_threshold         # USED TO PARTITION DETECTIONS IN SUBSEQUENT TRACKING PROCEDURE
        self._activation_threshold = activation_threshold       # USED TO ACTIVATE NEW TRACKS
        self._match_threshold = match_threshold                 # USED TO DETERMINE MATCHES BETWEEN TRACKS AND DETECTIONS
        
        # INIT COUNTERS
        self._frame_count = 0                                   # FRAME COUNT (UPDATED IN UPDATE FUNCTION)
        self._track_count = 1                                   # TRACK COUNT (UPDATED WHEN NEW TRACK IS ACTIVATED)
        
        # INIT TRACK LISTS
        self._new_tracks = []                                   # NEW TRACKS
        self._matched_tracks = []                               # MATCHED TRACKS
        self._lost_tracks = []                                  # LOST TRACKS
        self._reserved_tracks = []                              # RESERVED TRACKS
        
        if self.logger: self.logger.info(f"////////INIT TRACKER - D:{detection_threshold}, A:{activation_threshold}, M:{match_threshold}")

    ####### GETTERS/SETTERS #######
    @property
    def logger(self): return self._logger
    @property 
    def frame_count(self): return self._frame_count
    @property 
    def track_count(self): return self._track_count
    @property
    def detection_threshold(self): return self._detection_threshold
    @property
    def activation_threshold(self): return self._activation_threshold
    @property
    def match_threshold(self): return self._match_threshold
    @property
    def new_tracks(self): return self._new_tracks
    @property
    def matched_tracks(self): return self._matched_tracks
    @property
    def lost_tracks(self): return self._lost_tracks
    @property
    def reserved_tracks(self): return self._reserved_tracks

    @frame_count.setter
    def frame_count(self, value): self._frame_count = value
    @track_count.setter 
    def track_count(self, value): self._track_count = value
    @detection_threshold.setter
    def detection_threshold(self, value): self._detection_threshold = value
    @activation_threshold.setter
    def activation_threshold(self, value): self._activation_threshold = value
    @match_threshold.setter
    def match_threshold(self, value): self._match_threshold = value

    ####### UTILITY ####### 
    def increment_track_count(self): self.track_count += 1 # INCREMENT TRACK COUNT WHEN NEW TRACK IS ACTIVATED
    def reset_tracks(self):
        # ONLY NEW, MATCHED, LOST ARE CLEARED SINCE RESERVED TRACKS ARE NOT USED IN TRACKING PROCEDURE AS THEY STORE PERMANENTLY UNUSABLE TRACKS (THIS CAN BE CHANGED)
        self.new_tracks.clear()
        self.matched_tracks.clear()
        self.lost_tracks.clear()
    def reset(self):
        if self.logger: self.logger.info("////////RESET TRACKER")
        self.frame_count = 0
        self.track_count = 1
        self.reset_tracks()

    ####### MANAGE DETECTIONS #######
    def detection_condition(self, detection): return detection.confidence_score >= self.detection_threshold     # TO PARTITION DETECTIONS
    def partition_detections(self, detections):
        detections_high = [detection for detection in detections if self.detection_condition(detection)]
        detections_low = [detection for detection in detections if not self.detection_condition(detection)]
        return [detections_high, detections_low]
    
    ####### MANAGE TRACKS #######
    def deactivate_track(self, track):
        self.reserved_tracks.append(track)
        if self.logger: self.logger.info(f"////////////TRACK {track.id} DEACTIVATED.")
        self.output_tracks(self.reserved_tracks, "RESERVED TRACKS")

    def activation_condition(self, detection): return detection.confidence_score >= self.activation_threshold   # TO ACTIVATE TRACKS
    def activate_track(self, detection):
        track = Track(self.track_count)
        track.activate(self.frame_count, detection)
        self.increment_track_count()
        if self.logger: self.logger.info(f"////////////TRACK {track.id} ACTIVATED.")
        return track
    def activate_tracks(self, detections): # TODO: REMOVE TRACKS PARAMERTERS OR ADD DEFAULT TO SELF.NEWTRACKS
        if self.logger: self.logger.info("////////////ACTIVATING TRACKS")
        for detection in detections:
            if self.activation_condition(detection):
                track = self.activate_track(detection)
                self.new_tracks.append(track)
        
    def track_management(self, partition):

        if self.logger: self.logger.info("////////////MANAGE TRACKS")
        self.reset_tracks() # RESET MATCHED, LOST, NEW TRACKS

        self.matched_tracks.extend(partition.matches) # MATCHED TRACKS - (MATCHES ALREADY PROCESSED INSIDE ASSIGNMENT FUNCTION)

        for track in partition.unmatched_x: # UNMATCHED TRACKS
            track.update_unmatched()
            if track.track_state == TrackState.LOST: self.lost_tracks.append(track)
            elif track.track_state == TrackState.RESERVED: self.deactivate_track(track)
            else: raise ValueError("Invalid state for unmatched track")
        
        self.activate_tracks(partition.unmatched_y) # UNMATCHED DETECTIONS ARE POTENTIAL NEW TRACKS

    ####### ASSIGNMENT #######
    def assignment(self, tracks, detections, match_threshold):
        if self.logger: self.logger.info("////////////ASSIGNMENT")
        partition = linear_assignment(self.frame_count, tracks, detections, match_threshold) 
        return partition
    
    def initial_tracking(self, detections): # INITIAL TRACKING PROCEDURE
        if self.logger: self.logger.info("////////INITIAL TRACKING")
        partition = self.assignment(self.new_tracks, detections, self.match_threshold)
        self.track_management(partition)

    def subsequent_tracking(self, detections):  # GENERAL TRACKING PROCEDURE
        if self.logger: self.logger.info("////////SUBSEQUENT TRACKING")
        
        matched_lost_tracks = self.matched_tracks + self.lost_tracks            # MATCHED AND LOST TRACKS
        detections_high, detections_low = self.partition_detections(detections) # HIGH AND LOW SCORING DETECTIONS (BASED ON DETECTION THRESHOLD)

        partition1 = self.assignment(matched_lost_tracks, detections_high, self.match_threshold)    # MATCHED AND LOST TRACKS WITH HIGH SCORING DETECTIONS
        partition2 = self.assignment(partition1.unmatched_x, detections_low, self.match_threshold)  # UNMATCHED PARTITION1 TRACKS (MATCHED AND LOST), WITH LOW SCORING DETECTIONS
        partition3 = self.assignment(self.new_tracks, partition1.unmatched_y, self.match_threshold) # NEW TRACKS WITH UNMATCHED HIGH SCORING DETECTIONS

        matched_tracks = partition1.matches + partition2.matches + partition3.matches               # MATCHES FROM ALL ASSIGNMENTS
        unmatched_tracks = partition2.unmatched_x + partition3.unmatched_x                          # UNMATCHED TRACKS FROM SECOND AND THIRD ASSIGNMENT (FIRST ASSIGNMENT UNAMTCHED TRACKS ARE PASSED ON TO THE NEXT ASSIGNMENT)
        unmatched_detections = partition3.unmatched_y                                               # UNMATCHED DETECTIONS FROM THIRD ASSIGNMENT - ONLY HIGH SCORING UNMATCHED DETECTIONS ARE PASSED

        partition = Partition(matches=matched_tracks, unmatched_x=unmatched_tracks, unmatched_y=unmatched_detections)
        self.track_management(partition)

        ####### TODO: ASSIGNMENT CASCADE IMPLEMENTATION #######
        # # ASSIGNMENT CASCADE 1 - MATCHED AND LOST TRACKS
        # matched_tracks1, unmatched_tracks1, unmatched_detections_list1 = self.assignment(matched_lost_tracks, detections_list, match_thresholds1)
        # unmatched_detections11 = unmatched_detections_list1[0] # HIGH SCORING UNMATCHED DETECTIONS

        # # ASSIGNMENT CASCADE 2 -  NEW TRACKS
        # matched_tracks2, unmatched_tracks2, unmatched_detections_list2 = self.assignment(new_tracks, [unmatched_detections11], match_thresholds2)
        # unmatched_detections21 = unmatched_detections_list2[0]  # UNMATCHED HIGH SCORING UNMATCHED DETECTIONS 

        # matched_tracks_list = [matched_tracks1, matched_tracks2]
        # unmatched_tracks_list = [unmatched_tracks1, unmatched_tracks2]
        # unmatched_detections_list = [unmatched_detections21]
        # self.track_management(matched_tracks_list, unmatched_tracks_list, unmatched_detections_list)

    ####### UPDATE TRACKS #######
    def update(self, frame_count, detections):
        self.frame_count = frame_count

        num_new = len(self.new_tracks)  
        num_matched = len(self.matched_tracks)
        num_lost = len(self.lost_tracks)
        if self.logger: self.logger.info(f"////////UPDATING TRACKER: MATCHED:{num_matched}, LOST:{num_lost}, NEW:{num_new}")

        # IF NO MATCHED OR LOST TRACKS, THEN FOCUS ON NEW TRACKS
        if num_matched == 0 and num_lost == 0:
            # IF NO NEW TRACKS, ACTIVATE NEW TRACKS FROM DETECTIONS
            if num_new == 0:
                self.activate_tracks(detections)
            else:
                self.initial_tracking(detections)
        else:
            self.subsequent_tracking(detections)

        self.output_tracks(self.new_tracks, "NEW TRACKS")
        self.output_tracks(self.matched_tracks, "MATCHED TRACKS")
        self.output_tracks(self.lost_tracks, "LOST TRACKS")

    ####### OUTPUT #######
    def display_tracks(self, frame, mode='state'):
        for track in self.new_tracks: track.display(frame, mode=mode)
        for track in self.matched_tracks: track.display(frame, mode=mode)  
        for track in self.lost_tracks: track.display(frame, mode=mode) 
    
    def output_tracks(self, tracks, tracks_name):
       track_ids = [track.id for track in tracks]
       if self.logger: self.logger.info(f"////////{tracks_name} - {track_ids}")

# # TODO: THIS IS BASED ON THE TREE DIAGRAM I WAS DRAWING, SO MAYBE DRAW A TREE DIAGRAM TO EXPLAIN THIS
# # PERFORM A CASCADED ASSIGNMENT TO CONTINUOUSLY MATCH TRACKS IN A NUMBER OF STAGES - GIVEN A LIST OF DETECTIONS (AND MATCH THRESHOLDS)
# def cascaded_assignment(self, tracks, detections_list, match_thresholds):
#     self._logger.info("----ASSIGNMENT")
#     matched_tracks_all = []         # TRACKS MATCHED ACROSS ALL ASSIGNMENT STAGES
#     unmatched_detections_list = []  # UNMATCHED DETECTIONS LIST AT EACH ASSIGNMENT STAGE 
#     curr_unmatched_tracks = tracks  # CURRENT UNMATCHED TRACKS FOR FURTHER ASSIGNMENT

#     for detections, match_threshold in zip(detections_list, match_thresholds):

#         # LINEAR ASSIGNMENT
#         matched_tracks, matched_detections, unmatched_tracks, unmatched_detections = linear_assignment(curr_unmatched_tracks, detections, match_threshold=match_threshold)
#         # UPDATE MATCHED TRACKS WITH MATCHED DETECTIONS AND PLACE IN THE MATCHED TRACKS LIST
#         self.process_matched_tracks(matched_tracks, matched_detections, matched_tracks_all)
#         curr_unmatched_tracks = unmatched_tracks
#         unmatched_detections_list.append(unmatched_detections)
    
#     return matched_tracks_all, curr_unmatched_tracks, unmatched_detections_list





