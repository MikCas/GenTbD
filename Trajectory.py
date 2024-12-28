from collections import OrderedDict

# TODO: ADD FUNCTIONALITY TO GET BEGINNING AND END FRAME OF CONSECUTIVE MATCHES - WHEN LOST GET THE CONSECUTIVE MATCHES AND OUTPUT THOSE ALONG WITH SIZE OF LAST CONSECUTIVE TRACK (END - BEGINNING)
# MAYBE DEFINE THIS AS A TUPLE (BEGINNING, END") - SIZE = END - BEGINNING + 1, AND THIS TUPLE IS CALLED THE MATCHED_FRAMES (CONSECUTIVE) - THIS CAN BE USED TO CALCULATE THE AVERAGE LENGTH OF A TRACK
# OUTPUT THIS IN THE LABEL OF THE TRACK TO SHOW THE LENGTH OF THE TRACK
class Trajectory(OrderedDict):

    __slots__ = ['max']  # Define slots to restrict instance attributes to only those listed

    def __init__(self, *args, max_size=0, **kwargs):
        if max_size < 0:
            raise ValueError("Maximum size must be 0 or greater")
        self.max_size = max_size
        super().__init__(*args, **kwargs)
    
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        
        # REMOVE OLDEST ITEM IF MAX SIZE EXCEEDED
        if len(self) > self.max_size:
            self.popitem(last=False)

    def has_detection_at_frame(self, frame_number):
        return frame_number in self

    def get_detection_at_frame(self, frame_number):
        # Returns either the detection at the specified frame or None
        return self.get(frame_number)

    def get_most_recent_frame(self):
        return max(self.keys()) if self else None

    def get_most_recent_detection(self):
        return self.get(self.get_most_recent_frame())

    def __repr__(self):
        return f"Trajectory({list(self.items())})"
    
    
