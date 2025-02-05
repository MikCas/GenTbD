from collections import OrderedDict

# TODO: BE ABLE TO GET THE BEGIN AND END OF CONSECUTIVE MATCHES - WHEN LOST A NEW CONSECUTIVE MATCH IS STARTED
# CONSECUTIVE_MATCH = (BEGINNING, END), SIZE = END - BEGINNING + 1
# A CONSECUTIVE MATCH CAN BE USED TO CALCULATE THE AVERAGE LENGTH OF A TRACK

class Trajectory(OrderedDict):

    __slots__ = ['_max_size'] 

    def __init__(self, *args, max_size=0, **kwargs):
        if max_size < 0:
            raise ValueError("Maximum size must be 0 or greater")
        self._max_size = max_size
        super().__init__(*args, **kwargs)
    
    # PROPERTIES
    @property
    def max_size(self): return self._max_size
    
    # TRAJECTORY FUNCTIONALITY 
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        
        # REMOVE OLDEST ITEM IF MAX SIZE EXCEEDED
        if len(self) > self.max_size:
            self.popitem(last=False)
    
    def has_detection_at_frame(self, frame_number): return frame_number in self
    def get_detection_at_frame(self, frame_number): return self.get(frame_number)
    def get_most_recent_frame(self): return max(self.keys()) if self else None
    def get_most_recent_detection(self): return self.get(self.get_most_recent_frame())

    def __repr__(self): return f"Trajectory({list(self.items())})"
    
    
