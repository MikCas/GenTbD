from collections import OrderedDict

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
    
    
