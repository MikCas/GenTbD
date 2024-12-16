from collections import OrderedDict

class Trajectory(OrderedDict):
    def __init__(self, *args, max=0, **kwargs):
        if max < 0:
            raise ValueError("Maximum value must be 0 or greater")
        self.max = max
        super().__init__(*args, **kwargs)
    
    def __setitem__(self, key, value):
        super().__setitem__(key, value)

        # Remove the oldest item in the trajectory
        if len(self) > self.max:
            self.popitem(last=False)

    def hasDetectionAtFrame(self, key):
        return key in self.keys()

    def getDetectionAtFrame(self, key):
        # Returns either the detection at the specified frame or None
        return self.get(key)

    def getMostRecentFrame(self):
        return max(self.keys()) if len(self) > 0 else None

    def getMostRecentDetection(self):
        return self.get(max(self.keys()))

    def __repr__(self):
        return f"TRAJECTORY({list(self.items())})"
    
    
