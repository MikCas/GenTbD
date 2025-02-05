class Partition:

    # CONSTRUCTOR
    def __init__(self, matches=None, unmatched_x=None, unmatched_y=None):
        self._matches = matches if matches is not None else []
        self._unmatched_x = unmatched_x if unmatched_x is not None else []
        self._unmatched_y = unmatched_y if unmatched_y is not None else []
    
    # UNMATCHED PARTITION 
    @classmethod
    def create_unmatched(cls, xs, ys):
        return cls(matches=None, unmatched_x=xs, unmatched_y=ys)

    # PROPERTIES
    @property
    def matches(self): return self._matches
    @property
    def unmatched_x(self): return self._unmatched_x
    @property
    def unmatched_y(self): return self._unmatched_y

    # UTILITY
    def is_empty(self):
        return (len(self.matches) == 0 and 
                len(self.unmatched_x) == 0 and
                len(self.unmatched_y) == 0)

    def total_size(self):
        return (len(self.matches) + 
                len(self.unmatched_x) + 
                len(self.unmatched_y))

    def clear(self):
        self.matches.clear()
        self.unmatched_x.clear()
        self.unmatched_y.clear()

    def __str__(self):
        return (f"Partition:\n"
                f"  Matches: {len(self.matches)}\n"
                f"  Unmatched X: {len(self.unmatched_x)}\n"
                f"  Unmatched Y: {len(self.unmatched_y)}")