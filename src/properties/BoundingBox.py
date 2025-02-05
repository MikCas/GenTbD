class BoundingBox:
    # THIS BOUNDING BOX IMPLEMENTATION ASSUMES X INCREASES FROM LEFT TO RIGHT, AND Y INCREASES FROM TOP TO BOTTOM (SO (0, 0) IS AT TOP-LEFT CORNER)

    __slots__ = ['_x_min', '_y_min', '_x_max', '_y_max'] # ATTRIBUTES

    def __init__(self, x_min=0.0, y_min=0.0, x_max=0.0, y_max=0.0):
        self._x_min = float(x_min)
        self._y_min = float(y_min)
        self._x_max = float(x_max)
        self._y_max = float(y_max)

    # CONSTRUCTORS
    @classmethod # DEFAULT 
    def from_corners(cls, x_min, y_min, x_max, y_max):
        return cls(float(x_min), float(y_min), float(x_max), float(y_max))

    @classmethod 
    def from_center(cls, center_x, center_y, width, height):
        x_min = float(center_x - width / 2)
        y_min = float(center_y - height / 2)
        x_max = float(center_x + width / 2)
        y_max = float(center_y + height / 2)
        return cls(x_min, y_min, x_max, y_max)

    # PROPERTIES
    @property
    def width(self): return abs(self._x_max - self._x_min)
    @property
    def height(self): return abs(self._y_max - self._y_min)
    @property
    def area(self): return self.width * self.height
    @property
    def center(self): return (self._x_min + self._x_max) / 2, (self._y_min + self._y_max) / 2
    
    # REPRESENTATIONS
    def xyxy(self): return self._x_min, self._y_min, self._x_max, self._y_max
    def xywh(self): return self._x_min, self._y_min, self.width, self.height
    def cxcywh(self):
        center_x, center_y = self.center
        return center_x, center_y, self.width, self.height

    # SIMILARITY
    def intersect(self, other):
        x_min, y_min = max(self._x_min, other._x_min), max(self._y_min, other._y_min)
        x_max, y_max = min(self._x_max, other._x_max), min(self._y_max, other._y_max)

        # IF VALID INTERSECTION
        if x_min < x_max and y_min < y_max:
            return BoundingBox(x_min, y_min, x_max, y_max)
        
        return None

    def iou(self, other):
        intersection = self.intersect(other)
        if intersection is None:
            return 0.0

        intersection_area = intersection.area
        union_area = self.area + other.area - intersection_area
        return intersection_area / union_area

    # OUTPUT
    def __repr__(self, format='corners'):
        if format == 'corners':
            return f"BB(xyxy=[{self._x_min}, {self._y_min}, {self._x_max}, {self._y_max}])"
        elif format == 'center':
            return f"BB(cxcywh=[{self.center[0]}, {self.center[1]}, {self.width}, {self.height}])"
        else:
            return f"BB(xyxy=[{self._x_min}, {self._y_min}, {self._x_max}, {self._y_max}])"