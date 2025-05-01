from .Property import Property

class BoundingBox(Property):
    """
    A class representing a bounding box in 2D space
    Note that x increases from left to right (->) and y increases from top to bottom (v)

    The bounding box can be represented in different formats:
    - Corners, xyxy:  Tuple[Float] (x_min, y_min, x_max, y_max)   
    - TopLeft xywh:   Tuple[Float] (x_min, y_min, width, height)
    - Center, cxcywh: Tuple[Float] (center_x, center_y, width, height)

    Note that the default representation is Corners (xyxy)
    """

    ##### ATTRIBUTES #####
    __slots__ = ['_x_min', '_y_min', '_x_max', '_y_max']

    @property
    def width(self): return abs(self._x_max - self._x_min)
    @property
    def height(self): return abs(self._y_max - self._y_min)
    @property
    def area(self): return self.width * self.height
    @property
    def center(self): return (self._x_min + self._x_max) / 2, (self._y_min + self._y_max) / 2

    ##### SETUP #####
    def __init__(self, x_min: float = 0.0, y_min: float = 0.0, x_max: float = 0.0, y_max: float = 0.0):
        """
        Args:
            x_min (float): Minimum x-coordinate (left)
            y_min (float): Minimum y-coordinate (top)
            x_max (float): Maximum x-coordinate (right)
            y_max (float): Maximum y-coordinate (bottom)

        Raises:
            ValueError: If any of the coordinates are negative or if x_max < x_min or y_max < y_min
        """

        # Validate input
        if x_max < x_min:
            raise ValueError("x_max must be greater than or equal to x_min")
        if y_max < y_min:
            raise ValueError("y_max must be greater than or equal to y_min")
        
        self._x_min = float(x_min)
        self._y_min = float(y_min)
        self._x_max = float(x_max)
        self._y_max = float(y_max)
    
    @classmethod 
    def from_corners(cls, 
                     x_min: float, 
                     y_min: float, 
                     x_max: float, 
                     y_max: float):
        """
        Create a BoundingBox instance from corner coordinates (xyxy)

        Args:
            x_min (float): Minimum x-coordinate (left)
            y_min (float): Minimum y-coordinate (top)
            x_max (float): Maximum x-coordinate (right)
            y_max (float): Maximum y-coordinate (bottom)
        Returns:
            BoundingBox: A new instance of BoundingBox
        """
        return cls(x_min, y_min, x_max, y_max)
    def xyxy(self): return self._x_min, self._y_min, self._x_max, self._y_max
    
    @classmethod 
    def from_top_left(cls, 
                   x_min: float, 
                   y_min: float, 
                   width: float, 
                   height: float):
        """
        Create a BoundingBox instance from top-left coordinates (xywh)

        Args:
            x_min (float): Minimum x-coordinate (left)
            y_min (float): Minimum y-coordinate (top)
            width (float) : bounding box width
            height (float): bounding box height
        Returns:
            BoundingBox: A new instance of BoundingBox
        Raises:
            ValueError: If width or height is negative.
        """

        # Validate input
        if width < 0 or height < 0:
            raise ValueError("Width and height must be non-negative.")

        x_max = float(x_min + width)
        y_max = float(y_min + height)
        return cls(x_min, y_min, x_max, y_max)
    def xywh(self): return self._x_min, self._y_min, self.width, self.height
    
    @classmethod 
    def from_center(cls, 
                    center_x: float, 
                    center_y: float, 
                    width: float, 
                    height: float):
        """
        Create a BoundingBox instance from center coordinates (cxcywh)

        Args:
            center_x (float): center x-coordinate
            center_y (float): center y-coordinate
            width (float)   : bounding box width
            height (float)  : bounding box height
        Returns:
            BoundingBox: A new instance of BoundingBox
        Raises:
            ValueError: If width or height is negative.
        """
        # Validate input
        if width < 0 or height < 0:
            raise ValueError("Width and height must be non-negative.")
        
        x_min = float(center_x - width / 2)
        y_min = float(center_y - height / 2)
        x_max = float(center_x + width / 2)
        y_max = float(center_y + height / 2)
        return cls(x_min, y_min, x_max, y_max)
    def cxcywh(self): return self.center, self.width, self.height

    ####### FUNCTIONS #####
    def __eq__(self, other: 'BoundingBox') -> bool:
        """
        Check if two bounding boxes are equal.
        Args:
            other (BoundingBox): Another bounding box to compare with.
        Returns:
            bool: True if the bounding boxes are equal, False otherwise.
        """
        if not isinstance(other, BoundingBox):
            return False
        return (self._x_min == other._x_min and
                self._y_min == other._y_min and
                self._x_max == other._x_max and
                self._y_max == other._y_max)

    @staticmethod
    def iou(box1: 'BoundingBox', box2: 'BoundingBox') -> float:
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Args:
            box1 (BoundingBox): First bounding box.
            box2 (BoundingBox): Second bounding box.
        Returns:
            float: The IoU value between 0 and 1.
        Raises:
            TypeError: If either box1 or box2 is not an instance of BoundingBox.
        """
        if not isinstance(box1, BoundingBox) or not isinstance(box2, BoundingBox):
            raise TypeError("Both arguments must be instances of BoundingBox. Got {type(box1)} and {type(box2)}")
                
        # Calculate intersection
        x_min = max(box1._x_min, box2._x_min)
        y_min = max(box1._y_min, box2._y_min)
        x_max = min(box1._x_max, box2._x_max)
        y_max = min(box1._y_max, box2._y_max)

        # If there is no intersection, return 0.0
        if x_min >= x_max or y_min >= y_max:
            return 0.0

        # Calculate areas
        intersection_area = (x_max - x_min) * (y_max - y_min)
        union_area = box1.area + box2.area - intersection_area

        # Return IoU
        return intersection_area / union_area if union_area > 0 else 0.0

    def similarity(self, other: 'BoundingBox') -> float:
        """
        Calculate the similarity between this bounding box and another bounding box.

        Args:
            other (BoundingBox): Another bounding box to calculate similarity with.

        Returns:
            float: The IoU value between 0 and 1.
        """
        return BoundingBox.iou(self, other)

    ###### DISPLAY ######
    def __str__(self, format: str ='corners') -> str:
        """
        String representation of the bounding box in the specified format.
        Args:
            format (str): The format to use for the string representation. Options are 'corners', 'top_left', or 'center'.
        Returns:
            str: The string representation of the bounding box.
        Raises:
            ValueError: If an invalid format is specified.
        """
        if format == 'corners':
            return f"BB(xyxy=[{self._x_min:.2f}, {self._y_min:.2f}, {self._x_max:.2f}, {self._y_max:.2f}])"
        elif format == 'top_left':
            return f"BB(xywh=[{self._x_min:.2f}, {self._y_min:.2f}, {self.width:.2f}, {self.height:.2f}])"
        elif format == 'center':
            return f"BB(cxcywh=[{self.center[0]:.2f}, {self.center[1]:.2f}, {self.width:.2f}, {self.height:.2f}])"
        else:
            raise ValueError(f"Invalid format '{format}'. Use 'corners', 'top_left', or 'center'.")