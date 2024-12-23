class BoundingBox:
        
    def __init__(self, xMin=0, yMin=0, xMax=0, yMax=0):
        self.xMin = xMin
        self.yMin = yMin
        self.xMax = xMax
        self.yMax = yMax

    @classmethod
    def fromCorners(cls, xMin, yMin, xMax, yMax):
        """Create a BoundingBox using corner coordinates."""
        return cls(xMin, yMin, xMax, yMax)

    @classmethod
    def fromCenter(cls, centerX, centerY, width, height):
        """Create a BoundingBox using center coordinates and dimensions."""
        xMin = centerX - width / 2
        yMin = centerY - height / 2
        xMax = centerX + width / 2
        yMax = centerY + height / 2
        return cls(xMin, yMin, xMax, yMax)

    def width(self):
        return abs(self.xMax - self.xMin)

    def height(self):
        return abs(self.yMax - self.yMin)

    def area(self):
        return self.width() * self.height()
    
    def center(self):
        return [(self.xMin + self.xMax) / 2, (self.yMin + self.yMax) / 2]
    
    def xyxy(self):
        return [self.xMin, self.yMin, self.xMax, self.yMax]
    
    def xywh(self):
        return [self.xMin, self.yMin, self.width(), self.height()]
    
    def cxcywh(self):
        cx, cy = self.center()
        return [cx, cy, self.width(), self.height()]

    def intersect(self, other):
        xMin = max(self.xMin, other.xMin)
        yMin = max(self.yMin, other.yMin)
        xMax = min(self.xMax, other.xMax)
        yMax = min(self.yMax, other.yMax)

        if xMin < xMax and yMin < yMax:
            return BoundingBox(xMin, yMin, xMax, yMax)
        else:
            return None

    def iou(self, other):
        intersection = self.intersect(other)
        if intersection is None:
            return 0.0

        intersectionArea = intersection.area()
        unionArea = self.area() + other.area() - intersectionArea
        return intersectionArea / unionArea
    
    def __repr__(self):
        return f"BoundingBox(xMin={self.xMin}, yMin={self.yMin}, xMax={self.xMax}, yMax={self.yMax})"
        # return f"{self.xywh()}"
