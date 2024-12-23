class PartitionList(list):
    def __init__(self, inputList=None):
        # Initialize the base list and partitioned lists (matched and unmatched)
        super().__init__(inputList if inputList else [])
        self.matched = []
        self.unmatched = []
        self.trackMap = {}

    def addItem(self, item):
        """Adds an item to the list (inherited from list class)."""
        super().append(item)

    def clearAll(self):
        """Clears both matched and unmatched partitions along with the original list."""
        super().clear()
        self.matched.clear()
        self.unmatched.clear()

    def partition(self, matchedIndexes, unmatchedIndexes):
        """Partitions the list into matched and unmatched based on the provided indexes."""
        self.matched = [self[i] for i in matchedIndexes]
        self.unmatched = [self[i] for i in unmatchedIndexes]
        return self.matched, self.unmatched

    def moveToMatched(self, item):
        """Moves an item to the matched partition."""
        if item in self.unmatched:
            self.unmatched.remove(item)
            self.matched.append(item)

    def moveToUnmatched(self, item):
        """Moves an item to the unmatched partition."""
        if item in self.matched:
            self.matched.remove(item)
            self.unmatched.append(item)

    def __repr__(self):
        return f"PartitionList(List: {len(self)}, Matched: {len(self.matched)}, Unmatched: {len(self.unmatched)})"


# Example usage:
inputList = [1, 2, 3, 4, 5]
matchedIndexes = [0, 2, 4]
unmatchedIndexes = [1, 3]

# Create the PartitionList from the input list
partitionList = PartitionList(inputList)

# Partition the list into matched and unmatched
matched, unmatched = partitionList.partition(matchedIndexes, unmatchedIndexes)
print("Matched:", matched)  # Output: Matched: [1, 3, 5]
print("Unmatched:", unmatched)  # Output: Unmatched: [2, 4]

# Move an element to the matched list
partitionList.moveToMatched(2)
print("Matched after moving 2:", partitionList.matched)  # Matched should now include 2
print("Unmatched after moving 2:", partitionList.unmatched)  # Unmatched should no longer have 2

# Move an element to the unmatched list
partitionList.moveToUnmatched(1)
print("Matched after moving 1 to unmatched:", partitionList.matched)  # Matched should no longer have 1
print("Unmatched after moving 1 to unmatched:", partitionList.unmatched)  # Unmatched should now include 1
