from abc import ABC, abstractmethod

class Property:
    """
    Abstract class for identity-preserving properties of a detection. 
    All properties must implement the similarity method to compare with other detections during the tracking procedure.
    """

    ##### ABSTRACT METHODS #####
    @abstractmethod
    def similarity(self, other) -> float:
        """
        Computes the similarity between this property and another property.
        """
        pass

    @abstractmethod
    def __str__(self):
        """
        Returns a string representation of the property.
        """ 
        pass

