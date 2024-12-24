class PartitionList(list): 

    def __init__(self):
        self.list = []
        self.matched
        self.unmatched

    def init(self, list):
        self.list = list
        self.matched = PartitionList()
        self.unmatched = PartitionList()

    def partition(self, indexes):
        for i in indexes: 
            
        
