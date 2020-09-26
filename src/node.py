import numpy

class Node:

    def __init__(self, _id: int = 0, _x: float = 0.0, _y: float = 0.0):
        self.id = _id
        self.x = _x
        self.y = _y
    
    def from_str(self, numbers: str):
        numbers = numbers.rstrip().split(' ')
        self.id = int(numbers[0])
        self.x = float(numbers[1])
        self.y = float(numbers[2])

    def distance_to(self, other) -> float:
        return numpy.sqrt((self.x - other.x) * (self.x - other.x) + (self.y - other.y) * (self.y - other.y))

    def __str__(self):
        return "(ID) = {}; (x, y) = ({}, {})".format(self.id, self.x, self.y)
