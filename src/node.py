import numpy
import copy
import random

from typing import List

import parameters


class Node:
    def __init__(self, _id: int = 0, _x: float = 0.0, _y: float = 0.0):
        self.id = _id
        self.x = _x
        self.y = _y

    def from_str(self, numbers: str):
        numbers = numbers.rstrip().split(" ")
        self.id = int(numbers[0])
        self.x = float(numbers[1])
        self.y = float(numbers[2])

    def distance_to(self, other) -> float:
        return numpy.sqrt(
            (self.x - other.x) * (self.x - other.x)
            + (self.y - other.y) * (self.y - other.y)
        )

    def __deepcopy__(self, memo):
        return Node(self.id, self.x, self.y)

    def __str__(self):
        return "(ID) = {}; (x, y) = ({}, {})".format(self.id, self.x, self.y)


class Solution:
    def __init__(
        self, _order: List[Node] = [], _length: int = 0, _fitness: float = 0.0
    ):
        self.order: List[Node] = _order
        self.length: int = _length
        self.fitness: float = _fitness

    def get_order(self) -> List[int]:
        ret_order = []
        for node in self.order:
            ret_order.append(node.id)
        return ret_order

    def get_length(self) -> int:
        return self.length

    def get_fitness(self) -> (float or None):
        if self.fitness is None:
            parameters.MAX_FITNESS_EVALS -= 1
            self.fitness = 0.0
            prev: Node = self.order[self.length - 1]
            for node in self.order:
                self.fitness += node.distance_to(prev)
                prev = node
        return self.fitness

    def try_reverse_seg(self, left: int, right: int):
        assert left < right
        parameters.MAX_FITNESS_EVALS -= 1
        left_prev = (left - 1 + self.get_length()) % self.get_length()
        right_prev = right - 1
        return (
            self.get_fitness()
            - (
                self.order[left].distance_to(self.order[left_prev])
                + self.order[right].distance_to(self.order[right_prev])
            )
            + (
                self.order[right_prev].distance_to(self.order[left_prev])
                + self.order[left].distance_to(self.order[right])
            )
        )

    def reverse_seg(self, left: int, right: int):
        assert left < right
        self.fitness = self.try_reverse_seg(left, right)
        for i in range(left, (right - int((right - left + 1) / 2))):
            temp = self.order[i]
            self.order[i] = self.order[right - (i - left) - 1]
            self.order[right - (i - left) - 1] = temp

    def shuffle(self):
        self.fitness = None
        random.shuffle(self.order)

    def append(self, node: Node):
        self.order.append(node)
        self.length += 1
        self.fitness = None

    def __deepcopy__(self, memo):
        return Solution(copy.deepcopy(self.order), self.length, self.fitness)

    def __str__(self):
        ret_str: str = "["

        seperate = ""

        for node in self.order:
            ret_str += seperate + str(node)
            seperate = ",\n"
        ret_str += "]"
        return ret_str
