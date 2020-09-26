import sys

import random

from typing import List

from node import Node

from parsers import parse_input_data
from parsers import parse_args

from printer import print_response

def calculate_distance(nodes: List[Node]) -> float:
    ret_distance: float = 0.0

    prev = nodes[len(nodes) - 1]
    for node in nodes:
        ret_distance += node.distance_to(prev)
    return ret_distance

def get_order(nodes: List[Node]) -> List[int]:
    ret_order = []
    for node in nodes:
        ret_order.append(node.id)
    return ret_order

def main(args):
    nodes: List[Node] = parse_input_data(args.input_file)

    random.shuffle(nodes)



    print_response(calculate_distance(nodes), get_order(nodes))

if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))