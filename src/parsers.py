import argparse

from typing import List

from node import Node
from node import Solution


def parse_args(args):
    parser = argparse.ArgumentParser(description="TSP Solver")
    parser.add_argument(
        "input_file", type=str, help="The file of the input data for TSP"
    )
    parser.add_argument(
        "-p", "--max_population_size", type=int, help="Maximum Population Size"
    )
    parser.add_argument(
        "-f",
        "--max_fitness_evals",
        type=int,
        help="Maximum Number of Fitness Evaluations",
    )
    args = parser.parse_args(args)
    return args


def parse_input_data(file_path: str) -> Solution:
    ret_data: Solution = Solution()
    with open(file_path, "r") as input_data:
        for (i, line) in enumerate(input_data.readlines()):
            if i < 6 or line == "EOF\n":
                continue
            node = Node()
            node.from_str(line)
            ret_data.append(node)
    return ret_data
