import sys
import copy
import random

from typing import List

import parameters

from node import Node
from node import Solution

from local_searches import local_search_with_random

from parsers import parse_input_data
from parsers import parse_args
from printer import print_response


def main(args):
    solution: Solution = parse_input_data(args.input_file)

    if args.max_population_size is not None and args.max_population_size > 0:
        parameters.MAX_POPULATION_SIZE = args.max_population_size
    if args.max_fitness_evals is not None and args.max_fitness_evals > 0:
        parameters.MAX_FITNESS_EVALS = args.max_fitness_evals

    if solution.get_length() < 4:
        print_response(solution)
        return

    print_response(local_search_with_random(solution))


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
