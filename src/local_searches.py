from typing import List

import random
import copy

import parameters

from node import Node
from node import Solution


def local_search_with_random(solution: Solution) -> Solution:
    solution.get_tour_length()
    generation = 0
    while parameters.MAX_FITNESS_EVALS > 0:
        new_solution = copy.deepcopy(solution)
        new_solution.shuffle()
        new_solution = local_search(new_solution)
        if solution.get_tour_length() > new_solution.get_tour_length():
            solution = new_solution
        print(
            "Generation #{}: {}".format(generation, solution.get_tour_length())
        )
        generation += 1
    return solution


def local_search(best_solution: Solution) -> Solution:
    assert best_solution is not None
    assert best_solution.get_length() > 3

    outer_iterations = 0

    while outer_iterations < parameters.MAX_ITERATIONS_NUM:
        outer_iterations += 1
        length = random.randint(2, best_solution.get_length() - 2)
        mutation_left = None

        best_fitness: float = best_solution.get_tour_length()
        inner_iterations = 0
        while inner_iterations < parameters.MAX_ITERATIONS_NUM:
            inner_iterations += 1
            left = random.randint(0, best_solution.get_length() - length - 1)
            cur_fitness = best_solution.try_reverse_seg(left, left + length)

            if parameters.MAX_FITNESS_EVALS <= 0:
                mutation_left = None
                break

            if best_fitness > cur_fitness:
                best_fitness = cur_fitness
                mutation_left = left
        if mutation_left is None:
            break
        best_solution.reverse_seg(mutation_left, mutation_left + length)
        if parameters.MAX_FITNESS_EVALS <= 0:
            break

    return best_solution
