import parameters

from typing import List

from node import Solution


def print_response(solution: Solution):
    print(solution.get_tour_length())

    if solution.get_length() <= 0:
        return

    with open(parameters.SOLUTION_FILE, "w") as output_file:
        seperator = ""
        order = solution.get_order()
        for id in order:
            output_file.write(seperator + str(id))
            seperator = "\n"
