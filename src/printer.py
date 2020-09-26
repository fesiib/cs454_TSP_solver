from typing import List

SOLUTION_FILE = "solution.csv"

def print_response(distance: float, node_order: List[int]):
    print(distance)
    
    if len(node_order) <= 0:
        return

    with open(SOLUTION_FILE, "w") as output_file:
        seperator = ""
        for id in node_order:
            output_file.write(seperator + str(id))
            seperator = "\n"
