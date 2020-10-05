import random
import copy
from collections import deque
from typing import List
from node import Solution
from node import Node
from node import Population
import parameters


def find_closest(vers: Solution):
    n = vers.get_length()
    for i in range(n):
        v = vers.order[i]
        for j in range(i + 1, n):
            u = vers.order[j]
            v.try_add_closest(u)
            if j - i > 500:
                break


def generate_population(vers: Solution):
    ret_pop = Population()

    while ret_pop.get_size() < parameters.MAX_POPULATION_SIZE:
        new_vers = copy.deepcopy(vers)
        new_vers.shuffle()
        ret_pop.append(new_vers)
    return ret_pop


def ga_1(input_sol: Solution):
    find_closest(input_sol)

    population: Population = generate_population(input_sol)

    perm = []
    population_size = population.get_size()
    for i in range(population_size):
        perm.append(i)

    failed_gens = (
        0  # Gen's that did not improve best solution over last 1500/N_ch
    )
    recent_failed_gens = 0  # The most recent failed gens
    last_gen_results = deque()

    ret_sol = input_sol

    generations = 0

    while True:
        best_solution = population.best_by_tour_length()
        print(
            "Generation #{}: {}".format(
                generations, best_solution.get_tour_length()
            )
        )
        generations += 1
        # Stop Criteria
        if len(last_gen_results) >= parameters.R_stop / parameters.N_ch:
            if float(recent_failed_gens) > failed_gens / 10:
                ret_sol = best_solution
                break
        if parameters.MAX_ITERATIONS_NUM <= 0:
            ret_sol = best_solution
            break
        updated = False
        # Find the best solution
        random.shuffle(perm)

        for i in range(population_size):
            print("Starting {}".format(i))
            p_a = population.get_at(perm[i])
            p_b = population.get_at(perm[i - 1])
            children = p_a.crossover_local(p_b)
            best_child = best_by_fitness(population, perm[i], children, p_a)
            population.modify_at(perm[i], best_child)
            if best_child.get_tour_length() < best_solution.get_tour_length():
                updated = True

        if len(last_gen_results) >= parameters.R_stop / parameters.N_ch:
            if last_gen_results.popleft() is False:
                failed_gens -= 1
        if updated is False:
            failed_gens += 1
            recent_failed_gens += 1
        else:
            recent_failed_gens = 0
        last_gen_results.append(updated)
    assert ret_sol is not None
    return ret_sol


def best_by_fitness(
    population: Population,
    sol_pos: int,
    children: List[Solution],
    default: Solution,
) -> Solution:
    assert default is not None
    if children is None or len(children) == 0:
        return default
    for sol in children:
        sol.update_delta_avg_diversity(population.try_modify_at(sol_pos, sol))
        sol.update_delta_avg_tour_length(
            sol.get_delta_avg_tour_length() / population.get_size()
        )
    ret_sol = children[0]
    for sol in children:
        assert sol.get_fitness() is not None
        if sol.get_fitness() > ret_sol.get_fitness():
            ret_sol = sol
    if ret_sol.get_fitness() <= 0:
        ret_sol = default
    return ret_sol
