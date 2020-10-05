import numpy
import copy
import random
from collections import deque
import heapq

from typing import List, Tuple
from typing import Dict

import parameters


class Node:
    def __init__(self, _id: int = 0, _x: float = 0.0, _y: float = 0.0):
        self.id = _id
        self.x = _x
        self.y = _y
        self.closest: List[Node] = []

    def from_str(self, numbers: str):
        numbers = numbers.rstrip().split(" ")
        self.id = int(numbers[0]) - 1
        self.x = float(numbers[1])
        self.y = float(numbers[2])
        self.closest = []

    def distance_to(self, other) -> float:
        return numpy.sqrt(
            (self.x - other.x) * (self.x - other.x)
            + (self.y - other.y) * (self.y - other.y)
        )

    def try_add_closest(self, other):
        if len(self.closest) < parameters.N_close:
            heapq.heappush(self.closest, (-1 * self.distance_to(other), other))
        else:
            worst = heapq.heappop(self.closest)
            if -1 * worst[0] > self.distance_to(other):
                heapq.heappush(
                    self.closest, (-1 * self.distance_to(other), other)
                )
            else:
                heapq.heappush(self.closest, worst)

    def __deepcopy__(self, memo):
        return Node(self.id, self.x, self.y)

    def __str__(self):
        return "(ID) = {}; (x, y) = ({}, {})".format(self.id, self.x, self.y)

    def __lt__(self, other):
        return self.id < other.id


class Solution:
    def __init__(self, order: List[Node] = [], fitness: float = None):
        self.nodes: Dict[(int, Node)] = {}
        for node in order:
            self.nodes[node.id] = node
        self.order: List[Node] = order
        self.length: int = len(order)
        self.tour_length: float = None
        self.fitness: float = fitness
        self.delta_avg_diversity: float = None
        self.delta_avg_tour_length: float = None

    def get_order(self) -> List[int]:
        ret_order = []
        for node in self.order:
            ret_order.append(node.id + 1)
        return ret_order

    def get_node_by_id(self, id):
        if id in self.nodes:
            return self.nodes[id]
        assert False
        return None

    def get_length(self) -> int:
        return self.length

    def get_fitness(self) -> (float or None):
        if self.fitness is None:
            parameters.MAX_FITNESS_EVALS -= 1
            delta_L = self.get_delta_avg_tour_length()
            delta_D = self.get_delta_avg_diversity()

            # Cannot calculate fitness if we do not have delta_L and delta_D
            if delta_L is None or delta_D is None:
                assert False

            if delta_L < 0.0:
                if delta_D < 0.0:
                    self.fitness = delta_L / delta_D
                else:
                    self.fitness = (-1 * delta_L) / parameters.EPSILON
            else:
                self.fitness = -1 * delta_L
        return self.fitness

    def get_tour_length(self) -> float:
        if self.tour_length is None:
            self.tour_length = 0.0
            prev = self.order[self.get_length() - 1]
            for cur in self.order:
                self.tour_length += cur.distance_to(prev)
                prev = cur
        return self.tour_length

    def get_delta_avg_tour_length(self) -> (float or None):
        return self.delta_avg_tour_length

    def get_delta_avg_diversity(self) -> (float or None):
        return self.delta_avg_diversity

    def update_delta_avg_tour_length(self, delta_avg_L):
        self.delta_avg_tour_length = delta_avg_L
        self.fitness = None

    def update_delta_avg_diversity(self, delta_avg_D):
        self.delta_avg_diversity = delta_avg_D
        self.fitness = None

    def try_reverse_seg(self, left: int, right: int):
        assert left < right
        parameters.MAX_FITNESS_EVALS -= 1
        left_prev = (left - 1 + self.get_length()) % self.get_length()
        right_prev = right - 1
        return (
            self.get_tour_length()
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
        self.tour_length = self.try_reverse_seg(left, right)
        for i in range(left, (right - int((right - left + 1) / 2))):
            temp = self.order[i]
            self.order[i] = self.order[right - (i - left) - 1]
            self.order[right - (i - left) - 1] = temp

    def shuffle(self):
        random.shuffle(self.order)
        self.tour_length = None
        self.fitness = None
        self.delta_avg_diversity = None
        self.delta_avg_tour_length = None

    def append(self, node: Node):
        self.nodes[node.id] = node
        self.order.append(node)
        self.length += 1
        self.tour_length = None
        self.fitness = None
        self.delta_avg_diversity = None
        self.delta_avg_tour_length = None

    def crossover_local(self, other):
        if parameters.N_ch <= 0:
            return []
        ab_cycles: List[List[Node]] = get_AB_cycles(self, other)

        children = []

        while len(children) < parameters.N_ch and len(ab_cycles) > 0:
            children.append(single_local(self, ab_cycles))
        return children

    def crossover_global(self, other):
        if parameters.N_ch <= 0:
            return []
        ab_cycles: List[List[Node]] = get_AB_cycles(self, other)

        children = []

        while len(children) < parameters.N_ch and len(ab_cycles) > 0:
            children.append(k_global(self, ab_cycles))
        return children

    def __deepcopy__(self, memo):
        return Solution(copy.deepcopy(self.order), self.fitness)

    def __str__(self):
        ret_str: str = "["

        seperate = ""

        for node in self.order:
            ret_str += seperate + str(node)
            seperate = ",\n"
        ret_str += "]"
        return ret_str


class Population:
    def __init__(self, population: List[Solution] = [], edge_freq: dict = {}):
        self.population = population
        self.edge_freq = edge_freq
        self.calculated_edge_freq = False

    def get_population(self) -> List[Solution]:
        return self.population

    def get_size(self) -> int:
        return len(self.population)

    def get_edge_freq(self) -> dict:
        if self.calculated_edge_freq is False:
            self.calc_edge_freq()
        return self.edge_freq

    def calc_edge_freq(self):
        self.edge_freq.clear()
        self.calculated_edge_freq = True
        for sol in self.population:
            v = sol.order[sol.get_length() - 1]
            for u in sol.order:
                self.edge_freq[(v.id, u.id)] = 0
                self.edge_freq[(u.id, v.id)] = 0
                v = u

        for sol in self.population:
            v = sol.order[sol.get_length() - 1]
            for u in sol.order:
                self.edge_freq[(v.id, u.id)] += 1
                self.edge_freq[(u.id, v.id)] += 1
                v = u

    def get_single_edge_freq(self, v: int, u: int):
        if self.calculated_edge_freq is False:
            self.calc_edge_freq()
        if (v, u) not in self.edge_freq:
            return 0
        return self.edge_freq[(v, u)]

    def append(self, sol: Solution):
        self.population.append(sol)
        self.calculated_edge_freq = False

    def get_at(self, i: int):
        return self.population[i]

    def best_by_tour_length(self):
        ret_sol = None
        for sol in self.population:
            if ret_sol is None:
                ret_sol = sol
            elif sol.get_tour_length() < ret_sol.get_tour_length():
                ret_sol = sol
        assert ret_sol is not None
        return ret_sol

    def diversity_func(self, freq) -> float:
        if freq == 0:
            return 0.0
        avg_freq = freq / self.get_size()
        return -(avg_freq) * numpy.log(avg_freq)

    def modify_at(self, i: int, new_sol: Solution):
        prev_sol = self.population[i]
        self.population[i] = new_sol
        if self.calculated_edge_freq is False:
            return

        v = prev_sol.order[prev_sol.get_length() - 1]
        for u in prev_sol.order:
            self.edge_freq[(v.id, u.id)] -= 1
            self.edge_freq[(u.id, v.id)] -= 1
            v = u

        v = new_sol.order[new_sol.get_length() - 1]
        for u in new_sol.order:
            self.edge_freq[(v.id, u.id)] += 1
            self.edge_freq[(u.id, v.id)] += 1
            v = u
        # TODO: Change edge_freq

    def try_modify_at(self, i: int, new_sol: Solution) -> float:
        prev_sol = self.population[i]
        if self.calculated_edge_freq is False:
            self.calc_edge_freq()
        delta_avg_diversity = 0.0
        v = prev_sol.order[prev_sol.get_length() - 1]
        for u in prev_sol.order:
            freq = self.get_single_edge_freq(v.id, u.id)
            delta_avg_diversity += self.diversity_func(
                freq - 1
            ) - self.diversity_func(freq)
            freq = self.get_single_edge_freq(u.id, v.id)
            delta_avg_diversity += self.diversity_func(
                freq - 1
            ) - self.diversity_func(freq)
            v = u

        v = new_sol.order[new_sol.get_length() - 1]
        for u in new_sol.order:
            freq = self.get_single_edge_freq(v.id, u.id)
            delta_avg_diversity += self.diversity_func(
                freq + 1
            ) - self.diversity_func(freq)
            freq = self.get_single_edge_freq(u.id, v.id)
            delta_avg_diversity += self.diversity_func(
                freq + 1
            ) - self.diversity_func(freq)
            v = u
        return delta_avg_diversity


def add_edges(E: List[List[Tuple[Node, int]]], sol: Solution, ty: int):
    v = sol.order[sol.get_length() - 1]
    for u in sol.order:
        E[v.id].append((u, ty))
        E[u.id].append((v, ty))
        v = u


def remove_tuple(lst: List[tuple], v: Node, ty: int):
    for (v_lst, ty_lst) in lst:
        if v_lst.id == v.id and ty_lst == ty:
            lst.remove((v_lst, ty_lst))
            return
    assert False


def get_AB_cycles(p_a: Solution, p_b: Solution):
    assert p_a.get_length() == p_b.get_length()
    n = p_a.get_length()
    E = [[] for i in range(n)]

    add_edges(E, p_a, 0)
    add_edges(E, p_b, 1)

    ab_cycles: List[List[Node]] = []

    available_nodes = [i for i in range(n)]
    random.shuffle(available_nodes)

    visited = [-1 for i in range(n)]

    trace: List[(Node, int)] = deque()

    for start in available_nodes:
        if len(E[start]) == 0:
            continue
        trace.append((p_a.get_node_by_id(start), 0))
        visited[start] = 0
        while len(trace) > 0:
            (v, which) = trace[len(trace) - 1]
            appended = False
            while len(E[v.id]) > 0:
                (u, next_which) = E[v.id].pop()
                remove_tuple(E[u.id], v, next_which)
                if next_which ^ which == 0:
                    continue
                if visited[u.id] != -1 and visited[u.id] != 2:
                    if visited[u.id] != next_which:
                        # SUS
                        continue
                    ab_cycle: List[Node] = []

                    while len(trace) > 0:
                        (v_cycle, which_cycle) = trace.pop()
                        visited[v_cycle.id] = 2
                        ab_cycle.append(v_cycle)
                        if v_cycle.id == u.id:
                            trace.append((v_cycle, which_cycle))
                            break
                    assert len(trace) > 0
                    assert len(ab_cycle) % 2 == 0
                    if len(ab_cycle) > 2:
                        if next_which == 1:
                            last = ab_cycle.pop()
                            ab_cycle.reverse()
                            ab_cycle.append(last)
                        ab_cycles.append(ab_cycle)
                    appended = True
                    break
                visited[u.id] = next_which
                trace.append((u, next_which))
                appended = True
                break
            if appended is False:
                visited[v.id] = 2
                trace.pop()
    return ab_cycles


def single_local(sol: Solution, cycles: List[List[Node]]):
    prev_tour_length = sol.get_tour_length()
    choice = random.choice(cycles)
    cycles.remove(choice)
    intermediate_sol: List[List[Tuple[Node, int]]] = form_intermediate_sol(
        sol, [choice]
    )
    ret_sol = transform_into_solution(intermediate_sol, sol)
    post_tour_length = ret_sol.get_tour_length()
    ret_sol.update_delta_avg_tour_length(post_tour_length - prev_tour_length)
    return ret_sol


def k_global(sol: Solution, cycles: List[List[Node]]):
    prev_tour_length = sol.get_tour_length()
    choices = random.choices(cycles, k=parameters.K_GLOBAL)
    for choice in choices:
        cycles.remove(choice)
    intermediate_sol: List[List[Tuple[Node, int]]] = form_intermediate_sol(
        sol, choices
    )
    ret_sol = transform_into_solution(intermediate_sol, sol)
    post_tour_length = ret_sol.get_tour_length()
    ret_sol.update_delta_avg_tour_length(post_tour_length - prev_tour_length)
    return ret_sol


def form_intermediate_sol(sol: Solution, cycles: List[List[Node]]):
    assert len(cycles) >= 1
    n = sol.get_length()
    intermediate_sol = [[] for i in range(n)]
    add_edges(intermediate_sol, sol, 0)
    for cycle in cycles:
        assert len(cycle) >= 4
        v = cycle[len(cycle) - 1]
        ty = 0
        for u in cycle:
            if ty == 0:
                remove_tuple(intermediate_sol[v.id], u, 0)
                remove_tuple(intermediate_sol[u.id], v, 0)
            else:
                intermediate_sol[v.id].append((u, 0))
                intermediate_sol[u.id].append((v, 0))
            ty ^= 1
            v = u
    return intermediate_sol


def dfs(
    v: Node, cur_color: int, color: List[int], E: List[List[Tuple[Node, int]]]
) -> List[Node]:
    color[v.id] = cur_color
    assert len(E[v.id]) == 2
    for (u, _) in E[v.id]:
        if color[u.id] == -1:
            ret_list = dfs(u, cur_color, color, E)
            ret_list.append(v)
            return ret_list
    return [v]


def transform_into_solution(intermediate_sol, sol):
    ret_sol = Solution()
    n = len(intermediate_sol)
    tours = []
    color = [-1 for i in range(n)]
    tour_nodes = [[] for i in range(n)]
    cur_color = 0
    for i in range(n):
        if color[i] == -1:
            tour_nodes[cur_color] = dfs(
                sol.get_node_by_id(i), cur_color, color, intermediate_sol
            )
            heapq.heappush(tours, (len(tour_nodes[cur_color]), cur_color))
            cur_color += 1
    while len(tours) > 1:
        (length, cur_color) = heapq.heappop(tours)
        if length != len(tour_nodes[cur_color]):
            continue
        while len(tours) > 0 and len(tour_nodes[tours[0][1]]) == 0:
            heapq.heappop(tours)
        if len(tours) == 0:
            break
        opt_a = tour_nodes[tours[0][1]][0]
        opt_b = intermediate_sol[opt_a.id][0][0]
        opt_v = tour_nodes[cur_color][length - 1]
        opt_u = intermediate_sol[opt_v.id][0][0]
        found = False
        assert color[opt_a.id] != color[opt_v.id]
        for v in tour_nodes[cur_color]:
            for (u, _) in intermediate_sol[v.id]:
                for a in v.closest:
                    if color[a.id] == color[v.id]:
                        continue
                    for (b, _) in intermediate_sol[a.id]:
                        assert color[a.id] == color[b.id]
                        if calc_delta_tour_length(
                            v, u, a, b
                        ) < calc_delta_tour_length(v, u, opt_a, opt_b):
                            opt_a = a
                            opt_b = b
                            opt_v = v
                            opt_u = u
                            found = True

                for a in u.closest:
                    if color[a.id] == color[v.id]:
                        continue
                    for (b, _) in intermediate_sol[a.id]:
                        assert color[a.id] == color[b.id]
                        if calc_delta_tour_length(
                            v, u, a, b
                        ) < calc_delta_tour_length(v, u, opt_a, opt_b):
                            opt_a = a
                            opt_b = b
                            opt_v = v
                            opt_u = u
                            found = True
        new_color = color[opt_a.id]

        if found is False:
            # print("Maybe we should increase N_close")
            found = False
        remove_tuple(intermediate_sol[opt_v.id], opt_u, 0)
        remove_tuple(intermediate_sol[opt_a.id], opt_b, 0)
        intermediate_sol[opt_v.id].append((opt_a, 0))
        intermediate_sol[opt_a.id].append((opt_v, 0))

        remove_tuple(intermediate_sol[opt_u.id], opt_v, 0)
        remove_tuple(intermediate_sol[opt_b.id], opt_a, 0)
        intermediate_sol[opt_u.id].append((opt_b, 0))
        intermediate_sol[opt_b.id].append((opt_u, 0))
        while len(tour_nodes[cur_color]) > 0:
            v = tour_nodes[cur_color].pop()
            color[v.id] = new_color
            tour_nodes[new_color].append(v)
        heapq.heappush(tours, (len(tour_nodes[new_color]), new_color))

    color = [-1 for i in range(n)]
    cur_color = 0
    one_tour = None
    for i in range(n):
        if color[i] == -1:
            assert cur_color == 0
            one_tour = dfs(
                sol.get_node_by_id(i), cur_color, color, intermediate_sol
            )
            cur_color += 1
    assert one_tour is not None
    for v in one_tour:
        ret_sol.append(v)
    return ret_sol


def calc_delta_tour_length(v: Node, u: Node, a: Node, b: Node):
    return (
        v.distance_to(a)
        + u.distance_to(b)
        - (v.distance_to(u) + a.distance_to(b))
    )
