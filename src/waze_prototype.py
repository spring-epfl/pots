import sys
import attr
import osmnx as ox
import networkx as nx
import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KDTree

from ortools.linear_solver import pywraplp

MIN_SPEED_LIM = 15


@attr.s
class ArcCost:
    speed_lim = attr.ib()
    length = attr.ib()

    def get_time(self):
        return self.length / self.speed_lim


def get_speed_delta(target_delta_time, arc_cost):
    """What is the minimum speed limit delta that would make this arc slower by delta"""
    return arc_cost.length / (arc_cost.get_time() + target_delta_time) - arc_cost.speed_lim


def get_time_delta(arc_cost, target_speed_lim=MIN_SPEED_LIM):
    """What is the maximum time delta that would make this arc reach the min speed limit"""
    return arc_cost.length / target_speed_lim - arc_cost.get_time()


def dot(x, y):
    res = x[0] * y[0]
    for a, b in zip(x, y):
        res += a * b
    return res


def set_speed(type):
    if type == 'residential':
        return 25
    elif type == 'tertiary':
        return 45
    elif type == 'secondary':
        return 45
    elif type == 'primary':
        return 65
    elif type == 'motorway_link':
        return 55
    else:
        return 25


METERS_IN_MILE = 0.0006213712


def initialize_data():
    town_name = 'Leonia, NJ'
    source = '22 Fort Lee Rd, Leonia, NJ 07605'
    dest = '95 Hoefleys Ln, Leonia, NJ 07605'

    # town_name = 'Wind Gap, PA'
    # source = '951 Male Rd, Wind Gap, PA 18091'
    # dest = '316 Broadway, Wind Gap, PA 18091'

    # location to save the maps
    ox.config(use_cache=True, log_console=True)
    G = ox.graph_from_place(town_name, network_type='drive')
    nodes, edges = ox.graph_to_gdfs(G)


    tree = KDTree(nodes[['y', 'x']], metric='euclidean')

    origin = ox.geocode(source)
    origin = tree.query([origin], k=1, return_distance=False)[0]
    origin = int(nodes.iloc[origin].index.values[0])

    dest = ox.geocode(dest)
    dest = tree.query([dest], k=1, return_distance=False)[0]
    dest = int(nodes.iloc[dest].index.values[0])
    print('origin: %d' % origin)
    print('dest: %d' % dest)
    nodes_a = nodes[['osmid']].values
    nodes_fixed = []
    for node in nodes_a:
        if node == dest:
            nodes_fixed.append('T')
        if node == origin:
            nodes_fixed.append('S')
        nodes_fixed.append(node[0])
    edges['speed_limit'] = edges['highway'].apply(lambda x: set_speed(x))
    edges = edges[['u','v','length','speed_limit']].values
    # todo add speed limit
    arcs_a = {}
    for edge in edges:
        u = int(edge[0])
        v = int(edge[1])
        dist = edge[2]
        speed = edge[3]
        if u == origin:
            u = 'S'
        if v == origin:
            v = 'S'
        if u == dest:
            u = 'T'
        if v == dest:
            v = 'T'

        if u in nodes_fixed and v in nodes_fixed:
            arcs_a[u,v] = ArcCost(speed_lim=speed, length=dist * METERS_IN_MILE)

    # nodes_fixed = ["S", "A", "B", "C", "T"]
    # arcs_a = {
    #   ("S", "A"): ArcCost(speed_lim=35, length=0.5),
    #   ("S", "B"): ArcCost(speed_lim=50, length=0.6),
    #   ("A", "B"): ArcCost(speed_lim=35, length=0.5),
    #   ("B", "C"): ArcCost(speed_lim=35, length=0.3),
    #   ("A", "T"): ArcCost(speed_lim=35, length=0.6),
    #   ("B", "T"): ArcCost(speed_lim=50, length=0.8),
    #   ("B", "C"): ArcCost(speed_lim=35, length=0.5),
    #   ("C", "T"): ArcCost(speed_lim=35, length=0.4),
    # }

    return nodes_fixed, arcs_a


def solve_milp(nodes, weighted_arcs, min_time, delta=1):
    # Initialize solver.
    solver = pywraplp.Solver('waze',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
                             # pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    # Constants.
    interdiction_costs = [1 for _ in weighted_arcs]

    # Initialize interdiction variables.
    interdiction_vars = []
    interdiction_vars_by_arc = {}
    for i, (arc, arc_cost) in enumerate(weighted_arcs.items()):
        # upper_lim = get_time_delta(arc_cost)
        # var = solver.NumVar(0.0, upper_lim, "x%i" % i)
        var = solver.IntVar(0.0, 1.0, "x%i" % i)
        interdiction_vars.append(var)
        interdiction_vars_by_arc[arc] = var

    # Initialize shortest-path variables.
    shortest_path_vars = []
    path_vars_by_node = {}
    for i, node in enumerate(nodes):
        var = solver.NumVar(-solver.infinity(), solver.infinity(), "p%i" % i)
        shortest_path_vars.append(var)
        path_vars_by_node[node] = var

    # Specify constraints.
    solver.Add(path_vars_by_node["T"] - path_vars_by_node["S"] >= min_time)

    arc_time_deltas = {}
    for arc, arc_cost in weighted_arcs.items():
        interdiction_var = interdiction_vars_by_arc[arc]
        p = path_vars_by_node[arc[0]]
        q = path_vars_by_node[arc[1]]

        updated_speed_lim = arc_cost.speed_lim - delta
        time_delta = get_time_delta(arc_cost, target_speed_lim=updated_speed_lim)
        arc_time_deltas[arc] = time_delta
        if updated_speed_lim >= MIN_SPEED_LIM:
            solver.Add(q - p - interdiction_var * time_delta <= arc_cost.get_time())
        else:
            solver.Add(q - p <= arc_cost.get_time())

    # Specify the objective.
    solver.Minimize(dot(interdiction_costs, interdiction_vars))
    result_status = solver.Solve()
    assert result_status == pywraplp.Solver.OPTIMAL

    print("Target minimal time through town: %2.2f mins" % (min_time * 60))
    print("Solutions:")
    for arc, var in interdiction_vars_by_arc.items():
        time_delta = var.solution_value() * arc_time_deltas[arc]
        cost = weighted_arcs[arc]
        if get_speed_delta(time_delta, cost) < -1:
            print("Road: %s." % repr(arc), "Original speed lim: %2.2f -> New speed lim: %2.2f" % (
                cost.speed_lim, cost.speed_lim + get_speed_delta(time_delta, cost)))

    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())


def solve_lp(nodes, weighted_arcs, min_time):
    # Initialize solver.
    solver = pywraplp.Solver('waze',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    # Constants.
    interdiction_costs = [1 for _ in weighted_arcs]

    # Initialize interdiction variables.
    interdiction_vars = []
    interdiction_vars_by_arc = {}
    for i, (arc, arc_cost) in enumerate(weighted_arcs.items()):
        upper_lim = get_time_delta(arc_cost)
        var = solver.NumVar(0.0, upper_lim, "x%i" % i)
        interdiction_vars.append(var)
        interdiction_vars_by_arc[arc] = var

    # Initialize shortest-path variables.
    shortest_path_vars = []
    path_vars_by_node = {}
    for i, node in enumerate(nodes):
        var = solver.NumVar(-solver.infinity(), solver.infinity(), "p%i" % i)
        shortest_path_vars.append(var)
        path_vars_by_node[node] = var

    # Specify constraints.
    solver.Add(path_vars_by_node["T"] - path_vars_by_node["S"] >= min_time)

    for arc, arc_cost in weighted_arcs.items():
        interdiction_var = interdiction_vars_by_arc[arc]
        p = path_vars_by_node[arc[0]]
        q = path_vars_by_node[arc[1]]
        solver.Add(q - p - interdiction_var <= arc_cost.get_time())

    # Specify the objective.
    solver.Minimize(dot(interdiction_costs, interdiction_vars))
    result_status = solver.Solve()
    assert result_status == pywraplp.Solver.OPTIMAL

    print("Target minimal time through town: %2.2f mins" % (min_time * 60))
    print("Solutions:")
    for arc, var in interdiction_vars_by_arc.items():
        time_delta = var.solution_value()
        cost = weighted_arcs[arc]
        if get_speed_delta(time_delta, cost) < -0.1:
            print("Road: %s." % repr(arc), "Original speed lim: %2.2f -> New speed lim: %2.2f" % (
                cost.speed_lim, cost.speed_lim + get_speed_delta(time_delta, cost)))

    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())


if __name__ == '__main__':
    nodes, arcs = initialize_data()

    min_time = float(sys.argv[1])
    # Time through town: 0.0206 hours, or 1.24 mins.
    solve_milp(nodes, arcs, min_time=min_time, delta=15)
    # solve_lp(nodes, arcs, min_time=min_time)
