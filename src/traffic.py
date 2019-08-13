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

# Minimum possible speed limit in mph.
MIN_SPEED_LIM = 15


@attr.s
class EdgeData:
    """Data associated with an edge in a traffic network."""

    speed_lim = attr.ib()
    length = attr.ib()

    def get_time(self):
        return self.length / self.speed_lim


def get_speed_delta(edge_data, target_time_delta):
    """What is the minimum speed limit delta that would make this edge slower by delta"""
    return (
        edge_data.length / (edge_data.get_time() + target_time_delta)
        - edge_data.speed_lim
    )


def get_time_delta(edge_data, target_speed_lim=MIN_SPEED_LIM):
    """What is the maximum time delta that would make this edge reach the min speed limit"""
    return edge_data.length / target_speed_lim - edge_data.get_time()


def get_time_delta(edge_data, target_speed_lim=MIN_SPEED_LIM):
    speed_delta = edge_data.speed_lim - target_speed_lim
    return edge_data.length * speed_delta / (edge_data.speed_lim**2 - edge_data.speed_lim * speed_delta)


def dot(x, y):
    res = x[0] * y[0]
    for a, b in zip(x, y):
        res += a * b
    return res


def set_speed(type):
    if type == "residential":
        return 25
    elif type == "tertiary":
        return 45
    elif type == "secondary":
        return 45
    elif type == "primary":
        return 65
    elif type == "motorway_link":
        return 55
    else:
        return 25


METERS_IN_MILE = 0.0006213712
TOLERANCE = 0.001


def get_leonia_data():
    """Get data for Leonia, NJ."""
    town_name = "Leonia, NJ"
    source = "22 Fort Lee Rd, Leonia, NJ 07605"
    dest = "95 Hoefleys Ln, Leonia, NJ 07605"

    # town_name = 'Wind Gap, PA'
    # source = '951 Male Rd, Wind Gap, PA 18091'
    # dest = '316 Broadway, Wind Gap, PA 18091'

    # location to save the maps
    ox.config(use_cache=True, log_console=True)
    G = ox.graph_from_place(town_name, network_type="drive")
    nodes, edges = ox.graph_to_gdfs(G)

    tree = KDTree(nodes[["y", "x"]], metric="euclidean")

    origin = ox.geocode(source)
    origin = tree.query([origin], k=1, return_distance=False)[0]
    origin = int(nodes.iloc[origin].index.values[0])

    dest = ox.geocode(dest)
    dest = tree.query([dest], k=1, return_distance=False)[0]
    dest = int(nodes.iloc[dest].index.values[0])
    node_ids = nodes[["osmid"]].values
    processed_nodes = []
    for node_id in node_ids:
        if node_id == dest:
            processed_nodes.append("T")
        if node_id == origin:
            processed_nodes.append("S")
        processed_nodes.append(node_id[0])
    edges["speed_limit"] = edges["highway"].apply(lambda x: set_speed(x))
    edges = edges[["u", "v", "length", "speed_limit"]].values

    # TODO: Fix the speed limits.
    processed_edges = {}
    for edge in edges:
        u = int(edge[0])
        v = int(edge[1])
        dist = edge[2]
        speed = edge[3]
        if u == origin:
            u = "S"
        if v == origin:
            v = "S"
        if u == dest:
            u = "T"
        if v == dest:
            v = "T"

        if u in processed_nodes and v in processed_nodes:
            processed_edges[u, v] = EdgeData(
                speed_lim=speed, length=dist * METERS_IN_MILE
            )

    return processed_nodes, processed_edges


def solve_milp(
    nodes,
    edges,
    min_time,
    source="S",
    target="T",
    costs="uniform",
    delta=5,
    verbose=False,
):
    """
    Solve MILP for traffic interdiction.

    Args:
        nodes: List of node identifiers
        edges (dict): Traphic graph arcs. Mapping of node tuples that represent node
                identifiers (a, b) to :py:class:`Edge` objects.
        min_time: Minimum time to pass through town in hours
        delta: Speed limit delta for any interdiction.
        costs: Cost per interdiction. Either an iterable of costs, or one of ["uniform", "length"].
        verbose: Output the solutions.
    """
    if isinstance(delta, int) or isinstance(delta, float):
        delta_scheduler = lambda *args, **kwargs: delta
    else:
        delta_scheduler = delta

    # Initialize solver.
    solver = pywraplp.Solver("waze_milp", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    # Constants.
    if costs == "uniform":
        interdiction_costs = [1 for _ in edges]
    elif costs == "length":
        interdiction_costs = [edge_data.length for edge_data in edges.values()]
    else:
        interdiction_costs = costs

    # Initialize interdiction variables.
    interdiction_vars = []
    interdiction_vars_by_arc = {}
    for i, (edge, edge_data) in enumerate(edges.items()):
        var = solver.IntVar(0.0, 1.0, "x%i" % i)
        interdiction_vars.append(var)
        interdiction_vars_by_arc[edge] = var

    # Initialize shortest-path variables.
    shortest_path_vars = []
    path_vars_by_node = {}
    for i, node in enumerate(nodes):
        var = solver.NumVar(-solver.infinity(), solver.infinity(), "p%i" % i)
        shortest_path_vars.append(var)
        path_vars_by_node[node] = var

    # Specify constraints.
    solver.Add(path_vars_by_node[target] - path_vars_by_node[source] >= min_time)

    arc_time_deltas = {}
    for edge, edge_data in edges.items():
        interdiction_var = interdiction_vars_by_arc[edge]
        p = path_vars_by_node[edge[0]]
        q = path_vars_by_node[edge[1]]

        updated_speed_lim = edge_data.speed_lim - delta_scheduler(edge_data)
        time_delta = get_time_delta(edge_data, target_speed_lim=updated_speed_lim)
        arc_time_deltas[edge] = time_delta
        if updated_speed_lim >= MIN_SPEED_LIM:
            solver.Add(q - p <= edge_data.get_time() + interdiction_var * time_delta)
        else:
            solver.Add(q - p <= edge_data.get_time())

    # Specify the objective.
    solver.Minimize(dot(interdiction_costs, interdiction_vars))
    result_status = solver.Solve()
    if result_status != pywraplp.Solver.OPTIMAL:
        raise Exception("Solution not found")

    solution = []
    for edge, var in interdiction_vars_by_arc.items():
        time_delta = var.solution_value() * arc_time_deltas[edge]
        edge_data = edges[edge]
        if get_speed_delta(edge_data, time_delta) < -TOLERANCE:
            solution.append((edge, edge_data, get_speed_delta(edge_data, time_delta)))
            if verbose:
                print(
                    "Road: %s - %2.2f miles." % (repr(edge), edge_data.length),
                    "Original speed lim: %2.2f -> New speed lim: %2.2f"
                    % (
                        edge_data.speed_lim,
                        edge_data.speed_lim + get_speed_delta(edge_data, time_delta),
                    ),
                )

    return solution, solver.wall_time()

    # print('Problem solved in %f milliseconds' % solver.wall_time())
    # print('Problem solved in %d iterations' % solver.iterations())
    # print('Problem solved in %d branch-and-bound nodes' % solver.nodes())


def adaptive_delta(edge_data):
    speed_lim = edge_data.speed_lim
    if speed_lim <= 30:
        return speed_lim - MIN_SPEED_LIM
    if 30 < speed_lim <= 45:
        return speed_lim - 15
    if 45 < speed_lim <= 55:
        return speed_lim - 25
    if 55 < speed_lim <= 80:
        return speed_lim - 25


if __name__ == "__main__":
    nodes, arcs = get_leonia_data()

    min_time = float(sys.argv[1])
    # Time through town: 0.0206 hours, or 1.24 mins.
    print("Target min time through town: %2.2f mins" % (min_time * 60))
    solution, time = solve_milp(
        nodes,
        arcs,
        min_time=min_time,
        delta=adaptive_delta,
        costs="length",
        verbose=True,
    )
    print("Solution size:", len(solution))
    print("Solving time: %2.2f ms" % time)

