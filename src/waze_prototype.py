import attr
import osmnx as ox
import folium
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


def get_time_delta(arc_cost):
    """What is the maximum time delta that would make this arc reach the min speed limit"""
    return arc_cost.length / MIN_SPEED_LIM - arc_cost.get_time()

def km_to_miles(x):
    return x * 0.621371


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

def initialize_data():
    town_name = 'Leonia, NJ'
    source = '22 Fort Lee Rd, Leonia, NJ 07605'
    dest = '95 Hoefleys Ln, Leonia, NJ 07605'

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
    nodes_a = nodes.as_matrix(columns=['osmid'])
    nodes_fixed = []
    for node in nodes_a:
        if node == dest:
            nodes_fixed.append('T')
        if node == origin:
            nodes_fixed.append('S')
        nodes_fixed.append(node[0])

    edges['max_speed'] = edges['highway'].apply(lambda x: set_speed(x))

    arcs_a = {}
    for edge in edges.as_matrix(columns=['u','v','length','max_speed']):
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
            arcs_a[u,v] = ArcCost(speed_lim=speed, length=km_to_miles(dist))

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
    G = ox.gdfs_to_graph(nodes, edges)
    print_graph(G, 'before', origin, dest)
    return nodes_fixed, arcs_a


def print_graph(G, name, origin, dest):


    colors = ['#E51000', '#E72606', '#E93D0D', '#ED6B1A', '#F3B02E', '#FAF542'];

    for u, v, d in G.edges(data=True):
        d['weight'] =  d['length'] / d['max_speed']
        if d['max_speed'] < 25:
            d['edge_color'] = colors[0]
        elif d['max_speed'] < 35:
            d['edge_color'] = colors[1]
        elif d['max_speed'] < 45:
            d['edge_color'] = colors[2]
        elif d['max_speed'] < 55:
            d['edge_color'] = colors[3]
        elif d['max_speed'] < 65:
            d['edge_color'] = colors[4]
        else:
            d['edge_color'] = colors[4]


    nodes, edges = ox.graph_to_gdfs(G)

    route = nx.shortest_path(G, origin, dest, weight='weight')
    total_time = 0
    for node in route:
        total_time += nodes[node]['length']/nodes[node]['max_speed']
    print(total_time)
    quit()
    fig, ax = ox.plot_graph_route(G, route, route_color='b', fig_height=10, fig_width=10,
                                  edge_color=edges['edge_color'], node_size=0,
                                  show=True, close=False, route_linewidth=12)

    fig.savefig('%s.png' % (name), dpi=1000, format='pdf')



def solve(nodes, weighted_arcs):
    # Initialize solver.
    solver = pywraplp.Solver('waze',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    # Constants.
    min_time = 30  # greater than 0.6 / 50 + 0.8 / 50 = 0.024
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

    for arc, cost in weighted_arcs.items():
        interdiction_var = interdiction_vars_by_arc[arc]
        p = path_vars_by_node[arc[0]]
        q = path_vars_by_node[arc[1]]
        solver.Add(q - p - interdiction_var <= cost.get_time())
        # solver.Add(q - p >= cost.get_time())

    # Specify the objective.
    solver.Minimize(dot(interdiction_costs, interdiction_vars))
    result_status = solver.Solve()
    assert result_status == pywraplp.Solver.OPTIMAL

    print("Solutions.")
    print("Target minimal time through town: %2.2f mins" % (min_time))
    for arc, var in interdiction_vars_by_arc.items():
        time_delta = var.solution_value()
        cost = weighted_arcs[arc]
        if get_speed_delta(time_delta, cost) < -1:
            print("Road: %s." % repr(arc), "Original speed lim: %2.2f -> New speed lim: %2.2f" % (
                cost.speed_lim, cost.speed_lim + get_speed_delta(time_delta, cost)))

    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())


if __name__ == '__main__':
    nodes, arcs = initialize_data()
    solve(nodes, arcs)
