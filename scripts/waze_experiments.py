#! /usr/bin/env python

# CLI for running Waze experiments.

# Units of time: hours
# Units of distance: miles

import attr
import click
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pyroutelib3 import Router

from src.traffic_solver import solve_milp, get_speed_delta, EdgeData, SolutionNotFound


def meters_to_miles(meters):
    return meters * 0.0006213712


def set_speed(type, d):
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


def update_speedlimits(G_town, origin, dest, solution):
    for s in solution:
        su = s[0][0]
        sv = s[0][1]
        if su == "T":
            su = dest
        if sv == "T":
            sv = dest
        if su == "S":
            su = origin
        if sv == "S":
            sv = origin

        for u, v, d in G_town.edges(data=True):
            if u in (su, sv) and v in (su, sv):
                d["speed_limit"] = d["speed_limit"] + s[2]


def get_town_data(town_params):
    ox.config(use_cache=True, log_console=True)
    G_town = ox.graph_from_place(
        town_params.town_name, network_type="drive", simplify=False
    )
    G_town = ox.simplify.simplify_graph(G_town, strict=False)
    nodes, edges = ox.graph_to_gdfs(G_town)
    town_node_ids = set(nodes)

    node_ids = nodes[["osmid"]].values
    processed_nodes = []
    for node_id in node_ids:
        if node_id == town_params.dest:
            processed_nodes.append("T")
        if node_id == town_params.source:
            processed_nodes.append("S")
        processed_nodes.append(node_id[0])
    # change from meters to miles
    edges["length"] = edges["length"].apply(meters_to_miles)
    edges["speed_limit"] = edges.apply(lambda x: set_speed(x.highway, x.length), axis=1)

    processed_edges = {}
    for u, v, length, speed in edges[["u", "v", "length", "speed_limit"]].values:
        u = int(u)
        v = int(v)
        if u == town_params.source:
            u = "S"
        if v == town_params.source:
            v = "S"
        if u == town_params.dest:
            u = "T"
        if v == town_params.dest:
            v = "T"

        if u in processed_nodes and v in processed_nodes:
            e = EdgeData(speed_lim=speed, length=length)
            processed_edges[u, v] = e
    G = ox.gdfs_to_graph(nodes, edges)
    return (G, town_params.source, town_params.dest), (processed_nodes, processed_edges)


def print_graph(G_town, origin, dest, solution, title="map"):
    town_nodes, town_edges = ox.graph_to_gdfs(G_town)
    x, y = town_edges.unary_union.centroid.xy
    town_centroid = (y[0], x[0])

    G_area = ox.graph_from_point(
        town_centroid, distance=2500, simplify=False, network_type="drive"
    )

    G_area = ox.simplify.simplify_graph(G_area, strict=False)
    town_node_ids = set(G_town.nodes())

    # set default vals and make the weight d = rt -> t = d/r
    for u, v, d in G_area.edges(data=True):
        d["edge_color"] = (
            "#2b83ba" if u in town_node_ids and v in town_node_ids else "lightgray"
        )
        if (
            d["highway"] == "motorway"
            or d["highway"] == "trunk"
            or d["highway"] == "motorway_link"
        ):
            d["edge_color"] = "orange"
        d["delta"] = 0
        # d['edge_color'] = '#2b83ba'
        d["edge_linewidth"] = 0.5
    # solution.append((edge, edge_data, get_speed_delta(edge_data, time_delta)))

    for s in solution:
        su = s[0][0]
        sv = s[0][1]
        if su == "T":
            su = dest
        if sv == "T":
            sv = dest
        if su == "S":
            su = origin
        if sv == "S":
            sv = origin

        # find the edge
        found = False
        for u, v, d in G_area.edges(data=True):
            if u in (su, sv) and v in (su, sv):
                d["delta"] = s[2]
                d["edge_color"] = "k"
                d["edge_linewidth"] = 2
                # d['speed_limit'] = d['speed_limit'] + s[-1]
                found = True
        if not found:
            print("CANNOT FIND THIS EDGE IN THE ORIGINAL GRAPH: " + str(s))
            quit()

    nodes, edges = ox.graph_to_gdfs(G_area)

    router = Router("car")
    # start = router.findNode(lat, lon)
    # end = router.findNode(lat, lon)
    start = origin
    end = dest
    route = nx.shortest_path(G_area, origin, dest, weight="weight")

    fig, ax = ox.plot_graph_route(
        G_area,
        route,
        orig_dest_node_size=20,
        route_color="b",
        fig_height=5,
        fig_width=10,
        edge_color=edges["edge_color"],
        edge_alpha=1,
        node_size=0,
        save=True,
        close=True,
        file_format="pdf",
        filename="maps/%s" % (title),
        show=False,
        route_linewidth=0,
        edge_linewidth=edges["edge_linewidth"],
    )

    # fig.savefig("maps/%s.png" % (title), dpi=1000, format="pdf")


def speed_limit_change_by_15(edge_data):
    speed_lim = edge_data.speed_lim
    if speed_lim == 25:
        return 10
    elif speed_lim > 25:
        return 15


def speed_limit_change_to_15(edge_data):
    speed_lim = edge_data.speed_lim
    if speed_lim > 15:
        return speed_lim - 15


def time_coef(edge_data, coef=0.3333):
    return get_speed_delta(edge_data, coef * edge_data.get_time())


def all_pairs_cost(G_changed):
    """Compute average cost between any two points in town."""
    indexable_edges = {}
    for u, v, d in G_changed.edges(data=True):
        d["weight"] = (d["length"]) / d["speed_limit"]
        indexable_edges[(u, v)] = d

    # return nx.average_shortest_path_length(G_changed, weight='weight')
    routes = dict(nx.all_pairs_shortest_path(G_changed))
    # this data structure is hell.
    # So it's (s0, {d0:[r0], d1:[r1], d2:[r2]...}...)
    total_times = []
    for start_node, routes_from in routes.items():
        for end_node, route in routes_from.items():
            total_time = 0
            for i in range(0, len(route) - 1):
                total_time += indexable_edges[(route[i], route[i + 1])]["weight"]
            total_times.append(total_time * 60)

    return sum(total_times) / len(total_times)


@attr.s
class TownParams:
    town_name = attr.ib()
    dest = attr.ib()
    source = attr.ib()


TOWN_PARAMS_MAP = {
    "leonia": TownParams(town_name="Leonia, NJ", dest=103185965, source=4207193769),
    "lieusaint": TownParams(
        town_name="Lieusaint, France", source=1932540940, dest=1472509406
    ),
    "fremont": TownParams(
        town_name="Fremont, California", source=52986461, dest=53006402
    ),
}


@attr.s
class ExperimentParams:
    town_code = attr.ib()
    town_params = attr.ib()
    delta = attr.ib()
    interdiction_cost = attr.ib()
    eval_cost = attr.ib()


@click.group()
@click.option(
    "--town", default="leonia", type=click.Choice(["leonia", "lieusaint", "fremont"])
)
@click.option(
    "--delta",
    default="speed_limit_change_to_15",
    type=click.Choice(["speed_limit_change_to_15"]),
)
@click.option(
    "--interdiction_cost", default="length", type=click.Choice(["uniform", "length"])
)
@click.option(
    "--eval_cost", default="in_town", type=click.Choice(["in_town", "length", "uniform"])
)
@click.pass_context
def cli(ctx, town, delta, interdiction_cost, eval_cost):
    if isinstance(delta, str):
        delta = eval(delta)

    ctx.obj = ExperimentParams(
        town_code=town,
        town_params=TOWN_PARAMS_MAP[town],
        delta=delta,
        interdiction_cost=interdiction_cost,
        eval_cost=eval_cost,
    )

    print(ctx.obj)


@cli.command()
@click.option("--max_time", default=7, type=float)
@click.option("--min_time", default=3, type=float)
@click.option("--time_step", default=-0.1, type=float)
@click.pass_obj
def run_experiments(experiment_params, max_time, min_time, time_step):
    times = np.arange(max_time, min_time, time_step)
    costs = []
    for min_time_mins in times:
        print(min_time_mins)
        (G, s, t), (nodes, arcs) = get_town_data(experiment_params.town_params)
        min_time = min_time_mins / 60
        print("Target min time through town: %2.2f mins" % (min_time * 60))
        try:
            solution, time, milp_cost = solve_milp(
                nodes,
                arcs,
                min_time=min_time,
                delta=experiment_params.delta,
                costs=experiment_params.interdiction_cost,
                verbose=True,
            )

            update_speedlimits(G, s, t, solution)

        except SolutionNotFound:
            print("No Solution")
            costs.append(None)
            continue

        if len(solution) == 0:
            costs.append(None)
            # break
        else:
            if experiment_params.eval_cost == "in_town":
                costs.append(all_pairs_cost(G))
            if experiment_params.eval_cost == "uniform":
                costs.append(len(solution))
            if experiment_params.eval_cost == "length":
                # TODO: Is this right?
                costs.append(milp_cost)

    with open(
        "images/%s/%s_town_effects.csv"
        % (experiment_params.town_code, experiment_params.eval_cost),
        "w",
    ) as o:
        for t, c in zip(times, costs):
            if c == None:
                c = -1
            o.write("%2.2f,%2.2f\n" % (t, c))

    times_relative = []
    costs_relative = []
    first_time = None
    first_cost = None
    for t, c in zip(reversed(times), reversed(costs)):
        if c != None:
            if first_time == None:
                first_time = t
                first_cost = c
            times_relative.append(100 * ((t - first_time) / first_time))
            costs_relative.append(100 * ((c - first_cost) / first_cost))

    plt.plot(times_relative, costs_relative)

    if experiment_params.eval_cost == "uniform":
        plt.title("Cost of Delaying Through Traffic")
        plt.ylabel("Number of Road Segments Affected")

    if experiment_params.eval_cost == "in_town":
        plt.title("Effect of Changes on In-Town Travel")
        plt.ylabel("Percent Increase of Average In-Town Travel Time")

    plt.xlabel("Percent Increase of Through Traffic Travel")
    plt.tight_layout()
    plt.show()


@cli.command()
@click.option("--min_time", type=float)
@click.pass_obj
def run_one_experiment(experiment_params, min_time):
    (G, s, t), (nodes, arcs) = get_town_data(experiment_params.town_params)

    print("Target min time through town: %2.2f mins" % (min_time * 60))
    solution, time, value = solve_milp(
        nodes,
        arcs,
        min_time=min_time / 60,
        delta=experiment_params.delta,
        costs=experiment_params.interdiction_cost,
        verbose=True,
    )

    print_graph(G, s, t, solution)


if __name__ == "__main__":
    cli()
