from datetime import datetime

import attr
from ortools.linear_solver import pywraplp

# Minimum possible speed limit in mph.
MIN_SPEED_LIM = 15

TOLERANCE = 0.001
INTERSECTION_DELAY = 7 / 3600


@attr.s
class EdgeData:
    """Data associated with an edge in a traffic network."""

    speed_lim = attr.ib()
    length = attr.ib()

    def get_time(self):
        return self.length / self.speed_lim + INTERSECTION_DELAY


class SolutionNotFound(Exception):
    """Solution not found."""

    pass


def dot(x, y):
    """Dot product."""
    res = x[0] * y[0]
    for a, b in zip(x, y):
        res += a * b
    return res


def get_speed_delta(edge_data, target_time_delta):
    """What is the minimum speed limit delta that would make this edge slower by delta"""
    return (
        target_time_delta
        * edge_data.speed_lim ** 2
        / (target_time_delta * edge_data.speed_lim + edge_data.length)
    )


def get_time_delta(edge_data, target_speed_lim=MIN_SPEED_LIM):
    """What is the time delta that target speed limit will cause"""
    speed_delta = edge_data.speed_lim - target_speed_lim
    return (
        edge_data.length
        * speed_delta
        / (edge_data.speed_lim ** 2 - edge_data.speed_lim * speed_delta)
    )


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
    print("Solving... %s" % (datetime.now()))
    solver.Minimize(dot(interdiction_costs, interdiction_vars))
    result_status = solver.Solve()
    if result_status != pywraplp.Solver.OPTIMAL:
        raise SolutionNotFound()

    print("Solved %s" % (datetime.now()))
    solution = []
    for edge, var in interdiction_vars_by_arc.items():
        time_delta = var.solution_value() * arc_time_deltas[edge]
        edge_data = edges[edge]
        if get_speed_delta(edge_data, time_delta) >= TOLERANCE:
            solution.append((edge, edge_data, get_speed_delta(edge_data, time_delta)))
            if verbose:
                print(
                    "Road: %s - %2.2f miles." % (repr(edge), edge_data.length),
                    "Original speed lim: %2.2f -> New speed lim: %2.2f"
                    % (
                        edge_data.speed_lim,
                        edge_data.speed_lim - get_speed_delta(edge_data, time_delta),
                    ),
                )
    print(
        "solved with solution of len %d and cost %2.2f in %f seconds"
        % (len(solution), solver.Objective().Value(), solver.wall_time())
    )

    return solution, solver.wall_time(), solver.Objective().Value()

    # print('Problem solved in %f milliseconds' % solver.wall_time())
    # print('Problem solved in %d iterations' % solver.iterations())
    # print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
