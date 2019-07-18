import attr

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


def dot(x, y):
    res = x[0] * y[0]
    for a, b in zip(x, y):
        res += a * b
    return res


nodes = ["S", "A", "B", "C", "T"]
weighted_arcs = {
  ("S", "A"): ArcCost(speed_lim=35, length=0.5),
  ("S", "B"): ArcCost(speed_lim=50, length=0.6),
  ("A", "B"): ArcCost(speed_lim=35, length=0.5),
  ("B", "C"): ArcCost(speed_lim=35, length=0.3),
  ("A", "T"): ArcCost(speed_lim=35, length=0.6),
  ("B", "T"): ArcCost(speed_lim=50, length=0.8),
  ("B", "C"): ArcCost(speed_lim=35, length=0.5),
  ("C", "T"): ArcCost(speed_lim=35, length=0.4),
}

# Initialize solver.
solver = pywraplp.Solver('waze',
        pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

# Constants.
min_time = 0.05  # greater than 0.6 / 50 + 0.8 / 50 = 0.024
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
print("Time through town: %2.2f mins" % (0.026 * 60))
print("Target minimal time through town: %2.2f mins" % (min_time * 60))
for arc, var in interdiction_vars_by_arc.items():
    time_delta = var.solution_value()
    cost = weighted_arcs[arc]
    print("Road: %s." % repr(arc), "Original speed lim: %2.2f -> New speed lim: %2.2f" % (
        cost.speed_lim, cost.speed_lim + get_speed_delta(time_delta, cost)))

print('Problem solved in %f milliseconds' % solver.wall_time())
print('Problem solved in %d iterations' % solver.iterations())
print('Problem solved in %d branch-and-bound nodes' % solver.nodes())

