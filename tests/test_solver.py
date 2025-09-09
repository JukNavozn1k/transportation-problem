import math

import pulp
import solver


def _expand_with_dummy(costs, supply, demand):
    # replicate solver._add_dummy_if_unbalanced behaviour
    m = len(supply)
    n = len(demand)
    total_supply = sum(supply)
    total_demand = sum(demand)
    costs2 = [row[:] for row in costs]
    supply2 = supply[:]
    demand2 = demand[:]
    if total_supply > total_demand:
        diff = total_supply - total_demand
        for row in costs2:
            row.append(0)
        demand2 = demand2 + [diff]
    elif total_demand > total_supply:
        diff = total_demand - total_supply
        costs2 = costs2 + [[0] * n]
        supply2 = supply2 + [diff]
    return costs2, supply2, demand2


def _pulp_optimal_cost(costs, supply, demand):
    costs2, supply2, demand2 = _expand_with_dummy(costs, supply, demand)
    m = len(supply2)
    n = len(demand2)

    prob = pulp.LpProblem("transport", pulp.LpMinimize)
    x = pulp.LpVariable.dicts('x', (range(m), range(n)), lowBound=0, cat='Continuous')

    # objective
    prob += pulp.lpSum([costs2[i][j] * x[i][j] for i in range(m) for j in range(n)])

    # supply constraints
    for i in range(m):
        prob += pulp.lpSum([x[i][j] for j in range(n)]) == supply2[i]

    # demand constraints
    for j in range(n):
        prob += pulp.lpSum([x[i][j] for i in range(m)]) == demand2[j]

    solver_cmd = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver_cmd)
    if pulp.LpStatus[prob.status] != 'Optimal':
        raise RuntimeError('PuLP did not find optimal solution')
    return pulp.value(prob.objective)


def _allocation_sums_ok(costs, allocation, supply, demand):
    costs2, supply2, demand2 = _expand_with_dummy(costs, supply, demand)
    m = len(supply2)
    n = len(demand2)
    # check rows
    for i in range(m):
        s = sum((allocation[i][j] or 0) for j in range(n))
        if not math.isclose(s, supply2[i], rel_tol=1e-9, abs_tol=1e-9):
            return False
    # check cols
    for j in range(n):
        s = sum((allocation[i][j] or 0) for i in range(m))
        if not math.isclose(s, demand2[j], rel_tol=1e-9, abs_tol=1e-9):
            return False
    return True


def test_solver_matches_pulp_on_small_balanced():
    costs = [[8, 6, 10], [9, 7, 4]]
    supply = [20, 15]
    demand = [10, 15, 10]

    alloc, total, history, info = solver.solve_transportation(costs, supply, demand)

    # validate allocation satisfies supply/demand (after possible dummy expansion)
    assert _allocation_sums_ok(costs, alloc, supply, demand)

    # validate cost equals PuLP optimum (within tolerance)
    pulp_cost = _pulp_optimal_cost(costs, supply, demand)
    assert math.isclose(total, pulp_cost, rel_tol=1e-7, abs_tol=1e-7)


def test_solver_handles_unbalanced_and_validates_with_pulp():
    costs = [[3, 1], [2, 4]]
    supply = [30, 10]
    demand = [20, 10]

    # this is unbalanced (supply 40 vs demand 30) -> dummy demand col of 10
    alloc, total, history, info = solver.solve_transportation(costs, supply, demand)

    # dummy should be reported
    assert info['dummy'] is not None

    # allocation should satisfy expanded constraints
    assert _allocation_sums_ok(costs, alloc, supply, demand)

    # compare cost to PuLP on the expanded problem
    pulp_cost = _pulp_optimal_cost(costs, supply, demand)
    assert math.isclose(total, pulp_cost, rel_tol=1e-7, abs_tol=1e-7)
