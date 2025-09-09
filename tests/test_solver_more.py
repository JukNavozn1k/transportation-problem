import math

import pulp
import solver


def _expand_with_dummy(costs, supply, demand):
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

    prob += pulp.lpSum([costs2[i][j] * x[i][j] for i in range(m) for j in range(n)])

    for i in range(m):
        prob += pulp.lpSum([x[i][j] for j in range(n)]) == supply2[i]
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
    for i in range(m):
        s = sum((allocation[i][j] or 0) for j in range(n))
        if not math.isclose(s, supply2[i], rel_tol=1e-9, abs_tol=1e-9):
            return False
    for j in range(n):
        s = sum((allocation[i][j] or 0) for i in range(m))
        if not math.isclose(s, demand2[j], rel_tol=1e-9, abs_tol=1e-9):
            return False
    return True


def test_single_supplier_consumer_balanced():
    costs = [[5, 9]]
    supply = [10]
    demand = [4, 6]

    alloc, total, history, info = solver.solve_transportation(costs, supply, demand)
    assert _allocation_sums_ok(costs, alloc, supply, demand)
    pulp_cost = _pulp_optimal_cost(costs, supply, demand)
    assert math.isclose(total, pulp_cost, rel_tol=1e-7, abs_tol=1e-7)


def test_unbalanced_creates_dummy_row_and_satisfies_constraints():
    costs = [[2, 3], [4, 1]]
    supply = [5, 2]  # total 7
    demand = [4, 4]  # total 8 -> dummy row of 1 should be added

    alloc, total, history, info = solver.solve_transportation(costs, supply, demand)
    # info should report a dummy
    assert info['dummy'] is not None
    # allocation must satisfy expanded constraints
    assert _allocation_sums_ok(costs, alloc, supply, demand)
    pulp_cost = _pulp_optimal_cost(costs, supply, demand)
    assert math.isclose(total, pulp_cost, rel_tol=1e-7, abs_tol=1e-7)


def test_total_cost_from_allocation_matches_returned_total():
    costs = [[8, 6, 10], [9, 7, 4]]
    supply = [20, 15]
    demand = [10, 15, 10]

    alloc, total, history, info = solver.solve_transportation(costs, supply, demand)
    # verify solver.total_cost_from_allocation computes same total
    computed = solver.total_cost_from_allocation(costs if info['dummy'] is None else costs, alloc)
    assert math.isclose(total, computed, rel_tol=1e-12, abs_tol=1e-12)


def test_history_ends_with_optimal():
    costs = [[3, 1], [2, 4]]
    supply = [30, 10]
    demand = [20, 20]

    alloc, total, history, info = solver.solve_transportation(costs, supply, demand)
    assert len(history) > 0
    assert history[-1]['step'] == 'Optimal solution reached'
