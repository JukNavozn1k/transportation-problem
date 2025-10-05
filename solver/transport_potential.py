# transport_potential_fixed.py
from copy import deepcopy

def north_west_corner_solution(supply, demand):
    supply = supply[:]
    demand = demand[:]
    m, n = len(supply), len(demand)
    alloc = [[0.0]*n for _ in range(m)]

    i, j = 0, 0
    while i < m and j < n:
        qty = min(supply[i], demand[j])
        alloc[i][j] = float(qty)
        supply[i] -= qty
        demand[j] -= qty

        if supply[i] == 0 and demand[j] == 0:
            if i + 1 < m:
                i += 1
            elif j + 1 < n:
                j += 1
            else:
                break
        elif supply[i] == 0:
            i += 1
        elif demand[j] == 0:
            j += 1

    return alloc

def balance_transportation(supply, demand, cost):
    total_supply = sum(supply)
    total_demand = sum(demand)

    supply = supply[:]
    demand = demand[:]
    cost = [row[:] for row in cost]

    if total_supply > total_demand:
        diff = total_supply - total_demand
        demand.append(diff)
        for row in cost:
            row.append(0)
    elif total_demand > total_supply:
        diff = total_demand - total_supply
        supply.append(diff)
        cost.append([0] * len(demand))

    return supply, demand, cost

def make_non_degenerate_and_basis(alloc):
    m, n = len(alloc), len(alloc[0])
    basis = {(i, j) for i in range(m) for j in range(n) if alloc[i][j] > 0}
    needed = m + n - 1 - len(basis)
    epsilon = 1e-9

    if needed <= 0:
        return basis

    for i in range(m):
        for j in range(n):
            if (i, j) not in basis:
                alloc[i][j] = epsilon
                basis.add((i, j))
                needed -= 1
                if needed == 0:
                    return basis
    return basis

def calculate_potentials(alloc, cost, basis):
    m, n = len(alloc), len(alloc[0])
    u = [None] * m
    v = [None] * n
    u[0] = 0

    changed = True
    while changed:
        changed = False
        for (i, j) in basis:
            if u[i] is not None and v[j] is None:
                v[j] = cost[i][j] - u[i]
                changed = True
            elif v[j] is not None and u[i] is None:
                u[i] = cost[i][j] - v[j]
                changed = True

    u = [x if x is not None else 0 for x in u]
    v = [x if x is not None else 0 for x in v]
    return u, v

def find_entering_cell(alloc, cost, u, v, basis):
    m, n = len(alloc), len(alloc[0])
    min_delta = 0
    cell = None
    for i in range(m):
        for j in range(n):
            if (i, j) not in basis:
                delta = cost[i][j] - (u[i] + v[j])
                if delta < min_delta:
                    min_delta = delta
                    cell = (i, j)
    return cell, min_delta

def build_cycle(alloc, start, basis):
    """Строим цикл для MODI: движение по строкам и столбцам."""
    m, n = len(alloc), len(alloc[0])
    allowed = set(basis)
    allowed.add(start)
    i0, j0 = start

    def dfs(path, visited, horizontal):
        i, j = path[-1]
        if len(path) > 3 and (i, j) == (i0, j0):
            return path
        next_positions = []
        if horizontal:
            for jj in range(n):
                pos = (i, jj)
                if pos in allowed and (pos not in visited or pos == (i0, j0)):
                    next_positions.append(pos)
        else:
            for ii in range(m):
                pos = (ii, j)
                if pos in allowed and (pos not in visited or pos == (i0, j0)):
                    next_positions.append(pos)
        for pos in next_positions:
            if pos == path[-1]:
                continue
            new_path = path + [pos]
            res = dfs(new_path, visited | {pos}, not horizontal)
            if res:
                return res
        return None

    return dfs([start], {start}, horizontal=True)

def rebuild_basis_after_pivot(alloc):
    m, n = len(alloc), len(alloc[0])
    eps = 1e-12
    basis = {(i, j) for i in range(m) for j in range(n) if alloc[i][j] > eps}
    needed = m + n - 1 - len(basis)
    epsilon = 1e-9

    if needed > 0:
        for i in range(m):
            for j in range(n):
                if (i, j) not in basis:
                    alloc[i][j] = epsilon
                    basis.add((i, j))
                    needed -= 1
                    if needed == 0:
                        return basis
    return basis

def _format_matrix(M):
    return [[float(x) for x in row] for row in M]

def compute_reduced_costs(cost, u, v):
    m, n = len(cost), len(cost[0])
    deltas = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            deltas[i][j] = cost[i][j] - (u[i] + v[j])
    return deltas

def north_west_corner_with_steps(supply, demand):
    s = supply[:]
    d = demand[:]
    m, n = len(s), len(d)
    alloc = [[0.0] * n for _ in range(m)]
    log = []

    i, j = 0, 0
    step = 1
    while i < m and j < n:
        qty = min(s[i], d[j])
        alloc[i][j] = float(qty)
        log.append(
            f"Шаг {step}: Клетка (i={i+1}, j={j+1}). Берём min(поставка[{i+1}]={s[i]}, спрос[{j+1}]={d[j]}) = {qty}.\n"
            f"  - Заполняем x[{i+1}][{j+1}] = {qty}.\n"
            f"  - Обновляем: поставка[{i+1}] -= {qty} → {s[i]-qty}, спрос[{j+1}] -= {qty} → {d[j]-qty}."
        )
        s[i] -= qty
        d[j] -= qty

        if s[i] == 0 and d[j] == 0:
            if i + 1 < m:
                i += 1
                log.append("  Оба исчерпаны: переходим вниз к следующей строке.")
            elif j + 1 < n:
                j += 1
                log.append("  Оба исчерпаны: переходим вправо к следующему столбцу.")
            else:
                break
        elif s[i] == 0:
            i += 1
            log.append("  Поставка исчерпана: переходим вниз (следующая строка).")
        elif d[j] == 0:
            j += 1
            log.append("  Спрос исчерпан: переходим вправо (следующий столбец).")
        step += 1

    return alloc, log

def improve_plan_with_steps(alloc, cost):
    A = [list(map(float, row)) for row in alloc]
    m, n = len(A), len(A[0])
    eps = 1e-12
    basis = {(i, j) for i in range(m) for j in range(n) if A[i][j] > eps}
    if len(basis) < m + n - 1:
        basis = make_non_degenerate_and_basis(A)
    steps = []

    iter_no = 1
    while True:
        u, v = calculate_potentials(A, cost, basis)
        deltas = compute_reduced_costs(cost, u, v)

        entering, min_delta = find_entering_cell(A, cost, u, v, basis)
        step_log = {
            "iteration": iter_no,
            "u": u[:],
            "v": v[:],
            "reduced_costs": _format_matrix(deltas),
            "entering": entering,
            "min_delta": float(min_delta) if entering is not None else None,
            "cycle": None,
            "theta": None,
            "alloc_before": _format_matrix(A),
            "alloc_after": None,
        }

        if entering is None:
            step_log["note"] = "Все приведённые стоимости >= 0. План оптимален."
            steps.append(step_log)
            break

        cycle = build_cycle(A, entering, basis)
        step_log["cycle"] = cycle
        if cycle is None:
            step_log["note"] = "Не удалось построить цикл. Останавливаемся."
            steps.append(step_log)
            break

        if cycle[-1] == cycle[0]:
            cycle = cycle[:-1]

        plus = cycle[::2]
        minus = cycle[1::2]
        theta = min(A[i][j] for (i, j) in minus)
        step_log["theta"] = float(theta)

        for (i, j) in plus:
            A[i][j] += theta
        for (i, j) in minus:
            A[i][j] -= theta
            if abs(A[i][j]) < eps:
                A[i][j] = 0.0

        basis = rebuild_basis_after_pivot(A)
        step_log["alloc_after"] = _format_matrix(A)
        steps.append(step_log)

        iter_no += 1

    for i in range(m):
        for j in range(n):
            if abs(A[i][j]) < 1e-9:
                A[i][j] = 0.0

    return A, steps

def improve_plan(alloc, cost):
    alloc = [list(map(float, row)) for row in alloc]
    m, n = len(alloc), len(alloc[0])
    basis = {(i, j) for i in range(m) for j in range(n) if alloc[i][j] > 0}
    if len(basis) < m + n - 1:
        basis = make_non_degenerate_and_basis(alloc)

    while True:
        u, v = calculate_potentials(alloc, cost, basis)
        cell, delta = find_entering_cell(alloc, cost, u, v, basis)
        if cell is None:
            break

        cycle = build_cycle(alloc, cell, basis)
        if cycle is None:
            break

        if cycle[-1] == cycle[0]:
            cycle = cycle[:-1]

        plus = cycle[::2]
        minus = cycle[1::2]
        theta = min(alloc[i][j] for (i, j) in minus)
        for (i, j) in plus:
            alloc[i][j] += theta
        for (i, j) in minus:
            alloc[i][j] -= theta
            if abs(alloc[i][j]) < 1e-12:
                alloc[i][j] = 0.0

        basis = rebuild_basis_after_pivot(alloc)

    for i in range(m):
        for j in range(n):
            if abs(alloc[i][j]) < 1e-9:
                alloc[i][j] = 0.0

    return alloc

def transportation_problem(supply, demand, cost):
    supply_b, demand_b, cost_b = balance_transportation(supply, demand, cost)
    alloc = north_west_corner_solution(supply_b, demand_b)
    alloc = improve_plan(alloc, cost_b)
    alloc = [row[:len(demand)] for row in alloc[:len(supply)]]
    return alloc

if __name__ == "__main__":
    supply = [15,25 , 10]
    demand = [2, 20 , 18]
    cost = [
        [2,5,7],
        [8,12,2],
        [1,3,8],
    ]

    result = transportation_problem(supply, demand, cost)
    print("Оптимальный план (метод потенциалов):")
    for row in result:
        print(row)

    total_cost = sum(result[i][j] * cost[i][j] for i in range(len(supply)) for j in range(len(demand)))
    print("Оптимальные затраты:", total_cost)
