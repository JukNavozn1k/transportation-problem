def vogel_initial_solution(supply, demand, cost):
    supply = supply[:]
    demand = demand[:]
    m, n = len(supply), len(demand)
    alloc = [[0]*n for _ in range(m)]

    while any(s > 0 for s in supply) and any(d > 0 for d in demand):
        # штрафы
        row_penalties = []
        for i in range(m):
            if supply[i] > 0:
                row = [cost[i][j] for j in range(n) if demand[j] > 0]
                if len(row) >= 2:
                    sorted_row = sorted(row)
                    row_penalties.append((sorted_row[1] - sorted_row[0], 'row', i))
                elif len(row) == 1:
                    row_penalties.append((row[0], 'row', i))

        col_penalties = []
        for j in range(n):
            if demand[j] > 0:
                col = [cost[i][j] for i in range(m) if supply[i] > 0]
                if len(col) >= 2:
                    sorted_col = sorted(col)
                    col_penalties.append((sorted_col[1] - sorted_col[0], 'col', j))
                elif len(col) == 1:
                    col_penalties.append((col[0], 'col', j))

        candidates = row_penalties + col_penalties
        penalty, typ, idx = max(candidates, key=lambda x: x[0])

        if typ == 'row':
            j = min((j for j in range(n) if demand[j] > 0), key=lambda jj: cost[idx][jj])
            i = idx
        else:
            i = min((i for i in range(m) if supply[i] > 0), key=lambda ii: cost[ii][idx])
            j = idx

        qty = min(supply[i], demand[j])
        alloc[i][j] = qty
        supply[i] -= qty
        demand[j] -= qty

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


def build_cycle(alloc, start):
    """Строим цикл для улучшения плана (упрощённый поиск)."""
    from collections import deque
    m, n = len(alloc), len(alloc[0])
    i0, j0 = start

    # поиск цикла: чередуем строки и столбцы
    def dfs(path, used):
        i, j = path[-1]
        if len(path) > 3 and (i, j) == (i0, j0):
            return path
        if len(path) % 2 == 1:  # по строке
            for jj in range(n):
                if (i, jj) not in used and (alloc[i][jj] > 0 or (i, jj) == (i0, j0)):
                    res = dfs(path + [(i, jj)], used | {(i, jj)})
                    if res:
                        return res
        else:  # по столбцу
            for ii in range(m):
                if (ii, j) not in used and (alloc[ii][j] > 0 or (ii, j) == (i0, j0)):
                    res = dfs(path + [(ii, j)], used | {(ii, j)})
                    if res:
                        return res
        return None

    return dfs([start], {start})


def make_non_degenerate(alloc, cost):
    """Добавляем искусственные нули, чтобы число базисных клеток было m+n-1."""
    m, n = len(alloc), len(alloc[0])
    # список базисных клеток
    basis = [(i, j) for i in range(m) for j in range(n) if alloc[i][j] > 0]
    needed = m + n - 1 - len(basis)
    if needed <= 0:
        return alloc

    for i in range(m):
        for j in range(n):
            if alloc[i][j] == 0:
                alloc[i][j] = 0  # явно помечаем как базис
                basis.append((i, j))
                needed -= 1
                if needed == 0:
                    return alloc
    return alloc


def calculate_potentials(alloc, cost):
    m, n = len(alloc), len(alloc[0])
    u = [None] * m
    v = [None] * n
    u[0] = 0

    changed = True
    while changed:
        changed = False
        for i in range(m):
            for j in range(n):
                if alloc[i][j] is not None and alloc[i][j] >= 0:  # базис
                    if u[i] is not None and v[j] is None:
                        v[j] = cost[i][j] - u[i]
                        changed = True
                    elif v[j] is not None and u[i] is None:
                        u[i] = cost[i][j] - v[j]
                        changed = True
    # заполняем None нулями, чтобы не падало
    u = [x if x is not None else 0 for x in u]
    v = [x if x is not None else 0 for x in v]
    return u, v


def find_entering_cell(alloc, cost, u, v):
    m, n = len(alloc), len(alloc[0])
    min_delta = 0
    cell = None
    for i in range(m):
        for j in range(n):
            if alloc[i][j] == 0:  # только небазисные
                delta = cost[i][j] - (u[i] + v[j])
                if delta < min_delta:
                    min_delta = delta
                    cell = (i, j)
    return cell, min_delta


def improve_plan(alloc, cost):
    alloc = make_non_degenerate(alloc, cost)

    while True:
        u, v = calculate_potentials(alloc, cost)
        cell, delta = find_entering_cell(alloc, cost, u, v)
        if cell is None:
            break  # оптимум

        cycle = build_cycle(alloc, cell)
        if cycle is None:
            break  # на всякий случай защита

        plus = cycle[::2]
        minus = cycle[1::2]

        theta = min(alloc[i][j] for i, j in minus)
        for i, j in plus:
            alloc[i][j] += theta
        for i, j in minus:
            alloc[i][j] -= theta

    return alloc


def transportation_problem(supply, demand, cost):
    # балансируем
    supply, demand, cost = balance_transportation(supply, demand, cost)

    # начальный план Фогеля
    alloc = vogel_initial_solution(supply, demand, cost)

    # оптимизация методом потенциалов
    alloc = improve_plan(alloc, cost)

    # убираем фиктивные строки/столбцы
    alloc = [row[:len(demand)] for row in alloc[:len(supply)]]

    return alloc


if __name__ == "__main__":
    supply = [20, 30, 25]
    demand = [10, 10, 35, 20]
    cost = [
        [8, 6, 10, 9],
        [9, 12, 13, 7],
        [14, 9, 16, 5],
    ]

    result = transportation_problem(supply, demand, cost)

    print("Оптимальный план (метод потенциалов):")
    for row in result:
        print(row)

    total_cost = sum(result[i][j] * cost[i][j] for i in range(len(supply)) for j in range(len(demand)))
    print("Оптимальные затраты:", total_cost)