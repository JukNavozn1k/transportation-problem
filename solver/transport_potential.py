def vogel_initial_solution(supply, demand, cost):
    """
    Классический метод Фогеля. Возвращает только итоговый план.
    Для поэтапного вывода используйте vogel_initial_solution_with_steps.
    """
    _, steps = vogel_initial_solution_with_steps(supply, demand, cost)
    return steps[-1]["alloc"] if steps else [[0]*len(demand) for _ in range(len(supply))]


def vogel_initial_solution_with_steps(supply, demand, cost):
    """Метод Фогеля с поэтапным журналом действий.

    Возвращает кортеж (final_alloc, steps), где steps — список шагов.
    Каждый шаг — dict:
      - row_penalties: [(penalty, i)]
      - col_penalties: [(penalty, j)]
      - chosen: {"type": "row"|"col", "index": i|j, "cell": (i,j), "qty": q, "penalty": p}
      - supply: текущий вектор запасов после шага
      - demand: текущий вектор потребностей после шага
      - alloc: текущая матрица распределений после шага
    """
    supply_work = supply[:]
    demand_work = demand[:]
    m, n = len(supply_work), len(demand_work)
    alloc = [[0] * n for _ in range(m)]
    steps = []

    while any(s > 0 for s in supply_work) and any(d > 0 for d in demand_work):
        # штрафы
        row_penalties = []
        for i in range(m):
            if supply_work[i] > 0:
                row_costs = [cost[i][j] for j in range(n) if demand_work[j] > 0]
                if len(row_costs) >= 2:
                    sr = sorted(row_costs)
                    row_penalties.append((sr[1] - sr[0], i))
                elif len(row_costs) == 1:
                    row_penalties.append((row_costs[0], i))

        col_penalties = []
        for j in range(n):
            if demand_work[j] > 0:
                col_costs = [cost[i][j] for i in range(m) if supply_work[i] > 0]
                if len(col_costs) >= 2:
                    sc = sorted(col_costs)
                    col_penalties.append((sc[1] - sc[0], j))
                elif len(col_costs) == 1:
                    col_penalties.append((col_costs[0], j))

        # выбор максимального штрафа
        candidates = [(*p, 'row') for p in row_penalties] + [(*p, 'col') for p in col_penalties]
        penalty, idx, typ = max(candidates, key=lambda x: x[0])

        if typ == 'row':
            j = min((jj for jj in range(n) if demand_work[jj] > 0), key=lambda jj: cost[idx][jj])
            i = idx
        else:
            i = min((ii for ii in range(m) if supply_work[ii] > 0), key=lambda ii: cost[ii][idx])
            j = idx

        qty = min(supply_work[i], demand_work[j])
        alloc[i][j] += qty
        supply_work[i] -= qty
        demand_work[j] -= qty

        steps.append({
            "row_penalties": row_penalties[:],
            "col_penalties": col_penalties[:],
            "chosen": {"type": typ, "index": idx, "cell": (i, j), "qty": qty, "penalty": penalty},
            "supply": supply_work[:],
            "demand": demand_work[:],
            "alloc": [row[:] for row in alloc],
        })

    return alloc, steps


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
    """Оптимизация методом потенциалов. Возвращает итоговый план."""
    final_alloc, _ = improve_plan_with_steps(alloc, cost)
    return final_alloc


def improve_plan_with_steps(alloc, cost):
    """Метод потенциалов с поэтапным журналом.

    Возвращает (final_alloc, steps), где каждый шаг — dict:
      - u: список потенциалов по строкам
      - v: список потенциалов по столбцам
      - deltas: матрица оценок delta_ij = c_ij - (u_i+v_j)
      - entering: {"cell": (i,j), "delta": delta_min} или None, если оптимум
      - cycle: список клеток цикла в порядке обхода
      - plus: список клеток со знаком '+'
      - minus: список клеток со знаком '-'
      - theta: величина перераспределения
      - alloc: матрица после применения шага
    """
    m, n = len(alloc), len(alloc[0])
    work = [row[:] for row in alloc]
    steps = []

    work = make_non_degenerate(work, cost)

    while True:
        u, v = calculate_potentials(work, cost)
        # матрица оценок
        deltas = [[cost[i][j] - (u[i] + v[j]) for j in range(n)] for i in range(m)]
        cell, delta = find_entering_cell(work, cost, u, v)

        if cell is None:
            steps.append({
                "u": u[:],
                "v": v[:],
                "deltas": [row[:] for row in deltas],
                "entering": None,
                "alloc": [row[:] for row in work],
            })
            break

        cycle = build_cycle(work, cell)
        if cycle is None:
            # защита: нет цикла — выходим
            steps.append({
                "u": u[:],
                "v": v[:],
                "deltas": [row[:] for row in deltas],
                "entering": {"cell": cell, "delta": delta},
                "cycle": None,
                "alloc": [row[:] for row in work],
            })
            break

        plus = cycle[::2]
        minus = cycle[1::2]
        theta = min(work[i][j] for i, j in minus)

        # применяем сдвиг
        for i, j in plus:
            work[i][j] += theta
        for i, j in minus:
            work[i][j] -= theta

        steps.append({
            "u": u[:],
            "v": v[:],
            "deltas": [row[:] for row in deltas],
            "entering": {"cell": cell, "delta": delta},
            "cycle": cycle[:],
            "plus": plus[:],
            "minus": minus[:],
            "theta": theta,
            "alloc": [row[:] for row in work],
        })

    return work, steps


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
