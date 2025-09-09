"""
transportation_solver.py

Решение транспортной задачи:
- начальное приближение методом Фогеля (Vogel's approximation)
- оптимизация методом потенциалов (u-v)

Особенности:
- Автоматическое выравнивание (добавляется фиктивный поставщик/потребитель при несбалансированности)
- Обработка вырождений (добавляются нулевые базисные клетки, чтобы иметь m+n-1 базисных переменных)
- Пошаговая история: на каждой итерации оптимизации возвращается матрица распределений, базис, векторы потенциалов, матрица редуцированных стоимостей, и пояснение шага

Функции:
- solve_transportation(costs, supply, demand)
    -> вернёт (final_allocation, total_cost, history)

История представляет собой список словарей. Каждая запись содержит:
- 'step' : текстовое описание шага
- 'allocation' : матрица (список списков) текущих назначений (None или число)
- 'basis' : список пар (i,j) базисных клеток
- 'u' , 'v' : списки потенциалов
- 'reduced_costs' : матрица редуцированных стоимостей (c_ij - u_i - v_j)
- 'entering' : координата входящей клетки (если есть)
- 'leaving' : координата исключаемой клетки (если есть)

Пример использования внизу файла.

Автор: ChatGPT (генерировано автоматически)
"""
from copy import deepcopy
from collections import deque

INF = 10**12


def _add_dummy_if_unbalanced(costs, supply, demand):
    m = len(supply)
    n = len(demand)
    total_supply = sum(supply)
    total_demand = sum(demand)
    costs2 = [row[:] for row in costs]
    supply2 = supply[:]
    demand2 = demand[:]
    dummy_added = None
    if total_supply > total_demand:
        # add dummy demand (column)
        diff = total_supply - total_demand
        for row in costs2:
            row.append(0)  # zero cost to dummy
        demand2.append(diff)
        dummy_added = ('col', n)  # column index
    elif total_demand > total_supply:
        diff = total_demand - total_supply
        costs2.append([0]*n)  # new row
        supply2.append(diff)
        dummy_added = ('row', m)
    return costs2, supply2, demand2, dummy_added


def vogel_initial_solution(costs, supply, demand):
    """Вернуть начальное планирование методом Фогеля.
    Возвращает allocation (m x n) где None = пусто, значение = назначение.
    И список базисных клеток (i,j) с ненулевыми или нулевыми (добавленные при вырождении)
    """
    m = len(supply)
    n = len(demand)
    supply_left = supply[:]
    demand_left = demand[:]
    allocation = [[None for _ in range(n)] for __ in range(m)]
    basis = []

    # Helper: compute penalties
    rows_alive = [True]*m
    cols_alive = [True]*n

    while True:
        # check finished
        if all(not x for x in rows_alive) or all(not x for x in cols_alive):
            break
        # compute row penalties
        row_pen = [None]*m
        for i in range(m):
            if not rows_alive[i]:
                continue
            costs_row = [costs[i][j] for j in range(n) if cols_alive[j]]
            if len(costs_row) == 0:
                row_pen[i] = -1
            elif len(costs_row) == 1:
                row_pen[i] = costs_row[0]
            else:
                sorted_two = sorted(costs_row)[:2]
                row_pen[i] = sorted_two[1] - sorted_two[0]
        col_pen = [None]*n
        for j in range(n):
            if not cols_alive[j]:
                continue
            costs_col = [costs[i][j] for i in range(m) if rows_alive[i]]
            if len(costs_col) == 0:
                col_pen[j] = -1
            elif len(costs_col) == 1:
                col_pen[j] = costs_col[0]
            else:
                sorted_two = sorted(costs_col)[:2]
                col_pen[j] = sorted_two[1] - sorted_two[0]

        # find max penalty
        best_row_pen = max(((row_pen[i], i) for i in range(m) if rows_alive[i]), default=(None, None))
        best_col_pen = max(((col_pen[j], j) for j in range(n) if cols_alive[j]), default=(None, None))

        if best_row_pen[0] is None and best_col_pen[0] is None:
            break
        if best_row_pen[0] is None:
            choose = ('col', best_col_pen[1])
        elif best_col_pen[0] is None:
            choose = ('row', best_row_pen[1])
        else:
            if best_row_pen[0] > best_col_pen[0]:
                choose = ('row', best_row_pen[1])
            else:
                choose = ('col', best_col_pen[1])

        if choose[0] == 'row':
            i = choose[1]
            # select minimum cost in row among alive columns
            j = min((j for j in range(n) if cols_alive[j]), key=lambda jj: costs[i][jj])
        else:
            j = choose[1]
            i = min((i for i in range(m) if rows_alive[i]), key=lambda ii: costs[ii][j])

        q = min(supply_left[i], demand_left[j])
        allocation[i][j] = (allocation[i][j] or 0) + q
        basis.append((i, j))
        supply_left[i] -= q
        demand_left[j] -= q
        if abs(supply_left[i]) < 1e-9:
            rows_alive[i] = False
        if abs(demand_left[j]) < 1e-9:
            cols_alive[j] = False
    # handle remaining zeros if any supply/demand left due to numeric issues
    # ensure basis size <= m+n-1
    return allocation, basis


def _basis_size(m, n, basis):
    # count unique basis positions
    return len(set(basis))


def _ensure_non_degenerate(allocation, basis, m, n):
    """Если базис вырожден (меньше m+n-1), добавляем нулевые базисные клетки.
    Добавляем такие клетки, чтобы не образовывать циклы (т.е. не создавать замыкания),
    но при этом довести число базисов до m+n-1.
    Простая эвристика: пробуем добавить клетки (i,j) которые не создают цикла
    """
    needed = m + n - 1 - _basis_size(m, n, basis)
    if needed <= 0:
        return basis
    basis_set = set(basis)

    def _creates_cycle_if_added(bset, new_cell):
        # проверка: добавление new_cell к bset создаёт ли цикл?
        # цикл = существование замкнутого пути только по горизонталям/вертикалям
        # Мы можем проверить с помощью построения графа строк и столбцов.
        # Построим двудольный граф: строки -> столбцы для каждой базисной клетки.
        from collections import defaultdict, deque
        g = defaultdict(list)
        for (ii, jj) in bset:
            g[('r', ii)].append(('c', jj))
            g[('c', jj)].append(('r', ii))
        # добавим new_cell
        ii, jj = new_cell
        g[('r', ii)].append(('c', jj))
        g[('c', jj)].append(('r', ii))
        # цикл в двудольном графе означает, что граф содержит цикл четной длины.
        # Просто попробуем найти цикл через DFS track parent.
        visited = set()
        parent = {}
        for node in list(g.keys()):
            if node in visited:
                continue
            stack = [(node, None)]
            while stack:
                cur, par = stack.pop()
                if cur in visited:
                    # if we revisit and not parent -> cycle
                    if parent.get(cur) != par:
                        return True
                    continue
                visited.add(cur)
                parent[cur] = par
                for nb in g[cur]:
                    if nb == par:
                        continue
                    stack.append((nb, cur))
        return False

    # try all cells in row-major order
    for i in range(m):
        for j in range(n):
            if (i, j) in basis_set:
                continue
            if not _creates_cycle_if_added(basis_set, (i, j)):
                basis_set.add((i, j))
                needed -= 1
                if needed == 0:
                    return list(basis_set)
    # If couldn't avoid cycles by heuristic, just add arbitrary cells
    for i in range(m):
        for j in range(n):
            if (i, j) in basis_set:
                continue
            basis_set.add((i, j))
            needed -= 1
            if needed == 0:
                return list(basis_set)
    return list(basis_set)


def _compute_potentials(costs, basis, m, n):
    """Найти u (строки) и v (столбцы) такие, что для всех базисных клеток c_ij = u_i + v_j.
    Если система недоопределена, некоторые потенциалы остаются None.
    """
    u = [None]*m
    v = [None]*n
    # choose u0 = 0
    if not basis:
        return u, v
    u[basis[0][0]] = 0
    changed = True
    basis_set = set(basis)
    while changed:
        changed = False
        for (i, j) in basis_set:
            if u[i] is not None and v[j] is None:
                v[j] = costs[i][j] - u[i]
                changed = True
            elif v[j] is not None and u[i] is None:
                u[i] = costs[i][j] - v[j]
                changed = True
    return u, v


def _reduced_costs(costs, u, v):
    m = len(costs)
    n = len(costs[0])
    rc = [[None]*n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            ui = u[i] if u[i] is not None else INF
            vj = v[j] if v[j] is not None else INF
            if ui == INF or vj == INF:
                rc[i][j] = None
            else:
                rc[i][j] = costs[i][j] - (u[i] + v[j])
    return rc


def _find_entering_cell(rc, basis_set):
    # выбираем наиболее отрицательную редуцированную стоимость
    m = len(rc)
    n = len(rc[0])
    best = None
    for i in range(m):
        for j in range(n):
            if (i, j) in basis_set:
                continue
            if rc[i][j] is None:
                continue
            if rc[i][j] < 0:
                if best is None or rc[i][j] < rc[best[0]][best[1]]:
                    best = (i, j)
    return best


def _find_cycle(basis, entering, m, n):
    """Найти замкнутый цикл (альтернирующий по горизонтали/вертикали),
    включающий entering. Возвращает список клеток в порядке цикла.
    Алгоритм: используем рекурсивный поиск чередующихся ходов по строкам/столбцам.
    """
    basis_set = set(basis)
    all_cells = set(basis)
    all_cells.add(entering)

    # Build mapping by row and by col
    rows = {i: [] for i in range(m)}
    cols = {j: [] for j in range(n)}
    for (i, j) in all_cells:
        rows[i].append((i, j))
        cols[j].append((i, j))

    start = entering
    path = [start]

    def dfs(cur, visited, need_row_move):
        # need_row_move: if True, next move must change column (i.e., move within same row)
        i, j = cur
        if len(path) > 3 and cur == start:
            return True
        if need_row_move:
            # move to other cells in the same row
            for (ii, jj) in rows[i]:
                if (ii, jj) == cur:
                    continue
                if (ii, jj) in visited:
                    # can close if it's start and length>=4
                    if (ii, jj) == start and len(path) >= 4:
                        path.append((ii, jj))
                        return True
                    continue
                visited.add((ii, jj))
                path.append((ii, jj))
                if dfs((ii, jj), visited, not need_row_move):
                    return True
                path.pop()
                visited.remove((ii, jj))
        else:
            for (ii, jj) in cols[j]:
                if (ii, jj) == cur:
                    continue
                if (ii, jj) in visited:
                    if (ii, jj) == start and len(path) >= 4:
                        path.append((ii, jj))
                        return True
                    continue
                visited.add((ii, jj))
                path.append((ii, jj))
                if dfs((ii, jj), visited, not need_row_move):
                    return True
                path.pop()
                visited.remove((ii, jj))
        return False

    visited = set([start])
    if not dfs(start, visited, True):
        # try starting with column move
        path = [start]
        visited = set([start])
        if not dfs(start, visited, False):
            raise ValueError("Не удалось найти цикл (возможная ошибка в базисе)")
    # path ends with start repeated; normalize cycle (remove last repeated)
    if path[-1] == path[0]:
        path = path[:-1]
    # ensure alternating +/- positions; return path
    return path


def _apply_cycle(allocation, cycle):
    """Применить изменение по циклу: чередуем + и - начиная с входящей клетки (+)
    Найти минимальное значение на минус-позициях и скорректировать.
    Возвращает leaving_cell (координаты удаляемой базисной клетки) и amount.
    """
    # positions with '-' are every second cell starting from index 1
    minus_positions = [cycle[k] for k in range(1, len(cycle), 2)]
    vals = []
    for (i, j) in minus_positions:
        v = allocation[i][j] or 0
        vals.append(v)
    if not vals:
        raise ValueError("Пустые минус-позиции при применении цикла")
    theta = min(vals)
    # apply
    for idx, (i, j) in enumerate(cycle):
        if idx % 2 == 0:
            # plus
            cur = allocation[i][j] or 0
            allocation[i][j] = cur + theta
        else:
            cur = allocation[i][j] or 0
            allocation[i][j] = cur - theta
            # if becomes zero, set to 0 (not None) to allow degenerate basis
            if abs(allocation[i][j]) < 1e-12:
                allocation[i][j] = 0
    # leaving cell: any minus position that became zero and was in basis before
    leaving = None
    for (i, j) in minus_positions:
        if abs(allocation[i][j]) < 1e-9:
            leaving = (i, j)
            break
    return leaving, theta


def optimize_by_potentials(costs, allocation, basis, m, n, history):
    """Основной цикл оптимизации методом потенциалов.
    Модифицирует allocation и basis. Записывает шаги в history.
    """
    # ensure basis non-degenerate
    basis = _ensure_non_degenerate(allocation, basis, m, n)
    basis_set = set(basis)

    step_count = 0
    while True:
        step_count += 1
        # compute potentials
        u, v = _compute_potentials(costs, basis, m, n)
        rc = _reduced_costs(costs, u, v)
        history.append({
            'step': f'Compute potentials at iteration {step_count}',
            'allocation': deepcopy(allocation),
            'basis': sorted(list(basis_set)),
            'u': u[:],
            'v': v[:],
            'reduced_costs': deepcopy(rc),
            'entering': None,
            'leaving': None,
        })
        entering = _find_entering_cell(rc, basis_set)
        if entering is None:
            # optimal
            history.append({'step': 'Optimal solution reached', 'allocation': deepcopy(allocation), 'basis': sorted(list(basis_set)), 'u': u, 'v': v, 'reduced_costs': rc, 'entering': None, 'leaving': None})
            break
        # find cycle including entering
        cycle = _find_cycle(basis, entering, m, n)
        # ensure cycle starts with entering
        # rotate cycle so that cycle[0] == entering
        try:
            idx0 = cycle.index(entering)
            cycle = cycle[idx0:] + cycle[:idx0]
        except ValueError:
            pass
        # apply cycle changes
        leaving, theta = _apply_cycle(allocation, cycle)
        # update basis: add entering, remove leaving (if leaving equals entering shouldn't happen)
        basis_set.add(entering)
        if leaving is not None and leaving in basis_set:
            basis_set.remove(leaving)
        # after update, ensure non-degenerate
        basis = list(basis_set)
        basis = _ensure_non_degenerate(allocation, basis, m, n)
        basis_set = set(basis)
        history.append({
            'step': f'Pivot: entering={entering}, leaving={leaving}, theta={theta}',
            'allocation': deepcopy(allocation),
            'basis': sorted(list(basis_set)),
            'u': None,
            'v': None,
            'reduced_costs': None,
            'entering': entering,
            'leaving': leaving,
        })
    return allocation, basis, history


def total_cost_from_allocation(costs, allocation):
    m = len(costs)
    n = len(costs[0])
    total = 0
    for i in range(m):
        for j in range(n):
            if allocation[i][j] is not None and allocation[i][j] != 0:
                total += allocation[i][j] * costs[i][j]
    return total


def solve_transportation(costs, supply, demand):
    """Главная функция. Возвращает (final_allocation, total_cost, history, info)
    info содержит данные о добавлении фиктивного ряда/столбца, чтобы понять трансформации.
    """
    # copy inputs
    costs0 = [row[:] for row in costs]
    supply0 = supply[:]
    demand0 = demand[:]
    costs2, supply2, demand2, dummy = _add_dummy_if_unbalanced(costs0, supply0, demand0)
    m = len(supply2)
    n = len(demand2)
    allocation, basis = vogel_initial_solution(costs2, supply2, demand2)

    # If Vogel's left some None cells in basis, ensure basis size
    basis = _ensure_non_degenerate(allocation, basis, m, n)
    history = []

    # Record initial state
    history.append({'step': 'Initial Vogel approximation', 'allocation': deepcopy(allocation), 'basis': sorted(list(set(basis))), 'u': None, 'v': None, 'reduced_costs': None, 'entering': None, 'leaving': None})

    allocation_final, basis_final, history = optimize_by_potentials(costs2, allocation, basis, m, n, history)

    total = total_cost_from_allocation(costs2, allocation_final)

    # If dummy was added, produce mapping back to original sizes and indicate dummy
    info = {'dummy': dummy, 'original_m': len(supply0), 'original_n': len(demand0)}

    # Trim allocation to original dimensions and return full allocation including dummy
    return allocation_final, total, history, info


if __name__ == '__main__':
    # Пример из учебника
    costs = [
        [19, 30, 50, 10],
        [70, 30, 40, 60],
        [40, 8, 70, 20]
    ]
    supply = [7, 9, 18]
    demand = [5, 8, 7, 14]

    alloc, total, history, info = solve_transportation(costs, supply, demand)
    print('Final allocation:')
    for row in alloc:
        print(row)
    print('Total cost =', total)
    print('Dummy info:', info)
    print('\nHistory steps:', len(history))
    for step in history:
        print('STEP:', step['step'])
        # allocation printed briefly
        for r in step['allocation']:
            print(r)
        print()
