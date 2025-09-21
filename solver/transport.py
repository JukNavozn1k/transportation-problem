def vogel_initial_solution(supply, demand, cost):
    supply = supply[:]
    demand = demand[:]
    m, n = len(supply), len(demand)
    alloc = [[0]*n for _ in range(m)]

    while any(s > 0 for s in supply) and any(d > 0 for d in demand):
        # вычисляем штрафы
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

def transportation_to_lp(supply, demand, cost):
    m, n = len(supply), len(demand)
    c = [cost[i][j] for i in range(m) for j in range(n)]
    A, b, senses = [], [], []

    # ограничения по строкам (сумма по j = supply[i])
    for i in range(m):
        row = [0]*(m*n)
        for j in range(n):
            row[i*n + j] = 1
        A.append(row)
        b.append(supply[i])
        senses.append("==")

    # ограничения по столбцам (сумма по i = demand[j])
    for j in range(n):
        row = [0]*(m*n)
        for i in range(m):
            row[i*n + j] = 1
        A.append(row)
        b.append(demand[j])
        senses.append("==")

    return c, A, b, senses

if __name__ == "__main__":
    from dual import dual_simplex
    supply = [20, 30, 25]
    demand = [10, 10, 35, 20]
    cost = [
        [8, 6, 10, 9],
        [9, 12, 13, 7],
        [14, 9, 16, 5],
    ]

    # 1. Начальный план методом Фогеля
    init_plan = vogel_initial_solution(supply, demand, cost)
    print("Начальный план (Фогель):")
    for row in init_plan:
        print(row)

    # 2. Запуск симплекса
    c, A, b, senses = transportation_to_lp(supply, demand, cost)
    res = dual_simplex([-v for v in c], A, b, senses)

    print("Статус:", res.status)
    print("Оптимальное значение:", res.objective)
    print("Оптимальный план (x):", res.x)
