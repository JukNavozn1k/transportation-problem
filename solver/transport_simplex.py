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


from copy import deepcopy
from fractions import Fraction as F

class SimplexResult:
    def __init__(self, status, x=None, objective=None, alternative=False, tableau=None, history=None):
        self.status = status
        self.x = x or []
        self.objective = float(objective) if objective is not None else None
        self.alternative = alternative
        self.tableau = tableau
        self.history = history or []

def recover_basis_from_tableau(tableau):
    """
    Восстановить basis из tableau.
    Возвращает список length = m (число строк без строки Z), где
    basis[i] = j, если столбец j является единичным в строке i (unit column),
    иначе None.
    Работает с типами, которые можно преобразовать в Fraction (F).
    """
    if not tableau:
        return []

    m = len(tableau) - 1            # число строк ограничений
    cols = len(tableau[0]) - 1      # число столбцов без RHS
    basis = [None] * m

    for j in range(cols):
        one_row = None
        is_unit = True
        for i in range(m):
            v = F(tableau[i][j])
            if v == 1:
                if one_row is None:
                    one_row = i
                else:
                    # встретили ещё одну единицу — не unit-столбец
                    is_unit = False
                    break
            elif v == 0:
                continue
            else:
                # ненулевая отличная от единицы — не unit-столбец
                is_unit = False
                break
        if is_unit and one_row is not None:
            basis[one_row] = j

    return basis

def pivot(tableau, basis, row, col):
    piv = tableau[row][col]
    tableau[row] = [v / piv for v in tableau[row]]
    for r in range(len(tableau)):
        if r != row:
            factor = tableau[r][col]
            tableau[r] = [a - factor * b for a, b in zip(tableau[r], tableau[row])]
    basis[row] = col

def bland_rule_dual(tableau):
    best_row = None
    for i, row in enumerate(tableau[:-1]):
        rhs = row[-1]
        if rhs < 0:
            best_row = i
            break
    return best_row

def find_entering_variable_dual(tableau, row):
    best_col = None
    best_ratio = None
    last = tableau[-1]
    for j, coeff in enumerate(tableau[row][:-1]):
        if coeff < 0:
            rc = last[j]
            ratio = rc / coeff
            if best_ratio is None or ratio < best_ratio or (ratio == best_ratio and j < best_col):
                best_ratio = ratio
                best_col = j
    return best_col

def preprocess_constraints(A, b, senses):
    A2, b2, s2 = [], [], []
    for row, rhs, sense in zip(A, b, senses):
        if sense == '<=':
            A2.append(row)
            b2.append(rhs)
            s2.append('<=')  
        elif sense == '>=':
            A2.append([-c for c in row])
            b2.append(-rhs)
            s2.append('<=')  
        elif sense == '==':
            A2.append(row)
            b2.append(rhs)
            s2.append('<=')  
            A2.append([-c for c in row])
            b2.append(-rhs)
            s2.append('<=')  
        else:
            raise ValueError(f"Unknown sense: {sense}")
    return A2, b2, s2

def build_tableau(c, A, b, senses):
    m, n = len(A), len(c)
    slack_count = sum(1 for s in senses if s in ('<=', '>='))  
    tableau = []
    for i in range(m):
        row = list(map(F, A[i]))
        slack = [F(0)] * slack_count
        rhs = F(b[i])
        slack_pos = sum(1 for t in senses[:i] if t in ('<=', '>='))  
        if senses[i] == '<=':
            slack[slack_pos] = F(1)
        row += slack
        row.append(rhs)
        tableau.append(row)
    cost = list(map(lambda v: -F(v), c)) + [F(0)] * slack_count + [F(0)]
    tableau.append(cost)
    return tableau, slack_count

def extract_solution(tableau, basis, n):
    x = [0] * n
    for i, var in enumerate(basis):
        if var is not None and var < n:
            x[var] = float(tableau[i][-1])
    return x

def dual_simplex(c, A, b, senses=None):
    m, n = len(A), len(c)
    if senses is None:
        senses = ['<='] * m

    A2, b2, s2 = preprocess_constraints(A, b, senses)
    T, slack_count = build_tableau(c, A2, b2, s2)
    history = [deepcopy(T)]

    basis = [None] * len(A2)
    for j in range(len(T[0]) - 1):
        one_row = None
        is_unit = True
        for i in range(len(A2)):
            if T[i][j] == 1:
                if one_row is None:
                    one_row = i
                else:
                    is_unit = False
                    break
            elif T[i][j] != 0:
                is_unit = False
                break
        if is_unit and one_row is not None:
            basis[one_row] = j

    # двойственный симплекс
    while True:
        row = bland_rule_dual(T)
        if row is None:
            break
        col = find_entering_variable_dual(T, row)
        if col is None:
            return SimplexResult('infeasible', tableau=deepcopy(T), history=history)
        pivot(T, basis, row, col)
        history.append(deepcopy(T))

    # обычный симплекс
    while True:
        enter = None
        for j, coeff in enumerate(T[-1][:-1]):
            if coeff < 0:
                enter = j
                break
        if enter is None:
            break
        leave = None
        min_ratio = None
        for i, rowv in enumerate(T[:-1]):
            if rowv[enter] > 0:
                ratio = rowv[-1] / rowv[enter]
                if ratio >= 0 and (min_ratio is None or ratio < min_ratio):
                    min_ratio = ratio
                    leave = i
        if leave is None:
            return SimplexResult('unbounded', tableau=deepcopy(T), history=history)
        pivot(T, basis, leave, enter)
        history.append(deepcopy(T))

    x = extract_solution(T, basis, n)
    obj = float(T[-1][-1])
    alternative = any(T[-1][j] == 0 and j not in basis for j in range(len(T[0])-1))
    return SimplexResult('optimal', x, obj, alternative, tableau=deepcopy(T), history=history)




def balance_transportation(supply, demand, cost):
    total_supply = sum(supply)
    total_demand = sum(demand)

    supply = supply[:]
    demand = demand[:]
    cost = [row[:] for row in cost]  # копия

    if total_supply > total_demand:
        # добавляем фиктивного потребителя
        diff = total_supply - total_demand
        demand.append(diff)
        for row in cost:
            row.append(0)  # нулевая стоимость
    elif total_demand > total_supply:
        # добавляем фиктивного поставщика
        diff = total_demand - total_supply
        supply.append(diff)
        cost.append([0] * len(demand))

    return supply, demand, cost
def transportation_to_lp(supply, demand, cost):
    # сначала балансируем
    supply, demand, cost = balance_transportation(supply, demand, cost)

    m, n = len(supply), len(demand)
    c = [cost[i][j] for i in range(m) for j in range(n)]
    A, b, senses = [], [], []

    # ограничения по строкам (сумма по j = supply[i])
    for i in range(m):
        row = [0] * (m * n)
        for j in range(n):
            row[i * n + j] = 1
        A.append(row)
        b.append(supply[i])
        senses.append("==")

    # ограничения по столбцам (сумма по i = demand[j])
    for j in range(n):
        row = [0] * (m * n)
        for i in range(m):
            row[i * n + j] = 1
        A.append(row)
        b.append(demand[j])
        senses.append("==")

    return c, A, b, senses, supply, demand, cost

if __name__ == "__main__":
   
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
