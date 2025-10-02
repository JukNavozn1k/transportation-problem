# tests/test_transport.py
import pytest

from solver import transport_potential, transport_simplex


@pytest.fixture
def example_data():
    supply = [20, 30, 25]
    demand = [10, 10, 35, 20]
    cost = [
        [8, 6, 10, 9],
        [9, 12, 13, 7],
        [14, 9, 16, 5],
    ]
    return supply, demand, cost


def calc_total_cost(alloc, cost):
    return sum(
        alloc[i][j] * cost[i][j]
        for i in range(len(alloc))
        for j in range(len(alloc[0]))
    )


def test_same_result(example_data):
    supply, demand, cost = example_data

    # метод потенциалов
    alloc_pot = transport_potential.transportation_problem(supply, demand, cost)
    cost_pot = calc_total_cost(alloc_pot, cost)

    # симплекс
    c, A, b, senses, supply_b, demand_b, cost_b = transport_simplex.transportation_to_lp(
        supply, demand, cost
    )
    res = transport_simplex.dual_simplex([-v for v in c], A, b, senses)
    cost_simplex = -res.objective  # т.к. мы минимизируем

    # допускаем очень маленькую разницу из-за численных ошибок
    assert pytest.approx(cost_pot, rel=1e-6) == cost_simplex


def test_balance(example_data):
    """Проверка, что обе функции балансируют данные одинаково"""
    supply, demand, cost = example_data
    s1, d1, c1 = transport_potential.balance_transportation(supply, demand, cost)
    s2, d2, c2 = transport_simplex.balance_transportation(supply, demand, cost)
    assert sum(s1) == sum(d1)
    assert sum(s2) == sum(d2)
    assert sum(s1) == sum(s2)
    assert sum(d1) == sum(d2)


def test_initial_solution(example_data):
    """Начальный план должен полностью удовлетворять спрос/предложение"""
    supply, demand, cost = example_data
    s, d, c = transport_potential.balance_transportation(supply, demand, cost)
    alloc = transport_potential.north_west_corner_solution(s, d)

    assert all(sum(row) == s[i] or s[i] == 0 for i, row in enumerate(alloc))
    assert all(sum(alloc[i][j] for i in range(len(s))) == d[j] or d[j] == 0 for j in range(len(d)))
