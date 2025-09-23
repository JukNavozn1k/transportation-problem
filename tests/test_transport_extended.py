# tests/test_transport_extended.py
import pytest
import random
from solver import transport_potential, transport_simplex


def calc_total_cost(alloc, cost):
    m = min(len(alloc), len(cost))
    n = min(len(alloc[0]), len(cost[0]))
    return sum(
        alloc[i][j] * cost[i][j]
        for i in range(m)
        for j in range(n)
    )


def solve_with_potential(supply, demand, cost):
    alloc = transport_potential.transportation_problem(supply, demand, cost)
    return calc_total_cost(alloc, cost)


def solve_with_simplex(supply, demand, cost):
    c, A, b, senses, supply_b, demand_b, cost_b = transport_simplex.transportation_to_lp(
        supply, demand, cost
    )
    res = transport_simplex.dual_simplex([-v for v in c], A, b, senses)
    return -res.objective  # т.к. минимизация


@pytest.mark.parametrize("supply, demand, cost", [
    # 🔹 Базовый маленький случай 2x2
    ([5, 15], [10, 10],
     [[2, 3],
      [4, 1]]),

    # 🔹 Прямоугольный случай 2x3
    ([20, 30], [10, 25, 15],
     [[8, 6, 10],
      [9, 12, 7]]),

    # 🔹 Квадратный случай 3x3
    ([30, 25, 35], [20, 30, 40],
     [[3, 1, 7],
      [2, 6, 5],
      [9, 8, 4]]),

    # 🔹 Несбалансированный: предложение больше
    ([40, 30], [20, 25, 15],
     [[4, 8, 6],
      [5, 3, 7]]),

    # 🔹 Несбалансированный: спрос больше
    ([20, 25], [15, 10, 30],
     [[2, 7, 5],
      [6, 4, 3]]),

    # 🔹 Вырожденный случай — одинаковые стоимости
    ([10, 10], [10, 10],
     [[5, 5],
      [5, 5]]),
])
def test_potential_vs_simplex(supply, demand, cost):
    cost_pot = solve_with_potential(supply, demand, cost)
    cost_simplex = solve_with_simplex(supply, demand, cost)
    assert pytest.approx(cost_pot, rel=1e-6) == cost_simplex



@pytest.mark.parametrize("supply, demand, cost", [
    # 🔹 Большой перекос — всё забирает один потребитель
    ([50, 30, 20], [100], 
     [[2],
      [3],
      [1]]),

    # 🔹 Большой перекос — один поставщик обслуживает всех
    ([100], [40, 30, 30],
     [[4, 5, 6]]),

    # 🔹 Вырожденный случай: все стоимости разные, но базис неполный
    ([10, 20], [15, 15],
     [[1, 100],
      [100, 1]]),

    # 🔹 Более крупный случай 4x4
    ([20, 30, 25, 15], [10, 20, 30, 30],
     [[8, 6, 10, 9],
      [9, 12, 13, 7],
      [14, 9, 16, 5],
      [10, 11, 8, 6]]),
])
def test_potential_vs_simplex_extra(supply, demand, cost):
    cost_pot = solve_with_potential(supply, demand, cost)
    cost_smp = solve_with_simplex(supply, demand, cost)
    assert abs(cost_pot - cost_smp) < 1e-6
