import streamlit as st
import pandas as pd
from typing import List

from solver.transport import vogel_initial_solution, transportation_to_lp, balance_transportation
from solver.dual import dual_simplex


def to_int_if_possible(x: float) -> int | float:
    return int(x) if abs(x - round(x)) < 1e-9 else x


def format_matrix(mat: List[List[float]]) -> List[List[int | float]]:
    return [[to_int_if_possible(v) for v in row] for row in mat]


def main():
    st.set_page_config(page_title="Транспортная задача", page_icon="🚚", layout="wide")
    # Центрируем контент
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.title("🚚 Транспортная задача")
        st.caption("Ввод по центру. Каждый элемент — отдельное поле с проверкой (число ≥ 0).")
        st.subheader("Размерность задачи")
        c1, c2 = st.columns(2)
        with c1:
            m = st.number_input("Число поставщиков (m)", min_value=1, max_value=20, value=3, step=1)
        with c2:
            n = st.number_input("Число потребителей (n)", min_value=1, max_value=20, value=4, step=1)

        st.subheader("Матрица стоимостей C (m×n)")
        allow_float = st.checkbox("Разрешить дробные значения", value=False)
        # Сетка полей для матрицы C
        cost: List[List[float]] = []
        for i in range(m):
            cols = st.columns(n)
            row_vals = []
            for j in range(n):
                key = f"c_{i}_{j}"
                default = [
                    [8, 6, 10, 9],
                    [9, 12, 13, 7],
                    [14, 9, 16, 5],
                ]
                dval = default[i][j] if i < len(default) and j < len(default[0]) else 0.0
                if allow_float:
                    val = cols[j].number_input(
                        label=f"C[{i+1},{j+1}]",
                        min_value=0.0,
                        value=float(dval),
                        step=1.0,
                        key=key,
                        format="%.4f",
                    )
                else:
                    val = cols[j].number_input(
                        label=f"C[{i+1},{j+1}]",
                        min_value=0,
                        value=int(dval),
                        step=1,
                        key=key,
                        format="%d",
                    )
                row_vals.append(float(val))
            cost.append(row_vals)

        st.subheader("Запасы поставщиков (длина m)")
        supply_cols = st.columns(m)
        supply: List[float] = []
        for i in range(m):
            dval = [20, 30, 250][i] if i < 3 else 0.0
            if allow_float:
                v = supply_cols[i].number_input(
                    label=f"S[{i+1}]",
                    min_value=0.0,
                    value=float(dval),
                    step=1.0,
                    key=f"s_{i}",
                    format="%.4f",
                )
            else:
                v = supply_cols[i].number_input(
                    label=f"S[{i+1}]",
                    min_value=0,
                    value=int(dval),
                    step=1,
                    key=f"s_{i}",
                    format="%d",
                )
            supply.append(float(v))

        st.subheader("Потребности потребителей (длина n)")
        demand_cols = st.columns(n)
        demand: List[float] = []
        for j in range(n):
            dval = [10, 10, 35, 20][j] if j < 4 else 0.0
            if allow_float:
                v = demand_cols[j].number_input(
                    label=f"D[{j+1}]",
                    min_value=0.0,
                    value=float(dval),
                    step=1.0,
                    key=f"d_{j}",
                    format="%.4f",
                )
            else:
                v = demand_cols[j].number_input(
                    label=f"D[{j+1}]",
                    min_value=0,
                    value=int(dval),
                    step=1,
                    key=f"d_{j}",
                    format="%d",
                )
            demand.append(float(v))

        st.divider()
        options_col1, options_col2, options_col3 = st.columns([1, 1, 1])
        with options_col1:
            do_vogel = st.checkbox("Начальный план (Фогель)", value=True)
        with options_col2:
            do_lp = st.checkbox("Оптимальное решение (симплекс)", value=True)
        with options_col3:
            run = st.button("Решить", type="primary")

        obj_col1, obj_col2 = st.columns([1, 2])
        with obj_col1:
            st.write("")
        with obj_col2:
            # objective = st.radio("Целевая функция", ["Максимум", "Минимум"], index=0, horizontal=True)
            objective = st.radio("Целевая функция", ["Минимум"], index=0, horizontal=True)

    # Выходные данные — тоже по центру
    if not run:
        return

    with center:
        # Отображение исходных входных данных аккуратно
        total_supply = sum(supply)
        total_demand = sum(demand)
        st.subheader("Введённые данные")
        st.markdown("Матрица стоимостей C:")
        c_cols = [f"D{j+1}" for j in range(n)]
        c_idx = [f"S{i+1}" for i in range(m)]
        st.dataframe(pd.DataFrame(format_matrix(cost), columns=c_cols, index=c_idx), use_container_width=True)
        c_a, c_b = st.columns(2)
        with c_a:
            st.metric("Сумма запасов", to_int_if_possible(total_supply))
        with c_b:
            st.metric("Сумма потребностей", to_int_if_possible(total_demand))

        if do_vogel:
            st.subheader("Начальный план (метод Фогеля)")
            try:
                # Балансируем перед вычислением плана
                supply_v, demand_v, cost_v = balance_transportation(supply[:], demand[:], cost)
                m_v, n_v = len(supply_v), len(demand_v)
                init_plan = vogel_initial_solution(supply_v[:], demand_v[:], cost_v)
                st.caption("(Использованы сбалансированные данные, при необходимости добавлены фиктивные строки/столбцы с нулевой стоимостью)")
                cols_v = [f"D{j+1}" + (" (фикт.)" if j >= n else "") for j in range(n_v)]
                idx_v = [f"S{i+1}" + (" (фикт.)" if i >= m else "") for i in range(m_v)]
                st.dataframe(pd.DataFrame(format_matrix(init_plan), columns=cols_v, index=idx_v), use_container_width=True)
                init_cost = sum(init_plan[i][j] * cost_v[i][j] for i in range(m_v) for j in range(n_v))
                st.metric("Стоимость начального плана", to_int_if_possible(init_cost))
            except Exception as e:
                st.error(f"Ошибка вычисления начального плана: {e}")

        if do_lp:
            st.subheader("Оптимальное решение (симплекс)")
            try:
                c, A, b, senses, supply_b, demand_b, cost_b = transportation_to_lp(supply, demand, cost)
                m_b, n_b = len(supply_b), len(demand_b)
                # Знак цели: по умолчанию максимум (передаём c), для минимума — домножаем на -1
                if objective == "Минимум":
                    c_input = [-v for v in c]
                else:
                    c_input = c
                res = dual_simplex(c_input, A, b, senses)
                if res.status != "optimal":
                    st.warning(f"Статус решения: {res.status}")
                else:
                    st.success("Найдено оптимальное решение")

                st.caption("(Решение получено на сбалансированных данных)")
                st.subheader("Целевая функция")
                # Общее описание модели и цели
                if objective == "Максимум":
                    st.latex(r"Z = \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij} \, x_{ij} \;\to\; \max")
                else:
                    st.latex(r"Z = \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij} \, x_{ij} \;\to\; \min")
                    st.caption("Для решения через симплекс мы домножаем коэффициенты цели на −1: c' = −c.")
                st.markdown("Ограничения: ")
                st.latex(r"\sum_{j=1}^{n} x_{ij} = \text{supply}_i,\quad i=1..m;\quad \sum_{i=1}^{m} x_{ij} = \text{demand}_j,\quad j=1..n;\quad x_{ij} \ge 0.")

                st.markdown("Оптимальное значение целевой функции:")
                if res.objective is None:
                    display_obj = float('nan')
                else:
                    display_obj = res.objective if objective == "Максимум" else -res.objective
                st.metric("Z*", to_int_if_possible(display_obj))

                if res.x:
                    xmat = [res.x[i * n_b:(i + 1) * n_b] for i in range(m_b)]
                    st.markdown("Оптимальный план (матрица x):")
                    cols_b = [f"D{j+1}" + (" (фикт.)" if j >= n else "") for j in range(n_b)]
                    idx_b = [f"S{i+1}" + (" (фикт.)" if i >= m else "") for i in range(m_b)]
                    st.dataframe(pd.DataFrame(format_matrix(xmat), columns=cols_b, index=idx_b), use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка симплекс-решения: {e}")


if __name__ == "__main__":
    main()

