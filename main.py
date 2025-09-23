import streamlit as st
import pandas as pd
from typing import List

from solver.transport_simplex import vogel_initial_solution, transportation_to_lp, balance_transportation
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
        # st.caption("Ввод по центру. Каждый элемент — отдельное поле с проверкой (число ≥ 0).")
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
        ctrl_l, ctrl_c, ctrl_r = st.columns([1, 1, 1])
        with ctrl_l:
            objective = st.radio("Целевая функция", ["Максимум", "Минимум"], index=0, horizontal=True)
        with ctrl_c:
            st.write("")
        with ctrl_r:
            run = st.button("Решить", type="primary")

    # Выходные данные — тоже по центру
    if not run:
        return

    with center:
        # Отображение исходных входных данных аккуратно
        total_supply = sum(supply)
        total_demand = sum(demand)
        st.subheader("Введённые данные")
        st.markdown("Матрица стоимостей C:")
        c_cols = [f"D{j+1} ({to_int_if_possible(demand[j])})" for j in range(n)]
        c_idx = [f"S{i+1} ({to_int_if_possible(supply[i])})" for i in range(m)]
        st.dataframe(pd.DataFrame(format_matrix(cost), columns=c_cols, index=c_idx), use_container_width=True)
        c_a, c_b = st.columns(2)
        with c_a:
            st.metric("Сумма запасов", to_int_if_possible(total_supply))
        with c_b:
            st.metric("Сумма потребностей", to_int_if_possible(total_demand))

        # Оптимальное решение (симплекс)
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
                cols_b = [
                    f"D{j+1} ({to_int_if_possible(demand_b[j])})" + (" (фикт.)" if j >= n else "")
                    for j in range(n_b)
                ]
                idx_b = [
                    f"S{i+1} ({to_int_if_possible(supply_b[i])})" + (" (фикт.)" if i >= m else "")
                    for i in range(m_b)
                ]
                st.dataframe(pd.DataFrame(format_matrix(xmat), columns=cols_b, index=idx_b), use_container_width=True)

                # Красивое разложение целевой функции: Z = sum c_ij * x_ij = ...
                try:
                    terms = []
                    numeric_terms = []
                    for i in range(m_b):
                        for j in range(n_b):
                            xij = res.x[i * n_b + j]
                            cij = cost_b[i][j]
                            if abs(xij) > 1e-9:
                                cij_disp = to_int_if_possible(cij)
                                xij_disp = to_int_if_possible(xij)
                                terms.append(fr"{cij_disp}\,x_{{S{i+1},D{j+1}}}")
                                numeric_terms.append(fr"{cij_disp}\cdot {xij_disp}")
                    if terms:
                        latex_symbolic = " + ".join(terms)
                        latex_numeric = " + ".join(numeric_terms)
                        st.markdown("Разложение цели:")
                        st.latex(fr"Z = {latex_symbolic}")
                        st.latex(fr"Z = {latex_numeric} = {to_int_if_possible(sum(cost_b[i][j]*res.x[i*n_b+j] for i in range(m_b) for j in range(n_b)))}")
                        st.caption("Где x_{S_i,D_j} — поставка из S_i к D_j. Метки S/D соответствуют строкам и столбцам таблиц.")
                except Exception:
                    pass

            # Отображение итераций симплекс-метода во вкладке, которую можно скрыть
            if hasattr(res, 'history') and res.history:
                with st.expander("Итерации симплекс-метода (таблицы)", expanded=False):
                    # Подписи столбцов: x[i,j], затем s_k, затем RHS
                    n_dec = len(c)
                    var_cols = [f"x[{i+1},{j+1}]" for i in range(m_b) for j in range(n_b)]
                    # Определим число служебных столбцов по размерности табло
                    slack_count = (len(res.history[0][0]) - 1) - n_dec if res.history and res.history[0] else 0
                    slack_cols = [f"s{k+1}" for k in range(slack_count)]
                    all_cols = var_cols + slack_cols + ["RHS"]

                    for it, T in enumerate(res.history):
                        with st.expander(f"Итерация {it}", expanded=False):
                            row_count = len(T) - 1
                            row_names = [f"R{r+1}" for r in range(row_count)] + ["Z"]
                            df = pd.DataFrame(T, columns=all_cols, index=row_names)
                            df_fmt = df.applymap(lambda v: to_int_if_possible(float(v)))
                            st.dataframe(df_fmt, use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка симплекс-решения: {e}")


if __name__ == "__main__":
    main()

