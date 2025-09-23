import streamlit as st
import pandas as pd
from typing import List
import math

from solver.transport_potential import (
    balance_transportation,
    vogel_initial_solution_with_steps,
    improve_plan_with_steps,
)


def to_int_if_possible(x) -> int | float | str:
    """Пытается красиво привести число к целому, но безопасно для NaN/inf/нечисловых."""
    try:
        xf = float(x)
    except Exception:
        return x
    if not math.isfinite(xf):
        return x
    if abs(xf - round(xf)) < 1e-9:
        return int(round(xf))
    return xf


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

        # Показать сбалансированную таблицу с фиктивными участниками (если требуется)
        if abs(total_supply - total_demand) > 1e-9:
            st.markdown("Сбалансированная таблица (с учётом фиктивного поставщика/потребителя):")
            supply_b2, demand_b2, cost_b2 = balance_transportation(supply, demand, cost)
            m_b2, n_b2 = len(supply_b2), len(demand_b2)
            cols_b2 = [
                f"D{j+1} ({to_int_if_possible(demand_b2[j])})" + (" (фикт.)" if j >= n else "")
                for j in range(n_b2)
            ]
            idx_b2 = [
                f"S{i+1} ({to_int_if_possible(supply_b2[i])})" + (" (фикт.)" if i >= m else "")
                for i in range(m_b2)
            ]
            st.dataframe(pd.DataFrame(format_matrix(cost_b2), columns=cols_b2, index=idx_b2), use_container_width=True)

        # Оптимальное решение методом потенциалов с поэтапным выводом
        st.subheader("Оптимальное решение (метод потенциалов)")
        try:
            # Для максимизации инвертируем знак стоимостей
            if objective == "Максимум":
                cost_eff = [[-c for c in row] for row in cost]
            else:
                cost_eff = [row[:] for row in cost]

            # Балансировка (рабочая для оптимизации)
            supply_b, demand_b, cost_b = balance_transportation(supply, demand, cost_eff)
            # Балансировка (для показа исходной стоимости в разложении цели)
            supply_disp, demand_disp, cost_disp = balance_transportation(supply, demand, cost)
            m_b, n_b = len(supply_b), len(demand_b)

            # 1) Фогель — шаги
            init_alloc, vogel_steps = vogel_initial_solution_with_steps(supply_b, demand_b, cost_b)
            with st.expander("Метод Фогеля — поэтапно", expanded=False):
                # Держим состояние ДО шага, чтобы пояснять выбор на текущем шаге
                pre_supply = supply_b[:]
                pre_demand = demand_b[:]
                for k, step in enumerate(vogel_steps, start=1):
                    chosen = step["chosen"]
                    # Пояснение шага выбора
                    st.markdown(
                        f"Шаг {k}: выбран критерий максимального штрафа — {'строка' if chosen['type']=='row' else 'столбец'} {chosen['index']+1}. В выбранной {'строке' if chosen['type']=='row' else 'столбце'} берём ячейку с минимальной стоимостью: S{chosen['cell'][0]+1}-D{chosen['cell'][1]+1}. Распределяем количество = {to_int_if_possible(chosen['qty'])}. Штраф = {to_int_if_possible(chosen['penalty'])}."
                    )
                    # Штрафы по строкам в виде таблицы (для всех строк)
                    row_pen_map = {i: p for p, i in step.get('row_penalties', [])}
                    row_pen_df = pd.DataFrame({
                        'Штраф': [to_int_if_possible(row_pen_map.get(i, float('nan'))) for i in range(len(supply_b))]
                    }, index=[f"S{i+1}" for i in range(len(supply_b))])
                    st.caption("Штрафы по строкам (разность двух наименьших доступных стоимостей в строке):")
                    st.dataframe(row_pen_df, use_container_width=True)
                    # Штрафы по столбцам в виде таблицы (для всех столбцов)
                    col_pen_map = {j: p for p, j in step.get('col_penalties', [])}
                    col_pen_df = pd.DataFrame([
                        [to_int_if_possible(col_pen_map.get(j, float('nan'))) for j in range(len(demand_b))]
                    ], columns=[f"D{j+1}" for j in range(len(demand_b))])
                    st.caption("Штрафы по столбцам (разность двух наименьших доступных стоимостей в столбце):")
                    st.dataframe(col_pen_df, use_container_width=True)
                    # Объяснение выбора: где максимальный штраф на этом шаге
                    max_row = max(step.get('row_penalties', []), default=(float('-inf'), None))
                    max_col = max(step.get('col_penalties', []), default=(float('-inf'), None))
                    best_side = 'row' if (max_row[0] if max_row[1] is not None else float('-inf')) >= (max_col[0] if max_col[1] is not None else float('-inf')) else 'col'
                    best_val = max(max_row[0] if max_row[1] is not None else float('-inf'), max_col[0] if max_col[1] is not None else float('-inf'))
                    if best_side == 'row' and max_row[1] is not None:
                        st.info(f"Максимальный штраф = {to_int_if_possible(best_val)} в строке S{max_row[1]+1} — выбираем строку.")
                    elif best_side == 'col' and max_col[1] is not None:
                        st.info(f"Максимальный штраф = {to_int_if_possible(best_val)} в столбце D{max_col[1]+1} — выбираем столбец.")

                    # Показать доступные стоимости в выбранной строке/столбце ДО распределения шага и пояснить выбор минимальной
                    if chosen['type'] == 'row':
                        i = chosen['index']
                        avail = [(j, cost_b[i][j]) for j in range(n_b) if pre_demand[j] > 0]
                        if avail:
                            j_min, c_min = min(avail, key=lambda t: t[1])
                            avail_df = pd.DataFrame([[to_int_if_possible(cost) for _, cost in avail]], columns=[f"D{j+1}" for j, _ in avail], index=[f"S{i+1}"])
                            st.caption("Доступные стоимости в выбранной строке (до распределения текущего шага):")
                            st.dataframe(avail_df, use_container_width=True)
                            st.markdown(f"Минимальная стоимость в строке S{i+1}: ячейка D{j_min+1} со стоимостью {to_int_if_possible(c_min)} — её и выбираем.")
                    else:
                        j = chosen['index']
                        avail = [(i, cost_b[i][j]) for i in range(m_b) if pre_supply[i] > 0]
                        if avail:
                            i_min, c_min = min(avail, key=lambda t: t[1])
                            avail_df = pd.DataFrame({f"D{j+1}": [to_int_if_possible(cost) for _, cost in avail]}, index=[f"S{i+1}" for i, _ in avail])
                            st.caption("Доступные стоимости в выбранном столбце (до распределения текущего шага):")
                            st.dataframe(avail_df, use_container_width=True)
                            st.markdown(f"Минимальная стоимость в столбце D{j+1}: ячейка S{i_min+1} со стоимостью {to_int_if_possible(c_min)} — её и выбираем.")

                    # Подсказка выбора
                    st.caption("Алгоритм Фогеля: выбираем строку/столбец с наибольшим штрафом, затем в нём — ячейку с минимальной стоимостью среди доступных.")
                    c_cols2 = [f"D{j+1} ({to_int_if_possible(demand_b[j])})" for j in range(n_b)]
                    c_idx2 = [f"S{i+1} ({to_int_if_possible(supply_b[i])})" for i in range(m_b)]
                    st.dataframe(pd.DataFrame(format_matrix(step['alloc']), columns=c_cols2, index=c_idx2), use_container_width=True)
                    st.caption(
                        f"Текущие запасы: {list(map(to_int_if_possible, step['supply']))}; Потребности: {list(map(to_int_if_possible, step['demand']))}"
                    )

                    # Обновляем pre_* к следующему шагу на основании распределения текущего шага
                    qty = chosen['qty']
                    if chosen['type'] == 'row':
                        i = chosen['index']
                        j = chosen['cell'][1]
                    else:
                        j = chosen['index']
                        i = chosen['cell'][0]
                    pre_supply[i] = max(0.0, pre_supply[i] - qty)
                    pre_demand[j] = max(0.0, pre_demand[j] - qty)

            # 2) Потенциалы — шаги
            final_alloc, pot_steps = improve_plan_with_steps(init_alloc, cost_b)
            with st.expander("Метод потенциалов — поэтапно", expanded=True):
                for t, pstep in enumerate(pot_steps, start=1):
                    st.markdown(f"Итерация {t}")
                    st.markdown(
                        f"u: {list(map(to_int_if_possible, pstep.get('u', [])))}; v: {list(map(to_int_if_possible, pstep.get('v', [])))}"
                    )
                    if 'deltas' in pstep:
                        st.markdown("Матрица оценок Δ = c - (u+v):")
                        st.dataframe(pd.DataFrame(format_matrix(pstep['deltas'])), use_container_width=True)
                    ent = pstep.get('entering')
                    if ent is None:
                        st.success("Оптимальность достигнута (все Δ ≥ 0)")
                    else:
                        cell = ent['cell']
                        st.markdown(f"Входящая ячейка: S{cell[0]+1}-D{cell[1]+1}, Δ = {to_int_if_possible(ent['delta'])}")
                        if pstep.get('cycle'):
                            st.markdown(f"Цикл: {[(i+1, j+1) for i,j in pstep['cycle']]}")
                            st.markdown(
                                f"Плюс: {[(i+1, j+1) for i,j in pstep.get('plus', [])]} | Минус: {[(i+1, j+1) for i,j in pstep.get('minus', [])]}"
                            )
                            if 'theta' in pstep:
                                st.markdown(f"Θ = {to_int_if_possible(pstep['theta'])}")
                        c_cols3 = [f"D{j+1} ({to_int_if_possible(demand_b[j])})" for j in range(n_b)]
                        c_idx3 = [f"S{i+1} ({to_int_if_possible(supply_b[i])})" for i in range(m_b)]
                        st.dataframe(pd.DataFrame(format_matrix(pstep['alloc']), columns=c_cols3, index=c_idx3), use_container_width=True)

            # 3) Итоговый план и значение цели
            st.subheader("Итоговый план")
            cols_b = [
                f"D{j+1} ({to_int_if_possible(demand_b[j])})" + (" (фикт.)" if j >= n else "")
                for j in range(n_b)
            ]
            idx_b = [
                f"S{i+1} ({to_int_if_possible(supply_b[i])})" + (" (фикт.)" if i >= m else "")
                for i in range(m_b)
            ]
            st.dataframe(pd.DataFrame(format_matrix(final_alloc), columns=cols_b, index=idx_b), use_container_width=True)

            value_eff = sum(final_alloc[i][j] * cost_b[i][j] for i in range(m_b) for j in range(n_b))
            if objective == "Максимум":
                st.metric("Z*", to_int_if_possible(-value_eff))
            else:
                st.metric("Z*", to_int_if_possible(value_eff))

            # Разложение цели: Z = sum c_ij * x_ij на ИСХОДНОЙ матрице стоимостей (без знака)
            try:
                terms_symbolic = []
                terms_numeric = []
                for i in range(m_b):
                    for j in range(n_b):
                        xij = final_alloc[i][j]
                        if abs(xij) > 1e-12:
                            cij = cost_disp[i][j]
                            cij_disp = to_int_if_possible(cij)
                            xij_disp = to_int_if_possible(xij)
                            terms_symbolic.append(fr"{cij_disp}\,x_{{S{i+1},D{j+1}}}")
                            terms_numeric.append(fr"{cij_disp}\cdot {xij_disp}")
                if terms_symbolic:
                    st.markdown("Разложение цели Z в виде суммы произведений стоимости и объёма поставки:")
                    st.latex("Z = " + " + ".join(terms_symbolic))
                    total_original = sum(final_alloc[i][j] * cost_disp[i][j] for i in range(m_b) for j in range(n_b))
                    st.latex("Z = " + " + ".join(terms_numeric) + f" = {to_int_if_possible(total_original)}")
                    if objective == "Максимум":
                        st.caption("Так как задача максимизации была сведена к минимизации по -C, здесь Z вычислен по исходным C.")
            except Exception:
                pass

        except Exception as e:
            st.error(f"Ошибка метода потенциалов: {e}")


if __name__ == "__main__":
    main()

