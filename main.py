import streamlit as st
from typing import List
import pandas as pd

from solver import solve_transportation


def parse_matrix(inputs: List[List[float]]) -> List[List[float]]:
    return [[float(x) for x in row] for row in inputs]


def main():
    st.set_page_config(page_title="Решение транспортной задачи", layout="wide")
    st.title("Решение транспортной задачи")

    st.markdown("Введите размеры задачи, затем заполните матрицу стоимостей и векторы запасов и потребностей. Каждое значение вводится в отдельном поле.")

    col1, col2 = st.columns([1, 2])
    with col1:
        m = st.number_input("Число поставщиков (m)", min_value=1, max_value=12, value=3, step=1, key="m")
        n = st.number_input("Число потребителей (n)", min_value=1, max_value=12, value=4, step=1, key="n")
        st.write("")
        if st.button("Заполнить пример"):
            st.session_state['fill_sample'] = True
    with col2:
        st.write(" ")

    m = int(m)
    n = int(n)

    st.subheader("Матрица стоимостей (каждое значение — отдельное поле)")
    costs_inputs: List[List[float]] = []
    # строки
    for i in range(m):
        cols = st.columns(n)
        row_vals = []
        for j in range(n):
            key = f"cost_{i}_{j}"
            default = 0.0
            if st.session_state.get('fill_sample'):
                sample = [
                    [19, 30, 50, 10],
                    [70, 30, 40, 60],
                    [40, 8, 70, 20],
                ]
                if i < len(sample) and j < len(sample[0]):
                    default = float(sample[i][j])
            row_vals.append(cols[j].number_input(f"c[{i},{j}]", value=float(default), key=key, format="%.6f"))
        costs_inputs.append(row_vals)

    st.subheader("Запасы поставщиков (по одному полю на поставщика)")
    supply_cols = st.columns(m)
    supply_inputs: List[float] = []
    for i in range(m):
        key = f"s_{i}"
        default = 0.0
        if st.session_state.get('fill_sample'):
            sample_supply = [7, 9, 18]
            if i < len(sample_supply):
                default = float(sample_supply[i])
        supply_inputs.append(supply_cols[i].number_input(f"s[{i}]", value=float(default), key=key, format="%.6f"))

    st.subheader("Потребности потребителей (по одному полю на потребителя)")
    demand_cols = st.columns(n)
    demand_inputs: List[float] = []
    for j in range(n):
        key = f"d_{j}"
        default = 0.0
        if st.session_state.get('fill_sample'):
            sample_demand = [5, 8, 7, 14]
            if j < len(sample_demand):
                default = float(sample_demand[j])
        demand_inputs.append(demand_cols[j].number_input(f"d[{j}]", value=float(default), key=key, format="%.6f"))

    st.write("")
    solve = st.button("Решить")

    if solve:
        errors = []
        try:
            costs = parse_matrix(costs_inputs)
        except Exception:
            errors.append("Матрица стоимостей содержит некорректные значения.")

        supply = []
        demand = []
        try:
            supply = [float(x) for x in supply_inputs]
        except Exception:
            errors.append("Вектор запасов содержит некорректные значения.")
        try:
            demand = [float(x) for x in demand_inputs]
        except Exception:
            errors.append("Вектор потребностей содержит некорректные значения.")

        if any(x < 0 for x in supply):
            errors.append("Все значения запасов должны быть неотрицательны.")
        if any(x < 0 for x in demand):
            errors.append("Все значения потребностей должны быть неотрицательны.")

        if len(costs) != m or any(len(r) != n for r in costs):
            errors.append("Размер матрицы стоимостей не соответствует m x n.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            try:
                with st.spinner('Вычисление...'):
                    alloc, total, history, info = solve_transportation(costs, supply, demand)
            except Exception as exc:
                st.exception(exc)
                return

            st.success(f"Решено. Общая стоимость = {total:.6f}")

            df = pd.DataFrame(alloc)
            if info.get('dummy') is not None:
                typ, idx = info['dummy']
                st.info(f"Внимание: добавлен фиктивный { 'поставщик' if typ=='row' else 'потребитель' } в позицию {idx} для балансировки.")

            st.subheader("Итоговое распределение (строки = поставщики, столбцы = потребители)")
            st.dataframe(df.fillna(0))

            st.write("")
            st.subheader('История итераций')
            for k, step in enumerate(history):
                with st.expander(f"Шаг {k+1}: {step.get('step', '')}"):
                    alloc_step = step.get('allocation')
                    if alloc_step is not None:
                        st.table(pd.DataFrame(alloc_step).fillna(0))
                    if step.get('u') is not None or step.get('v') is not None:
                        st.write('u =', step.get('u'))
                        st.write('v =', step.get('v'))
                    if step.get('reduced_costs') is not None:
                        st.write('Редуцированные стоимости:')
                        st.table(pd.DataFrame(step.get('reduced_costs')))


if __name__ == '__main__':
    main()
