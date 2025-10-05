import streamlit as st
from typing import List, Tuple, Optional
import pandas as pd

# Use high-level step-by-step functions from the solver
from solver.transport_potential import (
    balance_transportation,
    north_west_corner_with_steps,
    improve_plan_with_steps,
)


# =====================
# Parsing helpers
# =====================
def parse_vector(text: str) -> List[float]:
    """Parse a comma/space/semicolon separated vector from user input."""
    text = (text or "").strip()
    if not text:
        return []
    # Replace various separators with spaces
    for sep in [",", ";", "\n", "\t"]:
        text = text.replace(sep, " ")
    parts = [p for p in text.split(" ") if p]
    return [float(p) for p in parts]


def parse_matrix(text: str) -> List[List[float]]:
    """Parse a matrix given as rows on separate lines, elements separated by space/comma/semicolon."""
    text = (text or "").strip()
    if not text:
        return []
    rows = [r for r in text.splitlines() if r.strip()]
    matrix: List[List[float]] = []
    for r in rows:
        row = r.strip()
        for sep in [",", ";", "\t"]:
            row = row.replace(sep, " ")
        parts = [p for p in row.split(" ") if p]
        matrix.append([float(p) for p in parts])
    # Validate rectangular
    if len({len(r) for r in matrix}) > 1:
        raise ValueError("Матрица стоимостей должна быть прямоугольной (одинаковая длина строк).")
    return matrix


def format_matrix(M: List[List[float]]) -> List[List[float]]:
    return [[float(x) for x in row] for row in M]


# =====================
# UI input helpers (center, per-cell)
# =====================
def example_data(m: int, n: int):
    ex_supply = [20, 30, 25]
    ex_demand = [10, 10, 35, 20]
    ex_cost = [
        [8, 6, 10, 9],
        [9, 12, 13, 7],
        [14, 9, 16, 5],
    ]
    # Fill defaults up to requested size when possible, otherwise zeros
    supply = [ex_supply[i] if i < len(ex_supply) else 0 for i in range(m)]
    demand = [ex_demand[j] if j < len(ex_demand) else 0 for j in range(n)]
    cost = [
        [ex_cost[i][j] if i < len(ex_cost) and j < len(ex_cost[0]) else 0 for j in range(n)]
        for i in range(m)
    ]
    return supply, demand, cost


def input_vector_grid(title: str, length: int, key_prefix: str, defaults: List[float]) -> List[float]:
    st.markdown(f"**{title}**")
    cols = st.columns(length)
    values: List[float] = []
    for j in range(length):
        key = f"{key_prefix}_{j}"
        val_str = cols[j].text_input(f"", value=str(defaults[j]), key=key, label_visibility="collapsed")
        try:
            values.append(float(val_str))
        except ValueError:
            values.append(0.0)
    return values


def input_matrix_grid(title: str, m: int, n: int, key_prefix: str, defaults: List[List[float]]) -> List[List[float]]:
    st.markdown(f"**{title} (размер {m}×{n})**")
    values: List[List[float]] = []
    # header
    header_cols = st.columns(n)
    for j in range(n):
        header_cols[j].markdown(f"<div style='text-align:center'><b>c{j+1}</b></div>", unsafe_allow_html=True)
    # rows
    for i in range(m):
        row_cols = st.columns(n)
        row_vals: List[float] = []
        for j in range(n):
            key = f"{key_prefix}_{i}_{j}"
            val_str = row_cols[j].text_input(
                "",
                value=str(defaults[i][j]),
                key=key,
                label_visibility="collapsed",
            )
            try:
                row_vals.append(float(val_str))
            except ValueError:
                row_vals.append(0.0)
        values.append(row_vals)
    return values


# =====================
# North-West Corner with steps
# =====================
##
# North-West corner with steps is now provided by solver.north_west_corner_with_steps
##


# =====================
# MODI (Potentials) with steps
# =====================
##
# Reduced costs and MODI steps are handled inside solver.improve_plan_with_steps
##


##
# MODI with steps is now provided by solver.improve_plan_with_steps
##


def total_cost(alloc: List[List[float]], cost: List[List[float]]) -> float:
    m, n = len(alloc), len(alloc[0])
    return sum(alloc[i][j] * cost[i][j] for i in range(m) for j in range(n))


# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="Транспортная задача — СЗУ + потенциалы", layout="wide")
st.title("Транспортная задача: метод северо-западного угла и метод потенциалов (шаг за шагом)")

st.subheader("Ввод параметров")
colA, colB = st.columns(2)
with colA:
    m = st.number_input("Количество поставщиков (m)", min_value=1, max_value=15, value=3, step=1)
with colB:
    n = st.number_input("Количество потребителей (n)", min_value=1, max_value=15, value=4, step=1)

def_sup, def_dem, def_cost = example_data(m, n)

st.markdown("---")
st.subheader("Заполнение данных (каждый элемент — отдельное поле)")
st.markdown("Сначала заполните матрицу стоимостей C, затем векторы поставки и спроса.")

cost = input_matrix_grid("Матрица стоимостей C", m, n, key_prefix="cost", defaults=def_cost)
supply = input_vector_grid("Поставка (длина m)", m, key_prefix="supply", defaults=def_sup)
demand = input_vector_grid("Спрос (длина n)", n, key_prefix="demand", defaults=def_dem)

run = st.button("Решить задачу", type="primary")


def render_matrix(name: str, M: List[List[float]]):
    st.markdown(f"**{name}**")
    if not M:
        st.info("Пусто")
        return
    st.dataframe({f"c{j+1}": [row[j] for row in M] for j in range(len(M[0]))})


def render_vector(name: str, v: List[float]):
    st.markdown(f"**{name}**: {v}")


if run:
    try:
        # Validate sizes
        if len(cost) != len(supply) or (len(cost) > 0 and len(cost[0]) != len(demand)):
            st.error("Размеры не согласованы: матрица должна быть размера m×n, где m=|поставка|, n=|спрос|.")
        else:
            st.subheader("Шаг 0. Балансировка задачи")
            st.write("Если сумма поставки не равна сумме спроса — добавляется фиктивный столбец/строка с нулевой стоимостью.")
            supply_b, demand_b, cost_b = balance_transportation(supply, demand, cost)

            c_sup, c_dem = sum(supply), sum(demand)
            c_sup_b, c_dem_b = sum(supply_b), sum(demand_b)
            st.write(f"Исходные суммы: Σпоставка={c_sup}, Σспрос={c_dem}")
            st.write(f"После балансировки: Σпоставка={c_sup_b}, Σспрос={c_dem_b}")

            # Determine dummy row/column
            dummy_row_idx: Optional[int] = None
            dummy_col_idx: Optional[int] = None
            note = ""
            if len(supply_b) > len(supply):
                dummy_row_idx = len(supply_b) - 1
                note = f"Добавлена фиктивная строка r{dummy_row_idx+1}."
            if len(demand_b) > len(demand):
                dummy_col_idx = len(demand_b) - 1
                note = f"Добавлен фиктивный столбец c{dummy_col_idx+1}." if not note else note + " Также добавлен фиктивный столбец c" + str(dummy_col_idx+1) + "."
            if note:
                st.info(note)

            st.markdown("**Матрица стоимостей (с пометкой фиктивных):**")
            # Labeled rendering
            row_labels = [f"r{i+1}" + (" (фиктивная)" if dummy_row_idx is not None and i == dummy_row_idx else "") for i in range(len(cost_b))]
            col_labels = [f"c{j+1}" + (" (фиктивный)" if dummy_col_idx is not None and j == dummy_col_idx else "") for j in range(len(cost_b[0]))]
            df_cost = pd.DataFrame(cost_b, index=row_labels, columns=col_labels)
            st.dataframe(df_cost)

            st.subheader("Шаг 1. Метод северо-западного угла")
            alloc0, nw_logs = north_west_corner_with_steps(supply_b, demand_b)
            cols = st.columns(2)
            with cols[0]:
                # Render with labels (include dummy marks)
                df_alloc0 = pd.DataFrame(alloc0, index=row_labels, columns=col_labels)
                st.markdown("**Начальный план (СЗУ)**")
                st.dataframe(df_alloc0)
            with cols[1]:
                st.markdown("**Пояснение шагов:**")
                for entry in nw_logs:
                    st.code(entry)

            st.subheader("Шаг 2. Метод потенциалов (MODI)")
            alloc_opt_b, modi_steps = improve_plan_with_steps(alloc0, cost_b)

            for step in modi_steps:
                with st.expander(f"Итерация {step['iteration']}"):
                    st.markdown(f"- **Потенциалы u**: {step['u']}")
                    st.markdown(f"- **Потенциалы v**: {step['v']}")
                    df_delta = pd.DataFrame(step["reduced_costs"], index=row_labels, columns=col_labels)
                    st.markdown("**Приведённые стоимости Δ = C - (u+v)**")
                    st.dataframe(df_delta)
                    st.markdown(f"- **Входящая клетка**: {step['entering']}")
                    st.markdown(f"- **Минимальная Δ**: {step['min_delta']}")
                    st.markdown(f"- **Цикл**: {step['cycle']}")
                    st.markdown(f"- **Θ (минимум по минус-позициям)**: {step['theta']}")
                    if step.get("alloc_before"):
                        df_before = pd.DataFrame(step["alloc_before"], index=row_labels, columns=col_labels)
                        st.markdown("**План до**")
                        st.dataframe(df_before)
                    if step.get("alloc_after"):
                        df_after = pd.DataFrame(step["alloc_after"], index=row_labels, columns=col_labels)
                        st.markdown("**План после**")
                        st.dataframe(df_after)
                    if step.get("note"):
                        st.info(step["note"])

            # Cut back to original size if was balanced with dummy
            alloc_opt = [row[: len(demand)] for row in alloc_opt_b[: len(supply)]]

            st.subheader("Результат")
            df_final = pd.DataFrame(alloc_opt, index=[f"r{i+1}" for i in range(len(alloc_opt))], columns=[f"c{j+1}" for j in range(len(alloc_opt[0]))])
            st.markdown("**Оптимальный план**")
            st.dataframe(df_final)
            st.markdown(f"**Оптимальные затраты:** {total_cost(alloc_opt, cost)}")

    except Exception as e:
        st.exception(e)

