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
# HTML/CSS table rendering helpers
TABLE_CSS = """
<style>
.tp-table { border-collapse: collapse; margin: 6px 0; font-family: Inter, system-ui, sans-serif; }
.tp-table th, .tp-table td { border: 1px solid #e0e0e0; padding: 6px 8px; text-align: center; vertical-align: middle; min-width: 64px; }
.tp-table th { background: #fafafa; font-weight: 600; }
.tp-basis { background: #fff7d6; } /* light yellow */
.tp-enter { box-shadow: inset 0 0 0 2px #7c3aed; } /* purple */
.tp-plus { box-shadow: inset 0 0 0 2px #10b981; } /* green */
.tp-minus { box-shadow: inset 0 0 0 2px #ef4444; } /* red */
.tp-delta-neg { color: #b91c1c; font-weight: 600; }
.tp-cell { position: relative; }
.tp-arrow { display: block; font-size: 14px; opacity: 0.85; }
.tp-small { font-size: 12px; color: #6b7280; }
.tp-val { font-weight: 600; }
.tp-legend { font-size: 13px; color: #374151; }
.tp-legend span { display: inline-block; margin-right: 12px; }
.tp-pill { display:inline-block; padding: 1px 6px; border-radius: 10px; font-size: 11px; border:1px solid #d1d5db; background:#f9fafb; }
</style>
"""

def basis_from_alloc(alloc: List[List[float]], eps: float = 1e-12):
    m, n = len(alloc), len(alloc[0])
    return {(i, j) for i in range(m) for j in range(n) if abs(alloc[i][j]) > eps}

def cycle_arrows(cycle: List[tuple]):
    # Returns mapping cell -> arrow indicating direction to next cell
    arrows = {}
    if not cycle:
        return arrows
    path = cycle[:]
    # close the loop for arrow continuity if needed
    if path[-1] != path[0]:
        path = path + [path[0]]
    for k in range(len(path) - 1):
        (i1, j1), (i2, j2) = path[k], path[k+1]
        if i1 == i2:
            arrows[(i1, j1)] = "→" if j2 > j1 else "←"
        elif j1 == j2:
            arrows[(i1, j1)] = "↓" if i2 > i1 else "↑"
    return arrows

def render_html_matrix(title: str,
                       M: List[List[float]],
                       row_labels: List[str],
                       col_labels: List[str],
                       basis: Optional[set] = None,
                       entering: Optional[tuple] = None,
                       plus_set: Optional[set] = None,
                       minus_set: Optional[set] = None,
                       arrows_map: Optional[dict] = None,
                       value_fmt: str = "{:.2f}"):
    m, n = (len(M), len(M[0])) if M else (0, 0)
    basis = basis or set()
    plus_set = plus_set or set()
    minus_set = minus_set or set()
    arrows_map = arrows_map or {}
    html = [TABLE_CSS]
    html.append(f"<div><b>{title}</b></div>")
    html.append("<table class='tp-table'>")
    # header
    html.append("<tr><th></th>" + "".join(f"<th>{cl}</th>" for cl in col_labels) + "</tr>")
    # rows
    for i in range(m):
        row = [f"<th>{row_labels[i]}</th>"]
        for j in range(n):
            classes = ["tp-cell"]
            if (i, j) in basis:
                classes.append("tp-basis")
            if entering and (i, j) == tuple(entering):
                classes.append("tp-enter")
            if (i, j) in plus_set:
                classes.append("tp-plus")
            if (i, j) in minus_set:
                classes.append("tp-minus")
            val = value_fmt.format(M[i][j]) if isinstance(M[i][j], (int, float)) else str(M[i][j])
            arrow = arrows_map.get((i, j), "")
            arrow_html = f"<span class='tp-arrow'>{arrow}</span>" if arrow else ""
            cell_html = f"<td class='{' '.join(classes)}'><div class='tp-val'>{val}</div>{arrow_html}</td>"
            row.append(cell_html)
        html.append("<tr>" + "".join(row) + "</tr>")
    html.append("</table>")
    return "\n".join(html)

def render_html_deltas(title: str,
                       D: List[List[float]],
                       row_labels: List[str],
                       col_labels: List[str],
                       entering: Optional[tuple] = None):
    m, n = (len(D), len(D[0])) if D else (0, 0)
    html = [TABLE_CSS]
    html.append(f"<div><b>{title}</b></div>")
    html.append("<table class='tp-table'>")
    html.append("<tr><th></th>" + "".join(f"<th>{cl}</th>" for cl in col_labels) + "</tr>")
    for i in range(m):
        row = [f"<th>{row_labels[i]}</th>"]
        for j in range(n):
            val = D[i][j]
            cls = "tp-delta-neg" if isinstance(val, (int, float)) and val < 0 else ""
            cls_enter = " tp-enter" if entering and (i, j) == tuple(entering) else ""
            txt = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
            row.append(f"<td class='{cls}{cls_enter}'>{txt}</td>")
        html.append("<tr>" + "".join(row) + "</tr>")
    html.append("</table>")
    return "\n".join(html)

st.set_page_config(page_title="Транспортная задача — СЗУ + потенциалы", layout="wide", page_icon="🚚")
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
            html_cost = render_html_matrix(
                title="",
                M=cost_b,
                row_labels=row_labels,
                col_labels=col_labels,
                basis=set(),
                value_fmt="{:.2f}",
            )
            st.markdown(html_cost, unsafe_allow_html=True)

            st.subheader("Шаг 1. Метод северо-западного угла")
            alloc0, nw_logs = north_west_corner_with_steps(supply_b, demand_b)
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Начальный план (СЗУ)**")
                html_alloc0 = render_html_matrix(
                    title="",
                    M=alloc0,
                    row_labels=row_labels,
                    col_labels=col_labels,
                    basis=basis_from_alloc(alloc0),
                )
                st.markdown(html_alloc0, unsafe_allow_html=True)
                st.markdown("<div class='tp-legend'><span class='tp-pill'>Базисные</span></div>", unsafe_allow_html=True)
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

                    # Render deltas with entering highlight
                    html_delta = render_html_deltas(
                        title="Приведённые стоимости Δ = C - (u+v)",
                        D=step["reduced_costs"],
                        row_labels=row_labels,
                        col_labels=col_labels,
                        entering=step.get("entering"),
                    )
                    st.markdown(html_delta, unsafe_allow_html=True)
                    st.markdown(f"- **Входящая клетка**: {step['entering']}")
                    st.markdown(f"- **Минимальная Δ**: {step['min_delta']}")

                    # Cycle visualization on matrices
                    cycle = step.get("cycle") or []
                    plus = set(cycle[::2]) if cycle else set()
                    minus = set(cycle[1::2]) if cycle else set()
                    arrows = cycle_arrows(cycle)

                    if step.get("alloc_before"):
                        st.markdown("**План до**")
                        html_before = render_html_matrix(
                            title="",
                            M=step["alloc_before"],
                            row_labels=row_labels,
                            col_labels=col_labels,
                            basis=basis_from_alloc(step["alloc_before"]),
                            entering=step.get("entering"),
                            plus_set=plus,
                            minus_set=minus,
                            arrows_map=arrows,
                        )
                        st.markdown(html_before, unsafe_allow_html=True)

                    st.markdown(f"- **Цикл**: {cycle}")
                    st.markdown(f"- **Θ (минимум по минус-позициям)**: {step['theta']}")

                    if step.get("alloc_after"):
                        st.markdown("**План после**")
                        html_after = render_html_matrix(
                            title="",
                            M=step["alloc_after"],
                            row_labels=row_labels,
                            col_labels=col_labels,
                            basis=basis_from_alloc(step["alloc_after"]),
                        )
                        st.markdown(html_after, unsafe_allow_html=True)

                    st.markdown("<div class='tp-legend'><span class='tp-pill'>Базисные</span><span class='tp-pill'>Входящая</span><span class='tp-pill'>Знак цикла: + / -</span><span class='tp-pill'>Стрелки — ход цикла</span></div>", unsafe_allow_html=True)
                    if step.get("note"):
                        st.info(step["note"])

            # Cut back to original size if was balanced with dummy
            alloc_opt = [row[: len(demand)] for row in alloc_opt_b[: len(supply)]]

            st.subheader("Результат")
            st.markdown("**Оптимальный план**")
            html_final = render_html_matrix(
                title="",
                M=alloc_opt,
                row_labels=[f"r{i+1}" for i in range(len(alloc_opt))],
                col_labels=[f"c{j+1}" for j in range(len(alloc_opt[0]))],
                basis=basis_from_alloc(alloc_opt),
            )
            st.markdown(html_final, unsafe_allow_html=True)

            # LaTeX: S = sum x_ij * c_ij = ... = value
            S = total_cost(alloc_opt, cost)
            terms = []
            for i in range(len(alloc_opt)):
                for j in range(len(alloc_opt[0])):
                    x = alloc_opt[i][j]
                    if abs(x) > 1e-12:
                        terms.append(f"x_{{{i+1}{j+1}}} \\cdot c_{{{i+1}{j+1}}}")
            terms_str = " + ".join(terms) if terms else "0"
            # numeric breakdown
            nterms = []
            for i in range(len(alloc_opt)):
                for j in range(len(alloc_opt[0])):
                    x = alloc_opt[i][j]
                    if abs(x) > 1e-12:
                        nterms.append(f"{x:g} \\cdot {cost[i][j]:g}")
            nterms_str = " + ".join(nterms) if nterms else "0"
            st.latex(rf"S = \sum_{{i,j}} x_{{ij}} c_{{ij}} = {terms_str} = {nterms_str} = {S:g}")

    except Exception as e:
        st.exception(e)

