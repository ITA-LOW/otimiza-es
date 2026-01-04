import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from multi_objetivo.cabling_v3 import analisar_layout_completo

# ============================
# CONFIGURAÇÕES DE ESTILO (PAPER)
# ============================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
})

# ============================
# FUNÇÕES AUXILIARES
# ============================

def load_solution(fname):
    with open(fname) as f:
        lines = f.readlines()
    xc = np.array(eval(lines[0].split(":")[1]))
    yc = np.array(eval(lines[1].split(":")[1]))
    return np.column_stack([xc, yc])


def knee_point(df):
    """Retorna índice do joelho (normalização + distância ao ideal)."""
    cost = df["Custo_USD"].values
    aep = df["AEP_Liquido_MWh"].values

    cost_n = (cost - cost.min()) / (cost.max() - cost.min())
    aep_n = (aep.max() - aep) / (aep.max() - aep.min())

    dist = np.sqrt(cost_n**2 + aep_n**2)
    return np.argmin(dist)


def plot_layout(ax, coords, title):
    SUB = np.array([[-1350, 0]])
    sub_idx = np.argmin(np.linalg.norm(coords - SUB, axis=1))

    plant, _ = analisar_layout_completo(coords, sub_idx)

    cmap = plt.colormaps["tab10"]
    for i, path in enumerate(plant.paths):
        x = coords[path, 0]
        y = coords[path, 1]
        ax.plot(x, y, "-o", lw=1.5, ms=4, color=cmap(i))

    ax.scatter(
        coords[sub_idx, 0], coords[sub_idx, 1],
        marker="*", s=120, c="gold", edgecolor="black", zorder=5
    )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])

# ============================
# LOAD DOS RESULTADOS
# ============================

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "pareto_front_results", "pareto_summary.csv")

df = pd.read_csv(csv_path)

idx_min_cost = df["Custo_USD"].idxmin()
idx_max_aep = df["AEP_Liquido_MWh"].idxmax()
idx_knee = knee_point(df)

solutions = {
    "Min Cost": df.loc[idx_min_cost],
    "Max AEP": df.loc[idx_max_aep],
    "Knee Point": df.loc[idx_knee],
}

# Load coordinates for each solution
coords_data = {}
for name, sol in solutions.items():
    file_path = sol["File"]
    # Handle relative paths
    if not os.path.isabs(file_path):
        file_path = os.path.join(script_dir, file_path)
    coords_data[name] = load_solution(file_path)

# ============================
# FIGURA FINAL (2x3 layout: 1 row for Pareto, 3 columns for layouts)
# ============================

fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.2], hspace=0.3, wspace=0.3)

# --- (a) Pareto Front ---
ax0 = fig.add_subplot(gs[0, :])

ax0.scatter(
    df["Custo_USD"] / 1e6,
    df["AEP_Liquido_MWh"],
    s=12,
    alpha=0.5,
    color="navy",
    label="Pareto Solutions"
)

ax0.scatter(
    df.loc[idx_min_cost, "Custo_USD"] / 1e6,
    df.loc[idx_min_cost, "AEP_Liquido_MWh"],
    c="red", s=60, label="Min Cost"
)
ax0.scatter(
    df.loc[idx_max_aep, "Custo_USD"] / 1e6,
    df.loc[idx_max_aep, "AEP_Liquido_MWh"],
    c="green", s=60, label="Max AEP"
)
ax0.scatter(
    df.loc[idx_knee, "Custo_USD"] / 1e6,
    df.loc[idx_knee, "AEP_Liquido_MWh"],
    c="orange", s=60, label="Knee Point"
)

ax0.set_xlabel("Total Cabling Cost (Million USD)")
ax0.set_ylabel("Net AEP (MWh/year)")
ax0.set_title("(a) Pareto Front: Net AEP vs. Cabling Cost")
ax0.grid(True, linestyle="--", alpha=0.4)
ax0.legend(frameon=False)

# --- (b)(c)(d) Layouts ---
titles = [
    "(b) Minimum Cost Solution",
    "(c) Maximum AEP Solution",
    "(d) Knee-Point Solution"
]

axes = [
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[1, 2])
]

for ax, (name, coords), title in zip(axes, list(coords_data.items()), titles):
    plot_layout(ax, coords, title)

plt.tight_layout()
output_path = os.path.join(script_dir, "pareto_and_layouts_top_tier.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to: {output_path}")
plt.show()
