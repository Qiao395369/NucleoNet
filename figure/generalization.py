import json
import numpy as np
import matplotlib.pyplot as plt
from tools.MLR import create_MLR_function

c_infer="blue"  # 1. 深蓝色（经典科研主色）
c_dense="red"  # 2. 橙色（高对比强调色）
size=15
x=0.75
linewidth=1
markersize=2
fontsize=15


phi4 = np.array([-2.34414547, -0.972469, -1.561777, -1.136, -1.3963, -0.819, -0.45, -3.36, 2.1])
MLR4 = create_MLR_function(phi=phi4,p=4,NS=6,NL=8)


def load_average(path):
    with open(path, "r") as f:
        return np.asarray(json.load(f)["average"])


LiCN_displacement=[3.590479, 3.779452, 3.968425, 4.346370, 4.724315]
LiCN_displacement_train=[3.590479,4.724315]

LiCN_dense = load_average("./data/LiCN/multi_5/statistics.json")
LiCN_infer = load_average("./data/LiCN/sparse_2/statistics.json")
LiCN_train = [LiCN_infer[0]] + [LiCN_infer[-1]]
LiCN_baseline = []
LiCN_baseline += [load_average("./data/LiCN/single/geometry_01/statistics.json")]
LiCN_baseline += [load_average("./data/LiCN/single/geometry_02/statistics.json")]
LiCN_baseline += [load_average("./data/LiCN/single/geometry_03/statistics.json")]
LiCN_baseline += [load_average("./data/LiCN/single/geometry_04/statistics.json")]
LiCN_baseline += [load_average("./data/LiCN/single/geometry_05/statistics.json")]
LiCN_baseline = np.asarray(LiCN_baseline)
LiCN_index_train = [0,4]
diff_LiCN_train = [(i-j)*1000 for (i,j) in zip(LiCN_train,LiCN_baseline[LiCN_index_train])]
diff_LiCN_infer = [(i-j)*1000 for (i,j) in zip(LiCN_infer,LiCN_baseline)]
diff_LiCN_dense = [(i-j)*1000 for (i,j) in zip(LiCN_dense,LiCN_baseline)]


N2_train = load_average("./data/N2/sparse_6/statistics6.json")
N2_infer = load_average("./data/N2/sparse_6/statistics32.json")
N2_dense = load_average("./data/N2/multi_32/statistics.json")
length_list = np.linspace(0.8, 3.2, 32).round(6)
N2_x_dense = [i*2 for i in length_list]
N2_x_train = np.linspace(1.6, 6, 6)
N2_baseline_dense = [MLR4(i) for i in N2_x_dense]
N2_baseline_infer = [MLR4(i) for i in N2_x_dense]
N2_baseline_train = [MLR4(i) for i in N2_x_train]
diff_N2_train = [(i-j)*1000 for (i,j) in zip(N2_train,N2_baseline_train)]
diff_N2_infer = [(i-j)*1000 for (i,j) in zip(N2_infer,N2_baseline_infer)]
diff_N2_dense = [(i-j)*1000 for (i,j) in zip(N2_dense,N2_baseline_dense)]


diff_N2_train = np.asarray(diff_N2_train)
diff_N2_infer = np.asarray(diff_N2_infer)

# ============================================================
# Plot
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(
    2, 1,
    figsize=(7, 8),
    constrained_layout=False,
)
plt.subplots_adjust(hspace=0.4)
# ------------------------------------------------------------
# (a) N2
# ------------------------------------------------------------
ax = axes[0]
ax.scatter(
    N2_x_train,
    diff_N2_train,
    marker="^",
    s=55,
    zorder=5,
    c=c_infer,
    label="Training geometries",
)

ax.plot(
    N2_x_dense,
    diff_N2_infer,
    marker="o",
    markersize=markersize,
    linestyle="--",
    linewidth=linewidth,
    c=c_infer,
    label="Optimized on training geometries",
)

ax.plot(
    N2_x_dense,
    diff_N2_dense,
    marker="o",
    markersize=markersize,
    linestyle="--",
    linewidth=linewidth,
    c=c_dense,
    label="Optimized on all geometries",
)

ax.text(
    x,
    0.85,
    f"MAE={np.mean([abs(i) for i in diff_N2_infer]):.2f}",
    transform=ax.transAxes,
    fontsize=size,
    c=c_infer,
    va="top"
)

ax.text(
    x,
    0.97,
    f"MAE={np.mean([abs(i) for i in diff_N2_dense]):.2f}",
    transform=ax.transAxes,
    fontsize=size,
    c=c_dense,
    va="top"
)
ax.axhline(0, linewidth=0.8, linestyle="--")
ax.set_ylim(-1.1,4.8)
ax.grid(True, alpha=0.3)
ax.text(0.03, 0.94, r"(a) N$_2$", transform=ax.transAxes, va="top", fontsize=fontsize)
ax.set_xlabel(r"Bond length ($a_0$)",fontsize=fontsize)
ax.set_ylabel(r"$\Delta E$ (m$E_h$)",fontsize=fontsize)
ax.tick_params(direction="in")

# ------------------------------------------------------------
# (b) LiCN
# ------------------------------------------------------------
ax = axes[1]
ax.scatter(
    LiCN_displacement_train,
    diff_LiCN_train,
    marker="^",
    s=55,
    zorder=5,
    c=c_infer,
    label="Training geometries",
)

ax.plot(
    LiCN_displacement,
    diff_LiCN_infer,
    marker="o",
    markersize=markersize,
    linewidth=linewidth,
    linestyle="--",
    c=c_infer,
    label="Optimized on training geometries",
)

ax.text(
    x,
    0.85,
    f"MSD={np.mean(diff_LiCN_infer):.2f}",
    transform=ax.transAxes,
    fontsize=size,
    c=c_infer,
    va="top"
)

ax.plot(
    LiCN_displacement,
    diff_LiCN_dense,
    marker="o",
    markersize=markersize,
    linewidth=linewidth,
    linestyle="--",
    c=c_dense,
    label="Optimized on all geometries",
)
ax.text(
    x,
    0.97,
    f"MSD={np.mean(diff_LiCN_dense):.2f}",
    transform=ax.transAxes,
    fontsize=size,
    c=c_dense,
    va="top"
)
ax.axhline(0, linewidth=0.8, linestyle="--")
ax.set_ylim(-0.12,0.34)
ax.set_xlim(3.5,4.85)
ax.grid(True, alpha=0.3)
ax.text(0.03, 0.94, r"(b) LiCN", transform=ax.transAxes, va="top", fontsize=fontsize)
ax.set_xlabel(r"Li-C distance ($a_0$)",fontsize=fontsize)
ax.set_ylabel(r"$\Delta E$ (m$E_h$)",fontsize=fontsize)
ax.tick_params(direction="in")


handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=1,
    frameon=False,
    columnspacing=1.0,
    fontsize=fontsize,
)


plt.savefig(
    "./figure/generalization.png",
    dpi=400,
    bbox_inches="tight",
)
