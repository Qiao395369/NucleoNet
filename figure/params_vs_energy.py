import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

HARTREE_TO_KCAL_MOL = 627.509474

selected_systems = [
"LiH",
r"Li$_2$",
r"NH$_3$",
r"CH$_4$",
"CO",
r"N$_2$",
"Ethene",
"Methylamine",
]

system_marker_dict = {
"LiH": "o",
r"Li$_2$": "s",
r"NH$_3$": "^",
r"CH$_4$": "v",
"CO": "D",
r"N$_2$": "p",
"Ethene": "h",
"Methylamine": "8",
}

model_color_dict = {
"FermiNet-Small": "#4477aa",
"FermiNet-Large": "#225577",
"Psiformer-Small": "#cc8833",
"Psiformer-Large": "#aa5522",
"LapNet": "#66aa55",
"NucleoNet": "crimson",
}

# =============================
# 1. baseline: CCSD(T) CBS
# =============================
baseline_energy = {
"LiH": -8.070696,
r"Li$_2$": -14.99507,
r"NH$_3$": -56.5644,
r"CH$_4$": -40.5150,
"CO": -113.3255,
r"N$_2$": -109.5425,
"Ethene": -78.5888,
"Methylamine": -95.8653,
}

# =============================
# 2.FermiNet  / Psiformer Small & Large, Lapnet
# =============================
records = [
# LiH
    ("LiH", "FermiNet-Small", -8.07050, 683744),
    ("LiH", "Psiformer-Small", -8.070528, 1593858),
    ("LiH", "LapNet", -8.070523, 1878018),
    ("LiH", "FermiNet-Large", -8.070515, 2610400),
    ("LiH", "Psiformer-Large", -8.070536, 6366210),
# Li2
    (r"Li$_2$", "FermiNet-Small", -14.99480, 700384),
    (r"Li$_2$", "Psiformer-Small", -14.99486, 1602178),
    (r"Li$_2$", "LapNet", -14.99485, 1894658),
    (r"Li$_2$", "FermiNet-Large", -14.99484, 2676448),
    (r"Li$_2$", "Psiformer-Large", -14.99485, 6399234),
# NH3
    (r"NH$_3$", "FermiNet-Small", -56.56347, 741088),
    (r"NH$_3$", "Psiformer-Small", -56.56367, 1621506),
    (r"NH$_3$", "LapNet", -56.56359, 1931266),
    (r"NH$_3$", "FermiNet-Large", -56.56387, 2823392),
    (r"NH$_3$", "Psiformer-Large", -56.56381, 6470658),
# CH4
    (r"CH$_4$", "FermiNet-Small", -40.51430, 744800),
    (r"CH$_4$", "Psiformer-Small", -40.51454, 1622850),
    (r"CH$_4$", "LapNet", -40.51445, 1932930),
    (r"CH$_4$", "FermiNet-Large", -40.51450, 2830816),
    (r"CH$_4$", "Psiformer-Large", -40.51461, 6473346),
# CO
    ("CO", "FermiNet-Small", -113.32354, 766944),
    ("CO", "Psiformer-Small", -113.32416, 1635458),
    ("CO", "LapNet", -113.32417, 1961218),
    ("CO", "FermiNet-Large", -113.32444, 2940640),
    ("CO", "Psiformer-Large", -113.32466, 6531330),
# N2
    (r"N$_2$", "FermiNet-Small", -109.54046, 766944),
    (r"N$_2$", "Psiformer-Small", -109.54137, 1635458),
    (r"N$_2$", "LapNet", -109.54128, 1961218),
    (r"N$_2$", "FermiNet-Large", -109.54148, 2940640),
    (r"N$_2$", "Psiformer-Large", -109.54179, 6531330),
# Ethene / C2H4
    ("Ethene", "FermiNet-Small", -78.58604, 799968),
    ("Ethene", "Psiformer-Small", -78.58762, 1649922),
    ("Ethene", "LapNet", -78.58721, 1986050),
    ("Ethene", "FermiNet-Large", -78.58701, 3039456),
    ("Ethene", "Psiformer-Large", -78.58794, 6576642),
# Methylamine
    ("Methylamine", "FermiNet-Small", -95.85917, 821344),
    ("Methylamine", "Psiformer-Small", -95.86050, 1660098),
    ("Methylamine", "LapNet", -95.86025, 2005378),
    ("Methylamine", "FermiNet-Large", -95.86010, 3114976),
    ("Methylamine", "Psiformer-Large", -95.86096, 6613378),
]

# =============================
# 3. NucleoNet
# =============================
extra_records = [
    ("LiH", "NucleoNet", -8.070544, 97776),
    (r"Li$_2$", "NucleoNet", -14.994913, 102000),
    (r"NH$_3$", "NucleoNet", -56.56389, 110448),
    (r"CH$_4$", "NucleoNet", -40.51471, 110448),
    ("CO", "NucleoNet", -113.32465, 118896),
    (r"N$_2$", "NucleoNet", -109.54195, 118896),
    ("Ethene", "NucleoNet", -78.58799, 123120),
    ("Methylamine", "NucleoNet", -95.86083, 127344),
]
records.extend(extra_records)


df = pd.DataFrame(records, columns=["system", "model", "energy_Eh", "params"], )
df = df[df["system"].isin(selected_systems)].copy()
df["params_M"] = (df["params"] / 1e6).round(3)
df["baseline_Eh"] = df["system"].map(baseline_energy)
df["error_kcal_mol"] = (np.abs(df["energy_Eh"] - df["baseline_Eh"]) * HARTREE_TO_KCAL_MOL)


brief_systems = sorted(df[df["model"] == "NucleoNet"]["system"].unique())
plot_df = df[df["system"].isin(brief_systems)].copy()
fig = plt.figure(figsize=(8.6, 8), dpi=250)
gs = gridspec.GridSpec(2, 1, height_ratios=[0.5, 7], hspace=0.1)
ax_legend = fig.add_subplot(gs[0])  
ax = fig.add_subplot(gs[1])     
ax_legend.axis("off")


for sys_name, sub_sys_df in plot_df.groupby("system"):
    mk = system_marker_dict[sys_name]
    for _, row in sub_sys_df.iterrows():
        ax.scatter(
            row["params_M"],
            row["error_kcal_mol"],
            s=115,
            marker=mk,
            color=model_color_dict[row["model"]],
            edgecolors="dimgray",
            linewidths=0.5,
            alpha=0.78,
            zorder=2,
        )


for system in brief_systems:
    sub_sys = plot_df[plot_df["system"] == system].copy()
    sub_sys_sorted = sub_sys.sort_values("params_M").reset_index(drop=True)
    for idx in range(len(sub_sys_sorted)-1):
        p1 = sub_sys_sorted.iloc[idx]
        p2 = sub_sys_sorted.iloc[idx+1]
        ax.plot(
            [p1["params_M"], p2["params_M"]],
            [p1["error_kcal_mol"], p2["error_kcal_mol"]],
            color="lightgray",
            linewidth=1,
            alpha=0.9,
            zorder=1,
        )


annotate_offset = {
"LiH": (-8, 0),
r"Li$_2$": (2, 6),
r"NH$_3$": (-6, 0),
r"CH$_4$": (-6, 1),
"CO": (-6, 1),
r"N$_2$": (-6, 1),
"Ethene": (-6, 0),
"Methylamine": (30,-8)
}

brief_df = plot_df[plot_df["model"] == "NucleoNet"].copy()
for _, row in brief_df.iterrows():
    dx, dy = annotate_offset[row["system"]]
    ax.annotate(
        row["system"],
        xy=(row["params_M"], row["error_kcal_mol"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=12,
        ha="right",
        va="bottom" if dy>0 else "top",
    )


ax.set_xscale("log")
ax.set_yscale("log")
custom_x_ticks = [0.1, 0.2, 0.7, 1, 2, 3, 7]
ax.set_xticks(custom_x_ticks,)
ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_xlabel("Number of parameters (M)", fontsize=18)
ax.set_ylabel("Energy error (kcal/mol)", fontsize=18,labelpad=-1)

x_min = plot_df["params_M"].min() * 0.75
x_max = plot_df["params_M"].max() * 1.6
y_min = plot_df["error_kcal_mol"].min() * 0.70
y_max = plot_df["error_kcal_mol"].max() * 1.2
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)


ax.axhline( 1.0, linestyle="--", linewidth=1.1, color="black", alpha=0.65, )
ax.text( x_min * 1.02, 1.05, "chemical accuracy: 1 kcal/mol", fontsize=13, va="bottom", ha="left", )
ax.grid(True, which="both", linestyle="--", alpha=0.25)


model_handles = []
for mname, c in model_color_dict.items():
    model_handles.append(Line2D([], [], marker="o", linestyle="", color=c, markersize=7, label=mname))

ax_legend.legend(handles=model_handles, loc="upper center", ncol=3,bbox_to_anchor=(0.5, 1.3), 
                        frameon=True, facecolor="white", edgecolor="gray", fontsize=16,borderpad=0.2,
                        columnspacing=1.0,  
                        handletextpad=0.1)


plt.savefig("./figure/params_vs_energy.png", dpi=250, bbox_inches='tight', facecolor="white")

