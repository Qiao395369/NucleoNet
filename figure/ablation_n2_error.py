import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
from tools.MLR import create_MLR_function
from matplotlib.lines import Line2D

phi4 = np.array([-2.34414547, -0.972469, -1.561777, -1.136, -1.3963, -0.819, -0.45, -3.36, 2.1])
MLR4 = create_MLR_function(phi=phi4,p=4,NS=6,NL=8)

length_list = np.linspace(0.8, 3.2, 32).round(6)
length_list = [i*2 for i in length_list]
energy_MLR = [MLR4(i) for i in length_list]

length_list_train = np.linspace(0.8, 3, 6).round(6)
length_list_train = [i*2 for i in length_list_train]
energy_MLR_train = [MLR4(i) for i in length_list_train]

with open("./data/N2/sparse_6_ablation/EB/statistics32.json", 'r') as file: 
    data = json.load(file)
    energy_eb_eval=data["average"]
diff_eb_eval = [(i-j)*1000 for (i,j) in zip(energy_eb_eval,energy_MLR)]

with open("./data/N2/sparse_6_ablation/EB/statistics6.json", 'r') as file:
    data = json.load(file)
    energy_eb_train=data["average"]
diff_eb_train = [(i-j)*1000 for (i,j) in zip(energy_eb_train,energy_MLR_train)]

with open("./data/N2/sparse_6_ablation/EB-NS/statistics32.json", 'r') as file: 
    data = json.load(file)
    energy_eb_ns_eval=data["average"]
diff_eb_ns_eval = [(i-j)*1000 for (i,j) in zip(energy_eb_ns_eval,energy_MLR)]

with open("./data/N2/sparse_6_ablation/EB-NS/statistics6.json", 'r') as file:  
    data = json.load(file)
    energy_eb_ns_train=data["average"]
diff_eb_ns_train = [(i-j)*1000 for (i,j) in zip(energy_eb_ns_train,energy_MLR_train)]

with open("./data/N2/sparse_6/statistics32.json", 'r') as file:
    data = json.load(file)
    energy_full_eval=data["average"]
diff_full_eval = [(i-j)*1000 for (i,j) in zip(energy_full_eval,energy_MLR)]

with open("./data/N2/sparse_6/statistics6.json", 'r') as file:
    data = json.load(file)
    energy_full_train=data["average"]
diff_full_train = [(i-j)*1000 for (i,j) in zip(energy_full_train,energy_MLR_train)]

with open("./data/N2/sparse_6_ablation/NS/statistics32.json", 'r') as file:   
    data= json.load(file)
    energy_ns_eval=data["average"]
diff_ns_eval = [(i-j)*1000 for (i,j) in zip(energy_ns_eval,energy_MLR)]

with open("./data/N2/sparse_6_ablation/NS/statistics6.json", 'r') as file:
    data= json.load(file)
    energy_ns_train=data["average"]
diff_ns_train = [(i-j)*1000 for (i,j) in zip(energy_ns_train,energy_MLR_train)]

fig = plt.figure(figsize=(8, 6),dpi=200)
ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])

mae_eb = np.mean(np.abs(np.asarray(diff_eb_eval)))
mae_eb_ns = np.mean(np.abs(np.asarray(diff_eb_ns_eval)))
mae_full = np.mean(np.abs(np.asarray(diff_full_eval)))
mae_ns = np.mean(np.abs(np.asarray(diff_ns_eval)))

ax.plot(
    length_list, diff_eb_ns_eval,
    linestyle="--", marker="o", markersize=3, c="brown", linewidth=1,
    label=f"NucleoNet-EB-NS, MAE={mae_eb_ns:.2f}",
)

ax.plot(
    length_list, diff_eb_eval,
    linestyle="--", marker="o", markersize=3, c="green", linewidth=1,
    label=f"NucleoNet-EB, MAE={mae_eb:.2f}",
)

ax.plot(
    length_list, diff_ns_eval,
    linestyle="--", marker="o", markersize=3, c="orange", linewidth=1,
    label=f"NucleoNet-NS, MAE={mae_ns:.2f}",
)

ax.plot(
    length_list, diff_full_eval,
    linestyle="--", marker="o", markersize=3, c="blue", linewidth=1,
    label=f"NucleoNet, MAE={mae_full:.2f}",
)


mae_eb_train = np.mean(np.abs(np.asarray(diff_eb_train)))
mae_full_train = np.mean(np.abs(np.asarray(diff_full_train)))
mae_ns_train = np.mean(np.abs(np.asarray(diff_ns_train)))
mae_eb_ns_train = np.mean(np.abs(np.asarray(diff_eb_ns_train)))

print("mae_eb_train :",mae_eb_train)
print("mae_full_train :",mae_full_train)
print("mae_ns_train :",mae_ns_train)
print("mae_eb_ns_train :",mae_eb_ns_train)

ax.scatter(length_list_train, diff_eb_train, marker="^", c="green",  s=90, zorder=5, label="__legend__")
ax.scatter(length_list_train, diff_ns_train, marker="^", c="orange", s=90, zorder=5, label="__legend__")
ax.scatter(length_list_train, diff_full_train, marker="^", c="blue",   s=90, zorder=5, label="__legend__")
ax.scatter(length_list_train, diff_eb_ns_train, marker="^", c="brown",  s=90, zorder=5, label="__legend__")

ax.axhline(0, color="black", linewidth=2)

training_handle = Line2D(
    [0], [0],
    marker="^", linestyle="None",
    markerfacecolor="0.35", markeredgecolor="0.35",
    markersize=8, label="Training geometries",
)
curve_handles, curve_labels = ax.get_legend_handles_labels()
ax.legend(
    curve_handles + [training_handle],
    curve_labels + ["Training geometries"],
    fontsize=14, loc="upper right", frameon=True, facecolor="white",
)


ax_inset = ax.inset_axes([0.06, 0.6, 0.33, 0.32])
ax_inset.plot(length_list, diff_eb_eval,  linestyle="--", c="green",  linewidth=0.9)
ax_inset.plot(length_list, diff_eb_ns_eval, linestyle="--", c="brown", linewidth=0.9)
ax_inset.plot(length_list, diff_ns_eval, linestyle="--", c="orange", linewidth=0.9)
ax_inset.plot(length_list, diff_full_eval, linestyle="--", c="blue",   linewidth=0.9)
ax_inset.axhline(0, color="black", linewidth=0.7)
ax_inset.set_title("Full range", fontsize=13, pad=2)
ax_inset.tick_params(axis="both", labelsize=11, direction="in")
ax_inset.grid(True, alpha=0.2)


ax.grid(True, alpha=0.3)
ax.set_xlabel(r'Bond length ($a_0$)', fontsize=18, labelpad=0)
ax.set_ylabel(r'Energy error (m$E_h$)', fontsize=18, labelpad=5, rotation=90)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

ax.set_xlim(1.2, 6.8)
ax.set_ylim(-0.5, 7.7)
# plt.tight_layout()

plt.savefig("./figure/ablation_n2_error.png",dpi=250)
