import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import json


with open("./data/LiCN/multi_5/statistics.json", 'r') as file:
    data = json.load(file)
    energy=data["average"]

sci_24color = [
    "#1f77b4",  # 1. 深蓝色（经典科研主色）
    "#ff7f0e",  # 2. 橙色（高对比强调色）
    "#2ca02c",  # 3. 深绿色（避免与红色冲突）
    "#d62728",  # 4. 深红色（醒目但不刺眼）
    "#9467bd",  # 5. 深紫色（辅助区分）
    "#8c564b",  # 6. 棕褐色（低饱和辅助色）
    "#e377c2",  # 7. 浅粉色（柔和区分色）
    "#7f7f7f",  # 8. 中灰色（中性辅助色）
    "#bcbd22",  # 9. 橄榄绿（低冲突辅助色）
    "#17becf",  # 10. 青蓝色（冷色调补充）
    "#8c6d31",  # 11. 黄棕色（原新增）
    "#c5b0d5",  # 12. 淡紫色（原新增）
    "#393b79",  # 13. 暗蓝紫（冷色延伸，区分主蓝色）
    "#ff9896",  # 14. 浅珊瑚红（低饱和红，区分主红色）
    "#637939",  # 15. 暗橄榄绿（区分主绿色）
    "#e7ba52",  # 16. 暖金色（低饱和黄，避免刺眼）
    "#ad494a",  # 17. 砖红色（深红延伸，低冲突）
    "#843c39",  # 18. 暗棕红（棕色系补充）
    "#ffffcc",  # 19. 米黄色（中性浅色调，低冲突背景/辅助色）
    "#9edae5",  # 20. 淡青色（冷色补充，柔和）
    "#7b4173",  # 21. 暗紫红（紫色系延伸，高区分度）
    "#fdbb84",  # 22. 浅橙棕色（橙色系柔和延伸，区分主橙色）
    "#31a354",  # 23. 翠绿色（绿色系延伸，区分深绿/橄榄绿）
    "#6baed6",  # 24. 天蓝色（蓝色系浅延伸，区分主深蓝/青蓝）
]

orb = [
-100.37641613260905,
-100.37584033203125,
-100.37229929199219,
-100.36224205423991,
-100.35008580265944,
]

ref=[
-100.3792953491211,
-100.37855529785156,
-100.37504577636719,
-100.36514282226562,
-100.35298156738281
]

LiCN_displacement=[
3.590479,
3.779452,
3.968425,
4.346370,
4.724315]

energy_single=[]
with open("./data/LiCN/single/geometry_01/statistics.json", 'r') as file:
    data = json.load(file)
    energy_single+=data["average"]
with open("./data/LiCN/single/geometry_02/statistics.json", 'r') as file:
    data = json.load(file)
    energy_single+=data["average"]
with open("./data/LiCN/single/geometry_03/statistics.json", 'r') as file:
    data = json.load(file)
    energy_single+=data["average"]
with open("./data/LiCN/single/geometry_04/statistics.json", 'r') as file:
    data = json.load(file)
    energy_single+=data["average"]
with open("./data/LiCN/single/geometry_05/statistics.json", 'r') as file:
    data = json.load(file)
    energy_single+=data["average"]

diff_orb=[(i-j)*1000 for i,j in zip(orb,energy_single)]
diff_nu=[(i-j)*1000 for i,j in zip(energy,energy_single)]
diff_ccsd=[(i-j)*1000 for i,j in zip(ref,energy_single)]

fig, ax = plt.subplots(figsize=(8, 5),)

ax.plot(LiCN_displacement, diff_orb, color="#174F57", marker="o", markersize=7, linewidth=1.2, linestyle="--" ,label="Orbformer, MSD=%.2f"%np.mean(diff_orb))
ax.plot(LiCN_displacement, diff_ccsd, marker="D", markersize=6, linewidth=1.2, linestyle="--" ,label="CCSD(T)/CBS, MSD=%.2f"%np.mean(diff_ccsd))
ax.plot(LiCN_displacement, diff_nu, linestyle="--", marker="^",markersize=7,c="red", linewidth=1.2, label="NucleoNet, MSD=%.2f"%np.mean(diff_nu))


ax.grid(True, alpha=0.3)
ax.axhline(0, color="black", linewidth=2, alpha=1)


ax.set_xlabel(r"Li-C distance ($a_0$)", fontsize=16, labelpad=5)
ax.set_ylabel(r'$\Delta E$ relative to ref (m$E_h$)', fontsize=16, labelpad=5, rotation=90)

ax.set_xlim(3.5,4.85)
ax.set_ylim(-0.3, 4.3) 

xtick_pos = np.arange(1, 6, 1)


ax.tick_params(axis="both", labelsize=15, length=4, width=0.8)
ax.legend(
    fontsize=17,
    loc="upper left",
    bbox_to_anchor=(0.42, 0.8)   
)


plt.tight_layout()

plt.savefig(
    "./figure/licn.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05
)
