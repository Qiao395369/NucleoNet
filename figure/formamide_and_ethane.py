import numpy as np
import matplotlib.pyplot as plt
import json
import os
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


top_image_path = "./figure/MEP_path.png" 
crop_top_image = True
top_gap_inch = 0.7
label=30
hartree_to_kcal = 627.509

color_ft = "#174F57"
color_gq = "#d62728"

plt.rcParams.update({
    "font.size": 16,
    "axes.linewidth": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# =====================================
# 顶部图片自动裁白边
# =====================================
def crop_white_margin(img, threshold=0.985, pad=2):
    arr = np.asarray(img)

    if arr.dtype.kind in "ui":
        rgb = arr[..., :3].astype(float) / np.iinfo(arr.dtype).max
    else:
        rgb = arr[..., :3].astype(float)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0

    nonwhite = np.any(rgb < threshold, axis=-1)

    if arr.ndim == 3 and arr.shape[-1] == 4:
        alpha = arr[..., 3].astype(float)
        if alpha.max() > 1.0:
            alpha = alpha / 255.0
        nonwhite &= alpha > 0.01

    rows = np.where(nonwhite.any(axis=1))[0]
    cols = np.where(nonwhite.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return img

    r0 = max(rows[0] - pad, 0)
    r1 = min(rows[-1] + pad + 1, arr.shape[0])
    c0 = max(cols[0] - pad, 0)
    c1 = min(cols[-1] + pad + 1, arr.shape[1])

    return arr[r0:r1, c0:c1]


# ===================== 分子贴图函 =====================
def annotate_mol_pic(ax, x_pt, y_pt, img_path, zoom=0.12, xy_offset=(0, 28)):
    try:
        img_arr = mpimg.imread(img_path)
        img_box = OffsetImage(img_arr, zoom=zoom)
        img_box.image.axes = ax
        anno_box = AnnotationBbox(
            img_box,
            xy=(x_pt, y_pt),
            xybox=xy_offset,
            xycoords="data",
            boxcoords="offset points",
            frameon=False,
            pad=0.0,
            arrowprops=dict(
                arrowstyle="<-",
                color="black",
                linewidth=1.2,
                shrinkA=8,
                shrinkB=6,
                mutation_scale=4
            )
        )
        anno_box.zorder = 2
        ax.add_artist(anno_box)
    except FileNotFoundError:
        print(f"[提示] 图片缺失跳过：{img_path}")


def load_case(filename, ref, orb_ft):
    with open(filename, "r") as file:
        data = json.load(file)
        energy = data["average"]

    energy = np.array(energy, dtype=float)
    ref = np.array(ref, dtype=float)
    ft = np.array(orb_ft, dtype=float)

    energy_kcal = energy * hartree_to_kcal
    ref_kcal = ref * hartree_to_kcal
    ft_kcal = ft * hartree_to_kcal

    error_gq = energy_kcal - ref_kcal
    error_ft = ft_kcal - ref_kcal

    me_gq = np.mean(error_gq)
    me_ft = np.mean(error_ft)

    x = np.arange(len(energy))

    return {
        "x": x,
        "energy": energy_kcal,
        "ref": ref_kcal,
        "ft": ft_kcal,
        "error_gq": error_gq,
        "error_ft": error_ft,
        "me_gq": me_gq,
        "me_ft": me_ft,
        "npoint": len(energy),
    }


formamide_ref = [
    -169.9031604200994,
    -169.90377725801045,
    -169.89903746646044,
    -169.88887370726093,
    -169.8681131643334,
    -169.73233954595185,
    -169.71585017933376,
    -169.83813087205826,
    -169.862349924223,
    -169.86517668336623,
    -169.8652736710562,
    -169.85040960247682,
    -169.73853021520065,
    -169.6944604986207,
    -169.69351603915453,
    -169.69160108337073,
    -169.69411058827865,
    -169.69242722677464,
    -169.69549315760554,
    -169.69410961349556,
]

formamide_orb_ft = [
    -169.89523932937828,
    -169.89510392927687,
    -169.8906026969449,
    -169.88164602251126,
    -169.86029793063173,
    -169.7240103395095,
    -169.70723197415816,
    -169.83146057891554,
    -169.8556696137735,
    -169.85879563993598,
    -169.85894290063112,
    -169.84432798326208,
    -169.73220294507595,
    -169.68524015433894,
    -169.6870588114614,
    -169.68806765204505,
    -169.68842229266465,
    -169.68867173276578,
    -169.6883629929746,
    -169.6885411558137,
]

formamide = load_case(
    filename = "./data/formamide/statistics.json",
    ref=formamide_ref,
    orb_ft=formamide_orb_ft
)


ethane_ref = [
    -79.82485,-79.82256,-79.81550,-79.77022,-79.70922,-79.65521,-79.63422,-79.60235,-79.59743,-79.63422,
    -79.64304,-79.64785,-79.64688,-79.64784,-79.64879,-79.64920,-79.64993,-79.65030,-79.65005,-79.64967
]

ethane_orb_ft = [
    -79.82279001193577,-79.8208029645284,-79.81405492502849,-79.76857064259848,-79.70757427249485,
    -79.65334869071113,-79.63152469092475,-79.59936870651245,-79.59572310663859,-79.6328892179701,
    -79.64165544170804,-79.64660120264689,-79.64638880761339,-79.64675515187581,-79.64784665739272,
    -79.64847104788886,-79.64868810636732,-79.64892591171265,-79.64924765048556,-79.64866351555719
]

ethane = load_case(
    filename = "./data/ethane/statistics.json",
    ref=ethane_ref,
    orb_ft=ethane_orb_ft
)


if not os.path.exists(top_image_path):
    raise FileNotFoundError(
        f"cant find:{top_image_path}\n"
    )

top_img = mpimg.imread(top_image_path)

if crop_top_image:
    top_img = crop_white_margin(top_img)

img_h, img_w = top_img.shape[:2]

fig_width_inch = 30.0
plot_height_inch = 6.2
bottom_space_inch = 2
top_space_inch = 0.10

left_margin = 0.04
right_margin = 0.99

plot_total_width_inch = fig_width_inch * (right_margin - left_margin)

top_image_height_inch = plot_total_width_inch * (img_h / img_w)

fig_height_inch = (
    bottom_space_inch
    + plot_height_inch
    + top_gap_inch
    + top_image_height_inch
    + top_space_inch
)

fig, axes = plt.subplots(
    1, 4,
    figsize=(fig_width_inch, fig_height_inch),
    dpi=400
)

# ========================================================================
# 子图 1：(a) Ethane - Relative Energy
# ========================================================================
ax = axes[0]

x = ethane["x"]
npoint = ethane["npoint"]

# 相对能量
rel_energy = ethane["energy"] - np.mean(ethane["energy"])
rel_ft = ethane["ft"] - np.mean(ethane["ft"])
rel_ref = ethane["ref"] - np.mean(ethane["ref"])

# 三条曲线
l1, = ax.plot(
    x, rel_energy,
    color=color_gq,
    marker="*",
    markersize=19,
    linewidth=2.5,
    linestyle="--",
    label="NucleoNet",
    zorder=3
)

l2, = ax.plot(
    x, rel_ft,
    color=color_ft,
    marker="o",
    markersize=11,
    linewidth=2.5,
    linestyle="--",
    label="Orbformer fine-tune LAC (400k)",
    zorder=3
)

l3, = ax.plot(
    x, rel_ref,
    color="black",
    linewidth=3.5,
    linestyle="-",
    label="Reference",
    zorder=3
)

# panel label
ax.text(
    0.03, 0.99,
    "(c) Ethane",
    transform=ax.transAxes,
    fontsize=30,
    ha="left",
    va="top"
)

# 坐标轴
ax.set_ylabel(
    "Relative E (kcal/mol)",
    fontsize=label
)

ax.set_xlabel(
    "Image ID on MEP",
    fontsize=label
)

ax.set_xlim(
    -0.5,
    npoint - 0.5
)
ax.set_ylim(-99,65)


step = 2
xtick_pos = np.arange(
    0,
    npoint,
    step
)

ax.set_xticks(xtick_pos)

ax.tick_params(
    axis="both",
    labelsize=20,
    length=6,
    width=1.2
)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)


# 保存句柄给总图例
handles = (l1, l2, l3)


# ========================================================================
# 子图 2：(b) Ethane - Energy Difference
# ========================================================================
ax = axes[1]

x = ethane["x"]
npoint = ethane["npoint"]

error_gq = ethane["error_gq"]
error_ft = ethane["error_ft"]

me_gq = ethane["me_gq"]
me_ft = ethane["me_ft"]

# 误差曲线
ax.plot(
    x,
    error_gq,
    color=color_gq,
    marker="*",
    markersize=19,
    linewidth=2.5,
    linestyle="--"
)

ax.plot(
    x,
    error_ft,
    color=color_ft,
    marker="o",
    markersize=9,
    linewidth=2.5,
    linestyle="--"
)

# 0 参考线
ax.axhline(
    0,
    color="black",
    linewidth=2
)

# panel label
ax.text(
    0.03,
    0.99,
    "(d) Ethane",
    transform=ax.transAxes,
    fontsize=30,
    ha="left",
    va="top"
)

# MSD
ax.text(
    0.6,
    0.92,
    f"MSD={me_gq:.2f}",
    transform=ax.transAxes,
    color=color_gq,
    fontsize=label,
    va="top"
)

ax.text(
    0.6,
    0.98,
    f"MSD={me_ft:.2f}",
    transform=ax.transAxes,
    color=color_ft,
    fontsize=label,
    va="top"
)

# 坐标轴
ax.set_ylabel(
    "E Difference (kcal/mol)",
    fontsize=label
)

ax.set_xlabel(
    "Image ID on MEP",
    fontsize=label
)

ax.set_xlim(
    -0.5,
    npoint - 0.5
)
ax.set_ylim(-1.1,2.2)

step = 2
xtick_pos = np.arange(
    0,
    npoint,
    step
)

ax.set_xticks(xtick_pos)

ytick_pos = np.arange(
    -1,
    3,
    1
)

ax.set_yticks(ytick_pos)

ax.tick_params(
    axis="both",
    labelsize=20,
    length=6,
    width=1.2
)
ax.grid(True, alpha=0.3)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)


# ========================================================================
# 子图 3：(c) Formamide - Relative Energy
# ========================================================================
ax = axes[2]

x = formamide["x"]
npoint = formamide["npoint"]

# 相对能量
rel_energy = formamide["energy"] - np.mean(formamide["energy"])
rel_ft = formamide["ft"] - np.mean(formamide["ft"])
rel_ref = formamide["ref"] - np.mean(formamide["ref"])

# 三条曲线
ax.plot(
    x,
    rel_energy,
    color=color_gq,
    marker="*",
    markersize=19,
    linewidth=2.5,
    linestyle="--",
    zorder=3
)

ax.plot(
    x,
    rel_ft,
    color=color_ft,
    marker="o",
    markersize=11,
    linewidth=2.5,
    linestyle="--",
    zorder=3
)

ax.plot(
    x,
    rel_ref,
    color="black",
    linewidth=3.5,
    linestyle="-",
    zorder=3
)

# panel label
ax.text(
    0.03,
    0.99,
    "(e) Formamide",
    transform=ax.transAxes,
    fontsize=30,
    ha="left",
    va="top"
)

# 坐标轴
ax.set_ylabel(
    "Relative E (kcal/mol)",
    fontsize=label
)

ax.set_xlabel(
    "Image ID on MEP",
    fontsize=label
)

ax.set_xlim(
    -0.5,
    npoint - 0.5
)

step = 2
xtick_pos = np.arange(
    0,
    npoint,
    step
)

ax.set_xticks(xtick_pos)

ax.tick_params(
    axis="both",
    labelsize=20,
    length=6,
    width=1.2
)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)


# ========================================================================
# 子图 4：(d) Formamide - Energy Difference
# ========================================================================
ax = axes[3]

x = formamide["x"]
npoint = formamide["npoint"]

error_gq = formamide["error_gq"]
error_ft = formamide["error_ft"]

me_gq = formamide["me_gq"]
me_ft = formamide["me_ft"]

# 误差曲线
ax.plot(
    x,
    error_gq,
    color=color_gq,
    marker="*",
    markersize=19,
    linewidth=2.5,
    linestyle="--"
)

ax.plot(
    x,
    error_ft,
    color=color_ft,
    marker="o",
    markersize=9,
    linewidth=2.5,
    linestyle="--"
)

# 0 参考线
ax.axhline(
    0,
    color="black",
    linewidth=2
)

# panel label
ax.text(
    0.03,
    0.99,
    "(f) Formamide",
    transform=ax.transAxes,
    fontsize=30,
    ha="left",
    va="top"
)

# MSD
ax.text(
    0.65,
    0.92,
    f"MSD={me_gq:.2f}",
    transform=ax.transAxes,
    color=color_gq,
    fontsize=label,
    va="top"
)

ax.text(
    0.65,
    0.98,
    f"MSD={me_ft:.2f}",
    transform=ax.transAxes,
    color=color_ft,
    fontsize=label,
    va="top"
)

# 坐标轴
ax.set_ylabel(
    "E Difference (kcal/mol)",
    fontsize=label
)

ax.set_xlabel(
    "Image ID on MEP",
    fontsize=label
)

ax.set_xlim(
    -0.5,
    npoint - 0.5
)

ax.set_ylim(
    -0.4,
    7
)

step = 2
xtick_pos = np.arange(
    0,
    npoint,
    step
)

ax.set_xticks(xtick_pos)

ytick_pos = np.arange(
    0,
    7,
    2
)

ax.set_yticks(ytick_pos)

ax.tick_params(
    axis="both",
    labelsize=20,
    length=6,
    width=1.2
)
ax.grid(True, alpha=0.3)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)




fig.legend(
    handles,
    [
        "NucleoNet",
        "Orbformer",
        "Reference"
    ],
    loc="lower center",
    ncol=3,
    fontsize=34,
    frameon=False,
    bbox_to_anchor=(0.5, 0),
    handlelength=2.5,
    columnspacing=2.0
)


plot_bottom = bottom_space_inch / fig_height_inch
plot_top = (bottom_space_inch + plot_height_inch) / fig_height_inch

plt.subplots_adjust(
    left=left_margin,
    right=right_margin,
    bottom=plot_bottom,
    top=plot_top,
    wspace=0.30,
)


fig.canvas.draw()

plots_left = axes[0].get_position().x0
plots_right = axes[-1].get_position().x1
plots_top = max(ax.get_position().y1 for ax in axes)
plots_width = plots_right - plots_left

top_img_height_frac = top_image_height_inch / fig_height_inch
gap_frac = top_gap_inch / fig_height_inch

ax_top = fig.add_axes([
    plots_left,
    plots_top + gap_frac,
    plots_width,
    top_img_height_frac
])

ax_top.imshow(
    top_img,
    aspect="equal",
    interpolation="none"
)

ax_top.set_xlim(-0.5, img_w - 0.5)
ax_top.set_ylim(img_h - 0.5, -0.5)
ax_top.margins(0)
ax_top.axis("off")


plt.savefig(
    "./figure/formamide_and_ethane.png",
    dpi=300,
)

# plt.show()
