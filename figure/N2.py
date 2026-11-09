import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tools.MLR import create_MLR_function
import json

phi3 = np.array([
    -3.125518, -1.338651, -1.926378, -1.053483, -1.088594,
    0.253611, 0.767715, -1.929323, 1.583792, 15.210659
    ])
MLR3 = create_MLR_function(phi=phi3,p=3,NS=9,NL=9)

#MLR_4(6,8)
phi4 = np.array([
-2.34414547, -0.972469, -1.561777, -1.136, -1.3963,
-0.819, -0.45, -3.36, 2.1
])
MLR4 = create_MLR_function(phi=phi4,p=4,NS=6,NL=8)


length_list = np.linspace(0.8, 3.2, 32).round(6)
length_list = [i*2 for i in length_list]
energy_MLR = [MLR4(i) for i in length_list]

length_list_train = np.linspace(0.8, 3, 6).round(6)
length_list_train = [i*2 for i in length_list_train]
energy_MLR_train = [MLR4(i) for i in length_list_train]

fig = plt.figure(figsize=(8, 6.5))
ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])

with open("./data/N2/multi_32/statistics.json", 'r') as file:
    data = json.load(file)
    energy=data["average"]
print("energy:",len(energy))
diff = [abs(i-j)*1000 for (i,j) in zip(energy,energy_MLR)]



x_pesnet = [
    1.60151, 1.70828, 1.81505, 1.92181, 2.02858, 2.13535,
    2.24212, 2.34889, 2.45565, 2.56242, 2.66919, 2.77595,
    2.88272, 2.98949, 3.09626, 3.20302, 3.30979, 3.41656,
    3.52333, 3.63009, 3.73686, 3.84363, 3.95040, 4.05716,
    4.16393, 4.27070, 4.37747, 4.48423, 4.59100, 4.69777,
    4.80454, 4.91130, 5.01807, 5.12484, 5.23161, 5.33837,
    5.87221, 6.40605,
]
y_pesnet = [
    0.00352832, 0.00343669, 0.00287605, 0.00284147, 0.00367587, 0.00387406,
    0.00266280, 0.00211453, 0.00243621, 0.00310587, 0.00375986, 0.00302045,
    0.00385466, 0.00427189, 0.00373975, 0.00631410, 0.00626827, 0.00705666,
    0.00830076, 0.00923855, 0.00960799, 0.00945501, 0.01038882, 0.01098299,
    0.00912387, 0.00920242, 0.00832464, 0.00674592, 0.00678292, 0.00582958,
    0.00508824, 0.00508810, 0.00467509, 0.00419918, 0.00336077, 0.00336845,
    0.00280661, 0.00232468,
]
y_pesnet = [1000 * y for y in y_pesnet]



ferminet_x_list = [
    1.6, 1.7, 1.8, 1.9, 2.0, 2.2,
    2.4, 2.6, 2.8, 3.0, 3.2, 3.4,
    3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 5.2, 5.4,
    5.6, 5.8, 6.0,
]
ferminet_y_list = [
    0.78, 1.26, 1.70, 1.72, 1.40, 1.62,
    1.85, 2.22, 1.97, 2.47, 3.04, 3.20,
    4.35, 5.03, 4.70, 5.30, 5.26, 4.18, 2.56, 2.28,
    1.85, 1.86, 1.08,
]


Pfaffian_x = [1.5987654320987654, 1.7098765432098766, 1.8148148148148149, 1.9197530864197532, 2.0246913580246915, 2.1358024691358026, 2.240740740740741, 2.351851851851852, 2.45679012345679, 2.5617283950617282, 2.6728395061728394, 2.7777777777777777, 2.882716049382716, 2.9876543209876543, 3.0925925925925926, 3.2037037037037037, 3.308641975308642, 3.419753086419753, 3.5246913580246915, 3.6296296296296298, 3.734567901234568, 3.8456790123456788, 3.950617283950617, 4.061728395061728, 4.160493827160494, 4.271604938271604, 4.382716049382716, 4.487654320987654, 4.592592592592593, 5.228395061728395, 5.339506172839506, 5.87037037037037, 6.407407407407407]
Pfaffian_y = [0.2377622377622377, 1.0769230769230769, 0.881118881118881, 0.46153846153846145, 0.7132867132867133, 0.6013986013986015, 0.5174825174825175, 0.43356643356643354, 0.6573426573426573, 0.8531468531468531, 0.9370629370629371, 1.0769230769230769, 1.0209790209790208, 1.2727272727272727, 1.832167832167832, 2.5314685314685317, 2.13986013986014, 2.7832167832167833, 3.3426573426573425, 3.93006993006993, 3.6783216783216783, 4.237762237762238, 4.013986013986014, 3.6503496503496504, 3.5664335664335667, 3.230769230769231, 3.2027972027972025, 2.895104895104895, 3.006993006993007, 1.904055944055944, 1.920374817726479, 1.2067832167832167, 1.008951048951049]


length_r12=[1.6,1.8,1.9,1.97,2.02,2.07,2.12,2.17,2.22,2.3,2.5,2.75,3,3.5,4,5,6]
energy_r=[-109.2151,-109.4563,-109.5086,-109.5277,-109.5346,-109.5369,
            -109.5355,-109.5310,-109.5240,-109.5088,-109.4581,-109.3878,-109.3238,
            -109.2346,-109.1947,-109.1765,-109.1738
            ]
MLR_r12 = [MLR4(i) for i in length_r12]
energy_r12 =[(i-j)*1000 for (i,j) in zip(energy_r,MLR_r12)]




ax.plot(x_pesnet,y_pesnet,linestyle="--", marker="v",markersize=7, linewidth=1, label="PESNet, MAE=%.2f"%np.mean(y_pesnet))
ax.plot(length_r12, energy_r12, linestyle="--", marker="^",markersize=7, linewidth=1, label=r"$r_{12}$-MR-ACPF, MAE=%.2f"%np.mean(energy_r12))
ax.plot(ferminet_x_list,ferminet_y_list,linestyle="--", marker="D",markersize=5, linewidth=1,c="black", label="FermiNet, MAE=%.2f"%np.mean(ferminet_y_list))
ax.plot(Pfaffian_x,Pfaffian_y,linestyle="--", marker="*",markersize=8, linewidth=1,c="purple", label="Pfaffian, MAE=%.2f"%np.mean(Pfaffian_y))
ax.plot(length_list, diff,  linestyle="--", marker="^",markersize=7,c="red", linewidth=1, label="NucleoNet, MAE=%.2f"%np.mean(diff))


ax.axhline(0, color="black", linewidth=2)
ax.legend(fontsize=17, loc="upper right", frameon=True, facecolor="white")
ax.grid(True, alpha=0.3)
ax.set_xlabel(r'Bond length ($a_0$)', fontsize=17, labelpad=0)
ax.set_ylabel(r'Energy error (m$E_h$)', fontsize=17, labelpad=5, rotation=90)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylim(-0.5, 13)

plt.tight_layout()

plt.savefig("./figure/N2.png",dpi=250)
