"""Plot optical force and torque vs. linear-polarization angle."""

import numpy as np
import matplotlib.pyplot as plt

# Polarization angle relative to x-axis (deg)
angle = np.array([0, 15, 30, 45, 60, 75, 90])

# Force (pN), columns = [Fx, Fy, Fz]
force = np.array([
    [1.992328, -0.000000, 3.751989],
    [1.877463, -0.019429, 4.006597],
    [1.563646, -0.033652, 4.702201],
    [1.134963, -0.038858, 5.652413],
    [0.706016, -0.033618, 6.602494],
    [0.392397, -0.019410, 7.298189],
    [0.277604,  0.000000, 7.552831],
])

# Torque (pN nm), columns = [Tx, Ty, Tz]
torque = np.array([
    [ 0.000000, -283.370905,  -0.000000],
    [-88.634848, -268.540013, -119.265660],
    [-153.520060, -228.021263, -206.574182],
    [-177.269696, -172.671622, -238.531319],
    [-153.811229, -117.232887, -206.263226],
    [-88.802955,  -76.781417, -119.086129],
    [ 0.000000,  -61.975152,   0.000000],
])

fig, (ax_f, ax_t) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

comp_labels = ["x", "y", "z"]
colors = ["#1f77b4", "#2ca02c", "#d62728"]

for i, (lbl, c) in enumerate(zip(comp_labels, colors)):
    ax_f.plot(angle, force[:, i], "o-", color=c, label=f"$F_{lbl}$")
    ax_t.plot(angle, torque[:, i], "s-", color=c, label=fr"$\tau_{lbl}$")

ax_f.set_ylabel("Force (pN)")
ax_f.legend(ncol=3, frameon=False)
ax_f.grid(alpha=0.3)
ax_f.axhline(0, color="k", lw=0.6)

ax_t.set_ylabel("Torque (pN·nm)")
ax_t.set_xlabel("Polarization angle (deg)")
ax_t.legend(ncol=3, frameon=False)
ax_t.grid(alpha=0.3)
ax_t.axhline(0, color="k", lw=0.6)

ax_t.set_xticks(angle)
fig.suptitle("Optical force & torque vs. polarization orientation")
fig.tight_layout()

fig.savefig("optical_force_torque.png", dpi=200)
plt.show()
