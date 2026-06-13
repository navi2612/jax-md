"""Fit optical force/torque components to the second harmonic a0 + a1 cos2φ + a2 sin2φ."""

import numpy as np

deg = np.array([0, 15, 30, 45, 60, 75, 90], dtype=float)
phi = np.deg2rad(deg)

force = np.array([                       # pN: [Fx, Fy, Fz]
    [1.992328, -0.000000, 3.751989],
    [1.877463, -0.019429, 4.006597],
    [1.563646, -0.033652, 4.702201],
    [1.134963, -0.038858, 5.652413],
    [0.706016, -0.033618, 6.602494],
    [0.392397, -0.019410, 7.298189],
    [0.277604,  0.000000, 7.552831],
])
torque = np.array([                      # pN·nm: [Tx, Ty, Tz]
    [  0.000000, -283.370905,   -0.000000],
    [-88.634848, -268.540013, -119.265660],
    [-153.520060, -228.021263, -206.574182],
    [-177.269696, -172.671622, -238.531319],
    [-153.811229, -117.232887, -206.263226],
    [-88.802955,  -76.781417, -119.086129],
    [  0.000000,  -61.975152,    0.000000],
])

# design matrix for a0 + a1 cos(2φ) + a2 sin(2φ)
M = np.column_stack([np.ones_like(phi), np.cos(2 * phi), np.sin(2 * phi)])

def fit(y):
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    resid = y - M @ coef
    rms = np.sqrt(np.mean(resid**2))
    a0, a1, a2 = coef
    R = np.hypot(a1, a2)               # amplitude of the 2φ modulation
    phi0 = 0.5 * np.arctan2(a2, a1)    # phase: a0 + R cos(2(φ - phi0))
    return a0, a1, a2, R, np.rad2deg(phi0), rms

print(f"{'comp':>4}  {'a0':>10} {'a1(cos2φ)':>11} {'a2(sin2φ)':>11} "
      f"{'R':>9} {'φ0(deg)':>8} {'rms':>9}")
labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
data = np.column_stack([force, torque])
for lbl, col in zip(labels, data.T):
    a0, a1, a2, R, p0, rms = fit(col)
    print(f"{lbl:>4}  {a0:10.4f} {a1:11.4f} {a2:11.4f} {R:9.4f} {p0:8.2f} {rms:9.2e}")
