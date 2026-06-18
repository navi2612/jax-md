"""Orientation-dependent optical active force/torque, fitted to the polarization sweep.

Each component is an exact second harmonic of the relative angle phi = phi_pol - theta,
where theta is the particle orientation and phi_pol the (fixed) lab polarization angle.

Mapping into the 2D sim: body force = (Fx, Fy), scalar torque (about z) = Tz.
Coefficients are in pN / pN.nm; constants below convert to SI (N, N.m).
"""

import jax.numpy as jnp

_PN = 1e-12      # pN  -> N
_PN_NM = 1e-21   # pN.nm -> N.m

#Fit coefficients A, B, C
_FX = [0.0, 0.0, 0.0]      # pN
_FY = [0.0,    0.0,    0.0]  # pN
_TZ = [-2608.94,    -2608.94,    0]  # pN.nm



def active_force_torque_body(theta, phi_pol):
    _PN = 1e-12  # pN  -> N
    _PN_NM = 1e-21
    alpha = phi_pol - theta # relative angle; note the minus theta
    c, s = jnp.cos(alpha), jnp.sin(alpha)

    tau_z = _TZ[0] * c*c + _TZ[1] * s*s + _TZ[2] * s*c

    Fx_lab = _FX[0] * c*c + _FX[1] * s*s + _FX[2] * s*c
    Fy_lab = _FY[0] * c*c + _FY[1] * s*s + _FY[2] * s*c
    F_lab  = jnp.stack([Fx_lab, Fy_lab], axis=-1)
    return F_lab*_PN*3, tau_z*_PN_NM*3