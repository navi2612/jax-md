"""Orientation-dependent optical active force/torque, fitted to the polarization sweep.

Each component is an exact second harmonic of the relative angle phi = phi_pol - theta,
where theta is the particle orientation and phi_pol the (fixed) lab polarization angle.

Mapping into the 2D sim: body force = (Fx, Fy), scalar torque (about z) = Tz.
Coefficients are in pN / pN.nm; constants below convert to SI.

The coefficients below are whichever design was fitted last. Pass FX/FY/TZ to use
a different one -- a saved run's metadata.json holds its own.
"""

import jax.numpy as jnp

_PN = 1e-12      # pN  -> N
_PN_NM = 1e-21   # pN.nm -> N.m

#Fit coefficients A, B, C
_FX = [0.0, 0.0, 0.0]      # pN
_FY = [0.0, 0.0, 0.0]  # pN
_TZ = [-2655.25, -2655.14, -0.109236]  # pN.nm


def active_force_torque_body(theta, phi_pol, FX=None, FY=None, TZ=None, power=3):
    FX = _FX if FX is None else FX
    FY = _FY if FY is None else FY
    TZ = _TZ if TZ is None else TZ

    alpha = phi_pol - theta # relative angle; note the minus theta
    c, s = jnp.cos(alpha), jnp.sin(alpha)

    tau_z = TZ[0] * c*c + TZ[1] * s*s + TZ[2] * s*c

    Fx_body = FX[0] * c*c + FX[1] * s*s + FX[2] * s*c
    Fy_body = FY[0] * c*c + FY[1] * s*s + FY[2] * s*c
    F_body  = jnp.stack([Fx_body, Fy_body], axis=-1)
    return F_body*_PN*power, tau_z*_PN_NM*power
