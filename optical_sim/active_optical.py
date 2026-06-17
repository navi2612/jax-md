"""Orientation-dependent optical active force/torque, fitted to the polarization sweep.

Each component is an exact second harmonic of the relative angle phi = phi_pol - theta,
where theta is the particle orientation and phi_pol the (fixed) lab polarization angle.

Mapping into the 2D sim: body force = (Fx, Fy), scalar torque (about z) = Tz.
Coefficients are in pN / pN.nm; constants below convert to SI (N, N.m).
"""

import jax.numpy as jnp

_PN = 1e-12      # pN  -> N
_PN_NM = 1e-21   # pN.nm -> N.m

# fitted second-harmonic coefficients  g(phi) = a0 + a1*cos(2phi) + a2*sin(2phi)
_FX = (1.1350, 0.8574, 0.0)      # pN
_FY = (0.0,    0.0,    -0.0388)  # pN
_TZ = (0.0,    0.0,    -238.4348)  # pN.nm


def _harmonic(phi, coef):
    a0, a1, a2 = coef
    return a0 + a1 * jnp.cos(2.0 * phi) + a2 * jnp.sin(2.0 * phi)


def active_force_torque_body(theta, phi_pol=0.0):
    """Active body-frame force (N) and scalar torque (N.m) for orientation theta.

    theta : (N,) particle orientations [rad]
    returns F_active_body (N, 2) and tau_active_body (N,)
    """
    # relative angle of polarization in the body frame; this sign makes the
    # polarization-aligned orientation (theta = phi_pol) the STABLE equilibrium,
    # so the torque damps Brownian angular fluctuations instead of amplifying them.
    phi = theta - phi_pol
    fbx = _harmonic(phi, _FX) * _PN
    fby = _harmonic(phi, _FY) * _PN
    tau = _harmonic(phi, _TZ) * _PN_NM
    F_active_body = jnp.stack([fbx, fby], axis=1)
    return F_active_body, tau


def active_force_torque_body_new(theta, phi_pol):
    alpha = phi_pol - theta          # relative angle; note the minus theta
    c, s = jnp.cos(alpha), jnp.sin(alpha)

    # --- torque (scalar, no frame rotation) ---
    tau_z = 598.194 * c*c + -262.874 * s*s + -0.003 * s*c
    # tau_z is already the physical torque about +z (counterclockwise positive),
    # as a function of the current relative angle. Plug directly into the
    # rotational EOM that uses "+tau increases theta".

    # --- active force: lab-frame from coeffs, then rotate to body frame ---
    Fx_lab = -0.2493 * c*c + 0.116281 * s*s + 0.93375 * s*c
    Fy_lab = 0.00218 * c*c + -0.00324 * s*s + -0.00340693 * s*c
    F_lab  = jnp.stack([Fx_lab, Fy_lab], axis=-1)
    F_body = vmap(lab_to_body_force)(theta, F_lab)

    return F_body, tau_z