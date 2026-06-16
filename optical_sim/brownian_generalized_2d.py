"""2D overdamped (Brownian) integrators for generalized coordinates.

Lifted out of a patched ``jax_md/simulate.py`` so this research code depends on
jax-md only as a normal package (its public submodules ``dataclasses`` and
``quantity``), not on a forked library. That means jax-md can be upgraded with a
plain ``pip install -U jax-md`` without ever touching these integrators.

Provides:
  - ``brownian_generalized_2d``       : point-like particles, ``q = [x, y, theta]``
  - ``brownian_generalized_rigid_2d`` : ``jax_md.rigid_body.RigidBody`` state

The dynamics are overdamped Langevin (Brownian) with a constant body-frame
mobility/resistance. The mobility's translation-rotation coupling produces a
thermal drift term, included below.
"""

from typing import Any

import jax.numpy as jnp
from jax import Array, random, vmap

from jax_md import dataclasses, quantity


# --------------------------------------------------------------------------
# rigid-body variant: state is a jax_md.rigid_body.RigidBody(center, orientation)
# --------------------------------------------------------------------------
@dataclasses.dataclass
class BrownianRigid2DState:
    position: Any
    rng: jnp.ndarray


def wrap_angle(theta):
    return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi


def brownian_generalized_rigid_2d(
    energy_or_force_fn,
    shift_fn,
    dt,
    kT,
    mobility_body=None,
    resistance_body=None,
):
    from jax_md import rigid_body

    if (mobility_body is None) == (resistance_body is None):
        raise ValueError("Specify exactly one of mobility_body or resistance_body.")
    if resistance_body is not None:
        mobility_body = jnp.linalg.inv(resistance_body)

    chol = jnp.linalg.cholesky(mobility_body)
    force_fn = quantity.canonicalize_force(energy_or_force_fn)

    def init_fn(key, body):
        return BrownianRigid2DState(body, key)

    def step_fn(state, **kwargs):
        _dt = kwargs.pop("dt", dt)
        _kT = kwargs.pop("kT", kT)

        body, key = state.position, state.rng
        key, split = random.split(key)

        theta = body.orientation
        R = rigid_body.rotation2d(theta)  # body -> lab
        # cant directly use .T as R is batched
        RT = jnp.swapaxes(R, -1, -2)  # lab -> body

        # take rigidbody position and return rigidbody force
        force = force_fn(body, **kwargs)  # RigidBody(force_xy, torque)

        # rigidbody can represent many things. so center can be Force and orientation can be Torque
        # RT (N,2,2) and Force (N,2)
        F_body_xy = jnp.einsum("nij,nj->ni", RT, force.center)
        W_body = jnp.concatenate([F_body_xy, force.orientation[:, None]], axis=-1)  # (N, 3)

        xi = random.normal(split, W_body.shape, W_body.dtype)

        dq_det_body = jnp.einsum("ni,ki->nk", W_body, mobility_body) * _dt
        dq_stoch_body = jnp.sqrt(2.0 * _kT * _dt) * jnp.einsum("ni,ji->nj", xi, chol)

        drift_body = _kT * _dt * jnp.array(
            [-mobility_body[1, 2], mobility_body[0, 2], 0.0],
            dtype=W_body.dtype,
        )

        dq_body = dq_det_body + dq_stoch_body + drift_body

        dR_lab = jnp.einsum("nij,nj->ni", R, dq_body[:, :2])
        dtheta = dq_body[:, 2]

        new_center = shift_fn(body.center, dR_lab, **kwargs)
        new_theta = wrap_angle(body.orientation + dtheta)

        new_body = rigid_body.RigidBody(new_center, new_theta)
        return BrownianRigid2DState(new_body, key)

    return init_fn, step_fn


# --------------------------------------------------------------------------
# point-like variant: state is q = [x, y, theta], force given in the body frame
# --------------------------------------------------------------------------
def rotation_2d_3x3(theta):
  c = jnp.cos(theta)
  s = jnp.sin(theta)
  return jnp.array([[c, -s, 0.0],
                    [s,  c, 0.0],
                    [0.0, 0.0, 1.0]], dtype=theta.dtype)


@dataclasses.dataclass
class BrownianGeneralized2DState:
  position: Array
  rng: Array


def brownian_generalized_2d(
  force_fn_body,
  shift_fn,
  dt,
  kT,
  mobility_body=None,
  resistance_body=None,
  include_drift=True,
):
  """Overdamped 2D Brownian integrator with a constant body-frame mobility.

  ``include_drift`` adds the thermal-drift term ``kT * grad . M_lab`` required
  for the scheme to sample the Boltzmann distribution when the mobility has a
  translation-rotation coupling. It is zero for a diagonal mobility, so it only
  matters for the anisotropic/chiral case; the flag exists mainly so validation
  can show its effect.
  """
  if (mobility_body is None) == (resistance_body is None):
    raise ValueError('Specify exactly one of mobility_body or resistance_body.')

  if resistance_body is not None:
    mobility_body = jnp.linalg.inv(resistance_body)

  chol_mobility_body = jnp.linalg.cholesky(mobility_body)
  rotation_2d_3x3_batched = vmap(rotation_2d_3x3)

  def init_fn(key, q):
    return BrownianGeneralized2DState(q, key)

  def step_fn(state, **kwargs):
    _dt = kwargs.pop('dt', dt)
    _kT = kwargs.pop('kT', kT)

    q, key = dataclasses.astuple(state)
    key, split = random.split(key)

    theta = q[:, 2]
    B = rotation_2d_3x3_batched(theta)

    F_body = force_fn_body(q, **kwargs)  # (n, 3)

    dq_det_body = (F_body @ mobility_body.T) * _dt

    xi = random.normal(split, q.shape, q.dtype)  # (n, 3)
    dq_stoch_body = (
        jnp.sqrt(2.0 * _kT * _dt) *
        (xi @ chol_mobility_body.T)
    )

    # Thermal drift for constant body-frame mobility with tr-coupling for 2d.
    if include_drift:
      drift_velocity_body = jnp.array([
          -mobility_body[1, 2],
          mobility_body[0, 2],
          0.0
      ], dtype=q.dtype)
      dq_drift_body = _kT * _dt * drift_velocity_body
    else:
      dq_drift_body = 0.0

    dq_body = dq_det_body + dq_stoch_body + dq_drift_body
    dq_lab = jnp.einsum('nij,nj->ni', B, dq_body)

    q = shift_fn(q, dq_lab, **kwargs)
    return BrownianGeneralized2DState(q, key)

  return init_fn, step_fn
