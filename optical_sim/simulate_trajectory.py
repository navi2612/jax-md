"""Reusable 2D overdamped (Brownian) trajectory simulation.

Wraps the local ``brownian_generalized_2d`` integrator into a small, reusable
API so you can simulate one (or many) point-like particles under arbitrary
optical forces/torques, starting from arbitrary positions/orientations. The
integrator lives in ``brownian_generalized_2d.py`` and uses jax-md only as a
normal package, so jax-md can be upgraded without touching this code.

The generalized coordinate of each particle is ``q = [x, y, theta]`` (metres,
metres, radians). A force function returns, *in the body frame*, the stacked
``[Fx, Fy, tau_z]`` of shape ``(N, 3)``.

Typical use
-----------
    from simulate_trajectory import simulate_and_save, make_active_optical_force

    force_fn = make_active_optical_force(phi_pol=0.0)
    run_dir, traj, state = simulate_and_save(
        force_fn, q0=[0.0, 0.0, 0.0], n_steps=200_000, label="linear_pol")

Each ``simulate_and_save`` call writes a self-documenting run folder under
``optical_sim/results/<timestamp>_<label>/`` containing ``trajectory.npz``,
``metadata.json`` (every parameter, the force/resistance config, date, versions)
and ``trajectory.png``. Use ``run_brownian_2d`` directly if you only want the
array back without saving.

Run this file directly (``python simulate_trajectory.py``) to reproduce the
single-particle example from the brownian_2d notebook and save a run folder.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax import random, lax, grad, vmap

from jax_md import energy, space

# Local integrator, decoupled from the jax-md package so jax-md can be upgraded
# freely (see brownian_generalized_2d.py).
from brownian_generalized_2d import brownian_generalized_2d

K_B = 1.380649e-23                                 # Boltzmann constant [J/K]


# --------------------------------------------------------------------------
# physical defaults (single 470 nm-ish bead, water at 300 K)
# --------------------------------------------------------------------------
KT_300K = jnp.float64(1.38e-23 * 300.0)            # k_B T  [J]

# Body-frame resistance (drag) matrix: diag(gamma_x, gamma_y, gamma_rot).
# Translational entries in N.s/m, rotational entry in N.m.s.
DEFAULT_RESISTANCE_BODY = jnp.array([
    [470e-9, 0.0,    0.0],
    [0.0,    470e-9, 0.0],
    [0.0,    0.0,    8000e-21],
], dtype=jnp.float64)


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def rotation_2d(theta):
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([[c, -s],
                      [s,  c]], dtype=theta.dtype)


def lab_to_body_force(theta, F_lab):
    """Rotate a lab-frame force into the particle body frame.

    The rotation matrix is orthogonal, so the lab->body map is its transpose.
    """
    return rotation_2d(theta).T @ F_lab


def shift_generalized_2d(q, dq, **kwargs):
    """Advance ``q`` by ``dq`` and wrap the orientation into ``(-pi, pi]``."""
    q_new = q + dq
    theta = (q_new[..., 2] + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return q_new.at[..., 2].set(theta)


def _as_q0(q0):
    """Normalize a start condition to a float64 ``(N, 3)`` array."""
    q0 = jnp.asarray(q0, dtype=jnp.float64)
    if q0.ndim == 1:
        q0 = q0[None, :]
    if q0.ndim != 2 or q0.shape[1] != 3:
        raise ValueError(f"q0 must be shape (3,) or (N, 3); got {q0.shape}")
    return q0


# --------------------------------------------------------------------------
# force-function builders  (each returns a force_fn_body: q -> (N, 3))
# --------------------------------------------------------------------------
def make_constant_force(Fx=0.0, Fy=0.0, tau=0.0):
    """Constant body-frame force/torque applied identically to every particle."""
    F_const = jnp.array([Fx, Fy, tau], dtype=jnp.float64)

    def force_fn_body(q, **kwargs):
        return jnp.broadcast_to(F_const, q.shape)

    force_fn_body.spec = {"type": "constant",
                          "Fx": float(Fx), "Fy": float(Fy), "tau": float(tau)}
    return force_fn_body


def make_active_optical_force(phi_pol=0.0, interaction_energy_fn=None):
    """Orientation-dependent optical force/torque from ``active_optical``.

    Args:
      phi_pol: fixed lab polarization angle [rad]; sets the stable orientation.
      interaction_energy_fn: optional ``q -> scalar`` lab-frame potential energy
        of the particle centres (e.g. soft-sphere repulsion + an optical trap).
        Its (lab-frame) gradient force is added to the optical force.

    Returns a ``force_fn_body(q) -> (N, 3)``.
    """
    import active_optical
    from active_optical import active_force_torque_body

    grad_energy_fn = grad(interaction_energy_fn) if interaction_energy_fn else None

    def force_fn_body(q, **kwargs):
        theta = q[:, 2]

        # orientation-dependent optical active force/torque (body frame)
        F_active_body, tau_active_body = active_force_torque_body(theta, phi_pol)
        F_active_body = F_active_body.astype(q.dtype)
        tau_active_body = tau_active_body.astype(q.dtype)

        if grad_energy_fn is not None:
            # interaction force in lab frame -> rotate into the body frame
            F_lab = -grad_energy_fn(q)[:, :2]
            F_inter_body = vmap(lab_to_body_force)(theta, F_lab)
        else:
            F_inter_body = jnp.zeros_like(F_active_body)

        F_body = F_active_body + F_inter_body
        tau = tau_active_body
        return jnp.concatenate([F_body, tau[:, None]], axis=1)

    # record the fitted model coefficients so a run is reproducible even if the
    # active_optical module is later re-fitted.
    force_fn_body.spec = {
        "type": "active_optical",
        "phi_pol": float(phi_pol),
        "fitted_coefficients": {
            "FX_pN": list(active_optical._FX),
            "FY_pN": list(active_optical._FY),
            "TZ_pN_nm": list(active_optical._TZ),
        },
        "interaction": getattr(interaction_energy_fn, "spec", None),
    }
    return force_fn_body


def make_soft_sphere_energy(sigma=2.0e-6, epsilon=1.0e-18, alpha=2.0,
                            k_trap=0.0):
    """Lab-frame centre energy: soft-sphere pair repulsion + optional trap.

    Returns ``energy_fn(q) -> scalar`` suitable for ``interaction_energy_fn``.
    """
    displacement_fn, _ = space.free()
    pair_energy_fn = energy.soft_sphere_pair(
        displacement_fn, sigma=sigma, epsilon=epsilon, alpha=alpha)

    def energy_fn(q):
        R = q[:, :2]
        U = pair_energy_fn(R)
        if k_trap:
            U = U + 0.5 * k_trap * jnp.sum(R ** 2)
        return U

    energy_fn.spec = {"type": "soft_sphere", "sigma": float(sigma),
                      "epsilon": float(epsilon), "alpha": float(alpha),
                      "k_trap": float(k_trap)}
    return energy_fn


def run_brownian_2d(
    force_fn_body,
    q0,
    n_steps=200_000,
    dt=1e-4,
    kT=KT_300K,
    resistance_body=DEFAULT_RESISTANCE_BODY,
    seed=0,
    shift_fn=shift_generalized_2d,
    include_drift=True,
):
    """Integrate overdamped 2D Brownian dynamics and return the trajectory.

    Args:
      force_fn_body: ``q -> (N, 3)`` body-frame ``[Fx, Fy, tau_z]``. Build one
        with the ``make_*`` helpers, or pass your own.
      q0: start condition, ``(3,)`` for a single particle or ``(N, 3)``.
      n_steps: number of integration steps to record.
      dt: time step [s].
      kT: thermal energy [J].
      resistance_body: ``(3, 3)`` body-frame drag matrix.
      seed: PRNG seed.
      shift_fn: position-update function (defaults to angle-wrapping shift).

    Returns:
      traj: ``(n_steps, N, 3)`` array of recorded positions.
      state: the final integrator state.
    """
    q0 = _as_q0(q0)

    init_fn, step_fn = brownian_generalized_2d(
        force_fn_body=force_fn_body,
        shift_fn=shift_fn,
        dt=dt,
        kT=kT,
        resistance_body=resistance_body,
        include_drift=include_drift,
    )

    state = init_fn(random.PRNGKey(seed), q0)

    def scan_step(state, _):
        state = step_fn(state)
        return state, state.position

    state, traj = lax.scan(scan_step, state, xs=None, length=n_steps)
    return traj, state


def quick_plot(traj, particle=0, pos_units="um", out_path=None, show=False):
    """Minimal sanity plot of one particle's xy path (no styling)."""
    import matplotlib.pyplot as plt

    xy = np.asarray(traj)[:, particle, :2]
    if pos_units == "um":
        xy = xy * 1e6
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(xy[:, 0], xy[:, 1], lw=1.0)
    ax.scatter(xy[0, 0], xy[0, 1], c="k", s=30, zorder=3, label="start")
    ax.set_aspect("equal")
    ax.set_xlabel("x  (µm)")
    ax.set_ylabel("y  (µm)")
    ax.legend(loc="best")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"saved {out_path}")
    if show:
        return fig
    plt.close(fig)
    return None


# --------------------------------------------------------------------------
# saving: one self-documenting folder per run
# --------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _json_default(o):
    """Fallback JSON encoder for numpy / jax scalars and arrays."""
    try:
        arr = np.asarray(o)
        return arr.item() if arr.ndim == 0 else arr.tolist()
    except Exception:
        return str(o)


def _git_commit():
    """Current short git commit hash, or None if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def simulate_and_save(
    force_fn_body,
    q0,
    n_steps=200_000,
    dt=1e-4,
    kT=KT_300K,
    resistance_body=DEFAULT_RESISTANCE_BODY,
    seed=0,
    shift_fn=shift_generalized_2d,
    label=None,
    results_dir=RESULTS_DIR,
    plot=True,
    extra_metadata=None,
):
    """Run a simulation and save it to its own timestamped results folder.

    Creates ``<results_dir>/<timestamp>[_label]/`` containing:
      - ``trajectory.npz`` : the trajectory ``(n_steps, N, 3)`` and ``q0``
      - ``metadata.json``  : every parameter, the force/resistance config, the
        date, jax version and git commit
      - ``trajectory.png`` : a quick-look plot (unless ``plot=False``)

    All ``run_brownian_2d`` arguments are accepted and recorded. ``label`` is an
    optional human-readable tag appended to the folder name; ``extra_metadata``
    is an optional dict merged into the saved metadata.

    Returns ``(run_dir, traj, state)``.
    """
    traj, state = run_brownian_2d(
        force_fn_body, q0, n_steps=n_steps, dt=dt, kT=kT,
        resistance_body=resistance_body, seed=seed, shift_fn=shift_fn)

    q0_arr = np.asarray(_as_q0(q0))
    now = datetime.now()
    meta = {
        "datetime": now.isoformat(timespec="seconds"),
        "label": label,
        "n_steps": int(n_steps),
        "dt": float(dt),
        "total_time_s": float(n_steps) * float(dt),
        "kT": float(kT),
        "temperature_K": float(kT) / K_B,
        "seed": int(seed),
        "n_particles": int(q0_arr.shape[0]),
        "q0": q0_arr.tolist(),
        "resistance_body": np.asarray(resistance_body).tolist(),
        "shift_fn": getattr(shift_fn, "__name__", str(shift_fn)),
        "force": getattr(force_fn_body, "spec",
                         {"type": "custom",
                          "name": getattr(force_fn_body, "__name__",
                                          repr(force_fn_body))}),
        "jax_version": jax.__version__,
        "jax_md_version": getattr(__import__("jax_md"), "__version__", None),
        "git_commit": _git_commit(),
    }
    if extra_metadata:
        meta["extra"] = extra_metadata

    # build a unique run directory: <timestamp>[_label][-i]
    stamp = now.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{label}" if label else stamp
    run_dir = Path(results_dir) / name
    i = 1
    while run_dir.exists():
        run_dir = Path(results_dir) / f"{name}-{i}"
        i += 1
    run_dir.mkdir(parents=True)

    np.savez_compressed(run_dir / "trajectory.npz",
                        traj=np.asarray(traj), q0=q0_arr)
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)
    if plot:
        quick_plot(traj, out_path=str(run_dir / "trajectory.png"))

    print(f"saved run to {run_dir}")
    return run_dir, traj, state


# --------------------------------------------------------------------------
# example: single point-like particle (reproduces the brownian_2d cell)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    energy_fn = make_soft_sphere_energy(sigma=2.0e-6, epsilon=1.0e-18, alpha=2.0)
    force_fn = make_active_optical_force(phi_pol=0.0, interaction_energy_fn=energy_fn)
    force_fn = make_constant_force(1.135e-12, 0.045e-12,440.44e-21)

    q0 = [0.0, 0.0, 0.0]  # single particle at the origin

    run_dir, traj, state = simulate_and_save(
        force_fn, q0, n_steps=1_000_000, dt=1e-4, label="circular_pol")
    print("trajectory shape:", traj.shape)
