"""Many interacting rigid-body metavehicles in 2D (overdamped Brownian).

Extracted and cleaned from ``notebooks/experiments/metavehicle_swarm.py``. Each
particle is a rigid union of beads (a filled square by default) that sterically
repels its neighbours and is driven by a body-frame optical force/torque. A
neighbor list keeps the interaction ~O(N), so it scales to many particles.

It is consistent with the single-particle code in this folder: same overdamped
integrator family (the local ``brownian_generalized_rigid_2d``), same kT and
default resistance matrix.

Coordinates: each particle is a ``jax_md.rigid_body.RigidBody`` with a centre
``(N, 2)`` [m] and an orientation ``(N,)`` [rad]. The drive is a constant
body-frame force ``F_body=(Fx, Fy)`` [N] and torque ``tau`` [N.m] (a constant
torque models circular polarization -> spinners); pass your own ``drive_fn`` for
an orientation-dependent drive.

Typical use
-----------
    from simulate_swarm import run_swarm
    out = run_swarm(n_particles=30, n_steps=100_000, label="circular")
    # out["centers"]: (frames, N, 2) [m]   out["angles"]: (frames, N) [rad]

Run directly (``python simulate_swarm.py``) to validate the forces and do a
small demo run saved under results/.
"""

import json
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random, lax, jit
from jax_md import space, energy, rigid_body, quantity

from brownian_generalized_2d import brownian_generalized_rigid_2d
from simulate_trajectory import (
    KT_300K, K_B, DEFAULT_RESISTANCE_BODY, _json_default, _git_commit, RESULTS_DIR,
)

# circular-polarization optical drive (body frame), from the swarm notebook
DEFAULT_F_BODY = (1.135e-12, 0.045e-12)        # (Fx, Fy)  [N]
DEFAULT_TAU = 440.44e-21                        # torque    [N.m]


# --------------------------------------------------------------------------
# geometry, drive, interaction  (small builders -> a force_fn(body))
# --------------------------------------------------------------------------
def square_shape(side=10e-6, n_grid=10):
    """A filled square rigid body: ``n_grid x n_grid`` beads tiling ``side``.

    Returns ``(shape, sigma)`` where ``sigma = side/n_grid`` is the bead diameter
    (== spacing -> a filled square). The bead-centre grid is inset by one radius
    so the bead surfaces reach exactly to +-side/2: steric repulsion then turns
    on right as two drawn squares touch.
    """
    sigma = side / n_grid
    span = side - sigma
    xs = jnp.linspace(-span / 2, span / 2, n_grid)
    gx, gy = jnp.meshgrid(xs, xs)
    pts = jnp.stack([gx.ravel(), gy.ravel()], axis=1).astype(jnp.float64)
    shape = rigid_body.point_union_shape(pts, jnp.ones(pts.shape[0]))
    return shape, sigma


def make_drive(F_body=DEFAULT_F_BODY, tau=DEFAULT_TAU):
    """Constant body-frame drive -> ``drive_fn(body) -> (f_lab (N,2), tau (N,))``."""
    F_body = jnp.asarray(F_body, dtype=jnp.float64)
    tau = jnp.float64(tau)

    def drive_fn(body):
        R = rigid_body.rotation2d(body.orientation)            # (N,2,2) body->lab
        f_lab = jnp.einsum("nij,j->ni", R, F_body)
        tau_arr = jnp.full(body.orientation.shape, tau, dtype=body.center.dtype)
        return f_lab, tau_arr

    return drive_fn


def make_modulated_drive(intensity_fn, F_body=DEFAULT_F_BODY, tau=DEFAULT_TAU):
    """Spatially-modulated optical drive: force AND torque scale with I(x,y)/I0.

    ``intensity_fn(centers)`` maps centres ``(N, 2)`` [m] to a non-negative
    scaling ``(N,)`` (the local intensity relative to a reference). Models an SLM
    pattern under the linear optical response. Returns the usual
    ``drive_fn(body) -> (f_lab (N,2), tau (N,))``.

    Note: because both F and tau scale with I, the orbit *radius* is unchanged;
    intensity sets the orbital *speed*. Spatial gradients drive a chiral-taxis
    drift, not simple accumulation.
    """
    F_body = jnp.asarray(F_body, dtype=jnp.float64)
    tau = jnp.float64(tau)

    def drive_fn(body):
        s = intensity_fn(body.center)                          # (N,) >= 0
        R = rigid_body.rotation2d(body.orientation)
        f_lab = jnp.einsum("nij,j->ni", R, F_body) * s[:, None]
        return f_lab, s * tau

    return drive_fn


# --- intensity-pattern builders: centres (N,2) [m] -> scaling (N,) ----------
def intensity_gradient(box, lo=0.5, hi=4.0, axis=0):
    """Linear ramp from ``lo`` (one edge) to ``hi`` (opposite edge)."""
    def fn(c):
        return lo + (hi - lo) * (c[:, axis] / box)
    return fn


def intensity_disk(center, radius, inside=4.0, outside=0.5):
    """Uniform ``inside`` within ``radius`` of ``center``, ``outside`` beyond."""
    center = jnp.asarray(center)
    def fn(c):
        r = jnp.linalg.norm(c - center, axis=1)
        return jnp.where(r < radius, inside, outside)
    return fn


def intensity_ring(center, r0, width, bright=4.0, dark=0.5):
    """Bright annulus of radius ``r0`` (thickness ``width``); dark elsewhere."""
    center = jnp.asarray(center)
    def fn(c):
        r = jnp.linalg.norm(c - center, axis=1)
        return jnp.where(jnp.abs(r - r0) < width / 2, bright, dark)
    return fn


def intensity_stripes(period, bright=4.0, dark=0.5, axis=0):
    """Alternating bright/dark stripes with spatial ``period`` [m]."""
    def fn(c):
        return jnp.where(jnp.sin(2 * jnp.pi * c[:, axis] / period) > 0, bright, dark)
    return fn


def _build_interaction(shape, sigma, epsilon, displacement_fn, box,
                       use_neighbor_list, dr_threshold=None,
                       capacity_multiplier=2.0):
    """Soft-sphere steric force between rigid bodies (lab frame).

    Returns ``(interaction_force_fn, nbr_fns)`` (``nbr_fns`` is None for the
    all-pairs path). The neighbor-list cutoff is ``sigma`` (soft_sphere is 0
    beyond sigma), so it returns the SAME forces as all-pairs -- only the cost
    scaling differs. ``dr_threshold`` is the Verlet skin (defaults to 0.2*sigma);
    use a *larger* skin when particles move fast (e.g. high light intensity) so
    the neighbor list does not rebuild every step.
    """
    if dr_threshold is None:
        dr_threshold = 0.2 * sigma
    if use_neighbor_list:
        neighbor_fn, ss_nl = energy.soft_sphere_neighbor_list(
            displacement_fn, box, sigma=sigma, epsilon=epsilon, alpha=2.0,
            dr_threshold=dr_threshold,     # Verlet skin in metres, not the 0.2 default
            capacity_multiplier=capacity_multiplier,
        )
        nbr_fns, rigid_energy = rigid_body.point_energy_neighbor_list(
            ss_nl, neighbor_fn, shape)
        return quantity.force(rigid_energy), nbr_fns

    pair_energy = energy.soft_sphere_pair(
        displacement_fn, sigma=sigma, epsilon=epsilon, alpha=2.0)
    rigid_energy = rigid_body.point_energy(pair_energy, shape)
    return quantity.force(rigid_energy), None


def _total_force(interaction_force_fn, drive_fn):
    """Combine steric interaction + active drive -> ``force_fn(body, **kwargs)``."""
    def force_fn(body, **kwargs):
        f_int = interaction_force_fn(body, **kwargs)   # kwargs carries neighbor=...
        f_lab, tau = drive_fn(body)
        return rigid_body.RigidBody(f_int.center + f_lab,
                                    f_int.orientation + tau)
    return force_fn


def grid_initial_condition(n_particles, box, seed=2):
    """Non-overlapping grid of centres with random orientations.

    Returns ``(body0, key)``; ``key`` seeds the Brownian noise.
    """
    m = int(np.ceil(np.sqrt(n_particles)))
    cell = box / m
    coords = (np.arange(m) + 0.5) * cell
    gx, gy = np.meshgrid(coords, coords)
    centers = np.stack([gx.ravel(), gy.ravel()], axis=1)[:n_particles]
    centers = jnp.asarray(centers, dtype=jnp.float64)

    key = random.PRNGKey(seed)
    key, okey = random.split(key)
    theta = random.uniform(okey, (n_particles,), minval=-jnp.pi, maxval=jnp.pi,
                           dtype=jnp.float64)
    return rigid_body.RigidBody(centers, theta), key


def droplet_initial_condition(n_particles, side, packing=0.6, box=None,
                              margin_factor=0.5, seed=2):
    """A dense disk ("droplet") of particles centred in an otherwise empty box.

    The droplet's own surface is the interface that lets co-rotating spinners
    drive a rim current -> collective rotation. ``packing`` is the area fraction
    inside the disk; the box is sized with empty margin around it (unless ``box``
    is given). Returns ``(body0, key, box)``.
    """
    spacing = side / np.sqrt(packing)               # square lattice at `packing`
    m = int(np.ceil(np.sqrt(n_particles))) + 4
    xs = (np.arange(m) - (m - 1) / 2) * spacing
    gx, gy = np.meshgrid(xs, xs)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    pts = pts[np.argsort(np.sum(pts ** 2, axis=1))[:n_particles]]   # N nearest -> disk
    r_drop = float(np.sqrt(np.max(np.sum(pts ** 2, axis=1))))
    if box is None:
        box = 2.0 * r_drop * (1.0 + margin_factor)  # empty margin around droplet
    centers = jnp.asarray(pts + box / 2.0, dtype=jnp.float64)

    key = random.PRNGKey(seed)
    key, okey = random.split(key)
    theta = random.uniform(okey, (n_particles,), minval=-jnp.pi, maxval=jnp.pi,
                           dtype=jnp.float64)
    return rigid_body.RigidBody(centers, theta), key, float(box)


# --------------------------------------------------------------------------
# main driver
# --------------------------------------------------------------------------
def run_swarm(
    n_particles=20,
    side=10e-6,
    n_grid=10,
    epsilon=5e-17,
    F_body=DEFAULT_F_BODY,
    tau=DEFAULT_TAU,
    area_fraction=0.165,
    box=None,
    init="grid",
    droplet_packing=0.6,
    n_steps=100_000,
    dt=1e-4,
    record_every=100,
    kT=KT_300K,
    resistance_body=DEFAULT_RESISTANCE_BODY,
    seed=2,
    use_neighbor_list=True,
    shape=None,
    sigma=None,
    drive_fn=None,
    label=None,
    save=True,
    plot=True,
    video=False,
    results_dir=RESULTS_DIR,
):
    """Simulate ``n_particles`` interacting rigid metavehicles and return the run.

    Returns a dict with ``centers`` (frames, N, 2) [m], ``angles`` (frames, N)
    [rad], ``box`` [m] and a ``meta`` dict. When ``save`` is True the run is
    written to ``results/<timestamp>_<label>/`` (trajectory.npz + metadata.json
    + snapshot.png), mirroring the single-particle ``simulate_and_save``.

    The common knobs are scalars (geometry, drive, counts, sim params). For a
    non-square particle pass ``shape`` + ``sigma``; for an orientation-dependent
    drive pass ``drive_fn(body) -> (f_lab (N,2), tau (N,))``.
    """
    # geometry
    if shape is None:
        shape, sigma = square_shape(side, n_grid)
    elif sigma is None:
        raise ValueError("pass `sigma` (interaction cutoff) with a custom `shape`")
    n_beads = int(np.asarray(shape.points).shape[0])

    # initial condition + periodic box
    if init == "droplet":
        body0, key, box = droplet_initial_condition(
            n_particles, side, packing=droplet_packing, box=box, seed=seed)
    else:
        if box is None:                             # density-matched (fixed phi)
            box = side * float(np.sqrt(n_particles / area_fraction))
        body0, key = grid_initial_condition(n_particles, box, seed)
    displacement_fn, shift_fn = space.periodic(box)

    custom_drive = drive_fn is not None
    if drive_fn is None:
        drive_fn = make_drive(F_body, tau)
    interaction_force_fn, nbr_fns = _build_interaction(
        shape, sigma, epsilon, displacement_fn, box, use_neighbor_list)
    force_fn = _total_force(interaction_force_fn, drive_fn)

    init_fn, step_fn = brownian_generalized_rigid_2d(
        energy_or_force_fn=force_fn, shift_fn=shift_fn, dt=dt, kT=kT,
        resistance_body=resistance_body)

    n_record = max(1, n_steps // record_every)
    overflowed = False

    if use_neighbor_list:
        @jit
        def run(state, nbrs):
            def inner(carry, _):
                s, nb, ovf = carry
                nb = nb.update(s.position)
                ovf = ovf | (nb.did_buffer_overflow != 0)
                s = step_fn(s, neighbor=nb)
                return (s, nb, ovf), None

            def outer(carry, _):
                carry, _ = lax.scan(inner, carry, xs=None, length=record_every)
                return carry, carry[0].position

            (state, nbrs, ovf), traj = lax.scan(
                outer, (state, nbrs, jnp.array(False)),
                xs=None, length=n_record)
            return state, traj, ovf

        extra_cap = 0
        while True:                         # grow the Verlet buffer if it overflows
            state = init_fn(key, body0)
            nbrs = nbr_fns.allocate(body0, extra_capacity=extra_cap)
            state, traj, ovf = run(state, nbrs)
            jax.block_until_ready(traj.center)
            if not bool(ovf):
                break
            extra_cap += 16
            if extra_cap > 256:
                overflowed = True
                print("WARNING: neighbor buffer still overflowing; results suspect")
                break
    else:
        @jit
        def run(state):
            def inner(s, _):
                return step_fn(s), None

            def outer(s, _):
                s, _ = lax.scan(inner, s, xs=None, length=record_every)
                return s, s.position

            return lax.scan(outer, state, xs=None, length=n_record)

        state = init_fn(key, body0)
        state, traj = run(state)
        jax.block_until_ready(traj.center)

    centers = np.asarray(traj.center)        # (frames, N, 2)
    angles = np.asarray(traj.orientation)    # (frames, N)

    meta = {
        "n_particles": int(n_particles),
        "n_beads_per_body": n_beads,
        "side_m": float(side), "n_grid": int(n_grid), "sigma_m": float(sigma),
        "epsilon_J": float(epsilon),
        "drive": {"type": "custom" if custom_drive else "constant_body",
                  "F_body_N": [float(f) for f in F_body], "tau_Nm": float(tau)},
        "box_m": float(box), "area_fraction": float(area_fraction),
        "init": init, "droplet_packing": float(droplet_packing),
        "n_steps": int(n_steps), "dt": float(dt), "record_every": int(record_every),
        "n_frames": int(centers.shape[0]),
        "kT": float(kT), "temperature_K": float(kT) / K_B,
        "resistance_body": np.asarray(resistance_body).tolist(),
        "seed": int(seed), "use_neighbor_list": bool(use_neighbor_list),
        "neighbor_buffer_overflow": bool(overflowed),
        "jax_version": jax.__version__, "git_commit": _git_commit(),
    }
    out = {"centers": centers, "angles": angles, "box": float(box),
           "side": float(side), "sigma": float(sigma), "meta": meta}

    print(f"swarm: {n_steps} steps x {n_particles} bodies ({n_beads} beads each) "
          f"-> {centers.shape[0]} frames; min gap (final) "
          f"{_min_gap(centers[-1], box) * 1e6:.1f} um (edge {side*1e6:.0f} um)")
    if save:
        run_dir = _save_run(out, label, results_dir, plot)
        if video:
            animate_swarm(out, out_path=str(run_dir / "animation.mp4"))
    elif video:
        animate_swarm(out, out_path="swarm_animation.mp4")
    return out


def _min_gap(centers, box):
    """Minimum centre-centre distance (minimum-image) -- overlap diagnostic."""
    d = centers[:, None, :] - centers[None, :, :]
    d -= box * np.round(d / box)
    dist = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


# --------------------------------------------------------------------------
# validation: neighbor-list forces must equal all-pairs; net force ~ 0
# --------------------------------------------------------------------------
def check_forces(n_particles=9, side=10e-6, n_grid=10, epsilon=5e-17,
                 F_body=DEFAULT_F_BODY, tau=DEFAULT_TAU, seed=2,
                 spacing_factor=0.6):
    """Validate the interaction: NL == all-pairs, and Newton's third law.

    Returns a dict of the discrepancies. The squares are packed at
    ``spacing_factor * side < side`` so they actually overlap -- otherwise every
    interaction force is trivially zero and the check is vacuous.
    """
    shape, sigma = square_shape(side, n_grid)
    m = int(np.ceil(np.sqrt(n_particles)))
    spacing = spacing_factor * side
    box = m * spacing + 4 * side                    # roomy: no image interaction
    coords = (np.arange(m) + 0.5) * spacing + 2 * side
    gx, gy = np.meshgrid(coords, coords)
    centers = jnp.asarray(np.stack([gx.ravel(), gy.ravel()], 1)[:n_particles],
                          dtype=jnp.float64)
    theta = random.uniform(random.PRNGKey(seed), (n_particles,),
                           minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float64)
    body0 = rigid_body.RigidBody(centers, theta)
    displacement_fn, _ = space.periodic(box)
    drive_fn = make_drive(F_body, tau)

    f_pair_int, _ = _build_interaction(shape, sigma, epsilon, displacement_fn,
                                       box, use_neighbor_list=False)
    f_nl_int, nbr_fns = _build_interaction(shape, sigma, epsilon, displacement_fn,
                                           box, use_neighbor_list=True)
    nbrs = nbr_fns.allocate(body0).update(body0)

    fp = _total_force(f_pair_int, drive_fn)(body0)
    fn = _total_force(f_nl_int, drive_fn)(body0, neighbor=nbrs)
    dF = float(jnp.max(jnp.abs(fp.center - fn.center)))
    dT = float(jnp.max(jnp.abs(fp.orientation - fn.orientation)))

    # interaction forces are internal -> they must sum (vectorially) to ~0
    fint = f_pair_int(body0)
    net = float(jnp.max(jnp.abs(jnp.sum(fint.center, axis=0))))
    scale = float(jnp.max(jnp.abs(fint.center)))

    return {"nl_vs_pair_force": dF, "nl_vs_pair_torque": dT,
            "net_interaction_force": net, "interaction_force_scale": scale,
            "buffer_overflow": bool(nbrs.did_buffer_overflow)}


# --------------------------------------------------------------------------
# saving + a minimal snapshot (the heavy poster/animation render stays in the
# notebooks; this is just a quick look)
# --------------------------------------------------------------------------
def _save_run(out, label, results_dir, plot):
    from pathlib import Path
    from datetime import datetime

    now = datetime.now()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{label}" if label else stamp
    run_dir = Path(results_dir) / f"swarm_{name}"
    i = 1
    while run_dir.exists():
        run_dir = Path(results_dir) / f"swarm_{name}-{i}"
        i += 1
    run_dir.mkdir(parents=True)

    out["meta"]["datetime"] = now.isoformat(timespec="seconds")
    # `side` and `sigma` go in the npz too, so it can be rendered on its own.
    np.savez_compressed(run_dir / "trajectory.npz", centers=out["centers"],
                        angles=out["angles"], box=out["box"],
                        side=out["side"], sigma=out["sigma"])
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(out["meta"], f, indent=2, default=_json_default)
    if plot:
        quick_plot_swarm(out, out_path=str(run_dir / "snapshot.png"))
    print(f"saved run to {run_dir}")
    return run_dir


def quick_plot_swarm(out, out_path=None, trail_frames=300, show=False):
    """Minimal snapshot: final oriented squares + recent centre trails."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    centers = out["centers"] * 1e6
    angles = out["angles"]
    box = out["box"] * 1e6
    side = out["side"] * 1e6
    N = centers.shape[1]
    cmap = plt.get_cmap("turbo")

    base = 0.5 * side * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for i in range(N):
        c = cmap(i / max(N - 1, 1))
        tr = centers[max(0, len(centers) - trail_frames):, i, :].astype(float)
        tr[np.abs(np.diff(tr, axis=0, prepend=tr[:1])).max(1) > box / 2] = np.nan
        ax.plot(tr[:, 0], tr[:, 1], color=c, lw=1.0, alpha=0.8)
        th = angles[-1, i]
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        verts = base @ rot.T + centers[-1, i]
        ax.add_patch(Polygon(verts, closed=True, facecolor=c, edgecolor="k",
                             linewidth=0.8, alpha=0.6))
    ax.set_aspect("equal")
    ax.set_xlim(0, box); ax.set_ylim(0, box)
    ax.set_xlabel("x  (µm)"); ax.set_ylabel("y  (µm)")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"saved {out_path}")
    if show:
        return fig
    plt.close(fig)
    return None


# --------------------------------------------------------------------------
# video: render an mp4/gif from a run (the out dict, a run folder, or a .npz)
# --------------------------------------------------------------------------
def load_run(run_dir):
    """Load a saved run (trajectory.npz [+ metadata.json]) into a run dict."""
    from pathlib import Path
    run_dir = Path(run_dir)
    d = np.load(run_dir / "trajectory.npz")
    out = {"centers": d["centers"], "angles": d["angles"], "box": float(d["box"])}
    if "side" in d:
        out["side"] = float(d["side"])
    meta_p = run_dir / "metadata.json"
    if meta_p.exists():
        out["meta"] = json.loads(meta_p.read_text())
        out.setdefault("side", out["meta"].get("side_m"))
    return out


def _as_run(run):
    """Normalize a dict / run-dir path / npz path into a run dict."""
    if isinstance(run, dict):
        return run
    from pathlib import Path
    p = Path(run)
    if p.is_dir():
        return load_run(p)
    d = np.load(p)
    side = float(d["side"]) if "side" in d.files else None
    if side is None:
        raise ValueError("this .npz has no `side`; pass the run folder instead")
    return {"centers": d["centers"], "angles": d["angles"],
            "box": float(d["box"]), "side": side}


def _pick_writer(out_path):
    """Choose mp4 (ffmpeg, pointed at the env binary) or fall back to gif."""
    import sys
    import matplotlib
    from pathlib import Path
    from matplotlib.animation import writers

    cand = Path(sys.executable).parent / "ffmpeg"      # ffmpeg next to this python
    if cand.exists():
        matplotlib.rcParams["animation.ffmpeg_path"] = str(cand)

    p = Path(out_path)
    if p.suffix.lower() == ".mp4":
        if writers.is_available("ffmpeg"):
            return str(p), "ffmpeg"
        p = p.with_suffix(".gif")
        print("ffmpeg unavailable -> writing GIF instead")
    return str(p), "pillow"


def animate_swarm(run, out_path="swarm_animation.mp4", target_frames=240,
                  trail_frames=80, fps=24, draw_ghosts=True, title=None):
    """Render a time-lapse video of a swarm run.

    ``run`` may be the dict returned by ``run_swarm``, a saved run folder, or a
    ``trajectory.npz`` path -- so you can render any old run without re-simulating.
    Periodic copies ("ghosts") are drawn so bodies crossing the box edge appear
    on both sides; trails are broken where they wrap. Writes mp4 if ffmpeg is
    available, otherwise a gif.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection, LineCollection
    from matplotlib.animation import FuncAnimation

    run = _as_run(run)
    centers = np.asarray(run["centers"]) * 1e6        # (T, N, 2) um
    angles = np.asarray(run["angles"])                # (T, N)
    box = run["box"] * 1e6
    side = run["side"] * 1e6
    T_all, N, _ = centers.shape

    skip = max(1, T_all // target_frames)
    C = centers[::skip]
    A = angles[::skip]
    T = C.shape[0]

    cmap = plt.get_cmap("turbo")
    face = np.array([cmap(i / max(N - 1, 1)) for i in range(N)])
    base = 0.5 * side * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    ghosts = (np.array([[dx, dy] for dx in (-box, 0, box) for dy in (-box, 0, box)])
              if draw_ghosts else np.zeros((1, 2)))
    G = len(ghosts)

    def frame_geometry(f):
        cos, sin = np.cos(A[f]), np.sin(A[f])
        R = np.stack([np.stack([cos, -sin], -1), np.stack([sin, cos], -1)], -2)  # (N,2,2)
        rb = np.einsum("kij,pj->kpi", R, base)                 # (N,4,2)
        cg = C[f][:, None, :] + ghosts[None, :, :]             # (N,G,2)
        verts = (rb[:, None, :, :] + cg[:, :, None, :]).reshape(-1, 4, 2)
        front = np.einsum("kij,j->ki", R, [side / 2, 0.0]) + C[f]   # (N,2)
        fg = front[:, None, :] + ghosts[None, :, :]            # (N,G,2)
        heads = np.stack([cg, fg], axis=2).reshape(-1, 2, 2)   # (N*G,2,2)
        return verts, heads

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal"); ax.set_xlim(0, box); ax.set_ylim(0, box)
    ax.set_xlabel("x  (µm)"); ax.set_ylabel("y  (µm)")

    verts0, heads0 = frame_geometry(0)
    squares = PolyCollection(verts0, facecolors=np.repeat(face, G, axis=0),
                             edgecolors="k", linewidths=0.7, alpha=0.6, zorder=5)
    headlc = LineCollection(heads0, colors="k", linewidths=1.0, zorder=6)
    ax.add_collection(squares); ax.add_collection(headlc)
    trails = [ax.plot([], [], color=face[i], lw=1.0, alpha=0.8, zorder=3)[0]
              for i in range(N)]
    ttl = ax.set_title(title or "")

    def update(f):
        verts, heads = frame_geometry(f)
        squares.set_verts(verts)
        headlc.set_segments(heads)
        lo = max(0, f - trail_frames)
        for i in range(N):
            tr = C[lo:f + 1, i, :].astype(float).copy()
            if len(tr) > 1:                       # break the trail where it wraps
                brk = np.where(np.abs(np.diff(tr, axis=0)).max(1) > box / 2)[0]
                if len(brk):
                    tr = np.insert(tr, brk + 1, np.nan, axis=0)
            trails[i].set_data(tr[:, 0], tr[:, 1])
        ttl.set_text((title + "  " if title else "") + f"frame {f + 1}/{T}")
        return []

    out_path, writer = _pick_writer(out_path)
    ani = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    ani.save(out_path, writer=writer, fps=fps, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")
    return out_path


if __name__ == "__main__":
    print("demo run + video...")
    run_swarm(n_particles=30, n_steps=100_000, label="demo_circular", video=True)
