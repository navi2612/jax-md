"""Light-driven particles in a finite arena, as in the experiment.

No periodic images that matter: the simulated square ``fov`` is the camera field
of view, and particles are kept in it by two independent things, both centred:

  light spot  radius ``r_light``  -- outside it the intensity (and so the force
              and torque) goes to zero; particles there are passive and, at
              D ~ 0.01 um^2/s, effectively frozen.
  wall        radius ``r_wall``   -- a soft circular wall on the body centre,
              so the particle number is conserved.

``r_light == r_wall``  clean arena, every particle always active.
``r_light <  r_wall``  the experimental beam edge: a dark rim where particles
                       stop and pile up; the wall only bounds how far they go.

Internally the box is still periodic (that is what ``swarm.make_runner``
builds), but the wall sits at least one body diameter inside the box edge, so
no two particles ever interact through the boundary.

Edit the CONFIG block in ``__main__`` and run:

    /opt/miniconda3/envs/pmeep/bin/python arena.py
"""
import numpy as np
import jax.numpy as jnp
from jax import random
from jax_md import rigid_body
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from swarm import (make_runner, make_drive, make_active_optical_drive,
                   single_bead_shape, grid_shape, perimeter_shape, shape_radius)
from trajectory import DEFAULT_RESISTANCE_BODY
from runs import save_run
from render_swarm import make_video


# --------------------------------------------------------------------------
# light spot and wall: wrappers around any drive_fn(body, light)
# --------------------------------------------------------------------------
def intensity_spot(center, radius, edge=2e-6):
    """Top-hat beam of ``radius`` with a smooth edge of width ~``edge`` [m].

    1 inside, 0 outside. A tanh edge instead of a step keeps the force smooth,
    so particles crossing the rim do not see a kick of one full dt.
    """
    center = jnp.asarray(center)

    def fn(c):
        r = jnp.linalg.norm(c - center, axis=1)
        return 0.5 * (1.0 - jnp.tanh((r - radius) / edge))

    fn.spec = {"pattern": "spot", "center_m": [float(x) for x in center],
               "radius_m": float(radius), "edge_m": float(edge)}
    return fn


def with_light_pattern(drive_fn, intensity_fn):
    """Scale force AND torque of any drive by the local intensity I(x, y).

    Works for every drive that is linear in intensity (uniform, linear
    polarization, ...), unlike ``swarm.make_modulated_drive`` which only wraps
    the uniform one.
    """
    def wrapped(body, light):
        f_lab, tau = drive_fn(body, light)
        s = intensity_fn(body.center)
        return f_lab * s[:, None], tau * s

    wrapped.spec = {"type": "light_pattern",
                    "intensity": getattr(intensity_fn, "spec", "custom"),
                    "drive": getattr(drive_fn, "spec", "custom")}
    return wrapped


def with_circular_wall(drive_fn, center, radius, shape_points, bead_radius, k=1e-4):
    """Soft circular wall acting on every bead: each bead surface stays inside
    ``radius``, pushed radially inward with stiffness ``k`` [N/m] per bead.

    ``shape_points`` are the body-frame bead centres in metres, (B, 2). Because
    the wall pushes the beads, not the centre, an extended body gets a torque
    from it (a square hitting the wall corner-first turns face-on); for a single
    bead this is the plain centre wall.
    """
    center = jnp.asarray(center)
    pts = jnp.asarray(shape_points)
    r_max = float(radius) - float(bead_radius)

    def wrapped(body, light):
        f_lab, tau = drive_fn(body, light)
        R = rigid_body.rotation2d(body.orientation)               # (N,2,2)
        arm = jnp.einsum("nij,bj->nbi", R, pts)                   # (N,B,2)
        d = body.center[:, None, :] + arm - center
        r = jnp.linalg.norm(d, axis=-1)
        push = k * jnp.maximum(r - r_max, 0.0) / jnp.maximum(r, 1e-12)
        f_bead = -push[..., None] * d                              # (N,B,2)
        t_bead = arm[..., 0] * f_bead[..., 1] - arm[..., 1] * f_bead[..., 0]
        return f_lab + f_bead.sum(1), tau + t_bead.sum(1)

    wrapped.spec = {"type": "circular_wall", "center_m": [float(x) for x in center],
                    "radius_m": float(radius), "bead_radius_m": float(bead_radius),
                    "k_N_per_m_per_bead": float(k),
                    "drive": getattr(drive_fn, "spec", "custom")}
    return wrapped


# --------------------------------------------------------------------------
# initial condition: random, non-overlapping, inside a circle
# --------------------------------------------------------------------------
def make_arena_init(center, radius, body_radius, margin=0.05, stages=40,
                    iters_per_stage=400):
    """Random centres inside ``radius - body_radius``, no overlaps (closer than
    ``2*body_radius*(1+margin)``), random headings.

    ``body_radius`` is the circumradius, so an extended body is kept apart by
    its circumscribed circle: fine for disks, but squares only pack up to
    phi ~ 0.5 this way (tested for grid_shape(3)).

    Placing particles one by one and rejecting overlaps jams at phi ~ 0.55, so
    instead all are dropped at random and then grown from 60% to full size,
    pushing overlapping pairs apart and pulling escapees back into the circle
    at every step. Works up to phi ~ 0.7.
    """
    center = np.asarray(center, dtype=float)
    dmin_full = 2 * body_radius * (1 + margin)
    r_max = radius - body_radius

    def init(n, box, seed):
        rng = np.random.default_rng(seed)
        r = r_max * np.sqrt(rng.uniform(size=n))          # uniform in the disk
        a = rng.uniform(0, 2 * np.pi, n)
        pos = center + r[:, None] * np.c_[np.cos(a), np.sin(a)]

        def relax(dmin, iters):
            nonlocal pos
            for _ in range(iters):
                pairs = cKDTree(pos).query_pairs(dmin, output_type="ndarray")
                if len(pairs) == 0:
                    return True
                dd = pos[pairs[:, 0]] - pos[pairs[:, 1]]
                d = np.maximum(np.hypot(*dd.T), 1e-12)
                push = 0.5 * (1.01 * dmin - d)[:, None] * dd / d[:, None]
                disp = np.zeros_like(pos)
                np.add.at(disp, pairs[:, 0], push)
                np.add.at(disp, pairs[:, 1], -push)
                pos = pos + disp
                rel = pos - center                        # back inside the circle
                rr = np.maximum(np.hypot(*rel.T), 1e-12)
                pos = center + rel * np.minimum(1.0, r_max / rr)[:, None]
            return False

        for g in np.linspace(0.6, 1.0, stages):
            relax(g * dmin_full, iters_per_stage)
        if not relax(dmin_full, 20 * iters_per_stage):
            raise RuntimeError(f"could not remove all overlaps for n={n}; "
                               "lower phi or enlarge the start radius")
        theta = rng.uniform(-np.pi, np.pi, n)
        return (rigid_body.RigidBody(jnp.asarray(pos), jnp.asarray(theta)),
                random.PRNGKey(seed))

    init.__name__ = "arena_init"
    return init


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def torque_for_radius(R_T, F, resistance_body=DEFAULT_RESISTANCE_BODY):
    """Torque giving turning radius R_T for propulsion force F."""
    g_t, g_r = float(resistance_body[0][0]), float(resistance_body[2][2])
    return F * g_r / (g_t * R_T)


def turning_radius(F, tau, resistance_body=DEFAULT_RESISTANCE_BODY):
    g_t, g_r = float(resistance_body[0][0]), float(resistance_body[2][2])
    return np.inf if tau == 0 else F * g_r / (g_t * abs(tau))


def body_area(shape, sigma):
    """Footprint area [m^2]: convex hull of the bead centres grown by one bead
    radius (exact for disks; a square of beads gets slightly rounded corners)."""
    from scipy.spatial import ConvexHull
    pts = np.unique(np.asarray(shape.points) * sigma, axis=0)
    a = np.pi * (sigma / 2) ** 2
    if len(pts) == 1:
        return a
    if len(pts) == 2 or np.linalg.matrix_rank(pts - pts[0]) < 2:   # a rod
        return a + sigma * np.ptp(np.linalg.norm(pts - pts[0], axis=1))
    hull = ConvexHull(pts)                     # 2D: .volume = area, .area = perimeter
    return hull.volume + hull.area * sigma / 2 + a


def n_for_phi(phi, radius, area):
    """Particles of footprint ``area`` giving area fraction ``phi`` inside a
    circle of ``radius``."""
    return int(round(phi * np.pi * radius ** 2 / area))


def summarize(centers, angles, center, r_light, d_contact):
    """A few numbers per frame: active fraction (inside the light), mean
    distance from the centre / r_light, largest contact cluster, polar order."""
    rows = []
    for c, th in zip(centers, angles):
        r = np.hypot(*(c - center).T)
        pairs = cKDTree(c).query_pairs(d_contact, output_type="ndarray")
        n = len(c)
        if len(pairs):
            adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                             shape=(n, n))
            _, lab = connected_components(adj, directed=False)
            big = np.bincount(lab).max() / n
        else:
            big = 1 / n
        rows.append((np.mean(r < r_light), np.mean(r) / r_light, big,
                     np.abs(np.mean(np.exp(1j * th)))))
    return np.array(rows)          # (T, 4)


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------
def run_arena(drive, label, *, fov, r_light, r_wall, n, shape, sigma, epsilon,
              T_total, t_sample, dt, seed=0, extra_capacity=16, video=True):
    """Confine ``drive`` to the spot + wall, simulate, save, render."""
    center = (fov / 2, fov / 2)
    body_r = shape_radius(shape, sigma)
    # The wall keeps every bead surface inside r_wall, so beads and their
    # periodic images stay >= sigma + fov - 2*r_wall apart -- more than the
    # sigma cutoff whenever r_wall < fov/2. The sigma/2 is headroom for beads
    # pressed into the soft wall.
    if r_wall > fov / 2 - sigma / 2:
        raise ValueError(f"r_wall must be <= fov/2 - sigma/2 = "
                         f"{(fov / 2 - sigma / 2) * 1e6:.1f} um, or particles "
                         "interact through the periodic edge")

    drive = with_circular_wall(with_light_pattern(drive, intensity_spot(center, r_light)),
                               center, r_wall, np.asarray(shape.points) * sigma,
                               sigma / 2)
    init = make_arena_init(center, min(r_light, r_wall), body_r)

    steps_per_sample = int(round(t_sample / dt))
    u = np.ones(int(round(T_total / t_sample)))      # constant intensity for now

    runner = make_runner(n_particles=n, sigma=sigma, box=fov, shape=shape,
                         drive_fn=drive, init=init, epsilon=epsilon, dt=dt,
                         steps_per_sample=steps_per_sample,
                         capacity_multiplier=5.0, initial_extra_capacity=extra_capacity,
                         progress=True)
    centers, angles, box = runner(u, seed=seed)

    run_dir = save_run(
        arrays={"centers": centers, "angles": angles, "light": u, "box": box,
                "sigma": sigma, "body_points": runner.shape_points},
        meta={**runner.meta, "n_samples": len(u), "seed": seed,
              "fov_m": fov, "r_light_m": r_light, "r_wall_m": r_wall},
        label=label, prefix="arena", source=__file__)

    d_eq = np.sqrt(4 * body_area(shape, sigma) / np.pi)     # equal-area diameter
    s = summarize(centers, angles, np.asarray(center), r_light, 1.1 * d_eq)
    q = max(1, len(s) // 4)
    print(f"\n{label}: n={n}, body radius {body_r*1e6:.1f} um, "
          f"T={T_total:g} s -> {run_dir}")
    print("            active   <r>/r_light   largest cluster   polar order")
    for name, part in (("first 1/4", s[:q]), ("last 1/4", s[-q:])):
        print(f"{name:>10}   " + "   ".join(f"{v:9.3f}" for v in part.mean(0)))

    if video:
        make_video(run_dir, out_path=str(run_dir / "animation.mp4"),
                   draw_ghosts=False, target_frames=min(len(u), 600), fps=24,
                   color_by="orientation")
    return run_dir


if __name__ == "__main__":
    # ===== CONFIG ===========================================================
    # --- arena ---------------------------------------------------------------
    fov     = 600e-6              # field of view = simulated square [m]
    r_light = 220e-6              # laser spot radius (homogeneous top hat)
    r_wall  = 290e-6              # wall radius (> r_light -> dark rim like the experiment)

    # --- particles -----------------------------------------------------------
    side    = 10e-6               # particle size: disk diameter / square side
    # shape, sigma = single_bead_shape(), side
    shape, sigma = grid_shape(3), side / 3           # filled square of 3x3 beads
    # shape, sigma = perimeter_shape(3, 0.5), side / 3  # hollow square, smoother faces
    epsilon = 5e-17 * (sigma / 1e-6) ** 2
    phi     = 0.3                # area fraction inside the spot
    n       = n_for_phi(phi, r_light, body_area(shape, sigma))

    # --- time ----------------------------------------------------------------
    T_total  = 1200.0              # [s]
    t_sample = 0.5                # saved frame spacing [s]
    dt       = 4e-4

    # --- drives: pick one, or loop over several -----------------------------
    F = 1.135e-12
    DRIVES = {
        # chiral circle swimmers, turning radius set directly
        "circ_RT20": lambda: make_drive(F_body=(F, 0.0), tau=torque_for_radius(20e-6, F)),
        "circ_RT80": lambda: make_drive(F_body=(F, 0.0), tau=torque_for_radius(80e-6, F)),
        "circ_RT40": lambda: make_drive(F_body=(F, 0.0), tau=torque_for_radius(40e-6, F)),
        "circ_RT140": lambda: make_drive(F_body=(F, 0.0), tau=torque_for_radius(140e-6, F)),
        # straight swimmers (no torque) -> only noise turns them
        "straight":  lambda: make_drive(F_body=(F, 0.0), tau=0.0),
        # linear polarization along x (FDTD fit, design1_res60_linear_30uW)
        "linear_x":  lambda: make_active_optical_drive(
            [1.24171, 0.443034, 0.0], [0.0, 0.0, 0.066536], [0, 0, 244.48],
            phi_pol=0.0, power=3),
    }
    RUN = ["circ_RT20"]           # e.g. list(DRIVES) for all of them
    # ========================================================================

    for name in RUN:
        run_arena(DRIVES[name](), name, fov=fov, r_light=r_light, r_wall=r_wall,
                  n=n, shape=shape, sigma=sigma, epsilon=epsilon,
                  T_total=T_total, t_sample=t_sample, dt=dt)
