"""Bead contacts for 2D rigid bodies: a body is a union of soft-sphere beads.

Same interface as ``spheropolygon.make_interaction``: ``interaction_fn(shape,
sigma, box, displacement_fn) -> (force_fn, neighbor_fns)``, so it plugs into
``swarm.make_runner``. Bumpy bead outlines act like contact friction; one bead
per body is a disk.

    python beads.py                # the ring of disks in __main__ below
"""
import numpy as np
import jax.numpy as jnp
from jax import random
from jax_md import energy, partition, quantity, rigid_body

from render_swarm import make_video
from runs import save_run
from swarm import make_active_optical_drive, make_drive, make_runner, single_bead_shape


# --------------------------------------------------------------------------
# particle geometry
# --------------------------------------------------------------------------
def grid_shape(n_grid=10):
    """A filled square rigid body: ``n_grid x n_grid`` beads.

    Bead centres are laid out so the bead *surfaces* reach exactly to the square
    edge, i.e. for a body of side ``side`` pass ``sigma = side / n_grid`` to
    ``make_runner``: steric repulsion then switches on as two drawn squares
    touch. Unlike ``single_bead_shape``, contacts here have a lever arm, so
    collisions produce torque and bodies tumble.
    """
    # positions are in units of sigma; make_runner scales them by sigma.
    span = n_grid - 1.0
    xs = jnp.linspace(-span / 2, span / 2, n_grid)
    gx, gy = jnp.meshgrid(xs, xs)
    pts = jnp.stack([gx.ravel(), gy.ravel()], axis=1).astype(jnp.float64)
    return rigid_body.point_union_shape(pts, jnp.ones(pts.shape[0]))


def perimeter_shape(side, spacing=0.25):
    """A hollow square rigid body: overlapping beads on the perimeter only.

    Same footprint as ``grid_shape(side)`` (``side`` is in units of sigma, bead
    surfaces reach the square edge), but the interior is left empty -- only the
    outer beads ever touch another body -- and neighbouring beads are
    ``spacing`` (< 1, in units of sigma) apart, so they overlap and the edge is
    smooth. Touching beads (``grid_shape``) mesh like gear teeth: sliding two
    faces past each other takes ~0.5 of the normal force, against ~0.2 at
    ``spacing=0.5`` and ~0.1 at ``spacing=0.25``.

    More beads touch per length of edge, so a contact is stiffer; pass
    ``epsilon * spacing`` to keep it about as stiff as with ``grid_shape``. Start
    from ``grid_initial_condition``: a body that begins inside another stays there.
    """
    if side <= 1:
        raise ValueError(f"side must be > 1 (in units of sigma), got {side}")
    h = (side - 1.0) / 2.0                     # bead-centre half-width
    n = int(np.ceil(2 * h / spacing)) + 1
    t = np.linspace(-h, h, n)
    full, rest = np.full(n, h), np.full(n - 2, h)
    pts = np.concatenate([np.stack([t, -full], 1), np.stack([t, full], 1),
                          np.stack([-rest, t[1:-1]], 1), np.stack([rest, t[1:-1]], 1)])
    pts = jnp.asarray(pts, dtype=jnp.float64)
    return rigid_body.point_union_shape(pts, jnp.ones(pts.shape[0]))


def make_ring_init(r0, beta, diameter, center=None):
    """N disks on a ring of radius r0, heading = ring angle + beta."""
    def init(n, box, seed):
        c = np.full(2, box / 2) if center is None else np.asarray(center)
        psi = 2 * np.pi * np.arange(n) / n
        pos = c + r0 * np.c_[np.cos(psi), np.sin(psi)]
        if 2 * r0 * np.sin(np.pi / n) < diameter:
            raise ValueError("Disks overlap on the initial ring; increase r0 or reduce n.")
        body = rigid_body.RigidBody(jnp.asarray(pos), jnp.asarray(psi + beta))
        return body, random.PRNGKey(seed)
    return init


# --------------------------------------------------------------------------
# the interaction: interaction_fn(shape, sigma, box, displacement_fn)
#                      -> (interaction_force_fn, nbr_fns or None)
# --------------------------------------------------------------------------
def soft_sphere_interaction(epsilon=5e-17, alpha=2.0, dr_threshold=None,
                            capacity_multiplier=2.0):
    """Soft-sphere steric repulsion through a Verlet neighbor list.

    The cutoff is ``sigma`` (soft_sphere is exactly 0 beyond it), so this gives
    the *same* forces as the all-pairs version -- only the cost scaling differs.
    ``dr_threshold`` is the Verlet skin [m], defaulting to ``0.2 * sigma``; use a
    larger skin when bodies move fast (high intensity) so the list does not
    rebuild every step, and raise ``capacity_multiplier`` if the neighbor buffer
    overflows in clumping regimes.
    """
    def interaction_fn(shape, sigma, box, displacement_fn):
        skin = 0.2 * sigma if dr_threshold is None else dr_threshold
        neighbor_fn, ss_nl = energy.soft_sphere_neighbor_list(
            displacement_fn, box, sigma=sigma, epsilon=epsilon, alpha=alpha,
            dr_threshold=skin, capacity_multiplier=capacity_multiplier)
        # The cell list behind the neighbor list needs every bead inside
        # [0, box). Beads of a body that straddles a periodic edge are not, and
        # would silently lose their neighbors (bodies then pass through each
        # other), so bead positions are wrapped first; the displacement is
        # periodic, so energies and forces are unchanged.
        def wrap(R):
            return jnp.mod(R, box)

        wrapped_neighbor_fn = partition.NeighborListFns(
            lambda R, **kw: neighbor_fn.allocate(wrap(R), **kw),
            lambda R, nbrs, **kw: neighbor_fn.update(wrap(R), nbrs, **kw))
        nbr_fns, rigid_energy = rigid_body.point_energy_neighbor_list(
            lambda R, **kw: ss_nl(wrap(R), **kw), wrapped_neighbor_fn, shape)
        return quantity.force(rigid_energy), nbr_fns

    interaction_fn.spec = {"type": "soft_sphere_neighbor_list",
                           "epsilon_J": float(epsilon), "alpha": float(alpha),
                           "dr_threshold_m": dr_threshold,
                           "capacity_multiplier": float(capacity_multiplier)}
    return interaction_fn


def soft_sphere_all_pairs(epsilon=5e-17, alpha=2.0):
    """Soft-sphere steric repulsion, all pairs (no neighbor list).

    Same forces as ``soft_sphere_interaction`` but O(N^2); simpler and immune to
    neighbor-buffer overflow, so it is the safer choice for small N or when
    checking that a neighbor-list result is trustworthy.
    """
    def interaction_fn(shape, sigma, box, displacement_fn):
        pair_energy = energy.soft_sphere_pair(
            displacement_fn, sigma=sigma, epsilon=epsilon, alpha=alpha)
        rigid_energy = rigid_body.point_energy(pair_energy, shape)
        return quantity.force(rigid_energy), None

    interaction_fn.spec = {"type": "soft_sphere_all_pairs",
                           "epsilon_J": float(epsilon), "alpha": float(alpha)}
    return interaction_fn


if __name__ == "__main__":
    seed = 0

    # --- particle --------------------------------------------------------------
    sigma = 10e-6
    shape = single_bead_shape()
    epsilon = 5e-17 * (sigma / 1e-6) ** 2          # eps/sigma^2 = 5e-5 N/m per bead pair

    # --- swarm and box ---------------------------------------------------------
    N   = 115
    box = 150e-6

    # --- time ------------------------------------------------------------------
    T_total  = 400.0                  # simulated time [s]
    t_sample = 0.5                    # time between saved frames [s]
    dt       = 4e-4
    steps_per_sample = int(round(t_sample / dt))   # 1250
    u = np.ones(int(round(T_total / t_sample)))    # 800 samples
    light_signal = "constant 1.0"

    # --- drive -----------------------------------------------------------------
    FX = [1.24171, 0.443034, 0.0] #desing1_res60_linear_30uW
    FY = [0.0, 0.0, 0.066536]
    TZ = [0, 0, 244.48]
    # drive = make_active_optical_drive(FX, FY, TZ, phi_pol=0.0)
    F_body = (1.135e-12, 0.0)
    tau    = 966e-21
    drive = make_drive(F_body=F_body, tau=tau)
    init = make_ring_init(88e-6, np.sign(tau) * np.pi / 2, sigma)
    runner = make_runner(n_particles=N, sigma=sigma, box=box,
                         init=init, shape=shape, drive_fn=drive,
                         interaction_fn=soft_sphere_interaction(epsilon=epsilon,
                                                                capacity_multiplier=5.0),
                         dt=dt, steps_per_sample=steps_per_sample, progress=True)

    centers, angles, box = runner(u, seed=seed)
    print(f"centers {centers.shape}, angles {angles.shape}, box {box:.2e} m")

    run_dir = save_run(
        arrays={"centers": centers, "angles": angles, "light": u,
                "box": box, "sigma": sigma, "body_points": runner.shape_points},
        meta={**runner.meta, "n_samples": len(u), "seed": seed,
              "light_signal": light_signal},
        label="ring", prefix="swarm", source=__file__)

    fps = 24
    target_frames = round(len(u))

    make_video(run_dir, out_path=str(run_dir / "animation.mp4"), body="disk",
               target_frames=target_frames, fps=fps)
