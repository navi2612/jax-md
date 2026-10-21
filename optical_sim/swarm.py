import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random, lax, jit
from jax_md import space, energy, partition, rigid_body, quantity

from render_swarm import make_video
from runs import save_run

from brownian_generalized_2d import brownian_generalized_rigid_2d
from trajectory import KT_300K, K_B, DEFAULT_RESISTANCE_BODY
from active_optical import active_force_torque_body

# Body-frame optical drive per unit intensity (circular polarization).
DEFAULT_F_BODY = (1.135e-12, 0.045e-12)        # (Fx, Fy)  [N]
DEFAULT_TAU = 5*440.44e-21                        # torque    [N.m]


# --------------------------------------------------------------------------
# particle geometry
# --------------------------------------------------------------------------
def single_bead_shape():
    """One point at the origin -> each rigid body is a single disk."""
    pts = jnp.zeros((1, 2), dtype=jnp.float64)
    return rigid_body.point_union_shape(pts, jnp.ones(1))


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

def make_disk_lane_init(diameter, tilt_deg=0.0, margin=0.05, max_tries=200_000):
    """Random non-overlapping disks (periodic box), half pointing +x, half -x."""
    dmin = diameter * (1 + margin)

    def init(n_particles, box, seed):
        rng = np.random.default_rng(seed)
        signs = rng.permutation(np.r_[np.ones(n_particles // 2),
                                      -np.ones(n_particles - n_particles // 2)])
        centers, thetas, tries = [], [], 0
        for s in signs:
            while True:
                tries += 1
                if tries > max_tries:
                    raise RuntimeError("Could not place all disks; lower the density.")
                c = rng.uniform(0, box, 2)
                if centers:
                    d = np.array(centers) - c
                    d -= box * np.round(d / box)          # minimum image
                    if np.min(np.hypot(d[:, 0], d[:, 1])) < dmin:
                        continue
                centers.append(c)
                thetas.append((0.0 if s > 0 else np.pi)
                              + np.deg2rad(rng.uniform(-tilt_deg, tilt_deg)))
                break
        body = rigid_body.RigidBody(jnp.asarray(np.array(centers)),
                                    jnp.asarray(np.array(thetas)))
        return body, random.PRNGKey(seed)

    return init

def make_disk_mips_init(diameter):
    """Square lattice, random vacancies, small jitter, random headings."""
    def init(n, box, seed):
        rng = np.random.default_rng(seed)
        m = int(np.ceil(np.sqrt(n)))
        a = box / m                                   # lattice spacing
        if a < 1.05 * diameter:
            raise ValueError("Too dense for a square-lattice start.")
        ij = np.stack(np.meshgrid(np.arange(m), np.arange(m)), -1).reshape(-1, 2)
        ij = ij[rng.choice(m * m, n, replace=False)]
        jit = 0.5 * (a - 1.05 * diameter)
        pos = (ij + 0.5) * a + rng.uniform(-jit, jit, (n, 2)) / np.sqrt(2)
        theta = rng.uniform(-np.pi, np.pi, n)
        return rigid_body.RigidBody(jnp.asarray(pos), jnp.asarray(theta)), random.PRNGKey(seed)
    return init

def make_ring_init(r0, beta, center=None):
    """N disks on a ring of radius r0, heading = ring angle + beta."""
    def init(n, box, seed):
        c = np.full(2, box / 2) if center is None else np.asarray(center)
        psi = 2 * np.pi * np.arange(n) / n
        pos = c + r0 * np.c_[np.cos(psi), np.sin(psi)]
        if 2 * r0 * np.sin(np.pi / n) < sigma:
            raise ValueError("Disks overlap on the initial ring; increase r0 or reduce n.")
        body = rigid_body.RigidBody(jnp.asarray(pos), jnp.asarray(psi + beta))
        return body, random.PRNGKey(seed)
    return init


def random_initial_condition(n_particles, box, seed):
    """Random centers in the box and random orientations -> (body0, key). Overlap can occur"""
    key = random.PRNGKey(seed)
    key, ckey, okey = random.split(key, 3)
    centers = random.uniform(ckey, (n_particles, 2), minval=0.0, maxval=box,
                             dtype=jnp.float64)
    theta = random.uniform(okey, (n_particles,), minval=-jnp.pi, maxval=jnp.pi,
                           dtype=jnp.float64)
    return rigid_body.RigidBody(centers, theta), key


def grid_initial_condition(n_particles, box, seed):
    """Non-overlapping grid of centers with random orientations -> (body0, key).
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


# --------------------------------------------------------------------------
# the light: drive_fn(body, light) -> (f_lab (N,2), tau (N,))
# --------------------------------------------------------------------------
def make_drive(F_body=DEFAULT_F_BODY, tau=DEFAULT_TAU):
    """Spatially uniform light: every body sees the same intensity ``light``.

    The body-frame force is rotated into the lab frame; force and torque both
    scale linearly with ``light``, so an input series of intensities is encoded
    directly into the drive.
    """
    F_body = jnp.asarray(F_body, dtype=jnp.float64)
    tau = jnp.float64(tau)

    def drive_fn(body, light):
        R = rigid_body.rotation2d(body.orientation)        # (N,2,2) body->lab
        f_lab = jnp.einsum("nij,j->ni", R, F_body) * light
        tau_arr = jnp.full(body.orientation.shape, tau * light,
                           dtype=body.center.dtype)
        return f_lab, tau_arr

    drive_fn.spec = {"type": "uniform", "F_body_N": [float(f) for f in F_body],
                     "tau_Nm": float(tau)}
    return drive_fn


def make_modulated_drive(intensity_fn, F_body=DEFAULT_F_BODY, tau=DEFAULT_TAU):
    """Spatially modulated light: force AND torque scale with ``I(x, y)``.

    ``intensity_fn(centers)`` maps centers ``(N, 2)`` [m] to a non-negative
    local scaling ``(N,)`` -- an SLM pattern under the linear optical response.
    The scalar input sample ``light`` multiplies the pattern, so the input sets
    the overall brightness while ``intensity_fn`` sets its shape. For a pattern
    whose *shape* also follows the input, write a ``drive_fn(body, light)`` by
    hand instead of using this helper.
    """
    F_body = jnp.asarray(F_body, dtype=jnp.float64)
    tau = jnp.float64(tau)

    def drive_fn(body, light):
        s = intensity_fn(body.center) * light                  # (N,) >= 0
        R = rigid_body.rotation2d(body.orientation)
        f_lab = jnp.einsum("nij,j->ni", R, F_body) * s[:, None]
        return f_lab, s * tau

    drive_fn.spec = {"type": "modulated", "F_body_N": [float(f) for f in F_body],
                     "tau_Nm": float(tau),
                     "intensity": getattr(intensity_fn, "spec", "custom (no .spec)")}
    return drive_fn


def shape_radius(shape, sigma):
    """Circumradius of a body in meters: furthest bead center + one bead radius.

    ``shape`` points are in units of sigma (as passed to ``make_runner``). This
    is what ``with_wall`` needs so the whole body stays inside, not just its
    center. For ``single_bead_shape`` it is ``sigma/2``.
    """
    pts = np.asarray(shape.points) * sigma
    return float(np.linalg.norm(pts, axis=-1).max()) + float(sigma) / 2


def make_active_optical_drive(FX_pN, FY_pN, TZ_pN_nm, phi_pol=0.0, power=3):
    """Orientation-dependent optical drive from a fitted design.

    Coefficients are the ``[A, B, C]`` of ``A cos^2(a) + B sin^2(a) + C sin(a)cos(a)``
    with ``a = phi_pol - theta``. 
    ``power`` is the 10 uW -> 30 uW scaling.

    Unlike ``make_drive``, the force varies with orientation, so it is
    rotated per particle into the lab frame.
    """
    def drive_fn(body, light):
        """light is the intensity over time. here we currently cant use light pattern like disk or gradient"""
        theta = body.orientation
        F_body, tau = active_force_torque_body(theta, phi_pol, FX=FX_pN,
                                               FY=FY_pN, TZ=TZ_pN_nm, power=power)
        R = rigid_body.rotation2d(theta)#body -> lab
        return jnp.einsum("nij,nj->ni", R, F_body) * light, tau * light

    drive_fn.spec = {"type": "active_optical", "phi_pol": float(phi_pol),
                     "power": float(power), "FX_pN": list(FX_pN),
                     "FY_pN": list(FY_pN), "TZ_pN_nm": list(TZ_pN_nm)}
    return drive_fn


def with_wall(drive_fn, box, k=5e-5, body_radius=0.0, margin=0.0):
    """Wrap a drive with a soft wall so bodies stay in the box instead of wrapping.

    The wall surface sits ``body_radius + margin`` in from each box edge, and it
    acts on the body *center* -- so pass ``body_radius=shape_radius(shape, sigma)``
    or the body will overhang the wall by its own radius. With the default
    ``margin=0`` the body then just touches the box edge.

    Acts per axis and only along the wall normal, so a body swims into the wall,
    stops, and slides along it rather than bouncing. Penetration is
    ``F_drive / k`` -- 23 nm at ``k=5e-5``, i.e. hard at this scale; much stiffer
    needs a smaller ``dt``.
    """
    edge = float(body_radius) + float(margin)

    def wrapped(body, light):
        f_lab, tau = drive_fn(body, light)
        c = body.center
        f_wall = k * (jnp.maximum(edge - c, 0.0)
                      - jnp.maximum(c - (box - edge), 0.0))
        f_wall = f_wall.at[:, 0].set(0.0)   # no wall along the drive (x)
        return f_lab + f_wall, tau

    wrapped.spec = {"type": "wall", "k_N_per_m": float(k),
                    "body_radius_m": float(body_radius), "margin_m": float(margin),
                    "wall_surface_m": edge, "box_m": float(box),
                    "drive": getattr(drive_fn, "spec", "custom (no .spec)")}
    return wrapped


# --- intensity patterns: centers (N,2) [m] -> non-negative scaling (N,) -----
def intensity_uniform(level=1.0):
    """Flat pattern, the identity case, useful as a sweep baseline."""
    def fn(c):
        return jnp.full(c.shape[0], level, dtype=c.dtype)
    fn.spec = {"pattern": "uniform", "level": float(level)}
    return fn


def intensity_gradient(box, lo=0.5, hi=4.0, axis=0):
    """Linear ramp from ``lo`` (one edge of the box) to ``hi`` (the opposite)."""
    def fn(c):
        return lo + (hi - lo) * (c[:, axis] / box)
    fn.spec = {"pattern": "gradient", "box_m": float(box), "lo": float(lo),
               "hi": float(hi), "axis": int(axis)}
    return fn


def intensity_disk(center, radius, inside=4.0, outside=0.5):
    """Uniform ``inside`` within ``radius`` of ``center``, ``outside`` beyond."""
    center = jnp.asarray(center)
    def fn(c):
        r = jnp.linalg.norm(c - center, axis=1)
        return jnp.where(r < radius, inside, outside)
    fn.spec = {"pattern": "disk", "center_m": [float(x) for x in center],
               "radius_m": float(radius), "inside": float(inside),
               "outside": float(outside)}
    return fn


def intensity_ring(center, r0, width, bright=4.0, dark=0.5):
    """Bright annulus of radius ``r0`` (thickness ``width``); dark elsewhere."""
    center = jnp.asarray(center)
    def fn(c):
        r = jnp.linalg.norm(c - center, axis=1)
        return jnp.where(jnp.abs(r - r0) < width / 2, bright, dark)
    fn.spec = {"pattern": "ring", "center_m": [float(x) for x in center],
               "r0_m": float(r0), "width_m": float(width), "bright": float(bright),
               "dark": float(dark)}
    return fn


def intensity_stripes(period, bright=4.0, dark=0.5, axis=0):
    """Alternating bright/dark stripes with spatial ``period`` [m]."""
    def fn(c):
        return jnp.where(jnp.sin(2 * jnp.pi * c[:, axis] / period) > 0,
                         bright, dark)
    fn.spec = {"pattern": "stripes", "period_m": float(period),
               "bright": float(bright), "dark": float(dark), "axis": int(axis)}
    return fn


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


def make_force_fn(interaction_force_fn, drive_fn):
    """Combine steric interaction + light drive -> ``force_fn(body, light, **kw)``.

    Returns ``RigidBody(force, torque)``. Any extra kwargs (notably
    ``neighbor=``) are forwarded to the interaction; the drive never sees them.
    """
    def force_fn(body, light=1.0, **kwargs):
        f_int = interaction_force_fn(body, **kwargs)
        f_lab, tau = drive_fn(body, light)
        return rigid_body.RigidBody(f_int.center + f_lab,
                                    f_int.orientation + tau)

    return force_fn


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------
def make_runner(n_particles, sigma=1e-6, area_fraction=0.3, box=None,
                steps_per_sample=100, dt=1e-4, kT=KT_300K,
                resistance_body=DEFAULT_RESISTANCE_BODY, F_body=DEFAULT_F_BODY,
                tau=DEFAULT_TAU, epsilon=5e-17, capacity_multiplier=2.0,
                shape=None, drive_fn=None, interaction_fn=None,
                init=random_initial_condition,
                grow_neighbor_buffer=True, max_extra_capacity=256,initial_extra_capacity=0,
                progress=False):
    """Build the simulator once; reuse it across seeds and light series.

    Returns ``run_one(light, seed) -> (centers, angles, box)``. The jitted
    simulation is compiled a single time, so looping over replicas (seeds) or
    over light series of the same length (e.g. a gain sweep) costs no
    recompilation, much faster than calling ``run`` in a loop.

    ``sigma`` is the *bead* diameter, which for ``single_bead_shape`` is also the
    particle diameter; a multi-bead ``shape`` is scaled by it, so a
    ``grid_shape(n)`` body has side ``n * sigma``.

    ``shape``, ``drive_fn`` and ``interaction_fn`` override the defaults built
    from the scalar arguments (``single_bead_shape()``, ``make_drive(F_body,
    tau)`` and ``soft_sphere_interaction(epsilon, capacity_multiplier)``
    respectively). ``F_body``/``tau`` are ignored when ``drive_fn`` is given, as
    are ``epsilon``/``capacity_multiplier`` when ``interaction_fn`` is given.

    ``init`` is the initial condition, a callable ``init(n_particles, box, seed)
    -> (body0, key)``: ``random_initial_condition`` (default) or
    ``grid_initial_condition`` (non-overlapping; better when dense)

    With ``grow_neighbor_buffer`` the Verlet buffer is re-allocated with more
    capacity and the run repeated whenever it overflows, up to
    ``max_extra_capacity`. So a clumping run self-corrects instead of
    silently returning wrong forces. The grown buffer is kept for later calls.
    Each growth step re-traces the jitted simulation, so a run that grows is
    slower;

    With ``progress`` a tqdm bar advances once per light sample and the wall
    time of the run is printed at the end.

    The returned ``run_one`` carries ``.meta`` (a dict of every parameter, ready
    for ``runs.save_run``) plus ``.box``, ``.sigma`` and ``.shape_points``.
    """
    if shape is None:
        shape = single_bead_shape()
    if drive_fn is None:
        drive_fn = make_drive(F_body, tau)
    if interaction_fn is None:
        interaction_fn = soft_sphere_interaction(
            epsilon=epsilon, capacity_multiplier=capacity_multiplier)

    # shape points are in units of sigma -> scale to meters
    shape = shape.set(points=shape.points * sigma)
    shape_points = np.asarray(shape.points)

    # The body's footprint: widest bead-center span plus one bead diameter.
    # For single_bead_shape the span is 0, so this is exactly sigma and the
    # default box below is identical to the old sigma-based one.
    body_side = float(np.max(shape_points.max(axis=0)
                             - shape_points.min(axis=0))) + float(sigma)

    if box is None:
        # area_fraction packs the BODY, not the bead: a grid_shape(n) body is
        # n*sigma across, so sizing by sigma alone would under-size the box n-fold.
        box = body_side * float(np.sqrt(n_particles / area_fraction))

    displacement_fn, shift_fn = space.periodic(box)
    interaction_force_fn, nbr_fns = interaction_fn(
        shape, sigma, box, displacement_fn)
    force_fn = make_force_fn(interaction_force_fn, drive_fn)
    init_fn, step_fn = brownian_generalized_rigid_2d(
        energy_or_force_fn=force_fn, shift_fn=shift_fn, dt=dt, kT=kT,
        resistance_body=resistance_body)

    use_nl = nbr_fns is not None
    if use_nl:
        # one reference allocation fixes the neighbor-buffer shape for every
        # seed (simulate re-fills it from the actual positions on its first step)
        ref_body, _ = init(n_particles, box, 0)
        nbrs0 = nbr_fns.allocate(ref_body,extra_capacity=initial_extra_capacity)
    else:
        nbrs0 = None

    bar = {}                                   # the tqdm bar of the current run

    def tick():
        bar["tqdm"].update(1)

    @jit
    def simulate(state, nbrs, light):
        def outer(carry, light_t):
            def inner(c, _):
                state, nbrs, ovf = c
                if use_nl:
                    nbrs = nbrs.update(state.position)
                    ovf = ovf | (nbrs.did_buffer_overflow != 0)
                    state = step_fn(state, neighbor=nbrs, light=light_t)
                else:
                    state = step_fn(state, light=light_t)
                return (state, nbrs, ovf), None

            carry, _ = lax.scan(inner, carry, xs=None, length=steps_per_sample)
            if progress:
                jax.debug.callback(tick) 
            state, nbrs, ovf = carry
            return (state, nbrs, ovf), (state.position.center,
                                        state.position.orientation)

        carry = (state, nbrs, jnp.array(False))
        (state, nbrs, ovf), (centers, angles) = lax.scan(outer, carry, light)
        return centers, angles, ovf

    buf = {"extra": int(initial_extra_capacity), "nbrs": nbrs0}   # grown in place, kept across calls

    def run_one(light, seed=0):
        light = jnp.asarray(light, dtype=jnp.float64)
        body0, key = init(n_particles, box, seed)
        state = init_fn(key, body0)
        overflowed = False
        t0 = time.time()
        while True:
            if progress:
                from tqdm import tqdm
                bar["tqdm"] = tqdm(total=len(light), unit="sample")
            centers, angles, ovf = simulate(state, buf["nbrs"], light)
            jax.block_until_ready(centers)
            if progress:
                bar["tqdm"].close()
            if not (use_nl and bool(ovf)):
                break
            if not grow_neighbor_buffer or buf["extra"] >= max_extra_capacity:
                overflowed = True
                print("WARNING: neighbor buffer still overflowing; results suspect")
                break
            buf["extra"] += 16
            buf["nbrs"] = nbr_fns.allocate(body0, extra_capacity=buf["extra"])
            print(f"neighbor buffer overflow -> retrying with "
                  f"extra_capacity={buf['extra']}")
        if progress:
            print(f"simulation took {time.time() - t0:.1f} s")
        run_one.meta["neighbor_buffer_overflow"] = overflowed
        run_one.meta["neighbor_extra_capacity"] = int(buf["extra"])
        return np.asarray(centers), np.asarray(angles), float(box)

    #function is object so we can add the attributes we later need in __main__
    run_one.box = float(box)
    run_one.sigma = float(sigma)
    run_one.shape_points = shape_points
    run_one.meta = {
        "n_particles": int(n_particles),
        "n_beads_per_body": int(shape_points.shape[0]),
        "body_side_m": body_side,
        "sigma_m": float(sigma),
        "box_m": float(box),
        "area_fraction": float(area_fraction),
        "init": getattr(init, "__name__", repr(init)),
        "steps_per_sample": int(steps_per_sample),
        "dt": float(dt),
        "kT": float(kT),
        "temperature_K": float(kT) / K_B,
        "resistance_body": np.asarray(resistance_body).tolist(),
        # drive/interaction describe themselves via a .spec attribute (same
        # convention as trajectory.make_*_force).
        "drive": getattr(drive_fn, "spec", "UNSPECIFIED (drive_fn has no .spec)"),
        "interaction": getattr(interaction_fn, "spec",
                               "UNSPECIFIED (interaction_fn has no .spec)"),
        "shape_points_sigma": (shape_points / sigma).tolist(),
        "use_neighbor_list": bool(use_nl),
        "jax_version": jax.__version__,
    }
    return run_one

if __name__ == "__main__":
    seed = 0

    # --- particle --------------------------------------------------------------
    body_side = 10e-6                 # side of one square body [m]
    n_grid    = 5                     # body side in units of sigma
    sigma     = body_side / n_grid    # bead diameter = 2 um
    spacing   = 0.5                   # bead spacing in units of sigma (was 0.25)
    soft      = 0.4                   # contact softening (2.5x softer than before)
    
    sigma = 10e-6
    shape = single_bead_shape()

    epsilon = 5e-17 * (sigma / 1e-6) ** 2          # eps/sigma^2 = 5e-5 N/m per bead pair
    # shape   = perimeter_shape(n_grid, spacing)
    # epsilon = epsilon * spacing * soft             # same face stiffness as grid_shape, x soft

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

    FX = [1.24171, 0.443034, 0.0] #desing1_res60_linear_30uW
    FY = [0.0, 0.0, 0.066536]
    TZ = [0, 0, 244.48]
    # drive = with_wall(make_active_optical_drive(FX, FY, TZ, phi_pol=0.0),
    #                 box, k=5e-5, body_radius=rad, margin=1e-7)
    drive = make_active_optical_drive(FX, FY, TZ, phi_pol=0.0)
    F_body = (1.135e-12, 0.0)
    tau    = 966e-21 
    drive = make_drive(F_body=F_body,tau=tau)
    # drive = with_wall(make_drive(), box, k=5e-5, body_radius=rad, margin=1e-7)
    init = make_ring_init(88e-6,np.sign(tau) * np.pi / 2)
    runner = make_runner(n_particles=N, sigma=sigma, box=box,
                     init=init,
                     shape=shape, drive_fn=drive, epsilon=epsilon,
                     dt=dt, steps_per_sample=steps_per_sample,
                     capacity_multiplier=5.0, progress=True)

    centers, angles, box = runner(u, seed=seed)
    print(f"centers {centers.shape}, angles {angles.shape}, box {box:.2e} m")

    run_dir = save_run(
        arrays={"centers": centers, "angles": angles, "light": u,
                "box": box, "sigma": sigma, "body_points": runner.shape_points},
        meta={**runner.meta, "n_samples": len(u), "seed": seed,
              "light_signal": light_signal},
        label="ring", prefix="swarm", source=__file__)

    fps = 24
    # target_frames = round(len(u) * steps_per_sample * dt * fps)
    target_frames = round(len(u))

    make_video(run_dir, out_path=str(run_dir / "animation.mp4"),body="disk",target_frames=target_frames,fps=fps)
