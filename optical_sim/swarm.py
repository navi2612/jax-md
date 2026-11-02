"""The simulation loop for 2D active rigid bodies, independent of the contact model.

``make_runner`` takes the contact model as ``interaction_fn``; see
``spheropolygon.py`` and ``beads.py``.
"""
import time

import numpy as np
import os
import jax
# float64 by default; OPTSIM_X64=0 runs in float32 (fast on consumer GPUs,
# whose float64 rate is ~1/64 of float32 -- validate before trusting it)
jax.config.update("jax_enable_x64", os.environ.get("OPTSIM_X64", "1") != "0")

import jax.numpy as jnp
from jax import random, lax, jit
from jax_md import space, rigid_body

from brownian_generalized_2d import brownian_generalized_rigid_2d
from trajectory import KT_300K, K_B, DEFAULT_RESISTANCE_BODY
from active_optical import active_force_torque_body

# Body-frame optical drive per unit intensity (circular polarization).
DEFAULT_F_BODY = (1.135e-12, 0.045e-12)        # (Fx, Fy)  [N]
DEFAULT_TAU = 5*440.44e-21                        # torque    [N.m]


def single_bead_shape():
    """One point at the origin -> each rigid body is a single disk."""
    pts = jnp.zeros((1, 2), dtype=jnp.float64)
    return rigid_body.point_union_shape(pts, jnp.ones(1))


def random_initial_condition(n_particles, box, seed):
    """Random centers in the box and random orientations -> (body0, key). Overlap can occur"""
    key = random.PRNGKey(seed)
    key, ckey, okey = random.split(key, 3)
    centers = random.uniform(ckey, (n_particles, 2), minval=0.0, maxval=box,
                             dtype=jnp.float64)
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


# --------------------------------------------------------------------------
# the simulation
# --------------------------------------------------------------------------
def make_force_fn(interaction_force_fn, drive_fn, wall_fn=None):
    """Combine steric interaction + optional wall + light drive -> ``force_fn(body, light, **kw)``.

    Returns ``RigidBody(force, torque)``. Any extra kwargs (notably
    ``neighbor=``) are forwarded to the interaction; the drive and the wall never
    see them.
    """
    def force_fn(body, light=1.0, **kwargs):
        f_int = interaction_force_fn(body, **kwargs)
        if wall_fn is not None:
            f_wall = wall_fn(body)
            f_int = rigid_body.RigidBody(f_int.center + f_wall.center,
                                         f_int.orientation + f_wall.orientation)
        f_lab, tau = drive_fn(body, light)
        return rigid_body.RigidBody(f_int.center + f_lab,
                                    f_int.orientation + tau)

    return force_fn


def make_runner(n_particles, interaction_fn, sigma=1e-6, area_fraction=0.3, box=None,
                steps_per_sample=100, dt=1e-4, kT=KT_300K,
                resistance_body=DEFAULT_RESISTANCE_BODY, F_body=DEFAULT_F_BODY,
                tau=DEFAULT_TAU, shape=None, drive_fn=None, wall_fn=None,
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

    ``interaction_fn`` is the contact model: ``spheropolygon.make_interaction``
    or ``beads.soft_sphere_interaction``. ``shape`` and ``drive_fn`` override the
    defaults (``single_bead_shape()`` and ``make_drive(F_body, tau)``);
    ``F_body``/``tau`` are ignored when ``drive_fn`` is given.

    ``wall_fn(body) -> RigidBody(force, torque)`` is the wall, e.g.
    ``spheropolygon.make_circular_wall``. Without one the box is periodic; with
    one the box is periodic underneath too, but the wall keeps the bodies away
    from its edges, so make the box larger than the arena.

    ``init`` is the initial condition, a callable ``init(n_particles, box, seed)
    -> (body0, key)``: ``random_initial_condition`` (default, can overlap) or
    ``spheropolygon.make_init``.

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
    force_fn = make_force_fn(interaction_force_fn, drive_fn, wall_fn)
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
        "wall": getattr(wall_fn, "spec", None),
        "shape_points_sigma": (shape_points / sigma).tolist(),
        "use_neighbor_list": bool(use_nl),
        "jax_version": jax.__version__,
    }
    return run_one
