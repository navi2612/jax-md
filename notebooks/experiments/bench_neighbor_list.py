"""Timing comparison: all-pairs (O(N^2)) vs neighbor-list (~O(N)) interaction
for the metavehicle swarm, at fixed area fraction as N grows.

Reports warmup-corrected ms/step (compile excluded) and the projected wall time
for a full 600k-step production run. Run:  python bench_neighbor_list.py
"""

import time
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax import random, lax, jit
from jax_md import space, energy, rigid_body, simulate, quantity

# ---- physics (identical to metavehicle_swarm.py) --------------------------
kT = jnp.float64(1.38e-23 * 300.0)
gamma_T, gamma_R = jnp.float64(470e-9), jnp.float64(8000e-21)
R_body = jnp.array([[gamma_T, 0, 0], [0, gamma_T, 0], [0, 0, gamma_R]],
                   dtype=jnp.float64)
F_active_body = jnp.array([1.135e-12, 0.045e-12], dtype=jnp.float64)
tau_active = jnp.float64(440.44e-21)
dt = jnp.float64(1e-4)

side, sigma, n_grid, epsilon = 10e-6, 2.0e-6, 5, jnp.float64(5e-17)
PHI = 0.165                                  # fixed area fraction
PROD_STEPS = 600_000                         # production length to project to

_span = side - sigma
_xs = jnp.linspace(-_span / 2, _span / 2, n_grid)
_gx, _gy = jnp.meshgrid(_xs, _xs)
square_pts = jnp.stack([_gx.ravel(), _gy.ravel()], 1).astype(jnp.float64)
square = rigid_body.point_union_shape(square_pts, jnp.ones(square_pts.shape[0]))


def _drive(body):
    R = rigid_body.rotation2d(body.orientation)
    fa = jnp.einsum("nij,j->ni", R, F_active_body)
    tau = jnp.full(body.orientation.shape, tau_active, dtype=body.center.dtype)
    return fa, tau


def build(N, use_nl):
    L = side * np.sqrt(N / PHI)
    disp, shift = space.periodic(L)
    if use_nl:
        nfn, e_nl = energy.soft_sphere_neighbor_list(
            disp, L, sigma=sigma, epsilon=epsilon, alpha=2.0,
            dr_threshold=0.4e-6, capacity_multiplier=2.0)
        nbr_fns, rigid_e = rigid_body.point_energy_neighbor_list(e_nl, nfn,
                                                                 square)
    else:
        pe = energy.soft_sphere_pair(disp, sigma=sigma, epsilon=epsilon,
                                     alpha=2.0)
        rigid_e = rigid_body.point_energy(pe, square)
        nbr_fns = None
    iforce = quantity.force(rigid_e)

    def force(body, **kw):
        f = iforce(body, **kw)
        fa, tau = _drive(body)
        return rigid_body.RigidBody(f.center + fa, f.orientation + tau)

    init_fn, step_fn = simulate.brownian_generalized_rigid_2d(
        force, shift, dt, kT, resistance_body=R_body)

    # initial grid of squares
    m = int(np.ceil(np.sqrt(N)))
    c = (np.arange(m) + 0.5) * (L / m)
    gx, gy = np.meshgrid(c, c)
    centers = jnp.array(np.stack([gx.ravel(), gy.ravel()], 1)[:N],
                        dtype=jnp.float64)
    key = random.PRNGKey(0)
    theta = random.uniform(key, (N,), minval=-jnp.pi, maxval=jnp.pi,
                           dtype=jnp.float64)
    body = rigid_body.RigidBody(centers, theta)
    state = init_fn(key, body)
    return nbr_fns, step_fn, state, body


def time_method(N, use_nl, K=1000):
    nbr_fns, step_fn, state, body = build(N, use_nl)
    if use_nl:
        nbrs = nbr_fns.allocate(body, extra_capacity=64)

        def run(state, nbrs):
            def stp(c, _):
                s, nb = c
                nb = nb.update(s.position)
                s = step_fn(s, neighbor=nb)
                return (s, nb), None
            (s, nb), _ = lax.scan(stp, (state, nbrs), xs=None, length=K)
            return s, nb
        rj = jit(run)
        s, nb = rj(state, nbrs)                 # warmup (compile)
        jax.block_until_ready(s.position.center)
        t = time.time()
        s, nb = rj(state, nbrs)                 # timed
        jax.block_until_ready(s.position.center)
        wall, ovf = time.time() - t, bool(nb.did_buffer_overflow)
    else:
        def run(state):
            def stp(s, _):
                return step_fn(s), None
            s, _ = lax.scan(stp, state, xs=None, length=K)
            return s
        rj = jit(run)
        s = rj(state)                           # warmup
        jax.block_until_ready(s.position.center)
        t = time.time()
        s = rj(state)                           # timed
        jax.block_until_ready(s.position.center)
        wall, ovf = time.time() - t, False
    return wall / K * 1e3, ovf                  # ms/step


if __name__ == "__main__":
    Ns = [20, 50, 100, 200]
    print(f"{'N':>5} {'points':>7} | {'pair ms/step':>12} {'pair 600k':>11} | "
          f"{'NL ms/step':>11} {'NL 600k':>9} | {'speedup':>7}")
    print("-" * 78)
    for N in Ns:
        pts = N * n_grid * n_grid
        ms_pair, _ = time_method(N, use_nl=False)
        ms_nl, ovf = time_method(N, use_nl=True)
        proj_pair = ms_pair * PROD_STEPS / 1e3
        proj_nl = ms_nl * PROD_STEPS / 1e3
        ov = " (overflow!)" if ovf else ""
        print(f"{N:>5} {pts:>7} | {ms_pair:>10.2f}   {proj_pair/60:>8.1f}m | "
              f"{ms_nl:>9.2f}   {proj_nl/60:>6.1f}m | {ms_pair/ms_nl:>6.1f}x"
              f"{ov}")
