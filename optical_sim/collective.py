"""Search for collective behaviour: disks, squares, triangles and their mixtures.

Every particle has the same footprint area (that of the 10 um x 10 um square)
and is built from beads of one size ``SIGMA``, so any mix of shapes can share a
single soft-sphere interaction (jax-md ``shape_species``). The outline is a
ring of overlapping beads (``SPACING`` < 1, like ``swarm.perimeter_shape``), so
faces are smooth and every shape has the same contact friction; one bead in the
middle keeps bodies from nesting.

Body frame: the drive pushes along body +x. A triangle points a vertex along
+x (``heading="tip"``) or its base (``heading="base"``).

Drive per particle type (all in the body frame, scaled by the light):
    force   F = (F_x, F_y)
    torque  tau + K * sin(2 * (phi_pol - theta))
``tau`` makes circle swimmers (turning radius R_T = F g_R / (g_T tau)); ``K`` is
the linear-polarisation aligning torque, which turns body +x towards +-phi_pol.

The arena is a circle of ``R_ARENA`` with a soft wall on every bead and light
everywhere inside (Phase 1 screening). A control run with interactions switched
off but the same drive, wall and start tells which numbers come from the
particles interacting.

Usage (pmeep env):
    python collective.py test
    python collective.py phase1 --worker 0 --nworkers 2
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from jax import random
from jax_md import energy, partition, quantity, rigid_body
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from swarm import make_runner
from trajectory import DEFAULT_RESISTANCE_BODY
from runs import save_run, RESULTS_DIR

# --------------------------------------------------------------------------
# particle geometry: equal area, one bead size
# --------------------------------------------------------------------------
SIDE = 10e-6                       # square side [m]; sets the common area
AREA = SIDE ** 2
SIGMA = SIDE / 3                   # bead diameter [m]
SPACING = 0.5                      # bead spacing along the outline / SIGMA
EPSILON = 5e-17 * (SIGMA / 1e-6) ** 2 * SPACING   # same face stiffness as grid_shape
G_T = float(DEFAULT_RESISTANCE_BODY[0][0])
G_R = float(DEFAULT_RESISTANCE_BODY[2][2])
F0 = 1.135e-12                     # propulsion force of the measured design [N]
V0 = F0 / G_T                      # ~2.4 um/s


def outline(kind, heading="tip"):
    """True outline in the body frame [m]: polygon vertices, or None for a disk."""
    if kind == "disk":
        return None
    if kind == "square":
        h = SIDE / 2
        return np.array([[h, h], [-h, h], [-h, -h], [h, -h]])
    if kind == "triangle":
        a = np.sqrt(4 * AREA / np.sqrt(3))            # side of the equal-area triangle
        rc = a / np.sqrt(3)                            # circumradius
        ang = np.deg2rad([0, 120, 240]) + (np.pi if heading == "base" else 0.0)
        return rc * np.c_[np.cos(ang), np.sin(ang)]
    raise ValueError(kind)


def rounded_outline(kind, heading="tip", n_arc=6):
    """Outline of the *collision* shape [m]: the bead-inset polygon grown by
    one bead radius, i.e. the true outline with corners rounded by SIGMA/2."""
    V = outline(kind, heading)
    if V is None:
        return None
    r = SIGMA / 2
    nv = len(V)
    rc = np.linalg.norm(V[0])
    Vi = V * (rc * np.cos(np.pi / nv) - r) / (rc * np.cos(np.pi / nv))
    pts = []
    for i in range(nv):
        a, b, c = Vi[i - 1], Vi[i], Vi[(i + 1) % nv]
        n1 = np.array([(b - a)[1], -(b - a)[0]]); n1 /= np.linalg.norm(n1)
        n2 = np.array([(c - b)[1], -(c - b)[0]]); n2 /= np.linalg.norm(n2)
        # vertices run counter-clockwise, so (dy, -dx) is the outward normal
        a1, a2 = np.arctan2(n1[1], n1[0]), np.arctan2(n2[1], n2[0])
        da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
        for t in np.linspace(0, 1, n_arc):
            ang = a1 + t * da
            pts.append(b + r * np.array([np.cos(ang), np.sin(ang)]))
    return np.array(pts)


def disk_radius():
    return np.sqrt(AREA / np.pi)


def bead_points(kind, heading="tip"):
    """Bead centres in the body frame, in units of SIGMA.

    Beads sit on the outline shrunk by one bead radius, so bead *surfaces*
    reach the true outline (corners come out rounded by SIGMA/2)."""
    r = SIGMA / 2
    step = SPACING * SIGMA
    if kind == "disk":
        R = disk_radius() - r
        n = int(np.ceil(2 * np.pi * R / step))
        a = 2 * np.pi * np.arange(n) / n
        ring = R * np.c_[np.cos(a), np.sin(a)]
    else:
        V = outline(kind, heading)
        nv = len(V)
        # inset a regular polygon: apothem shrinks by r
        rc = np.linalg.norm(V[0])
        scale = (rc * np.cos(np.pi / nv) - r) / (rc * np.cos(np.pi / nv))
        Vi = V * scale
        ring = []
        for i in range(nv):
            p, q = Vi[i], Vi[(i + 1) % nv]
            m = int(np.ceil(np.linalg.norm(q - p) / step))
            t = np.arange(m) / m
            ring.append(p + t[:, None] * (q - p))
        ring = np.concatenate(ring)
    pts = np.vstack([ring, [[0.0, 0.0]]])             # + one centre bead
    return pts / SIGMA


def shape_of(kind, heading="tip"):
    pts = jnp.asarray(bead_points(kind, heading), dtype=jnp.float64)
    return rigid_body.point_union_shape(pts, jnp.ones(len(pts)))


# --------------------------------------------------------------------------
# particle types and the drive
# --------------------------------------------------------------------------
@dataclass
class PType:
    kind: str                       # disk | square | triangle
    count: int = 0
    heading: str = "tip"            # triangle only: tip | base first
    F: tuple = (F0, 0.0)            # body-frame force [N]
    tau: float = 0.0                # chiral torque [N m]
    K: float = 0.0                  # linear-pol aligning torque amplitude [N m]
    phi_pol: float = 0.0

    @property
    def R_T(self):
        return np.inf if self.tau == 0 else np.hypot(*self.F) * G_R / (G_T * abs(self.tau))


def tau_for_RT(R_T, F=F0):
    return F * G_R / (G_T * R_T)


def make_species_drive(types):
    """Per-particle force/torque: particle i belongs to type species[i]."""
    species = np.concatenate([np.full(t.count, s) for s, t in enumerate(types)])
    F = jnp.asarray(np.array([t.F for t in types])[species])           # (N,2)
    tau = jnp.asarray(np.array([t.tau for t in types])[species])       # (N,)
    K = jnp.asarray(np.array([t.K for t in types])[species])
    phi = jnp.asarray(np.array([t.phi_pol for t in types])[species])

    def drive_fn(body, light):
        th = body.orientation
        R = rigid_body.rotation2d(th)
        f_lab = jnp.einsum("nij,nj->ni", R, F) * light
        torque = (tau + K * jnp.sin(2.0 * (phi - th))) * light
        return f_lab, torque

    drive_fn.spec = {"type": "species", "types": [asdict(t) for t in types]}
    return drive_fn, species


# --------------------------------------------------------------------------
# interaction (+ wall) for a mixture
# --------------------------------------------------------------------------
def make_interaction(species, center, r_wall, interacting=True, k_wall=1e-4,
                     capacity_multiplier=5.0):
    """Bead-bead soft spheres between bodies + a soft circular wall on every
    bead, both as energies so jax-md returns force *and* torque.

    ``interacting=False`` keeps only the wall: the control run."""
    species = None if species is None else np.asarray(species)   # None = one shape
    center = jnp.asarray(center)

    def interaction_fn(shape, sigma, box, displacement_fn):
        r_max = None if r_wall is None else r_wall - sigma / 2

        def wall_energy(body, **kw):
            if r_max is None:                  # periodic box: no wall
                return 0.0 * jnp.sum(body.center)
            p, _ = rigid_body.union_to_points(body, shape, species)
            over = jnp.maximum(jnp.linalg.norm(p - center, axis=-1) - r_max, 0.0)
            return 0.5 * k_wall * jnp.sum(over ** 2)

        if not interacting:
            return quantity.force(wall_energy), None

        neighbor_fn, ss = energy.soft_sphere_neighbor_list(
            displacement_fn, box, sigma=sigma, epsilon=EPSILON, alpha=2.0,
            dr_threshold=0.2 * sigma, capacity_multiplier=capacity_multiplier)

        def wrap(R):                       # cell list needs beads inside [0, box)
            return jnp.mod(R, box)

        wrapped = partition.NeighborListFns(
            lambda R, **kw: neighbor_fn.allocate(wrap(R), **kw),
            lambda R, nbrs, **kw: neighbor_fn.update(wrap(R), nbrs, **kw))
        nbr_fns, pair_e = rigid_body.point_energy_neighbor_list(
            lambda R, **kw: ss(wrap(R), **{k: v for k, v in kw.items()
                                          if k != "species"}),
            wrapped, shape, species)

        def total(body, neighbor, **kw):
            return pair_e(body, neighbor=neighbor) + wall_energy(body)

        return quantity.force(total), nbr_fns

    interaction_fn.spec = {"type": "mixture_soft_sphere+circular_wall",
                           "interacting": bool(interacting), "epsilon_J": EPSILON,
                           "k_wall_N_per_m_per_bead": k_wall, "r_wall_m": r_wall}
    return interaction_fn


# --------------------------------------------------------------------------
# initial condition: random, non-overlapping *as shapes*, inside the circle
# --------------------------------------------------------------------------
def make_init(types, species, center, r_wall, margin=0.08, stages=30, iters=300):
    """Random centres and headings, then grow every body from 50% to full size,
    pushing apart bodies whose beads overlap and pulling escapees back in.
    Works with any mix of shapes (overlap is tested bead by bead)."""
    pts_by_type = [bead_points(t.kind, t.heading) * SIGMA for t in types]
    rmax_by_type = [np.linalg.norm(p, axis=1).max() + SIGMA / 2 for p in pts_by_type]
    center = np.asarray(center, float)
    n = len(species)
    body_of_bead = np.concatenate([np.full(len(pts_by_type[s]), i)
                                   for i, s in enumerate(species)])

    def beads(pos, th, g):
        out = []
        for i, s in enumerate(species):
            c, si = np.cos(th[i]), np.sin(th[i])
            p = pts_by_type[s] * g
            out.append(pos[i] + p @ np.array([[c, si], [-si, c]]))
        return np.concatenate(out)

    def init(n_particles, box, seed):
        assert n_particles == n
        rng = np.random.default_rng(seed)
        rlim = np.array([r_wall - rmax_by_type[s] for s in species])
        rr = rlim * np.sqrt(rng.uniform(size=n))
        a = rng.uniform(0, 2 * np.pi, n)
        pos = center + rr[:, None] * np.c_[np.cos(a), np.sin(a)]
        th = rng.uniform(-np.pi, np.pi, n)

        def relax(g, n_iter):
            nonlocal pos
            dmin = SIGMA * (1 + margin) * g
            for _ in range(n_iter):
                b = beads(pos, th, g)
                pairs = cKDTree(b).query_pairs(dmin, output_type="ndarray")
                if len(pairs):
                    i, j = body_of_bead[pairs[:, 0]], body_of_bead[pairs[:, 1]]
                    keep = i != j
                    pairs, i, j = pairs[keep], i[keep], j[keep]
                if len(pairs) == 0:
                    return True
                dd = b[pairs[:, 0]] - b[pairs[:, 1]]
                d = np.maximum(np.hypot(*dd.T), 1e-12)
                push = 0.5 * (1.05 * dmin - d)[:, None] * dd / d[:, None]
                disp = np.zeros_like(pos)
                cnt = np.zeros(n)
                np.add.at(disp, i, push); np.add.at(disp, j, -push)
                np.add.at(cnt, i, 1); np.add.at(cnt, j, 1)
                pos = pos + disp / np.maximum(cnt, 1)[:, None] * 1.5
                rel = pos - center
                r = np.maximum(np.hypot(*rel.T), 1e-12)
                pos = center + rel * np.minimum(1.0, rlim / r)[:, None]
            return False

        for g in np.linspace(0.5, 1.0, stages):
            relax(g, iters)
        if not relax(1.0, 20 * iters):
            raise RuntimeError(f"could not remove overlaps (n={n}); lower phi")
        return (rigid_body.RigidBody(jnp.asarray(pos), jnp.asarray(th)),
                random.PRNGKey(seed))

    init.__name__ = "collective_init"
    return init


# --------------------------------------------------------------------------
# one run (interacting + control)
# --------------------------------------------------------------------------
COLL_DIR = Path(RESULTS_DIR) / "collective"
LEDGER = COLL_DIR / "LEDGER.md"


def counts_for_phi(phi, r_arena, fractions):
    n = int(round(phi * np.pi * r_arena ** 2 / AREA))
    c = np.floor(np.asarray(fractions) * n).astype(int)
    c[0] += n - c.sum()
    return c


def simulate(types, label, *, r_arena=None, L=None, T_total, t_sample=1.0, dt=4e-4,
             seed=0, interacting=True, tag=""):
    """Run one configuration, save it under results/collective. Returns run_dir.

    ``r_arena``: circular walled arena of that radius. ``L``: periodic box of
    side L instead (no wall; a patch inside a large homogeneous spot)."""
    periodic = L is not None
    box = L if periodic else 2 * r_arena + 2 * SIGMA
    center = (box / 2, box / 2)
    drive, species = make_species_drive(types)
    order = np.argsort(species, kind="stable")
    assert np.all(order == np.arange(len(species))), "types must be contiguous"
    shapes = [shape_of(t.kind, t.heading) for t in types]
    shape = rigid_body.concatenate_shapes(*shapes) if len(shapes) > 1 else shapes[0]
    sp = species if len(shapes) > 1 else None
    inter = make_interaction(sp, center, None if periodic else r_arena,
                             interacting=interacting)
    init = (make_periodic_init(types, species, L) if periodic
            else make_init(types, species, center, r_arena))
    steps = int(round(t_sample / dt))
    u = np.ones(int(round(T_total / t_sample)))
    runner = make_runner(n_particles=len(species), sigma=SIGMA, box=box,
                         shape=shape, drive_fn=drive, interaction_fn=inter,
                         init=init, dt=dt, steps_per_sample=steps,
                         initial_extra_capacity=32, progress=True)
    centers, angles, box = runner(u, seed=seed)
    meta = {**runner.meta, "n_samples": len(u), "seed": seed, "t_sample": t_sample,
            "r_arena_m": r_arena, "L_m": L, "periodic": periodic,
            "center_m": list(center), "interacting": interacting,
            "types": [asdict(t) | {"R_T_m": t.R_T} for t in types],
            "area_m2": AREA, "v0_m_s": V0, "tag": tag}
    run_dir = save_run(
        arrays={"centers": centers, "angles": angles, "species": species,
                "light": u, "box": box, "sigma": SIGMA,
                **{f"beads_{s}": bead_points(t.kind, t.heading) * SIGMA
                   for s, t in enumerate(types)}},
        meta=meta, label=label + ("" if interacting else "_ctrl"),
        prefix="coll", results_dir=COLL_DIR, source=__file__)
    return run_dir


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def load(run_dir):
    run_dir = Path(run_dir)
    z = np.load(run_dir / "trajectory.npz")
    meta = json.loads((run_dir / "metadata.json").read_text())
    nt = len(meta["types"])
    beads = [z[f"beads_{s}"] for s in range(nt)]
    return z["centers"], z["angles"], z["species"], beads, meta


class _Static:
    """Per-frame structure (clusters, density cells) for one arena + species."""

    def __init__(self, sp, beads, cen, r_ar):
        self.sp, self.beads, self.cen = sp, beads, np.asarray(cen)
        nb = np.array([len(beads[s]) for s in sp])
        self.body_of_bead = np.repeat(np.arange(len(sp)), nb)
        self.cell = 3 * SIDE
        self.g = np.arange(self.cen[0] - r_ar, self.cen[0] + r_ar, self.cell)
        gx, gy = np.meshgrid(self.g, self.g)
        c = self.cell
        self.ok = np.all([np.hypot(gx + dx - self.cen[0], gy + dy - self.cen[1]) < r_ar
                          for dx in (0, c) for dy in (0, c)], axis=0)

    def frame(self, c, th):
        """-> largest cluster / N, weighted mean cluster size, local-phi std."""
        rot = np.stack([np.stack([np.cos(th), -np.sin(th)], -1),
                        np.stack([np.sin(th), np.cos(th)], -1)], -2)
        b = np.concatenate([c[i] + self.beads[s] @ rot[i].T
                            for i, s in enumerate(self.sp)])
        pairs = cKDTree(b).query_pairs(1.15 * SIGMA, output_type="ndarray")
        n = len(self.sp)
        lab = np.arange(n)
        if len(pairs):
            bi, bj = self.body_of_bead[pairs[:, 0]], self.body_of_bead[pairs[:, 1]]
            m = bi != bj
            if m.any():
                adj = coo_matrix((np.ones(m.sum()), (bi[m], bj[m])), shape=(n, n))
                _, lab = connected_components(adj, directed=False)
        sizes = np.bincount(lab)
        edges = np.r_[self.g, self.g[-1] + self.cell]
        h, _, _ = np.histogram2d(c[:, 0], c[:, 1], bins=[edges, edges])
        dstd = (h.T[self.ok] * AREA / self.cell ** 2).std() if self.ok.any() else 0.0
        return sizes.max() / n, (sizes ** 2).sum() / sizes.sum(), dstd


def rsa_placement(types, species, center, r_wall, rng, max_tries=400_000):
    """Random sequential placement: drop bodies one at a time at a random
    position and heading, reject any bead overlap (< SIGMA) or wall overlap.
    Unlike ``make_init`` (which pushes bodies until they just touch) this is
    an unbiased random excluded-volume configuration. Returns (pos, th) or
    None if it jams before all are placed."""
    pts_by_type = [bead_points(t.kind, t.heading) * SIGMA for t in types]
    center = np.asarray(center, float)
    cell = SIGMA
    grid = {}
    pos, th = [], []
    tries = 0
    for s in species:
        p0 = pts_by_type[s]
        rlim = r_wall - SIGMA / 2
        while True:
            tries += 1
            if tries > max_tries:
                return None
            r = (r_wall) * np.sqrt(rng.uniform())
            a = rng.uniform(0, 2 * np.pi)
            c = center + r * np.array([np.cos(a), np.sin(a)])
            t = rng.uniform(-np.pi, np.pi)
            ro = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            b = c + p0 @ ro.T
            if np.any(np.hypot(*(b - center).T) > rlim):
                continue
            keys = np.floor(b / cell).astype(int)
            bad = False
            for (kx, ky), q in zip(keys, b):
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for o in grid.get((kx + dx, ky + dy), ()):
                            if (o[0] - q[0]) ** 2 + (o[1] - q[1]) ** 2 < SIGMA ** 2:
                                bad = True
                                break
                        if bad: break
                    if bad: break
                if bad: break
            if bad:
                continue
            for (kx, ky), q in zip(keys, b):
                grid.setdefault((kx, ky), []).append(q)
            pos.append(c); th.append(t)
            break
    return np.array(pos), np.array(th)


def metrics_random(run_dir, seeds=(101, 102, 103)):
    """Structure of random non-overlapping (RSA) placements at the same
    composition: the reference for clustering and density."""
    C, A, sp, beads, meta = load(run_dir)
    types = [PType(**{k: v for k, v in t.items() if k != "R_T_m"}) for t in meta["types"]]
    st = _Static(sp, beads, meta["center_m"], meta["r_arena_m"])
    rows = []
    for s in seeds:
        res = rsa_placement(types, sp, meta["center_m"], meta["r_arena_m"],
                            np.random.default_rng(s))
        if res is not None:
            rows.append(st.frame(*res))
    if not rows:            # jammed: fall back to the (contact-rich) start
        rows = [st.frame(C[0], A[0])]
        print("RSA jammed; structure reference = start configuration")
    largest, csize, dstd = np.mean(rows, axis=0)
    return {"largest": float(largest), "csize": float(csize), "dens_std": float(dstd),
            "rsa_ok": len(rows)}


def metrics(run_dir, t_from=0.5, t_to=1.0, stride_s=2.0):
    """Observables averaged over the fraction [t_from, t_to) of the run."""
    C, A, sp, beads, meta = load(run_dir)
    ts = meta["t_sample"]
    k = max(1, int(round(stride_s / ts)))
    i0, i1 = int(t_from * len(C)), int(t_to * len(C)) - k
    cen = np.asarray(meta["center_m"])
    st = _Static(sp, beads, cen, meta["r_arena_m"])
    out = {k_: [] for k_ in ("largest", "csize", "omega", "polar", "nematic",
                             "local_align", "speed", "dens_std")}
    for f in range(i0, i1, k):
        c, th = C[f], A[f]
        largest, csize, dstd = st.frame(c, th)
        out["largest"].append(largest)
        out["csize"].append(csize)
        out["dens_std"].append(dstd)
        v = (C[f + k] - c) / (k * ts)
        rel = c - cen
        rr = np.maximum(np.linalg.norm(rel, axis=1), 1e-12)
        vth = (v[:, 1] * rel[:, 0] - v[:, 0] * rel[:, 1]) / rr
        out["omega"].append(vth.mean() / V0)
        out["speed"].append(np.linalg.norm(v, axis=1).mean() / V0)
        e = np.exp(1j * th)
        out["polar"].append(np.abs(e.mean()))
        out["nematic"].append(np.abs((e ** 2).mean()))
        pp = cKDTree(c).query_pairs(2.5 * SIDE, output_type="ndarray")
        out["local_align"].append(np.cos(th[pp[:, 0]] - th[pp[:, 1]]).mean()
                                  if len(pp) else 0.0)
    return {k_: float(np.mean(v_)) for k_, v_ in out.items()}


def wall_split(run_dir, t_from=0.5, stride_s=10.0):
    """Separate wall accumulation from bulk clustering (2nd half of the run).

    Returns the fraction of particles in clusters touching the wall, the
    largest cluster that does NOT touch it (/ N), and the fraction of centres
    more than two body sizes from the wall."""
    C, A, sp, beads, meta = load(run_dir)
    cen = np.asarray(meta["center_m"])
    R = meta["r_arena_m"]
    stride = max(1, int(round(stride_s / meta["t_sample"])))
    nb = np.array([len(beads[s]) for s in sp])
    own = np.repeat(np.arange(len(sp)), nb)
    n = len(sp)
    rows = []
    for f in range(int(t_from * len(C)), len(C), stride):
        c, th = C[f], A[f]
        rot = np.stack([np.stack([np.cos(th), -np.sin(th)], -1),
                        np.stack([np.sin(th), np.cos(th)], -1)], -2)
        b = np.concatenate([c[i] + beads[s] @ rot[i].T for i, s in enumerate(sp)])
        rb = np.linalg.norm(b - cen, axis=1)
        at_wall = np.zeros(n, bool)
        at_wall[own[rb > R - SIGMA / 2 - 0.15 * SIGMA]] = True
        pr = cKDTree(b).query_pairs(1.15 * SIGMA, output_type="ndarray")
        lab = np.arange(n)
        if len(pr):
            bi, bj = own[pr[:, 0]], own[pr[:, 1]]
            m = bi != bj
            if m.any():
                _, lab = connected_components(
                    coo_matrix((np.ones(m.sum()), (bi[m], bj[m])), shape=(n, n)),
                    directed=False)
        on_wall = np.isin(lab, np.unique(lab[at_wall]))
        bulk = np.bincount(lab[~on_wall]).max() / n if (~on_wall).any() else 0.0
        r = np.linalg.norm(c - cen, axis=1)
        rows.append((on_wall.mean(), bulk, np.mean(r < R - 2 * SIDE)))
    w, bulk, inner = np.mean(rows, axis=0)
    return {"wall_attached": float(w), "bulk_largest": float(bulk),
            "interior": float(inner)}


THRESH = {   # what counts as "collective" against the interactions-off control
    "largest": lambda x, c: x >= max(0.2, 2 * c),
    "omega": lambda x, c: abs(x) - abs(c) >= 0.15,     # more rotation, not less
    "polar": lambda x, c: x - c >= 0.2,
    "local_align": lambda x, c: x - c >= 0.15,
    "dens_std": lambda x, c: x >= 1.5 * c,
}


def reference(mc, mr):
    """Structure metrics from random placements, dynamic ones from the control."""
    return {**mc, **mr}


def verdict(m, ref):
    return [k for k, f in THRESH.items() if f(m[k], ref[k])]


def describe(types):
    parts = []
    for t in types:
        d = f"{t.count}x{t.kind}"
        if t.kind == "triangle":
            d += f"({t.heading})"
        if t.tau:
            d += f" R_T={t.R_T * 1e6:.0f}um"
        if t.K:
            d += f" K={t.K:.1e}"
        if t.F[1]:
            d += f" Fy={t.F[1]:.2e}"
        parts.append(d)
    return " + ".join(parts)


def write_ledger(name, types, phi, run_dir, ctrl_dir, m, mc, m_q3, hits, wall_s):
    COLL_DIR.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        LEDGER.write_text(
            "# Collective-behaviour campaign ledger\n\n"
            "Metrics: 2nd half of the run. In brackets the reference (ref): for "
            "largest and dstd, random non-overlapping placements at the same "
            "composition; for omega, polar, align, the same run with interactions "
            "switched off. "
            "largest = largest contact cluster / N; omega = mean tangential velocity / v0 "
            "(global rotation); polar = |<e^(i theta)>|; align = <cos dtheta> for pairs "
            "< 25 um; dstd = std of local area fraction (30 um cells); speed = <|v|>/v0. "
            "q3 = same metric over the 3rd quarter (convergence check).\n\n"
            "| name | config | phi | largest (ref, q3) | omega (ref) | polar (ref) "
            "| align (ref) | dstd (ref) | speed | hits | run | min |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    f = lambda k: f"{m[k]:.2f} ({mc[k]:.2f})"
    row = (f"| {name} | {describe(types)} | {phi:.2f} "
           f"| {m['largest']:.2f} ({mc['largest']:.2f}, {m_q3['largest']:.2f}) "
           f"| {m['omega']:+.2f} ({mc['omega']:+.2f}) | {f('polar')} | {f('local_align')} "
           f"| {f('dens_std')} | {m['speed']:.2f} | {', '.join(hits) or '-'} "
           f"| {Path(run_dir).name} | {wall_s / 60:.0f} |\n")
    with open(LEDGER, "a") as fh:
        fh.write(row)


# --------------------------------------------------------------------------
# video: true outlines, coloured by heading
# --------------------------------------------------------------------------
def make_movie(run_dir, every_s=2.0, fps=24, trail_s=20.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon, Circle

    run_dir = Path(run_dir)
    C, A, sp, beads, meta = load(run_dir)
    types = meta["types"]
    k = max(1, int(round(every_s / meta["t_sample"])))
    frames = range(0, len(C), k)
    periodic = bool(meta.get("periodic"))
    if periodic:
        Lu = meta["L_m"] * 1e6
        C = np.mod(C, meta["L_m"])               # show every body inside the box
    else:
        cen = np.asarray(meta["center_m"]) * 1e6
        R = meta["r_arena_m"] * 1e6
    outl = [rounded_outline(t["kind"], t["heading"]) for t in types]
    cmap = plt.cm.hsv

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    if periodic:
        ax.set_xlim(0, Lu); ax.set_ylim(0, Lu)
    else:
        ax.set_xlim(cen[0] - R - 5, cen[0] + R + 5)
        ax.set_ylim(cen[1] - R - 5, cen[1] + R + 5)
        ax.add_patch(Circle(cen, R, fill=False, lw=1.5, color="k"))
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ttl = ax.set_title("")
    coll = PatchCollection([], cmap=cmap, edgecolor="k", linewidth=0.4)
    coll.set_clim(0, 2 * np.pi)
    ax.add_collection(coll)
    from matplotlib.collections import LineCollection
    ticks = LineCollection([], colors="k", linewidths=0.8, zorder=6)
    ax.add_collection(ticks)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 360))
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="heading (deg)")

    def patches(f):
        ps = []
        for i, s in enumerate(sp):
            c, th = C[f, i] * 1e6, A[f, i]
            if outl[s] is None:
                ps.append(Circle(c, disk_radius() * 1e6))
            else:
                ro = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                ps.append(Polygon(c + outl[s] * 1e6 @ ro.T, closed=True))
        return ps

    def update(f):
        coll.set_paths(patches(f))
        coll.set_array(np.mod(A[f], 2 * np.pi))
        c = C[f] * 1e6
        hd = c + 0.45 * SIDE * 1e6 * np.c_[np.cos(A[f]), np.sin(A[f])]
        ticks.set_segments(np.stack([c, hd], 1))
        ttl.set_text(f"{run_dir.name}   t = {f * meta['t_sample']:.0f} s")
        return coll, ttl, ticks

    ffmpeg = Path(sys.executable).parent / "ffmpeg"
    if ffmpeg.exists():
        plt.rcParams["animation.ffmpeg_path"] = str(ffmpeg)
        writer, out = animation.FFMpegWriter(fps=fps, bitrate=2400), run_dir / "animation.mp4"
    else:
        writer, out = animation.PillowWriter(fps=fps), run_dir / "animation.gif"
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    ani.save(out, writer=writer, dpi=110)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# campaign driver
# --------------------------------------------------------------------------
def run_config(name, types, phi, *, r_arena, T_total, seed=0, video=True):
    """Interacting run + control, analyse, ledger. Skips if already in the ledger."""
    if LEDGER.exists() and f"| {name} |" in LEDGER.read_text():
        print(f"skip {name} (in ledger)")
        return None
    t0 = time.time()
    print(f"\n##### {name}: {describe(types)}  phi={phi}", flush=True)
    rd = simulate(types, name, r_arena=r_arena, T_total=T_total, seed=seed)
    cd = simulate(types, name, r_arena=r_arena, T_total=T_total, seed=seed,
                  interacting=False)
    m, mq3 = metrics(rd), metrics(rd, 0.5, 0.75)
    mc_dyn, mr = metrics(cd), metrics_random(rd)
    mc = reference(mc_dyn, mr)
    hits = verdict(m, mc)
    (Path(rd) / "summary.json").write_text(json.dumps(
        {"metrics": m, "reference": mc, "control_dynamics": mc_dyn,
         "random_structure": mr, "q3": mq3, "hits": hits,
         "control_dir": str(cd), "phi": phi}, indent=1))
    if video:
        try:
            make_movie(rd)
        except Exception as e:                       # never lose a run to the video
            print("video failed:", e)
    write_ledger(name, types, phi, rd, cd, m, mc, mq3, hits, time.time() - t0)
    print(f"{name}: hits={hits}  m={m}  ctrl={mc}", flush=True)
    return rd


def phase1_configs():
    """Shape x drive x density screen, walled arena, light everywhere inside."""
    R_ARENA = 150e-6
    drives = {
        "RT20": dict(tau=tau_for_RT(20e-6)),
        "RT60": dict(tau=tau_for_RT(60e-6)),
        "straight": dict(),
        "linpol": dict(K=3e-19),
    }
    shapes = [("disk", "tip"), ("square", "tip"), ("triangle", "tip")]
    cfgs = []
    for phi in (0.25, 0.45):
        for kind, head in shapes:
            for dname, kw in drives.items():
                n = counts_for_phi(phi, R_ARENA, [1.0])[0]
                name = f"p1_{kind}_{dname}_phi{int(phi * 100):02d}"
                cfgs.append((name, [PType(kind, n, head, **kw)], phi))
    return cfgs, dict(r_arena=R_ARENA, T_total=900.0)


# ==========================================================================
# periodic box (Phase 2): bulk behaviour without walls
# ==========================================================================
def rsa_periodic(types, species, L, rng, max_tries=2_000_000):
    """Random sequential placement in a periodic box: random position and
    heading, reject any bead overlap (< SIGMA, minimum image). Unbiased random
    excluded-volume configuration. Returns (pos, th) or None if it jams."""
    pts_by_type = [bead_points(t.kind, t.heading) * SIGMA for t in types]
    ncell = max(3, int(L // SIGMA))
    cell = L / ncell
    grid = {}
    pos, th, tries = [], [], 0
    for s in species:
        p0 = pts_by_type[s]
        while True:
            tries += 1
            if tries > max_tries:
                return None
            c = rng.uniform(0, L, 2)
            t = rng.uniform(-np.pi, np.pi)
            ro = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            b = np.mod(c + p0 @ ro.T, L)
            keys = np.floor(b / cell).astype(int) % ncell
            bad = False
            for (kx, ky), q in zip(keys, b):
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for o in grid.get(((kx + dx) % ncell, (ky + dy) % ncell), ()):
                            d = o - q
                            d -= L * np.round(d / L)
                            if d[0] * d[0] + d[1] * d[1] < SIGMA ** 2:
                                bad = True
                                break
                        if bad: break
                    if bad: break
                if bad: break
            if bad:
                continue
            for (kx, ky), q in zip(keys, b):
                grid.setdefault((kx, ky), []).append(q)
            pos.append(c); th.append(t)
            break
    return np.array(pos), np.array(th)


def make_periodic_init(types, species, L):
    def init(n_particles, box, seed):
        res = rsa_periodic(types, species, L, np.random.default_rng(seed))
        if res is None:
            raise RuntimeError("random placement jammed; lower phi")
        pos, th = res
        return (rigid_body.RigidBody(jnp.asarray(pos), jnp.asarray(th)),
                random.PRNGKey(seed))
    init.__name__ = "rsa_periodic"
    return init


def counts_for_phi_box(phi, L, fractions):
    n = int(round(phi * L ** 2 / AREA))
    c = np.floor(np.asarray(fractions) * n).astype(int)
    c[0] += n - c.sum()
    return c


class _StaticP:
    """Per-frame structure in a periodic box: clusters, density cells, lanes."""

    def __init__(self, sp, beads, L):
        self.sp, self.beads, self.L = sp, beads, L
        nb = np.array([len(beads[s]) for s in sp])
        self.own = np.repeat(np.arange(len(sp)), nb)
        self.ncell = max(1, int(round(L / (3 * SIDE))))
        self.cell = L / self.ncell

    def _wrap(self, x):
        x = np.mod(x, self.L)
        x[x >= self.L] = 0.0
        return x

    def frame(self, c, th):
        L, n = self.L, len(self.sp)
        rot = np.stack([np.stack([np.cos(th), -np.sin(th)], -1),
                        np.stack([np.sin(th), np.cos(th)], -1)], -2)
        b = self._wrap(np.concatenate([c[i] + self.beads[s] @ rot[i].T
                                       for i, s in enumerate(self.sp)]))
        pr = cKDTree(b, boxsize=L).query_pairs(1.15 * SIGMA, output_type="ndarray")
        lab = np.arange(n)
        if len(pr):
            bi, bj = self.own[pr[:, 0]], self.own[pr[:, 1]]
            m = bi != bj
            if m.any():
                _, lab = connected_components(
                    coo_matrix((np.ones(m.sum()), (bi[m], bj[m])), shape=(n, n)),
                    directed=False)
        sizes = np.bincount(lab)
        cw = self._wrap(c.copy())
        h, _, _ = np.histogram2d(cw[:, 0], cw[:, 1], bins=self.ncell,
                                 range=[[0, L], [0, L]])
        phi_loc = h * AREA / self.cell ** 2
        idx = np.minimum((cw / self.cell).astype(int), self.ncell - 1)
        phi_of_particle = phi_loc[idx[:, 0], idx[:, 1]]
        # pairs of centres for alignment / lanes
        pp = cKDTree(cw, boxsize=L).query_pairs(2.5 * SIDE, output_type="ndarray")
        return dict(largest=sizes.max() / n, csize=(sizes ** 2).sum() / sizes.sum(),
                    dens_std=float(phi_loc.std()),
                    f_dense=float(np.mean(phi_of_particle > 0.55)),
                    pairs=pp)


def _lane_index(th, pp):
    """Segregation of +x and -x movers: (f_same - f0) / (1 - f0) over close pairs,
    f0 = what random mixing of the two populations gives. 0 = mixed, 1 = lanes."""
    sgn = np.cos(th) > 0
    if len(pp) == 0:
        return 0.0
    p = sgn.mean()
    f0 = p ** 2 + (1 - p) ** 2
    if f0 >= 1:
        return 0.0
    f_same = np.mean(sgn[pp[:, 0]] == sgn[pp[:, 1]])
    return float((f_same - f0) / (1 - f0))


P_KEYS = ("largest", "csize", "dens_std", "f_dense", "polar", "nematic", "flock",
          "local_align", "lanes", "speed")


def metrics_periodic(run_dir, t_from=0.5, t_to=1.0, stride_s=2.0, series_s=10.0):
    """Bulk observables over [t_from, t_to) of the run + coarse time series."""
    C, A, sp, beads, meta = load(run_dir)
    L, ts = meta["L_m"], meta["t_sample"]
    k = max(1, int(round(stride_s / ts)))
    st = _StaticP(sp, beads, L)
    out = {key: [] for key in P_KEYS}
    for f in range(int(t_from * len(C)), int(t_to * len(C)) - k, k):
        c, th = C[f], A[f]
        r = st.frame(c, th)
        for key in ("largest", "csize", "dens_std", "f_dense"):
            out[key].append(r[key])
        d = C[f + k] - c
        d -= L * np.round(d / L)
        v = d / (k * ts)
        e = np.exp(1j * th)
        out["polar"].append(np.abs(e.mean()))
        out["nematic"].append(np.abs((e ** 2).mean()))
        out["flock"].append(np.linalg.norm(v.mean(0)) / V0)
        out["speed"].append(np.linalg.norm(v, axis=1).mean() / V0)
        pp = r["pairs"]
        out["local_align"].append(np.cos(th[pp[:, 0]] - th[pp[:, 1]]).mean() if len(pp) else 0.0)
        out["lanes"].append(_lane_index(th, pp))
    return {key: float(np.mean(v)) for key, v in out.items()}


def series_periodic(run_dir, every_s=10.0):
    """Polar order, largest cluster and dense fraction vs time (for plots)."""
    C, A, sp, beads, meta = load(run_dir)
    st = _StaticP(sp, beads, meta["L_m"])
    k = max(1, int(round(every_s / meta["t_sample"])))
    rows = []
    for f in range(0, len(C), k):
        r = st.frame(C[f], A[f])
        rows.append([f * meta["t_sample"], float(np.abs(np.exp(1j * A[f]).mean())),
                     r["largest"], r["f_dense"], _lane_index(A[f], r["pairs"])])
    return {"t_polar_largest_fdense_lanes": rows}


def metrics_random_periodic(run_dir, seeds=(101, 102, 103)):
    """Structure of random (RSA) placements at the same composition."""
    C, A, sp, beads, meta = load(run_dir)
    types = [PType(**{k: v for k, v in t.items() if k != "R_T_m"}) for t in meta["types"]]
    st = _StaticP(sp, beads, meta["L_m"])
    rows = []
    for s in seeds:
        res = rsa_periodic(types, sp, meta["L_m"], np.random.default_rng(s))
        if res is not None:
            r = st.frame(*res)
            rows.append([r["largest"], r["csize"], r["dens_std"], r["f_dense"]])
    if not rows:
        r = st.frame(C[0], A[0])
        rows = [[r["largest"], r["csize"], r["dens_std"], r["f_dense"]]]
    m = np.mean(rows, axis=0)
    return dict(zip(("largest", "csize", "dens_std", "f_dense"), map(float, m)))


THRESH_P = {   # vs reference: structure -> random placement, dynamics -> control
    "cluster": lambda m, r: m["largest"] >= max(0.2, 2 * r["largest"]),
    "mips": lambda m, r: (m["f_dense"] - r["f_dense"] >= 0.2
                          and m["dens_std"] >= 1.5 * r["dens_std"]),
    "polar": lambda m, r: m["polar"] - r["polar"] >= 0.2,
    "flock": lambda m, r: m["flock"] - r["flock"] >= 0.2,
    "align": lambda m, r: m["local_align"] - r["local_align"] >= 0.15,
    "lanes": lambda m, r: m["lanes"] - r["lanes"] >= 0.2,
}

LEDGER_P = COLL_DIR / "LEDGER_periodic.md"


def write_ledger_p(name, types, phi, L, rd, m, ref, mq3, hits, wall_s):
    COLL_DIR.mkdir(parents=True, exist_ok=True)
    if not LEDGER_P.exists():
        LEDGER_P.write_text(
            "# Periodic-box (bulk) ledger\n\n"
            "2nd half of the run; in brackets the reference: largest, f_dense -> "
            "random non-overlapping placement; polar, flock, align, lanes -> same run "
            "with interactions off. largest = largest contact cluster / N; f_dense = "
            "fraction of particles in 30 um cells with local phi > 0.55; flock = "
            "|<v>|/v0 (whole-system drift); lanes = +x/-x mover segregation (0 mixed, "
            "1 lanes); q3 = polar over the 3rd quarter (convergence).\n\n"
            "| name | config | phi | L um | largest (ref) | f_dense (ref) | polar (ref, q3) "
            "| flock (ref) | align (ref) | lanes (ref) | speed | hits | run | min |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    g = lambda k_, f="{:.2f}": f"{f.format(m[k_])} ({f.format(ref[k_])})"
    row = (f"| {name} | {describe(types)} | {phi:.2f} | {L * 1e6:.0f} | {g('largest')} "
           f"| {g('f_dense')} | {m['polar']:.2f} ({ref['polar']:.2f}, {mq3['polar']:.2f}) "
           f"| {g('flock')} | {g('local_align', '{:+.2f}')} | {g('lanes', '{:+.2f}')} "
           f"| {m['speed']:.2f} | {', '.join(hits) or '-'} | {Path(rd).name} "
           f"| {wall_s / 60:.0f} |\n")
    with open(LEDGER_P, "a") as fh:
        fh.write(row)


def run_config_periodic(name, types, phi, *, L, T_total, seed=0, video=True):
    """Interacting run + interactions-off control in a periodic box; analyse."""
    if LEDGER_P.exists() and f"| {name} |" in LEDGER_P.read_text():
        print(f"skip {name} (in ledger)")
        return None
    t0 = time.time()
    print(f"\n##### {name}: {describe(types)}  phi={phi}  L={L * 1e6:.0f} um", flush=True)
    rd = simulate(types, name, L=L, T_total=T_total, seed=seed)
    cd = simulate(types, name, L=L, T_total=T_total, seed=seed, interacting=False)
    m, mq3 = metrics_periodic(rd), metrics_periodic(rd, 0.5, 0.75)
    mc, mr = metrics_periodic(cd), metrics_random_periodic(rd)
    ref = {**mc, **mr}
    hits = [k_ for k_, f in THRESH_P.items() if f(m, ref)]
    (Path(rd) / "summary.json").write_text(json.dumps(
        {"metrics": m, "reference": ref, "control_dynamics": mc, "random_structure": mr,
         "q3": mq3, "hits": hits, "control_dir": str(cd), "phi": phi,
         **series_periodic(rd)}, indent=1))
    if video:
        try:
            make_movie(rd)
        except Exception as e:
            print("video failed:", e)
    write_ledger_p(name, types, phi, L, rd, m, ref, mq3, hits, time.time() - t0)
    print(f"{name}: hits={hits}", flush=True)
    return rd


def phase2_configs():
    """Bulk screen: periodic box, straight / linear-pol / chiral, two densities."""
    L = 400e-6
    cases = [
        ("tri_straight", PType("triangle", heading="tip")),
        ("tribase_straight", PType("triangle", heading="base")),
        ("square_straight", PType("square")),
        ("disk_straight", PType("disk")),
        ("tri_linpol", PType("triangle", heading="tip", K=3e-19)),
        ("square_linpol", PType("square", K=3e-19)),
        ("tri_RT60", PType("triangle", heading="tip", tau=tau_for_RT(60e-6))),
    ]
    cfgs = []
    for phi in (0.25, 0.45):
        n = counts_for_phi_box(phi, L, [1.0])[0]
        for cname, t in cases:
            tt = PType(**{**asdict(t), "count": n})
            cfgs.append((f"p2_{cname}_phi{int(phi * 100):02d}", [tt], phi))
    return cfgs, dict(L=L, T_total=1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma-separated config names")
    a = ap.parse_args()

    if a.phase == "test":
        types = [PType("disk", 6, tau=tau_for_RT(30e-6)),
                 PType("square", 6, K=3e-19),
                 PType("triangle", 6, "tip")]
        dirs = []
        for interacting in (True, False):
            rd = simulate(types, "test_mix", r_arena=40e-6, T_total=20.0,
                          interacting=interacting)
            print(rd, metrics(rd))
            dirs.append(rd)
        print(make_movie(dirs[0], every_s=1.0))
        return

    runner = run_config
    if a.phase == "phase1":
        cfgs, kw = phase1_configs()
    elif a.phase == "phase2":
        cfgs, kw = phase2_configs()
        runner = run_config_periodic
    else:
        raise SystemExit(f"unknown phase {a.phase}")
    if a.only:
        keep = set(a.only.split(","))
        cfgs = [c for c in cfgs if c[0] in keep]
    for i, (name, types, phi) in enumerate(cfgs):
        if i % a.nworkers == a.worker:
            runner(name, types, phi, **kw)


if __name__ == "__main__":
    main()
