"""Frictionless spheropolygon contacts for 2D rigid bodies.

A body is a convex polygon *core* (vertices in the body frame) swept by a disk
of radius ``r`` (Minkowski sum): a polygon with corners rounded by ``r``. A disk
is a one-vertex core with ``r`` = disk radius. Contacts follow Alonso-Marroquin,
EPL 83, 14001 (2008): every vertex of one body against every edge of the other,

    E = 1/2 k max(0, r_i + r_j - d)^2,   d = distance(vertex, edge),

so each force acts along the line between the two closest points, i.e. normal
to both surfaces: no friction, and the only torques are those the geometry
gives. Disks therefore collide without any torque; a triangle hitting
off-centre turns.

Counting: summed over vertex(i)-edge(j) for every ordered neighbour pair, so a
face-face contact gets up to 4 terms and a corner-corner one up to 4, a
corner-face one 1 -- the stiffness varies by a small factor with the contact
type, the direction of the force does not. A disk-disk pair (both one-vertex)
would count twice and is weighted by 1/2.

``make_interaction`` gives ``interaction_fn(shape, sigma, box, displacement_fn)
-> (force_fn, neighbor_fns)``, so it plugs into ``swarm.make_runner`` unchanged;
the bead ``shape`` handed in is ignored. ``make_circular_wall`` is the optional
arena wall and ``make_init`` the matching non-overlapping start.
"""
import numpy as np
import jax.numpy as jnp
from jax import random
from jax_md import dataclasses, partition, quantity, rigid_body, space
from scipy.spatial import cKDTree

SIDE = 10e-6            # square side [m]; sets the common area of all shapes
AREA = SIDE ** 2
R_ROUND = 1.0e-6        # corner rounding of polygons [m]
K_CONTACT = 2.5e-5      # stiffness per vertex-edge term [N/m] (= one bead-bead contact)
K_WALL = 1.0e-4         # wall stiffness per vertex [N/m] (as the bead wall)


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


def disk_radius():
    return np.sqrt(AREA / np.pi)


def core_of(kind, heading="tip", r=R_ROUND):
    """(core vertices [m] in the body frame, sweep radius [m]) of one shape.
    The rounded body has the true outline of ``outline`` with corners rounded
    by ``r``; a disk is its centre point with r = disk radius."""
    V = outline(kind, heading)
    if V is None:
        return np.zeros((1, 2)), disk_radius()
    nv = len(V)
    apothem = np.linalg.norm(V[0]) * np.cos(np.pi / nv)
    return V * (apothem - r) / apothem, r


def rounded_outline(core, r, n_arc=8):
    """Closed outline of a spheropolygon [m]: core edges offset by r + corner arcs."""
    if len(core) == 1:
        a = np.linspace(0, 2 * np.pi, 4 * n_arc, endpoint=False)
        return core[0] + r * np.c_[np.cos(a), np.sin(a)]
    pts = []
    n = len(core)
    for i in range(n):
        a, b, c_ = core[i - 1], core[i], core[(i + 1) % n]
        n1 = np.array([(b - a)[1], -(b - a)[0]]); n1 /= np.linalg.norm(n1)
        n2 = np.array([(c_ - b)[1], -(c_ - b)[0]]); n2 /= np.linalg.norm(n2)
        a1, a2 = np.arctan2(n1[1], n1[0]), np.arctan2(n2[1], n2[0])
        da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
        for t in np.linspace(0, 1, n_arc):
            pts.append(b + r * np.array([np.cos(a1 + t * da), np.sin(a1 + t * da)]))
    return np.array(pts)


class Shapes:
    """Per-type core geometry, padded to a common vertex count."""

    def __init__(self, cores, radii):
        nt = len(cores)
        V = max(len(c) for c in cores)
        self.n = np.array([len(c) for c in cores])
        P = np.zeros((nt, V, 2))
        nxt = np.zeros((nt, V), int)
        vmask = np.zeros((nt, V), bool)
        emask = np.zeros((nt, V), bool)
        for s, c in enumerate(cores):
            k = len(c)
            P[s, :k] = c
            P[s, k:] = c[0]                          # padding: harmless copies
            nxt[s] = np.arange(V)
            nxt[s, :k] = (np.arange(k) + 1) % k      # edge v -> next vertex
            vmask[s, :k] = True
            emask[s, :k] = True                      # one-vertex core: degenerate edge
        self.P, self.nxt, self.vmask, self.emask = P, nxt, vmask, emask
        self.r = np.asarray(radii, float)
        self.reach = float(max(np.linalg.norm(c, axis=1).max() + r
                               for c, r in zip(cores, radii)))   # max centre-to-surface

    @classmethod
    def from_types(cls, types, r=R_ROUND):
        cr = [core_of(t.kind, t.heading, r) for t in types]
        return cls([c for c, _ in cr], [q for _, q in cr])


def _point_segment(p, a, b):
    """Distance from p to segment ab (broadcast over leading axes). A zero-length
    segment (one-vertex core) is a point; the double ``where`` keeps its
    gradient finite in float32 too."""
    e = b - a
    ee = jnp.sum(e * e, -1)
    real = ee > 1e-20                                    # edges are ~10 um, ee ~ 1e-10 m^2
    t = jnp.where(real, jnp.clip(jnp.sum((p - a) * e, -1) / jnp.where(real, ee, 1.0),
                                 0.0, 1.0), 0.0)
    q = a + t[..., None] * e
    return jnp.sqrt(jnp.sum((p - q) ** 2, -1) + 1e-30)


def _lab_vertices(P, theta):
    """Body-frame vertices (..., V, 2) rotated by theta (...)."""
    c, s = jnp.cos(theta)[..., None], jnp.sin(theta)[..., None]
    return jnp.stack([c * P[..., 0] - s * P[..., 1], s * P[..., 0] + c * P[..., 1]], -1)


def make_interaction(shapes, species, interacting=True, k=K_CONTACT,
                     dr_threshold=2e-6, capacity_multiplier=2.0):
    """Spheropolygon contacts between bodies.

    ``shapes``: ``Shapes``; ``species``: type index per body (None = all type 0).
    ``interacting=False`` switches the contacts off (the control run)."""
    P = jnp.asarray(shapes.P)
    nxt = jnp.asarray(shapes.nxt)
    vmask = jnp.asarray(shapes.vmask)
    emask = jnp.asarray(shapes.emask)
    rad = jnp.asarray(shapes.r)
    one = jnp.asarray(shapes.n == 1)

    def build(box, displacement_fn):
        """-> (energy_fn(body, neighbor=None), neighbor_fns or None)."""
        def geometry(body, sp):
            Vb = _lab_vertices(P[sp], body.orientation)          # (N, V, 2) rel. to centre
            Eb = jnp.take_along_axis(Vb, nxt[sp][..., None], axis=-2)
            return Vb, Eb

        def species_of(body):
            nb = body.center.shape[0]
            return (jnp.zeros(nb, int) if species is None else jnp.asarray(species))

        if not interacting:
            def ctrl(body, **kw):
                return 0.0 * jnp.sum(body.center)
            return ctrl, None

        cutoff = 2 * shapes.reach
        neighbor_fn = partition.neighbor_list(
            displacement_fn, box, cutoff, dr_threshold=dr_threshold,
            capacity_multiplier=capacity_multiplier)

        def wrap(R):                       # cell list needs centres inside [0, box)
            return jnp.mod(R, box)

        def allocate(body, **kw):
            nbrs = neighbor_fn.allocate(wrap(body.center), **kw)
            return dataclasses.replace(nbrs, update_fn=update)

        def update(body, nbrs, **kw):
            return neighbor_fn.update(wrap(body.center), nbrs, **kw)

        nbr_fns = partition.NeighborListFns(allocate, update)
        d_nbr = space.map_neighbor(displacement_fn)

        def pair_energy(body, neighbor):
            sp = species_of(body)
            N = body.center.shape[0]
            idx = neighbor.idx                                   # (N, K), N = empty
            valid = idx < N
            j = jnp.where(valid, idx, 0)
            Vi, Ei = geometry(body, sp)                          # (N, V, 2)
            Vj, Ej = Vi[j], Ei[j]                                # (N, K, V, 2)
            dR = d_nbr(body.center, body.center[j])              # R_j - R_i  (N, K, 2)
            Aj = Vj + dR[:, :, None, :]                          # j's edges, i's frame
            Bj = Ej + dR[:, :, None, :]
            # vertex a of i  vs  edge b of j  ->  (N, K, V, V)
            d = _point_segment(Vi[:, None, :, None, :], Aj[:, :, None, :, :],
                               Bj[:, :, None, :, :])
            reach = rad[sp][:, None, None, None] + rad[sp[j]][:, :, None, None]
            over = jnp.maximum(reach - d, 0.0)
            m = (valid[:, :, None, None] & vmask[sp][:, None, :, None]
                 & emask[sp[j]][:, :, None, :])
            w = jnp.where(one[sp][:, None] & one[sp[j]], 0.5, 1.0)[:, :, None, None]
            return 0.5 * k * jnp.sum(jnp.where(m, w * over ** 2, 0.0))

        def total(body, neighbor, **kw):
            return pair_energy(body, neighbor)

        return total, nbr_fns

    def interaction_fn(shape, sigma, box, displacement_fn):
        energy_fn, nbr_fns = build(box, displacement_fn)
        return quantity.force(energy_fn), nbr_fns

    interaction_fn.build = build
    interaction_fn.spec = {"type": "spheropolygon",
                           "interacting": bool(interacting), "k_N_per_m": k,
                           "radii_m": shapes.r.tolist(), "n_vertices": shapes.n.tolist(),
                           "reference": "Alonso-Marroquin, EPL 83, 14001 (2008)"}
    return interaction_fn


def make_circular_wall(shapes, center, r_wall, species=None, k_wall=K_WALL):
    """Soft circular wall of radius ``r_wall``, acting on every vertex of a body.

    ``wall_fn(body) -> RigidBody(force, torque)`` for ``swarm.make_runner``."""
    P = jnp.asarray(shapes.P)
    vmask = jnp.asarray(shapes.vmask)
    rad = jnp.asarray(shapes.r)
    c0 = jnp.asarray(center)

    def wall_energy(body):
        nb = body.center.shape[0]
        sp = jnp.zeros(nb, int) if species is None else jnp.asarray(species)
        x = body.center[:, None, :] + _lab_vertices(P[sp], body.orientation) - c0
        dist = jnp.sqrt(jnp.sum(x ** 2, -1) + 1e-30)              # finite gradient at 0
        over = jnp.maximum(dist + rad[sp][:, None] - r_wall, 0.0)
        return 0.5 * k_wall * jnp.sum(jnp.where(vmask[sp], over ** 2, 0.0))

    wall_force = quantity.force(wall_energy)

    def wall_fn(body):
        return wall_force(body)

    wall_fn.spec = {"type": "circular", "center_m": [float(x) for x in center],
                    "r_wall_m": float(r_wall), "k_wall_N_per_m_per_vertex": float(k_wall)}
    return wall_fn


def make_init(core, r, center=None, r_wall=None, margin=0.08, stages=30, iters=300):
    """Non-overlapping start for ``swarm.make_runner``, one shape.

    With ``center`` and ``r_wall``: inside that circular wall. Without them:
    across the whole periodic box.

    Random centres and headings, then every body grows from 50% to full size
    while overlapping pairs are pushed apart along their separating axis and
    escapees pulled back in; headings stay as drawn. The overlap test works on
    the convex cores, so it also sees a body buried inside another, which the
    contact energy cannot. ``margin``: spare gap as a fraction of r_i + r_j.
    ``core``, ``r``: from ``core_of``."""
    periodic = r_wall is None
    core = np.asarray(core, float)
    r_core = np.linalg.norm(core, axis=1).max()
    if not periodic:
        center = np.asarray(center, float)
        r_lim = r_wall - (r_core + r)
    normals = np.zeros((0, 2))
    if len(core) > 1:
        e = np.roll(core, -1, axis=0) - core
        normals = np.c_[e[:, 1], -e[:, 0]] / np.linalg.norm(e, axis=1, keepdims=True)
        normals *= np.sign(np.sum(normals * core, axis=1, keepdims=True))   # outward

    def rotated(p, th):
        """Body-frame points p (K, 2) at headings th (P,) -> (P, K, 2)."""
        co, si = np.cos(th)[:, None], np.sin(th)[:, None]
        return np.stack([co * p[:, 0] - si * p[:, 1], si * p[:, 0] + co * p[:, 1]], -1)

    def separation(pos, th, g, i, j, box):
        """Largest gap between the cores of pairs (i, j) over the candidate
        axes, and that axis (unit, i -> j). A lower bound on their distance."""
        pj = pos[j]
        if periodic:
            pj = pj - box * np.round((pj - pos[i]) / box)            # nearest image of j
        d = pj - pos[i]
        axes = (d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12))[:, None]
        axes = np.concatenate([axes, rotated(normals, th[i]), -rotated(normals, th[j])], 1)
        Vi = pos[i][:, None] + g * rotated(core, th[i])
        Vj = pj[:, None] + g * rotated(core, th[j])
        ax_T = axes.transpose(0, 2, 1)
        gap = (Vj @ ax_T).min(1) - (Vi @ ax_T).max(1)                # (P, n_axes)
        best = gap.argmax(1)
        k = np.arange(len(d))
        return gap[k, best], axes[k, best]

    def init(n_particles, box, seed):
        rng = np.random.default_rng(seed)
        if periodic:
            pos = rng.uniform(0, box, (n_particles, 2))
        else:
            rr = r_lim * np.sqrt(rng.uniform(size=n_particles))
            a = rng.uniform(0, 2 * np.pi, n_particles)
            pos = center + rr[:, None] * np.c_[np.cos(a), np.sin(a)]
        th = rng.uniform(-np.pi, np.pi, n_particles)

        def relax(g, n_iter):
            nonlocal pos
            need = 2 * r * g * (1 + margin)
            for _ in range(n_iter):
                tree = cKDTree(pos, boxsize=box if periodic else None)
                pairs = tree.query_pairs(2 * g * r_core + need, output_type="ndarray")
                if len(pairs) == 0:
                    return True
                i, j = pairs.T
                gap, ax = separation(pos, th, g, i, j, box)
                keep = gap < need
                if not keep.any():
                    return True
                i, j, gap, ax = i[keep], j[keep], gap[keep], ax[keep]
                push = 0.5 * (1.05 * need - gap)[:, None] * ax
                disp = np.zeros_like(pos)
                cnt = np.zeros(n_particles)
                np.add.at(disp, i, -push); np.add.at(disp, j, push)
                np.add.at(cnt, i, 1); np.add.at(cnt, j, 1)
                pos = pos + disp / np.maximum(cnt, 1)[:, None] * 1.5
                if periodic:
                    pos = np.mod(pos, box)
                    pos[pos >= box] = 0.0                            # mod can round up to box
                else:
                    rel = pos - center
                    dist = np.maximum(np.hypot(*rel.T), 1e-12)
                    pos = center + rel * np.minimum(1.0, r_lim / dist)[:, None]
            return False

        for g in np.linspace(0.5, 1.0, stages):
            relax(g, iters)
        if not relax(1.0, 20 * iters):
            raise RuntimeError(f"could not remove overlaps (n={n_particles}); lower phi")
        return (rigid_body.RigidBody(jnp.asarray(pos), jnp.asarray(th)),
                random.PRNGKey(seed))

    init.__name__ = "spheropolygon_periodic_init" if periodic else "spheropolygon_arena_init"
    return init
