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

Same interface as ``collective.make_interaction``: ``interaction_fn(shape,
sigma, box, displacement_fn) -> (force_fn, neighbor_fns)``, so it plugs into
``swarm.make_runner`` unchanged; the bead ``shape`` handed in is ignored.
"""
import numpy as np
import jax.numpy as jnp
from jax_md import dataclasses, partition, quantity, rigid_body, space

R_ROUND = 1.0e-6        # corner rounding of polygons [m]
K_CONTACT = 2.5e-5      # stiffness per vertex-edge term [N/m] (= one bead-bead contact)
K_WALL = 1.0e-4         # wall stiffness per vertex [N/m] (as the bead wall)


def core_of(kind, heading="tip", r=R_ROUND):
    """(core vertices [m] in the body frame, sweep radius [m]) of one shape.
    The rounded body has the true outline of ``collective.outline`` with
    corners rounded by ``r``; a disk is its centre point with r = disk radius."""
    import collective as c
    V = c.outline(kind, heading)
    if V is None:
        return np.zeros((1, 2)), c.disk_radius()
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


def make_interaction(shapes, species, center=None, r_wall=None, interacting=True,
                     k=K_CONTACT, k_wall=K_WALL, dr_threshold=2e-6,
                     capacity_multiplier=2.0):
    """Spheropolygon contacts (+ optional circular wall of radius ``r_wall``).

    ``shapes``: ``Shapes``; ``species``: type index per body (None = all type 0).
    ``interacting=False`` keeps only the wall (the control run)."""
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

        def wall_energy(body, sp):
            if r_wall is None:
                return 0.0 * jnp.sum(body.center)
            Vb, _ = geometry(body, sp)
            x = body.center[:, None, :] + Vb - jnp.asarray(center)
            dist = jnp.sqrt(jnp.sum(x ** 2, -1) + 1e-30)          # finite gradient at 0
            over = jnp.maximum(dist + rad[sp][:, None] - r_wall, 0.0)
            return 0.5 * k_wall * jnp.sum(jnp.where(vmask[sp], over ** 2, 0.0))

        def species_of(body):
            nb = body.center.shape[0]
            return (jnp.zeros(nb, int) if species is None else jnp.asarray(species))

        if not interacting:
            def ctrl(body, **kw):
                return wall_energy(body, species_of(body))
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
            return pair_energy(body, neighbor) + wall_energy(body, species_of(body))

        return total, nbr_fns

    def interaction_fn(shape, sigma, box, displacement_fn):
        energy_fn, nbr_fns = build(box, displacement_fn)
        return quantity.force(energy_fn), nbr_fns

    interaction_fn.build = build
    interaction_fn.spec = {"type": "spheropolygon" + ("+circular_wall" if r_wall else ""),
                           "interacting": bool(interacting), "k_N_per_m": k,
                           "k_wall_N_per_m_per_vertex": k_wall, "r_wall_m": r_wall,
                           "radii_m": shapes.r.tolist(), "n_vertices": shapes.n.tolist(),
                           "reference": "Alonso-Marroquin, EPL 83, 14001 (2008)"}
    return interaction_fn
