"""Many interacting 10x10 um metavehicles under CIRCULAR polarization.

Outlook figure for the poster: a periodic "tank" of identical square
metavehicles. Each square is a rigid union of beads (so they sterically repel
as filled squares, not points), driven by the SAME circular-polarization
optical force/torque used in the single-particle panel -> every square is a
spinner (orbit R_T ~ 42 um) and they collide/deflect into a swirling crowd.

Physics is kept consistent with the single-particle notebook cells:
  - translational drag gamma_T = 470 nN.s/m, rotational gamma_R = 8000e-21
  - circular drive: body force [1.135, 0.045] pN, torque 440.44 pN.nm
  - kT at 300 K
Only the interaction is changed to PURELY REPULSIVE soft spheres (no LJ tail).

Run:  python metavehicle_swarm.py            (full run -> png + mp4)
      QUICK=1 python metavehicle_swarm.py    (tiny run, for timing/debug)
"""

import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax import random, lax, jit
from jax_md import space, energy, rigid_body, simulate, quantity, partition

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Arc

QUICK = bool(int(os.environ.get("QUICK", "0")))
RENDER_ONLY = bool(int(os.environ.get("RENDER_ONLY", "0")))  # reuse saved traj
# NL=1 -> neighbor-list interaction (~O(N), scales to many particles)
# NL=0 -> all-pairs interaction (O(N^2), the original path)
USE_NL = bool(int(os.environ.get("NL", "1")))
TAG = "nl" if USE_NL else "pair"
# IMG=1 -> draw the real particle.png at each metavehicle instead of a colored
# square (static figure only; the video always uses squares for speed)
USE_IMAGE = bool(int(os.environ.get("IMG", "0")))
PARTICLE_IMG = "particle.png"

# --------------------------------------------------------------------------
# physical constants  (match the single-particle simulation in the notebook)
# --------------------------------------------------------------------------
kT = jnp.float64(1.38e-23 * 300.0)
gamma_T = jnp.float64(470e-9)      # translational drag  [N.s/m]
gamma_R = jnp.float64(8000e-21)    # rotational drag     [N.m.s/rad]
R_body = jnp.array([[gamma_T, 0.0, 0.0],
                    [0.0, gamma_T, 0.0],
                    [0.0, 0.0, gamma_R]], dtype=jnp.float64)

# circular-polarization optical drive, body frame (same as notebook cell 5)
F_active_body = jnp.array([1.135e-12, 0.045e-12], dtype=jnp.float64)
tau_active = jnp.float64(440.44e-21)

dt = jnp.float64(1e-4)

# --------------------------------------------------------------------------
# geometry: N identical 10 x 10 um square rigid bodies, periodic box
# --------------------------------------------------------------------------
side = 10e-6                       # metavehicle edge length (drawn size)
N = int(os.environ.get("N", "20"))            # number of metavehicles
# density-matched box: hold the area fraction (~0.165) fixed as N grows, so the
# neighbor-list speedup reflects scaling and not just a sparser box
L_box = side * np.sqrt(N / 0.165)  # periodic "tank" side
# n_grid x n_grid beads tile each square. sigma = side/n_grid makes the bead
# spacing equal the bead diameter -> a filled square. A finer grid (smaller
# beads) pushes the corner bead closer to the drawn corner: the residual
# corner overlap between touching squares is ~ 4.14 / n_grid um, so n_grid=10
# (~0.41 um) roughly halves the n_grid=5 (~0.83 um) corner clipping.
n_grid = int(os.environ.get("NGRID", "10"))
sigma = side / n_grid              # bead diameter (== spacing -> filled square)
epsilon = jnp.float64(5e-17)       # repulsion strength (stiff enough vs drive)

# Data/figure label so finer-grid runs don't overwrite earlier ones. Default
# encodes the bead grid (e.g. "g10"); override with VARIANT=... to pin a name.
VARIANT = os.environ.get("VARIANT", f"g{n_grid}")
TRAJ_FILE = f"metavehicle_swarm_traj_{TAG}_{VARIANT}.npz"

# Inset the bead-center grid by one bead radius so the bead *surfaces* reach
# exactly to +-side/2 (the drawn edge). Then steric repulsion switches on right
# when the two drawn squares touch -- no phantom gap, and they can nestle.
_span = side - sigma
_xs = jnp.linspace(-_span / 2, _span / 2, n_grid)
_gx, _gy = jnp.meshgrid(_xs, _xs)
square_pts = jnp.stack([_gx.ravel(), _gy.ravel()], axis=1).astype(jnp.float64)
square = rigid_body.point_union_shape(square_pts, jnp.ones(square_pts.shape[0]))

displacement_fn, shift_fn = space.periodic(L_box)

# ---- all-pairs steric interaction (O(N^2)) --------------------------------
pair_energy = energy.soft_sphere_pair(
    displacement_fn, sigma=sigma, epsilon=epsilon, alpha=2.0,
)
rigid_energy_pair = rigid_body.point_energy(pair_energy, square)
interaction_force_pair = quantity.force(rigid_energy_pair)

# ---- neighbor-list steric interaction (~O(N)) -----------------------------
# Cutoff is `sigma` (soft_sphere is exactly 0 beyond sigma), so this returns
# the SAME forces as all-pairs -- only the cost scaling changes. dr_threshold
# is the Verlet skin: must be small in metres (a fraction of sigma), not the
# default 0.2 which would be 0.2 m and pull in every point.
neighbor_fn, ss_nl_energy = energy.soft_sphere_neighbor_list(
    displacement_fn, L_box,
    sigma=sigma, epsilon=epsilon, alpha=2.0,
    dr_threshold=0.2 * sigma,      # Verlet skin, scaled to the bead size
    capacity_multiplier=2.0,       # headroom for transient clusters
)
nbr_fns, rigid_energy_nl = rigid_body.point_energy_neighbor_list(
    ss_nl_energy, neighbor_fn, square,
)
interaction_force_nl = quantity.force(rigid_energy_nl)


def _active_drive(body):
    # circular optical drive: body-frame force rotated into the lab frame
    R = rigid_body.rotation2d(body.orientation)            # (N, 2, 2) body->lab
    f_active_lab = jnp.einsum("nij,j->ni", R, F_active_body)
    tau = jnp.full(body.orientation.shape, tau_active, dtype=body.center.dtype)
    return f_active_lab, tau


def force_pair(body, **kwargs):
    f_int = interaction_force_pair(body, **kwargs)
    fa, tau = _active_drive(body)
    return rigid_body.RigidBody(f_int.center + fa, f_int.orientation + tau)


def force_nl(body, **kwargs):
    # kwargs carries neighbor=nbrs, threaded in from the integrator step
    f_int = interaction_force_nl(body, **kwargs)
    fa, tau = _active_drive(body)
    return rigid_body.RigidBody(f_int.center + fa, f_int.orientation + tau)


total_force_fn = force_nl if USE_NL else force_pair

init_fn, step_fn = simulate.brownian_generalized_rigid_2d(
    energy_or_force_fn=total_force_fn,
    shift_fn=shift_fn,
    dt=dt,
    kT=kT,
    resistance_body=R_body,
)

# --------------------------------------------------------------------------
# initial condition: non-overlapping grid of squares, random orientations
# --------------------------------------------------------------------------
key = random.PRNGKey(2)
m = int(np.ceil(np.sqrt(N)))
cell = L_box / m
coords = (np.arange(m) + 0.5) * cell
gxx, gyy = np.meshgrid(coords, coords)
centers0 = np.stack([gxx.ravel(), gyy.ravel()], axis=1)[:N]
centers0 = jnp.array(centers0, dtype=jnp.float64)

key, okey = random.split(key)
theta0 = random.uniform(okey, (N,), minval=-jnp.pi, maxval=jnp.pi,
                        dtype=jnp.float64)
body0 = rigid_body.RigidBody(centers0, theta0)
state = init_fn(key, body0)

# --------------------------------------------------------------------------
# run with strided recording (keep memory small)
# --------------------------------------------------------------------------
record_every = 100
n_steps = int(os.environ.get("STEPS", "4000" if QUICK else "600000"))
n_record = max(1, n_steps // record_every)


# --- all-pairs run: simple strided scan -------------------------------------
def _run_pair(state):
    def inner(s, _):
        return step_fn(s), None

    def outer(s, _):
        s, _ = lax.scan(inner, s, xs=None, length=record_every)
        return s, s.position

    return lax.scan(outer, state, xs=None, length=n_record)


# --- neighbor-list run: thread + update nbrs, track buffer overflow ----------
def _run_nl(state, nbrs):
    def inner(carry, _):
        s, nb, ovf = carry
        nb = nb.update(s.position)
        ovf = ovf | (nb.did_buffer_overflow != 0)
        s = step_fn(s, neighbor=nb)
        return (s, nb, ovf), None

    def outer(carry, _):
        carry, _ = lax.scan(inner, carry, xs=None, length=record_every)
        s = carry[0]
        return carry, s.position

    (state, nbrs, ovf), traj = lax.scan(
        outer, (state, nbrs, jnp.array(False)), xs=None, length=n_record)
    return state, traj, ovf


_run_pair_j = jit(_run_pair)
_run_nl_j = jit(_run_nl)

if RENDER_ONLY and os.path.exists(TRAJ_FILE):
    data = np.load(TRAJ_FILE)
    centers, angles = data["centers"], data["angles"]
    print(f"loaded {TRAJ_FILE}: {centers.shape[0]} frames x {centers.shape[1]} "
          "squares")
else:
    # sanity check: neighbor-list and all-pairs must give identical forces
    if QUICK:
        nbrs0 = nbr_fns.allocate(body0)
        fp = force_pair(body0)
        fn = force_nl(body0, neighbor=nbrs0.update(body0))
        dF = float(jnp.max(jnp.abs(fp.center - fn.center)))
        dT = float(jnp.max(jnp.abs(fp.orientation - fn.orientation)))
        print(f"force check  |pair - nl|: dF={dF:.2e} N, dTau={dT:.2e} N.m "
              f"(buffer_overflow={bool(nbrs0.did_buffer_overflow)})")

    t0 = time.time()
    if USE_NL:
        extra_cap = 0
        while True:
            state = init_fn(key, body0)
            nbrs = nbr_fns.allocate(body0, extra_capacity=extra_cap)
            state, traj, ovf = _run_nl_j(state, nbrs)
            jax.block_until_ready(traj.center)
            if not bool(ovf):
                break
            extra_cap += 16
            print(f"  neighbor buffer overflow -> retry extra_capacity="
                  f"{extra_cap}")
            if extra_cap > 256:
                print("  WARNING: still overflowing; results may be wrong")
                break
    else:
        state, traj = _run_pair_j(state)
        jax.block_until_ready(traj.center)
    print(f"sim [{TAG}]: {n_steps} steps x {N} squares  ->  "
          f"{time.time() - t0:.1f} s wall, {n_record} frames")
    centers = np.asarray(traj.center)        # (T, N, 2)  metres
    angles = np.asarray(traj.orientation)    # (T, N)
    np.savez_compressed(TRAJ_FILE, centers=centers, angles=angles)

# --- overlap diagnostic: closest center distance vs square size -------------
final = centers[-1]
d = final[:, None, :] - final[None, :, :]
d -= L_box * np.round(d / L_box)         # minimum image
dist = np.linalg.norm(d, axis=-1)
np.fill_diagonal(dist, np.inf)
print(f"min center-center distance (final): {dist.min() * 1e6:.1f} um "
      f"(square edge {side * 1e6:.0f} um)")

# ==========================================================================
# rendering helpers
# ==========================================================================
Lum = L_box * 1e6
centers_um = centers * 1e6
side_um = side * 1e6


def _periodic_com(xy, L):
    """Circular (toroidal) center of mass of points xy (N, 2) in a box L."""
    ang = 2.0 * np.pi * xy / L
    com = np.arctan2(np.sin(ang).mean(0), np.cos(ang).mean(0))
    return (com % (2.0 * np.pi)) * L / (2.0 * np.pi)


# one constant shift (from the final-frame crowd center) keeps the interacting
# group centered in the frame without distorting relative motion or trails
_shift = Lum / 2.0 - _periodic_com(centers_um[-1], Lum)
centers_w = np.mod(centers_um + _shift, Lum)

cmap = plt.get_cmap("turbo")
colors = [cmap(i / max(N - 1, 1)) for i in range(N)]

_square_body = np.array([[-side_um / 2, -side_um / 2],
                         [+side_um / 2, -side_um / 2],
                         [+side_um / 2, +side_um / 2],
                         [-side_um / 2, +side_um / 2]])
_GHOSTS = np.array([[dx, dy] for dx in (-Lum, 0, Lum) for dy in (-Lum, 0, Lum)])


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _square_verts(cx, cy, theta):
    return (_square_body @ _rot(theta).T) + np.array([cx, cy])


def _unwrap_trail(xy):
    """Break a periodic trail wherever it jumps across the box (insert NaN)."""
    jumps = np.where(np.abs(np.diff(xy, axis=0)) > Lum / 2)[0]
    if len(jumps) == 0:
        return xy
    out = xy.astype(float).copy()
    out = np.insert(out, jumps + 1, np.nan, axis=0)
    return out


def _circular_glyph(ax, handedness=+1, lw=2.6, fontsize=22):
    """Circular-polarization symbol, placed bottom-left just above the scale bar.
    The SYMBOL itself (arc span 20 deg -> 300 deg and arrowhead-tip position on
    the circle) is kept identical to the single-particle figure
    (plot_trajectories._circular_glyph), so the polarization illustration is the
    same across figures -- only the location differs. The arrowhead is drawn via
    annotate (robust) rather than the reference's degenerate FancyArrowPatch."""
    cx, cy = 0.12 * Lum, 0.20 * Lum                # bottom-left, above scale bar
    r = 0.055 * Lum
    t1, t2 = (20, 300) if handedness > 0 else (240, -40)
    ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=min(t1, t2),
                     theta2=max(t1, t2), color="crimson", lw=lw, zorder=10))
    # arrowhead at the leading end (t2): same tip position on the circle as ref
    aend = np.radians(t2)
    tip = (cx + r * np.cos(aend), cy + r * np.sin(aend))
    aback = np.radians(t2 - handedness * 14.0)     # a step back along the arc
    back = (cx + r * np.cos(aback), cy + r * np.sin(aback))
    ax.annotate("", xy=tip, xytext=back,
                arrowprops=dict(arrowstyle="-|>", color="crimson", lw=lw,
                                mutation_scale=16, shrinkA=0, shrinkB=0),
                zorder=10)
    ax.text(cx, cy + r + 0.012 * Lum, r"$E_{pol}$", color="crimson",
            ha="center", va="bottom", fontsize=fontsize, zorder=10)


def _place_image(ax, img, center, span_um, angle_deg, border_color=None):
    """Draw the (square) particle image rotated by angle_deg, spanning
    span_um x span_um centred at `center`, clipped to the panel. Assumes equal
    x/y data scale (true here)."""
    from matplotlib.transforms import Affine2D
    h = 0.5 * span_um
    im = ax.imshow(img, extent=[-h, h, -h, h], aspect="auto", zorder=5,
                   interpolation="bilinear")
    im.set_transform(Affine2D().rotate_deg(angle_deg)
                     .translate(center[0], center[1]) + ax.transData)
    im.set_clip_path(ax.patch)
    if border_color is not None:
        verts = _square_verts(center[0], center[1], np.radians(angle_deg))
        poly = Polygon(verts, closed=True, fill=False, edgecolor=border_color,
                       linewidth=1.6, zorder=6)
        poly.set_clip_path(ax.patch)
        ax.add_patch(poly)


def _style_axes(ax):
    ax.set_aspect("equal")
    ax.set_xlim(0, Lum)
    ax.set_ylim(0, Lum)
    ax.set_facecolor("#f7f4ef")
    # clean panel: no ticks/labels -- the scale bar carries the size reference
    ax.set_xticks([])
    ax.set_yticks([])


def _scale_bar(ax, length_um=20.0, fontsize=19):
    """Black scale bar in the bottom-left corner with a length label above it."""
    x0, y0 = 0.06 * Lum, 0.06 * Lum
    ax.plot([x0, x0 + length_um], [y0, y0], color="black", lw=4,
            solid_capstyle="butt", zorder=11)
    ax.text(x0 + length_um / 2, y0 + 0.015 * Lum, f"{length_um:g} µm",
            ha="center", va="bottom", fontsize=fontsize, zorder=11)


# ==========================================================================
# 1) static poster snapshot: final state + trails
# ==========================================================================
def _visible_ghosts(cx, cy):
    """Periodic copies whose square could fall inside the panel (skip the rest
    so we don't draw ~9x the work, which matters a lot in image mode)."""
    out = []
    for off in _GHOSTS:
        x, y = cx + off[0], cy + off[1]
        if -side_um <= x <= Lum + side_um and -side_um <= y <= Lum + side_um:
            out.append((x, y))
    return out


particle_img = None
if USE_IMAGE:
    import matplotlib.image as mpimg
    particle_img = mpimg.imread(PARTICLE_IMG)

plt.rcParams.update({"font.size": 14})
fig, ax = plt.subplots(figsize=(7.6, 7.6), dpi=300)
_style_axes(ax)

trail_len = min(len(centers_w), 3000)    # recent ~30 s -> visible orbit arcs
for i in range(N):
    tr = _unwrap_trail(centers_w[-trail_len:, i, :])
    ax.plot(tr[:, 0], tr[:, 1], color=colors[i], lw=1.6, alpha=0.9, zorder=3)
    th = angles[-1, i]
    for x, y in _visible_ghosts(*centers_w[-1, i]):
        if USE_IMAGE:
            _place_image(ax, particle_img, (x, y), side_um, np.degrees(th),
                         border_color=colors[i])
        else:
            verts = _square_verts(x, y, th)
            poly = Polygon(verts, closed=True, facecolor=colors[i],
                           edgecolor="black", linewidth=1.0, alpha=0.6,
                           zorder=5)
            poly.set_clip_path(ax.patch)
            ax.add_patch(poly)
            # heading tick (front of the vehicle, local +x)
            front = np.array([x, y]) + _rot(th) @ np.array([side_um / 2, 0.0])
            ln, = ax.plot([x, front[0]], [y, front[1]], color="black", lw=1.4,
                          zorder=6)
            ln.set_clip_path(ax.patch)

_circular_glyph(ax, handedness=+1)
_scale_bar(ax, 20.0)
fig.tight_layout()
_style = "_img" if USE_IMAGE else ""
_base = f"metavehicle_swarm_{TAG}_{VARIANT}{_style}"
# PNG for quick preview; SVG (vector) for the poster -- text + trails stay sharp
# at print size, the embedded particle images come along at native resolution
fig.savefig(f"{_base}.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{_base}.svg", bbox_inches="tight")
print(f"saved {_base}.png and {_base}.svg")
plt.close(fig)

# ==========================================================================
# 2) animation -> mp4
# ==========================================================================
from matplotlib.animation import FuncAnimation

target_frames = 60 if QUICK else 360
skip = max(1, len(centers_w) // target_frames)
cw = centers_w[::skip]
ang = angles[::skip]
Tn = cw.shape[0]
trail_frames = min(Tn, 150)              # growing trail, ~24 s of history

fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=150)
_style_axes(ax)
_circular_glyph(ax, handedness=+1, lw=2.4, fontsize=19)
_scale_bar(ax, 20.0, fontsize=16)
ax.set_title("Interacting metavehicles — circular polarization", fontsize=15,
             pad=8)

polys = [[Polygon(np.zeros((4, 2)), closed=True, facecolor=colors[i],
                  edgecolor="black", linewidth=1.0, alpha=0.55, zorder=5)
          for _ in _GHOSTS] for i in range(N)]
heads = [[ax.plot([], [], color="black", lw=1.3, zorder=6)[0] for _ in _GHOSTS]
         for i in range(N)]
trails = [ax.plot([], [], color=colors[i], lw=1.4, alpha=0.85, zorder=3)[0]
          for i in range(N)]
for i in range(N):
    for p in polys[i]:
        p.set_clip_path(ax.patch)
        ax.add_patch(p)
    for h in heads[i]:
        h.set_clip_path(ax.patch)


def update(frame):
    lo = max(0, frame - trail_frames)
    for i in range(N):
        cx, cy = cw[frame, i]
        th = ang[frame, i]
        for g, off in enumerate(_GHOSTS):
            polys[i][g].set_xy(_square_verts(cx + off[0], cy + off[1], th))
            front = np.array([cx + off[0], cy + off[1]]) + _rot(th) @ np.array(
                [side_um / 2, 0.0])
            heads[i][g].set_data([cx + off[0], front[0]],
                                 [cy + off[1], front[1]])
        tr = _unwrap_trail(cw[lo:frame + 1, i, :])
        trails[i].set_data(tr[:, 0], tr[:, 1])
    return []


ani = FuncAnimation(fig, update, frames=Tn, interval=40, blit=False)
ani.save(f"metavehicle_swarm_{TAG}_{VARIANT}.mp4", writer="ffmpeg", dpi=150)
plt.close(fig)
print(f"saved metavehicle_swarm_{TAG}_{VARIANT}.mp4")
