"""Checks for the spheropolygon contact model (spheropolygon.py).

    python validate_spheropolygon.py            # numerical tests (CPU is fine)
    python validate_spheropolygon.py --videos   # + side-by-side scenario videos

Numerical tests:
  1. disk-disk: energy = 1/2 k (2R - d)^2 exactly, zero torque (smooth sphere).
  2. tip on the middle of a face: force exactly normal to the face (no friction).
  3. random overlapping mixture: total force = 0, angular momentum conserved,
     autodiff force/torque = finite differences of the energy.
  4. wall: a disk pressed on the wall gets a radial force and no torque.

Videos (results/validate_spheropolygon/): the same start simulated with the
bead model (left, beads drawn) and spheropolygons (right, true outline), so
the difference in contact friction is visible directly.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax_md import space, rigid_body, quantity

import collective as c
import spheropolygon as S
from swarm import make_runner

OUT = Path(__file__).parent / "results" / "validate_spheropolygon"
UM = 1e-6


def _energy(types, species, box=300 * UM, center=None, r_wall=None, interacting=True):
    disp, _ = space.periodic(box)
    it = S.make_interaction(S.Shapes.from_types(types), species, center=center, r_wall=r_wall,
                            interacting=interacting)
    return it.build(box, disp)


def _forces(E, nf, pos, th):
    b = rigid_body.RigidBody(jnp.asarray(pos, float), jnp.asarray(th, float))
    nb = nf.allocate(b) if nf is not None else None
    e = lambda bb: E(bb, nb) if nb is not None else E(bb)
    return b, e, quantity.force(e)(b)


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def test_disk_pair():
    R = c.disk_radius()
    types = [c.PType("disk", 1), c.PType("disk", 1)]
    E, nf = _energy(types, np.arange(2))
    d = 2 * R - 0.4 * UM
    b, e, F = _forces(E, nf, [[100 * UM, 100 * UM], [100 * UM + d * np.cos(0.7), 100 * UM + d * np.sin(0.7)]],
                      [0.3, -2.0])
    ana = 0.5 * S.K_CONTACT * (2 * R - d) ** 2
    En = float(e(b))
    tq = float(np.abs(np.asarray(F.orientation)).max())
    fdir = np.asarray(F.center[1]) / np.linalg.norm(np.asarray(F.center[1]))
    return all([
        check("disk-disk energy", abs(En - ana) < 1e-6 * ana, f"{En:.4e} vs analytic {ana:.4e}"),
        check("disk-disk torque", tq == 0.0, f"max |torque| = {tq:.1e} N m"),
        check("disk-disk force along centres", abs(abs(fdir @ [np.cos(0.7), np.sin(0.7)]) - 1) < 1e-9,
              f"direction {fdir.round(6)}"),
    ])


def test_face_normal():
    types = [c.PType("triangle", 1), c.PType("square", 1)]
    E, nf = _energy(types, np.arange(2))
    tip = S.core_of("triangle")[0][0, 0] + S.R_ROUND                 # rounded tip, body frame
    x_sq = 100 * UM + tip + c.SIDE / 2 - 0.3 * UM                     # 0.3 um overlap
    _, _, F = _forces(E, nf, [[100 * UM, 100 * UM], [x_sq, 101.5 * UM]], [0.0, 0.0])
    f = np.asarray(F.center[0])
    return check("tip on face: force normal to face", abs(f[1]) <= 1e-12 * abs(f[0]) and f[0] < 0,
                 f"F = {f} N (y must be 0)")


def test_random_mixture(seed=0):
    rng = np.random.default_rng(seed)
    kinds = ["triangle", "square", "disk", "triangle", "square", "disk"]
    types = [c.PType(k, 1, "tip") for k in kinds]
    n = len(types)
    # place on a tight ring so most neighbours overlap a little
    a = 2 * np.pi * np.arange(n) / n
    pos = 150 * UM + 10.5 * UM * np.c_[np.cos(a), np.sin(a)] + rng.normal(0, 0.3 * UM, (n, 2))
    th = rng.uniform(-np.pi, np.pi, n)
    E, nf = _energy(types, np.arange(n))
    b, e, F = _forces(E, nf, pos, th)
    Fc, T = np.asarray(F.center), np.asarray(F.orientation)
    scale = np.abs(Fc).max()
    L = np.cross(pos, Fc).sum() + T.sum()
    # finite differences (float64)
    h, h_t = 1e-10, 1e-6
    fd_c = np.zeros_like(pos)
    fd_t = np.zeros(n)
    for i in range(n):
        for k in range(2):
            p1, p2 = pos.copy(), pos.copy()
            p1[i, k] += h; p2[i, k] -= h
            fd_c[i, k] = -(float(e(rigid_body.RigidBody(jnp.asarray(p1), jnp.asarray(th))))
                           - float(e(rigid_body.RigidBody(jnp.asarray(p2), jnp.asarray(th))))) / (2 * h)
        t1, t2 = th.copy(), th.copy()
        t1[i] += h_t; t2[i] -= h_t
        fd_t[i] = -(float(e(rigid_body.RigidBody(jnp.asarray(pos), jnp.asarray(t1))))
                    - float(e(rigid_body.RigidBody(jnp.asarray(pos), jnp.asarray(t2))))) / (2 * h_t)
    err_c = np.abs(fd_c - Fc).max() / scale
    err_t = np.abs(fd_t - T).max() / max(np.abs(T).max(), 1e-30)
    return all([
        check("mixture has contacts", float(e(b)) > 0, f"E = {float(e(b)):.2e} J"),
        check("total force = 0", np.abs(Fc.sum(0)).max() < 1e-9 * scale, f"{Fc.sum(0)}"),
        check("angular momentum conserved", abs(L) < 1e-9 * scale * 1e-5, f"sum(r x F) + sum(tau) = {L:.1e}"),
        check("force = finite difference", err_c < 1e-4, f"rel. error {err_c:.1e}"),
        check("torque = finite difference", err_t < 1e-4, f"rel. error {err_t:.1e}"),
    ])


def test_wall():
    Rw, cen = 50 * UM, (100 * UM, 100 * UM)
    types = [c.PType("disk", 1)]
    E, _ = _energy(types, None, center=cen, r_wall=Rw, interacting=False)   # wall only
    a = 1.1
    r = Rw - c.disk_radius() + 0.2 * UM
    b = rigid_body.RigidBody(jnp.asarray([[cen[0] + r * np.cos(a), cen[1] + r * np.sin(a)]]), jnp.asarray([2.0]))
    F = quantity.force(lambda bb: E(bb))(b)
    f = np.asarray(F.center[0])
    return all([
        check("wall force radial, inward", abs(f @ [np.sin(a), -np.cos(a)]) < 1e-12 * np.linalg.norm(f)
              and f @ [np.cos(a), np.sin(a)] < 0, f"F = {f}"),
        check("wall torque on disk = 0", float(F.orientation[0]) == 0.0, f"{float(F.orientation[0]):.1e}"),
    ])


# --------------------------------------------------------------------------
# scenario videos: beads (left) vs spheropolygons (right), same start
# --------------------------------------------------------------------------
def simulate_scenario(model, types, pos, th, T, t_sample=0.2, dt=1.6e-3, r_wall=None,
                      box=300 * UM, seed=0):
    drive, species = c.make_species_drive(types)
    center = (box / 2, box / 2)
    shapes = [c.shape_of(t.kind, t.heading) for t in types]
    if len(types) > 1:
        shape, sp = rigid_body.concatenate_shapes(*shapes), species
    else:
        shape, sp = shapes[0], None
    if model == "beads":
        inter = c.make_interaction(sp, center, r_wall)
    else:
        inter = S.make_interaction(S.Shapes.from_types(types), sp, center=center, r_wall=r_wall)
    pos = np.asarray(pos, float) + np.asarray(center)
    init = lambda n, b, s: (rigid_body.RigidBody(jnp.asarray(pos), jnp.asarray(th, float)),
                            random.PRNGKey(s))
    runner = make_runner(n_particles=len(species), sigma=c.SIGMA, box=box, shape=shape,
                         drive_fn=drive, interaction_fn=inter, init=init, dt=dt,
                         steps_per_sample=int(round(t_sample / dt)), initial_extra_capacity=8)
    C, A, _ = runner(np.ones(int(round(T / t_sample))), seed=seed)
    return C - np.asarray(center), A, species


def render(name, title, types, runs, t_sample, r_wall=None, lim=40, fps=25, trail=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.patches import Polygon, Circle
    OUT.mkdir(parents=True, exist_ok=True)
    (Cb, Ab, sp), (Cs, As, _) = runs
    beads = [c.bead_points(t.kind, t.heading) * c.SIGMA / UM for t in types]
    sph = [S.rounded_outline(*S.core_of(t.kind, t.heading)) / UM for t in types]
    cols = plt.cm.tab10(np.arange(10))
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.6))
    fig.suptitle(title, fontsize=11)
    for ax, lab in zip(axs, ("bead model (beads drawn)", "spheropolygon (true outline)")):
        ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(lab, fontsize=10); ax.set_xlabel("x (µm)")
    axs[0].set_ylabel("y (µm)")
    txt = fig.text(0.5, 0.02, "", ha="center")

    def rot(t):
        return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    def draw(f):
        for ax, (C, A), is_bead in zip(axs, ((Cb, Ab), (Cs, As)), (True, False)):
            for p in list(ax.patches) + list(ax.lines):
                p.remove()
            if r_wall is not None:
                ax.add_patch(Circle((0, 0), r_wall / UM, fill=False, lw=1.5, color="k"))
            for i, s in enumerate(sp):
                c0, R = C[f, i] / UM, rot(A[f, i])
                col = cols[i % 10]
                if trail:
                    ax.plot(C[:f + 1, i, 0] / UM, C[:f + 1, i, 1] / UM, "-", color=col, lw=0.7, alpha=0.6)
                if is_bead:
                    for q in beads[s] @ R.T + c0:
                        ax.add_patch(Circle(q, c.SIGMA / 2 / UM, fc=col, ec="k", lw=0.3, alpha=0.55))
                else:
                    ax.add_patch(Polygon(sph[s] @ R.T + c0, closed=True, fc=col, ec="k", lw=0.6, alpha=0.6))
                h = c0 + 6 * np.array([np.cos(A[f, i]), np.sin(A[f, i])])
                ax.annotate("", h, c0, arrowprops=dict(arrowstyle="->", lw=1.2, color="k"))
        txt.set_text(f"t = {f * t_sample:.1f} s     arrows: heading (= drive direction)")
        return []

    ani = animation.FuncAnimation(fig, draw, frames=range(len(Cb)), blit=False)
    out = OUT / f"{name}.mp4"
    import sys
    ff = Path(sys.executable).parent / "ffmpeg"
    if ff.exists():
        plt.rcParams["animation.ffmpeg_path"] = str(ff)
        ani.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=1800), dpi=90)
    else:
        out = out.with_suffix(".gif")
        ani.save(out, writer=animation.PillowWriter(fps=fps), dpi=70)
    plt.close(fig)
    return out


def scenarios():
    """(name, title, types, pos [m] rel. to centre, theta, T [s], r_wall)."""
    P = lambda k, **kw: c.PType(k, 1, kw.pop("heading", "tip"), **kw)
    passive = dict(F=(0.0, 0.0))
    return [
        ("01_disks_headon_offset", "two disks, head-on, 3 µm off-centre (straight swimmers)",
         [P("disk"), P("disk")], [[-15 * UM, 0], [15 * UM, 3 * UM]], [0.0, np.pi], 40.0, None),
        ("02_triangles_tip_to_tip", "two triangles, tip to tip, 2 µm off-centre",
         [P("triangle"), P("triangle")], [[-16 * UM, 0], [16 * UM, 2 * UM]], [0.0, np.pi], 60.0, None),
        ("03_triangle_into_square_face", "triangle (active) hits a passive square off-centre",
         [P("triangle"), P("square", **passive)], [[-20 * UM, 0], [6 * UM, 3 * UM]], [0.0, 0.0], 60.0, None),
        ("04_squares_face_to_face", "two squares, face to face, 3 µm off-centre",
         [P("square"), P("square")], [[-14 * UM, 0], [14 * UM, 3 * UM]], [0.0, np.pi], 60.0, None),
        ("05_chiral_disk_at_wall", "circle swimmer (R_T = 25 µm) in a 30 µm arena: stuck at the wall or sliding?",
         [P("disk", tau=c.tau_for_RT(25 * UM))], [[0, 0]], [0.0], 120.0, 30 * UM),
        ("06_triangle_at_wall", "straight triangle hits the wall at 30°",
         [P("triangle")], [[0, 0]], [np.deg2rad(30)], 80.0, 30 * UM),
        ("07_chiral_disk_cluster", "7 circle swimmers (R_T = 30 µm) start as a touching cluster, heads inward",
         [P("disk", tau=c.tau_for_RT(30 * UM)) for _ in range(7)],
         [[0, 0]] + [[11.4 * UM * np.cos(a), 11.4 * UM * np.sin(a)] for a in np.arange(6) * np.pi / 3],
         [0.0] + [a + np.pi for a in np.arange(6) * np.pi / 3], 120.0, None),
    ]


def make_videos():
    t_sample = 0.2
    for name, title, types, pos, th, T, r_wall in scenarios():
        runs = [simulate_scenario(m, types, pos, th, T, t_sample, r_wall=r_wall)
                for m in ("beads", "spheropolygon")]
        out = render(name, title, types, runs, t_sample, r_wall=r_wall,
                     lim=40 if r_wall is None else (r_wall / UM + 3))
        print("video:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", action="store_true")
    a = ap.parse_args()
    jax.config.update("jax_enable_x64", True)
    ok = all([test_disk_pair(), test_face_normal(), test_random_mixture(), test_wall()])
    print("ALL PASS" if ok else "SOME TESTS FAILED")
    if a.videos:
        make_videos()
