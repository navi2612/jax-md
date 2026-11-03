"""Checks for the spheropolygon contact model (spheropolygon.py).

    python validate_spheropolygon.py

Always float64: these are exact identities, tighter than float32 rounding.

  1. disk-disk: energy = 1/2 k (2R - d)^2 exactly, zero torque (smooth sphere).
  2. tip on the middle of a face: force exactly normal to the face (no friction).
  3. random overlapping mixture: total force = 0, angular momentum conserved,
     autodiff force/torque = finite differences of the energy.
  4. wall: a disk pressed on the wall gets a radial force and no torque.
"""
import sys
from collections import namedtuple

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax_md import space, rigid_body, quantity

import spheropolygon as S

UM = 1e-6
Type = namedtuple("Type", "kind heading", defaults=("tip",))


def _energy(types, species, box=300 * UM):
    disp, _ = space.periodic(box)
    return S.make_interaction(S.Shapes.from_types(types), species).build(box, disp)


def _forces(E, nf, pos, th):
    b = rigid_body.RigidBody(jnp.asarray(pos, float), jnp.asarray(th, float))
    nb = nf.allocate(b) if nf is not None else None
    e = lambda bb: E(bb, nb) if nb is not None else E(bb)
    return b, e, quantity.force(e)(b)


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def test_disk_pair():
    R = S.disk_radius()
    types = [Type("disk"), Type("disk")]
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
    types = [Type("triangle"), Type("square")]
    E, nf = _energy(types, np.arange(2))
    tip = S.core_of("triangle")[0][0, 0] + S.R_ROUND                 # rounded tip, body frame
    x_sq = 100 * UM + tip + S.SIDE / 2 - 0.3 * UM                     # 0.3 um overlap
    _, _, F = _forces(E, nf, [[100 * UM, 100 * UM], [x_sq, 101.5 * UM]], [0.0, 0.0])
    f = np.asarray(F.center[0])
    return check("tip on face: force normal to face", abs(f[1]) <= 1e-12 * abs(f[0]) and f[0] < 0,
                 f"F = {f} N (y must be 0)")


def test_random_mixture(seed=0):
    rng = np.random.default_rng(seed)
    kinds = ["triangle", "square", "disk", "triangle", "square", "disk"]
    types = [Type(k) for k in kinds]
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
    wall = S.make_circular_wall(S.Shapes.from_types([Type("disk")]), cen, Rw)
    a = 1.1
    r = Rw - S.disk_radius() + 0.2 * UM
    b = rigid_body.RigidBody(jnp.asarray([[cen[0] + r * np.cos(a), cen[1] + r * np.sin(a)]]), jnp.asarray([2.0]))
    F = wall(b)
    f = np.asarray(F.center[0])
    return all([
        check("wall force radial, inward", abs(f @ [np.sin(a), -np.cos(a)]) < 1e-12 * np.linalg.norm(f)
              and f @ [np.cos(a), np.sin(a)] < 0, f"F = {f}"),
        check("wall torque on disk = 0", float(F.orientation[0]) == 0.0, f"{float(F.orientation[0]):.1e}"),
    ])


if __name__ == "__main__":
    ok = all([test_disk_pair(), test_face_normal(), test_random_mixture(), test_wall()])
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
