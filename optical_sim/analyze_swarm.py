"""Collective-behaviour diagnostics for a swarm trajectory.

Answers "is there collective behaviour, and on what length scale?" from a saved
run (the dict from ``run_swarm``, a run folder, or a ``trajectory.npz``). It
computes, from the centre trajectories:

  - per-particle velocities (minimum-image, so periodic wrapping is handled)
  - coarse-grained velocity + vorticity field        -> flows / vortices / edge currents
  - velocity spatial correlation C(r) + length xi     -> how far motion is correlated
  - collective rotation Omega about the crowd centre  -> net co-rotation (chiral signal)
  - pair correlation g(r)                              -> local order / crystallization

A correlation length ``xi`` much larger than the particle size, a structured
vorticity field, or a non-zero collective ``Omega`` are the quantitative
signatures of collective behaviour (vs. independent spinners, where xi ~ one
particle and the field is noise).

Usage:  python analyze_swarm.py results/swarm_<...>/
"""

import json
import numpy as np

from simulate_swarm import _as_run


def _frame_dt(run):
    """Seconds between recorded frames, from metadata if present (else 1)."""
    meta = run.get("meta", {})
    if "dt" in meta and "record_every" in meta:
        return float(meta["dt"]) * float(meta["record_every"])
    return 1.0


def velocities(centers, box, frame_dt=1.0):
    """Minimum-image velocities, shape (T-1, N, 2)."""
    d = np.diff(centers, axis=0)
    d -= box * np.round(d / box)
    return d / frame_dt


def velocity_correlation(centers, vel, box, nbins=40, frames=None):
    """Normalized velocity correlation C(r) = <v_i . v_j>(r) / <v^2>.

    Averaged over a set of frames and all pairs. Returns (r_centers, C).
    """
    T = vel.shape[0]
    frames = frames if frames is not None else np.linspace(0, T - 1, 15).astype(int)
    rmax = box / 2
    edges = np.linspace(0, rmax, nbins + 1)
    rc = 0.5 * (edges[:-1] + edges[1:])
    num = np.zeros(nbins)
    cnt = np.zeros(nbins)
    v2_acc, v2_n = 0.0, 0
    for f in frames:
        c = centers[f]
        v = vel[f] - vel[f].mean(0)                    # fluctuations (drop drift)
        d = c[:, None, :] - c[None, :, :]
        d -= box * np.round(d / box)
        r = np.linalg.norm(d, axis=-1)
        vv = v @ v.T                                   # (N, N) dot products
        iu = np.triu_indices(len(c), k=1)
        b = np.digitize(r[iu], edges) - 1
        ok = (b >= 0) & (b < nbins)
        np.add.at(num, b[ok], vv[iu][ok])
        np.add.at(cnt, b[ok], 1.0)
        v2_acc += np.sum(v ** 2); v2_n += len(v)
    v2 = v2_acc / max(v2_n, 1)
    C = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan) / max(v2, 1e-300)
    return rc, C


def correlation_length(rc, C):
    """First r where C(r) drops below 1/e (linear interp); NaN if it never does.

    Empty small-r bins (no pairs closer than ~one particle) are dropped first.
    """
    thr = 1.0 / np.e
    m = np.isfinite(C)
    rc, C = rc[m], C[m]
    below = np.where(C < thr)[0]
    if len(below) == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float(rc[0])
    x0, x1, y0, y1 = rc[i - 1], rc[i], C[i - 1], C[i]
    return float(x0 + (thr - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)


def coarse_field(centers, vel, box, nb=14, frames=None):
    """Coarse-grained mean velocity field and its vorticity on an nb x nb grid."""
    T = vel.shape[0]
    frames = frames if frames is not None else np.linspace(0, T - 1, 30).astype(int)
    edges = np.linspace(0, box, nb + 1)
    vx = np.zeros((nb, nb)); vy = np.zeros((nb, nb)); cnt = np.zeros((nb, nb))
    for f in frames:
        c = centers[f] % box
        ix = np.clip(np.digitize(c[:, 0], edges) - 1, 0, nb - 1)
        iy = np.clip(np.digitize(c[:, 1], edges) - 1, 0, nb - 1)
        np.add.at(vx, (ix, iy), vel[f][:, 0])
        np.add.at(vy, (ix, iy), vel[f][:, 1])
        np.add.at(cnt, (ix, iy), 1.0)
    cnt = np.maximum(cnt, 1)
    vx /= cnt; vy /= cnt
    cell = box / nb
    dvy_dx = np.gradient(vy, cell, axis=0)
    dvx_dy = np.gradient(vx, cell, axis=1)
    vort = dvy_dx - dvx_dy
    xc = 0.5 * (edges[:-1] + edges[1:])
    return xc, vx, vy, vort


def collective_rotation(centers, vel, box):
    """Mean angular velocity of the crowd about its centre, per frame (T-1,)."""
    T = vel.shape[0]
    out = np.zeros(T)
    for f in range(T):
        c = centers[f]
        com = c.mean(0)
        d = c - com
        d -= box * np.round(d / box)
        r2 = np.sum(d ** 2, axis=1)
        cross = d[:, 0] * vel[f][:, 1] - d[:, 1] * vel[f][:, 0]
        out[f] = np.sum(cross) / max(np.sum(r2), 1e-300)
    return out


def azimuthal_profile(centers, vel, box, nbins=18, frames=None):
    """Mean azimuthal (tangential) velocity vs radius from the crowd centre.

    A rim current / collective rotation shows up as a clear non-zero v_az(r),
    typically peaking near the droplet edge. Returns (r_centers [m], v_az [m/s]).
    """
    T = vel.shape[0]
    frames = frames if frames is not None else np.linspace(0, T - 1, 40).astype(int)
    rsum = np.zeros(nbins); vsum = np.zeros(nbins); cnt = np.zeros(nbins)
    rmax_acc = []
    for f in frames:
        c = centers[f]
        com = c.mean(0)
        d = c - com
        d -= box * np.round(d / box)
        r = np.linalg.norm(d, axis=1)
        rmax_acc.append(r.max())
        that = np.stack([-d[:, 1], d[:, 0]], axis=1) / (r[:, None] + 1e-30)  # tangent
        v_az = np.sum(vel[f] * that, axis=1)
        edges = np.linspace(0, np.max(r) + 1e-30, nbins + 1)
        b = np.clip(np.digitize(r, edges) - 1, 0, nbins - 1)
        np.add.at(rsum, b, r); np.add.at(vsum, b, v_az); np.add.at(cnt, b, 1.0)
    rc = rsum / np.maximum(cnt, 1)
    v_az = vsum / np.maximum(cnt, 1)
    return rc, v_az


def pair_correlation(centers, box, side, nbins=60, frames=None):
    """Radial distribution g(r) of centres (2D), in units of the particle side."""
    T = centers.shape[0]
    frames = frames if frames is not None else np.linspace(0, T - 1, 15).astype(int)
    N = centers.shape[1]
    rmax = box / 2
    edges = np.linspace(0, rmax, nbins + 1)
    rc = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(nbins)
    for f in frames:
        c = centers[f]
        d = c[:, None, :] - c[None, :, :]
        d -= box * np.round(d / box)
        r = np.linalg.norm(d, axis=-1)
        r = r[np.triu_indices(N, k=1)]
        hist += np.histogram(r, bins=edges)[0]
    ring = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    rho = N / box ** 2
    norm = len(frames) * 0.5 * N * rho * ring
    g = hist / np.maximum(norm, 1e-300)
    return rc / side, g


def analyze(run, out_path=None, show=False):
    """Run all diagnostics and (optionally) save a 4-panel figure."""
    run = _as_run(run)
    centers = np.asarray(run["centers"])
    box = run["box"]
    side = run.get("side") or run.get("meta", {}).get("side_m", 1.0)
    fdt = _frame_dt(run)
    vel = velocities(centers, box, fdt)
    cen = centers[:-1]                                 # align with vel frames

    rc, C = velocity_correlation(cen, vel, box)
    xi = correlation_length(rc, C)
    xg, vx, vy, vort = coarse_field(cen, vel, box)
    Omega = collective_rotation(cen, vel, box)
    rg, g = pair_correlation(centers, box, side)
    r_az, v_az = azimuthal_profile(cen, vel, box)

    print(f"velocity correlation length xi = {xi*1e6:.1f} um  "
          f"({xi/side:.1f} particle sizes)")
    print(f"collective rotation Omega = {np.nanmean(Omega):.3e} rad/s "
          f"(particle spin omega ~ 0.055 rad/s)")
    print(f"peak azimuthal speed |v_az| = {np.nanmax(np.abs(v_az))*1e6:.3f} um/s "
          f"(rim-current signal; particle self-propulsion ~ 2.4 um/s)")
    print(f"g(r) first peak  = {rg[np.nanargmax(g)]:.2f} particle sizes")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax[1, 2].axis("off")

    a = ax[0, 0]
    sp = np.hypot(vx, vy)
    im = a.imshow(vort.T, origin="lower", extent=[0, box*1e6, 0, box*1e6],
                  cmap="RdBu_r", vmin=-np.nanmax(np.abs(vort)),
                  vmax=np.nanmax(np.abs(vort)))
    a.quiver(np.repeat(xg*1e6, len(xg)), np.tile(xg*1e6, len(xg)),
             vx.ravel(), vy.ravel(), color="k", scale_units="xy", alpha=0.6)
    a.set_title("velocity field + vorticity"); a.set_xlabel("x (µm)"); a.set_ylabel("y (µm)")
    fig.colorbar(im, ax=a, fraction=0.046, label="vorticity (1/s)")

    a = ax[0, 1]
    a.plot(rc*1e6, C, lw=1.5)
    a.axhline(1/np.e, color="0.6", ls="--", lw=1, label="1/e")
    if np.isfinite(xi):
        a.axvline(xi*1e6, color="crimson", ls=":", label=f"xi={xi*1e6:.0f} µm")
    a.axhline(0, color="0.8", lw=0.8)
    a.set_xlabel("r (µm)"); a.set_ylabel("C(r)")
    a.set_title("velocity correlation"); a.legend()

    a = ax[1, 0]
    a.plot(np.arange(len(Omega))*fdt, Omega, lw=1.0)
    a.axhline(0, color="0.7", lw=0.8)
    a.set_xlabel("t (s)"); a.set_ylabel("Ω about crowd centre (rad/s)")
    a.set_title("collective rotation")

    a = ax[1, 1]
    a.plot(rg, g, lw=1.5); a.axhline(1, color="0.7", ls="--", lw=1)
    a.set_xlabel("r / particle size"); a.set_ylabel("g(r)")
    a.set_title("pair correlation"); a.set_xlim(0, 6)

    a = ax[0, 2]
    a.plot(r_az * 1e6, v_az * 1e6, lw=1.5, marker="o", ms=3)
    a.axhline(0, color="0.7", lw=0.8)
    a.set_xlabel("r from crowd centre (µm)"); a.set_ylabel("mean azimuthal v (µm/s)")
    a.set_title("edge / rim current")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"saved {out_path}")
    if show:
        return fig
    plt.close(fig)
    return {"xi": xi, "Omega_mean": float(np.nanmean(Omega)),
            "rc": rc, "C": C, "g_r": (rg, g)}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    run = sys.argv[1] if len(sys.argv) > 1 else None
    if run is None:
        raise SystemExit("usage: python analyze_swarm.py <run_dir | trajectory.npz>")
    out = str(Path(run) / "analysis.png") if Path(run).is_dir() else "analysis.png"
    analyze(run, out_path=out)
