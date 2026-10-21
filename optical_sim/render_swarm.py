"""Render a swarm run to a pretty mp4/gif.

Takes a run dict, a run folder written by ``runs.save_run``, or a bare ``.npz``,
and animates it: bodies drawn in their actual geometry, periodic "ghost" copies
so they wrap smoothly across the box edges, fading center trails, and a turbo
colormap.

Three body styles, chosen automatically by ``body="auto"``:

``"beads"``   every bead of the rigid body, rotated into the lab frame -- the
              faithful picture of a ``grid_shape(n)`` body. Used when the run
              carries ``body_points`` with more than one bead.
``"square"``  an oriented square outline of side ``body_side``; cheaper and
              cleaner than beads for large N. Used when a side is known but the
              bead layout is not.
``"disk"``    one disk of diameter ``sigma`` per body -- the single-bead case.

    python render_swarm.py                      # renders swarm_run.npz
    from render_swarm import make_video
    make_video("results/swarm_20260914-.../")   # any saved run
    make_video(run, body="square")              # force a style
"""

import numpy as np

from runs import as_run


def _pick_writer(out_path):
    """Choose mp4 (ffmpeg, pointed at the env binary) or fall back to gif."""
    import sys
    import matplotlib
    from pathlib import Path
    from matplotlib.animation import writers

    cand = Path(sys.executable).parent / "ffmpeg"      # ffmpeg next to this python
    if cand.exists():
        matplotlib.rcParams["animation.ffmpeg_path"] = str(cand)

    p = Path(out_path)
    if p.suffix.lower() == ".mp4":
        if writers.is_available("ffmpeg"):
            return str(p), "ffmpeg"
        p = p.with_suffix(".gif")
        print("ffmpeg unavailable -> writing GIF instead")
    return str(p), "pillow"


def _body_geometry(run, body, body_points, body_side, sigma):
    """Resolve the drawing style -> (mode, points (B,2) [um], side [um]).

    Looks for ``body_points`` / ``side`` / ``body_side_m`` in the run (top level
    or in ``meta``) unless given explicitly.
    """
    meta = run.get("meta", {}) if isinstance(run.get("meta", {}), dict) else {}

    if body_points is None:
        body_points = run.get("body_points")
    if body_points is not None:
        body_points = np.asarray(body_points, dtype=float).reshape(-1, 2) * 1e6

    if body_side is None:
        body_side = run.get("side", meta.get("body_side_m"))
    if body_side is not None:
        body_side = float(body_side) * 1e6
    elif body_points is not None and len(body_points) > 1:
        body_side = float(body_points.max() - body_points.min()) + sigma

    if body in (None, "auto"):
        if body_points is not None and len(body_points) > 1:
            body = "beads"
        elif body_side is not None and body_side > sigma * 1.01:
            body = "square"
        else:
            body = "disk"

    if body == "beads" and (body_points is None or len(body_points) < 1):
        raise ValueError("body='beads' needs `body_points` (the run has none)")
    if body == "square" and body_side is None:
        raise ValueError("body='square' needs `body_side` (the run has none)")
    if body == "disk":
        body_points = np.zeros((1, 2))
    return body, body_points, body_side


def make_video(run, out_path="swarm.mp4", target_frames=240,
               trail_frames=80, fps=24, draw_ghosts=True, title=None,
               body="auto", body_points=None, body_side=None,
               color_by="index"):
    """Render a time-lapse video of a swarm run.

    ``run`` may be the dict returned by ``run``/``run_one``, a run folder, or a
    ``trajectory.npz`` path -- so any old run can be re-rendered without
    re-simulating. See the module docstring for the ``body`` styles.

    ``color_by``:
      ``"index"``        turbo over particle index (fixed, carries no information)
      ``"orientation"``  cyclic hsv over the current heading, recoloured every
                         frame -> polar order, lanes, vortices show as patches;
                         trails are drawn grey
      ``"start_angle"``  cyclic hsv over the polar angle of the starting position
                         around the box centre (fixed) -> shows mixing
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import (EllipseCollection, LineCollection,
                                        PolyCollection)
    from matplotlib.animation import FuncAnimation

    run = as_run(run)
    centers = np.asarray(run["centers"]) * 1e6        # (T, N, 2) um
    angles = np.asarray(run["angles"])                # (T, N)
    box = float(run["box"]) * 1e6
    sigma = float(run["sigma"]) * 1e6
    T_all, N, _ = centers.shape

    mode, pts, side = _body_geometry(run, body, body_points, body_side, sigma)
    head_len = 0.5 * (side if mode == "square" else sigma)

    skip = max(1, int(T_all // target_frames))
    C = centers[::skip]
    A = angles[::skip]
    T = C.shape[0]

    if color_by == "index":
        cmap = plt.get_cmap("turbo")
        face = np.array([cmap(i / max(N - 1, 1)) for i in range(N)])
    elif color_by == "orientation":
        cmap = plt.get_cmap("hsv")
        face = cmap(np.mod(A[0], 2 * np.pi) / (2 * np.pi))
    elif color_by == "start_angle":
        cmap = plt.get_cmap("hsv")
        d0 = centers[0] - box / 2
        face = cmap(np.mod(np.arctan2(d0[:, 1], d0[:, 0]), 2 * np.pi) / (2 * np.pi))
    else:
        raise ValueError(f"color_by must be index|orientation|start_angle, got {color_by!r}")
    ghosts = (np.array([[dx, dy] for dx in (-box, 0, box) for dy in (-box, 0, box)])
              if draw_ghosts else np.zeros((1, 2)))
    G = len(ghosts)
    B = 4 if mode == "square" else len(pts)
    rep = G if mode == "square" else B * G   # face colours per body
    square = 0.5 * (side or sigma) * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])

    def _rot(f):
        cos, sin = np.cos(A[f]), np.sin(A[f])
        return np.stack([np.stack([cos, -sin], -1),
                         np.stack([sin, cos], -1)], -2)      # (N,2,2)

    def frame_geometry(f):
        R = _rot(f)
        # body-frame points -> lab, then + center, then + every ghost offset
        local = np.einsum("nij,bj->nbi", R, square if mode == "square" else pts)
        world = C[f][:, None, :] + local                      # (N,B,2)
        world = world[:, :, None, :] + ghosts[None, None, :, :]   # (N,B,G,2)
        head = C[f] + np.einsum("nij,j->ni", R, [head_len, 0.0])
        cg = (C[f][:, None, :] + ghosts[None, :, :]).reshape(-1, 2)
        hg = (head[:, None, :] + ghosts[None, :, :]).reshape(-1, 2)
        ticks = np.stack([cg, hg], axis=1)                    # (N*G,2,2)
        if mode == "square":
            verts = world.transpose(0, 2, 1, 3).reshape(-1, 4, 2)   # (N*G,4,2)
            return verts, ticks
        return world.reshape(-1, 2), ticks                    # (N*B*G,2) offsets

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal"); ax.set_xlim(0, box); ax.set_ylim(0, box)
    ax.set_xlabel("x  (µm)"); ax.set_ylabel("y  (µm)")

    geom0, ticks0 = frame_geometry(0)
    if mode == "square":
        bodies = PolyCollection(geom0, facecolors=np.repeat(face, G, axis=0),
                                edgecolors="k", linewidths=0.7, alpha=0.6,
                                zorder=5)
    else:
        bodies = EllipseCollection(
            widths=sigma, heights=sigma, angles=0.0, units="xy",
            offsets=geom0, offset_transform=ax.transData,
            facecolors=np.repeat(face, B * G, axis=0), edgecolors="k",
            linewidths=0.4 if B > 1 else 0.7, alpha=0.6, zorder=5)
    ticklc = LineCollection(ticks0, colors="k", linewidths=1.0, zorder=6)
    ax.add_collection(bodies); ax.add_collection(ticklc)
    trail_col = ["0.6"] * N if color_by == "orientation" else face
    trails = [ax.plot([], [], color=trail_col[i], lw=1.0, alpha=0.8, zorder=3)[0]
              for i in range(N)]
    if color_by != "index":
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 360))
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 90, 180, 270, 360])
        cb.set_label("heading (deg)" if color_by == "orientation"
                     else "start position angle (deg)")
    ttl = ax.set_title(title or "")

    def update(f):
        geom, ticks = frame_geometry(f)
        if mode == "square":
            bodies.set_verts(geom)
        else:
            bodies.set_offsets(geom)
        ticklc.set_segments(ticks)
        if color_by == "orientation":
            bodies.set_facecolors(np.repeat(
                cmap(np.mod(A[f], 2 * np.pi) / (2 * np.pi)), rep, axis=0))
        lo = max(0, f - trail_frames)
        for i in range(N):
            tr = C[lo:f + 1, i, :].astype(float).copy()
            if len(tr) > 1:                       # break the trail where it wraps
                brk = np.where(np.abs(np.diff(tr, axis=0)).max(1) > box / 2)[0]
                if len(brk):
                    tr = np.insert(tr, brk + 1, np.nan, axis=0)
            trails[i].set_data(tr[:, 0], tr[:, 1])
        ttl.set_text((title + "  " if title else "") + f"frame {f + 1}/{T}")
        return []

    out_path, writer = _pick_writer(out_path)
    ani = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    ani.save(out_path, writer=writer, fps=fps, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}  [{mode}, N={N}, {T} frames]")
    return out_path


if __name__ == "__main__":
    make_video("swarm_run.npz", "swarm.mp4")
