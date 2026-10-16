"""MIPS / collective-behaviour scan for active disks.

Two drive modes:
  circular  constant torque -> chiral disks on circles of radius R_T (default)
  linear    polarization-dependent force and aligning torque from your FDTD
            coefficients -> disks lock to +x or -x (counter-propagating streams)

Usage in your main script:

    from mips_scan import run_mips_scan
    results = run_mips_scan()                      # default plan, prints a table

or a single point:

    from mips_scan import run_mips_point
    res = run_mips_point(R_T=44e-6, phi=0.4, box=150e-6, T_total=1200)

Every run's trajectories are saved to ``out_dir`` so new observables can be
computed later without rerunning.
"""
import os
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# ---- change this import to the file that defines make_runner etc. -----------
SIM_MODULE = "swarm"                    # e.g. "swarm" for swarm.py

DIAMETER = 10e-6                        # disk diameter [m]
F_PROP = 1.135e-12                      # propulsion force [N] (circular pol.)

# (R_T [m], box [m], T_total [s]) -- box >= ~3 R_T, run >= ~10 periods
DEFAULT_PLAN = [
    (20e-6, 150e-6, 600.0),
    (44e-6, 150e-6, 1200.0),
    (80e-6, 250e-6, 2000.0),
]
DEFAULT_PHIS = (0.2, 0.4, 0.6)

# linear polarization: FDTD coefficients [A, B, C] of A cos^2 + B sin^2 + C sin cos
# (design1_res60_linear_30uW) and the power factor used in make_active_optical_drive
LIN_FX = (1.24171, 0.443034, 0.0)       # pN
LIN_FY = (0.0, 0.0, 0.066536)           # pN
LIN_TZ = (0.0, 0.0, 244.48)             # pN nm
LIN_POWER = 3.0


# =============================================================================
# helpers
# =============================================================================
def torque_for_radius(R_T, F, resistance_body):
    """tau giving turning radius R_T for force F:  R_T = F*gamma_R / (tau*gamma_t)."""
    g_t, g_r = float(resistance_body[0][0]), float(resistance_body[2][2])
    return F * g_r / (g_t * R_T)


def n_for_phi(phi, box, diameter=DIAMETER):
    """Number of disks for area fraction phi in a square box."""
    return int(round(phi * box**2 / (np.pi * (diameter / 2) ** 2)))


def disk_mips_positions(n, box, diameter, rng, method="random"):
    """Homogeneous start with uniformly random headings.

    method="random" (default): uniformly random positions, overlaps removed by
    pushing overlapping pairs apart (no lattice rows -> no artificial channels
    for straight swimmers). method="lattice": jittered square lattice.
    """
    if method == "random":
        return _random_relaxed_positions(n, box, diameter, rng), rng.uniform(-np.pi, np.pi, n)
    m = int(np.ceil(np.sqrt(n)))
    a = box / m
    if a < 1.05 * diameter:
        raise ValueError(f"Too dense for a square-lattice start (spacing {a*1e6:.1f} um).")
    ij = np.stack(np.meshgrid(np.arange(m), np.arange(m)), -1).reshape(-1, 2)
    ij = ij[rng.choice(m * m, n, replace=False)]
    jit = 0.5 * (a - 1.05 * diameter)
    pos = (ij + 0.5) * a + rng.uniform(-jit, jit, (n, 2)) / np.sqrt(2)
    theta = rng.uniform(-np.pi, np.pi, n)
    return pos, theta


def _random_relaxed_positions(n, box, diameter, rng, margin=1.05, stages=40,
                              iters_per_stage=400):
    """Random positions, then grow the disks step by step from 60% to
    margin*diameter, pushing overlapping pairs apart at every step (periodic).
    No lattice, no rows. Works up to phi ~ 0.7."""
    pos = rng.uniform(0, box, (n, 2))

    def relax(dmin, iters):
        nonlocal pos
        for _ in range(iters):
            tree = cKDTree(pos, boxsize=box)
            pairs = tree.query_pairs(dmin, output_type="ndarray")
            if len(pairs) == 0:
                return True
            dd = pos[pairs[:, 0]] - pos[pairs[:, 1]]
            dd -= box * np.round(dd / box)
            r = np.maximum(np.hypot(dd[:, 0], dd[:, 1]), 1e-12)
            push = 0.5 * (1.01 * dmin - r)[:, None] * dd / r[:, None]   # aim slightly past dmin
            disp = np.zeros_like(pos)
            np.add.at(disp, pairs[:, 0], push)
            np.add.at(disp, pairs[:, 1], -push)
            pos = np.mod(pos + disp, box)
        return False

    for g in np.linspace(0.6, 1.0, stages):
        relax(g * margin * diameter, iters_per_stage)
    if not relax(margin * diameter, 20 * iters_per_stage):
        raise RuntimeError("Could not remove all overlaps; density too high.")
    return pos


def make_disk_mips_init(diameter=DIAMETER):
    """init(n, box, seed) -> (RigidBody, key) for make_runner."""
    import jax.numpy as jnp
    from jax import random
    from jax_md import rigid_body

    def init(n, box, seed):
        pos, theta = disk_mips_positions(n, box, diameter, np.random.default_rng(seed))
        return (rigid_body.RigidBody(jnp.asarray(pos), jnp.asarray(theta)),
                random.PRNGKey(seed))

    init.__name__ = "disk_mips_init"
    return init


# =============================================================================
# observables
# =============================================================================
def largest_cluster_fraction(pos, box, r_contact):
    """Fraction of particles in the largest contact cluster (periodic box)."""
    n = len(pos)
    tree = cKDTree(np.mod(pos, box), boxsize=box)
    pairs = tree.query_pairs(r_contact, output_type="ndarray")
    if len(pairs) == 0:
        return 1.0 / n
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    return np.bincount(labels).max() / n


def local_area_fractions(pos, box, diameter, cell):
    """Area fraction in square cells of side ~cell (counts disk centres)."""
    nb = max(int(box // cell), 1)
    counts, _, _ = np.histogram2d(pos[:, 0] % box, pos[:, 1] % box, bins=nb,
                                  range=[[0, box], [0, box]])
    return (counts * np.pi * (diameter / 2) ** 2 / (box / nb) ** 2).ravel()


def bimodality_coefficient(x):
    """Sarle's coefficient: > 0.555 hints at a bimodal distribution."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 4 or x.std() == 0:
        return np.nan
    z = (x - x.mean()) / x.std()
    g = np.mean(z**3)
    k = np.mean(z**4) - 3.0
    return (g**2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def lane_segregation(pos, s, box, half_width):
    """Mean fraction of same-direction particles among those within |dy| <
    half_width (whole box length in x). 0.5 = mixed, 1 = perfect lanes."""
    dy = pos[:, 1][:, None] - pos[:, 1][None, :]
    dy -= box * np.round(dy / box)
    m = np.abs(dy) < half_width
    np.fill_diagonal(m, False)
    n = m.sum(1)
    same = ((s[:, None] == s[None, :]) & m).sum(1)
    ok = n > 0
    return float(np.mean(same[ok] / n[ok])) if ok.any() else np.nan


def profile_contrast(pos, box, nb, axis):
    """Std/mean of the particle count in nb strips along `axis`, divided by the
    value for random positions (sqrt(nb/N)). 1 = random, >>1 = bands."""
    counts, _ = np.histogram(pos[:, axis] % box, bins=nb, range=(0, box))
    return float(counts.std() / counts.mean() / np.sqrt(nb / len(pos)))


def analyze_run(centers, box, t_sample, v0, diameter=DIAMETER,
                t_start=None, r_contact_factor=1.05, cell_factor=3.0,
                blocked_threshold=0.5, angles=None):
    """Steady-state observables from one run (averaged over t >= t_start).

    centers: (T, N, 2) [m], angles: (T, N) [rad] (optional, needed for the
    drive-direction observables). Returns a dict of floats plus time series.
    """
    T = len(centers)
    t = np.arange(T) * t_sample
    if t_start is None:
        t_start = 0.5 * t[-1]
    sel = np.where(t >= t_start)[0]

    lcf_t = np.array([largest_cluster_fraction(c, box, r_contact_factor * diameter)
                      for c in centers])
    lcf = lcf_t[sel]

    phi_loc = np.concatenate([local_area_fractions(centers[k], box, diameter,
                                                   cell_factor * diameter)
                              for k in sel])

    # speeds from minimum-image displacements between saved frames
    d = np.diff(centers, axis=0)
    d -= box * np.round(d / box)
    speed = np.hypot(d[..., 0], d[..., 1]) / t_sample            # (T-1, N)
    sp = speed[sel[sel > 0] - 1]

    out_dir_obs = {}
    if angles is not None:
        s = np.sign(np.cos(angles))                                # +1 / -1 along x
        drift = s[:-1] * d[..., 0] / t_sample                      # velocity along own direction
        out_dir_obs["drive_mob"] = float(drift[sel[sel > 0] - 1].mean() / v0)
        out_dir_obs["nematic_S"] = float(np.abs(np.exp(2j * angles[sel]).mean(1)).mean())
        sub = sel[:: max(1, len(sel) // 100)]
        out_dir_obs["lane_seg"] = float(np.nanmean(
            [lane_segregation(np.mod(centers[k], box), s[k], box, 0.5 * diameter)
             for k in sub]))
        nb = max(int(box // (cell_factor * diameter)), 1)
        out_dir_obs["band_x"] = float(np.mean([profile_contrast(centers[k], box, nb, 0)
                                               for k in sub]))
        out_dir_obs["band_y"] = float(np.mean([profile_contrast(centers[k], box, nb, 1)
                                               for k in sub]))

    return {
        **out_dir_obs,
        "lcf_mean": float(lcf.mean()),
        "lcf_std": float(lcf.std()),
        "lcf_max": float(lcf.max()),
        "phi_loc_std": float(phi_loc.std()),
        "bimodality": float(bimodality_coefficient(phi_loc)),
        "speed_rel": float(sp.mean() / v0),
        "blocked_frac": float(np.mean(sp < blocked_threshold * v0)),
        "lcf_t": lcf_t,
        "t": t,
    }


SUMMARY_KEYS = ["lcf_mean", "lcf_std", "lcf_max", "phi_loc_std",
                "bimodality", "speed_rel", "blocked_frac",
                "drive_mob", "nematic_S", "lane_seg", "band_x", "band_y"]


# =============================================================================
# visuals
# =============================================================================
def cluster_sizes_per_particle(pos, box, r_contact):
    """For each particle, the size of the contact cluster it belongs to."""
    n = len(pos)
    tree = cKDTree(np.mod(pos, box), boxsize=box)
    pairs = tree.query_pairs(r_contact, output_type="ndarray")
    if len(pairs) == 0:
        return np.ones(n, int)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    return np.bincount(labels)[labels]


def render_run(npz_path, out_path=None, fps=20, max_frames=600, diameter=DIAMETER,
               r_contact_factor=1.05, dpi=100):
    """Video of a saved run. Disks coloured by the size of their cluster
    (grey = single or pair), black line = heading. Returns the output path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.collections import EllipseCollection, LineCollection

    z = np.load(npz_path)
    linear = "drive" in z.files and str(z["drive"]) == "linear"
    centers, angles = z["centers"] * 1e6, z["angles"]           # um
    box = float(z["box"]) * 1e6
    t_sample = float(z["t_sample"])
    N = centers.shape[1]
    d = diameter * 1e6
    frames = np.unique(np.linspace(0, len(centers) - 1,
                                   min(max_frames, len(centers))).astype(int))

    fig, ax = plt.subplots(figsize=(6.4, 6.8))
    ax.set_xlim(0, box); ax.set_ylim(0, box); ax.set_aspect("equal")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    cmap = plt.get_cmap("plasma_r")
    disks = EllipseCollection(np.full(N, d), np.full(N, d), np.zeros(N), units="xy",
                              offsets=np.zeros((N, 2)), offset_transform=ax.transData,
                              edgecolors="k", linewidths=0.4)
    ax.add_collection(disks)
    heads = LineCollection([], colors="k", linewidths=1.0)
    ax.add_collection(heads)
    if not linear:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03, label="cluster size / N")
    title = ax.set_title("")
    params = (_run_label(z) + f", φ = {float(z['phi']):.2f}, N = {N}"
              + ("   (red → +x, blue → −x)" if linear else ""))

    def draw(k):
        p = np.mod(centers[k], box)
        size = cluster_sizes_per_particle(p * 1e-6, box * 1e-6,
                                          r_contact_factor * diameter)
        if linear:
            col = np.where(np.cos(angles[k])[:, None] > 0,
                           np.array([[0.84, 0.15, 0.16, 0.9]]),
                           np.array([[0.12, 0.37, 0.75, 0.9]]))
        else:
            col = np.where(size[:, None] >= 3, cmap(size / N),
                           np.array([[0.78, 0.78, 0.78, 1.0]]))
        disks.set_offsets(p)
        disks.set_facecolors(col)
        tip = p + 0.5 * d * np.c_[np.cos(angles[k]), np.sin(angles[k])]
        heads.set_segments(np.stack([p, tip], 1))
        title.set_text(f"{params}\nt = {k*t_sample:.0f} s,  largest cluster = "
                       f"{size.max()/N:.2f}")
        return disks, heads, title

    out_path = out_path or str(npz_path).replace(".npz", ".mp4")
    anim = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
    try:
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
    except (FileNotFoundError, RuntimeError):            # no ffmpeg -> gif
        out_path = out_path.rsplit(".", 1)[0] + ".gif"
        anim.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out_path


def plot_run_summary(npz_path, out_path=None, diameter=DIAMETER, cell_factor=3.0):
    """Largest-cluster fraction vs time + local-density histogram (second half)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(npz_path)
    linear = "drive" in z.files and str(z["drive"]) == "linear"
    centers, box = z["centers"], float(z["box"])
    t = np.arange(len(centers)) * float(z["t_sample"])
    omega = float(z["omega"])
    lcf_t = z["lcf_t"]
    sel = np.where(t >= 0.5 * t[-1])[0]
    phi_loc = np.concatenate([local_area_fractions(centers[k], box, diameter,
                                                   cell_factor * diameter) for k in sel])

    if linear:
        fig, axs = plt.subplots(2, 2, figsize=(11, 7.4))
        ax = axs[0]
    else:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(t, lcf_t, lw=1)
    ax[0].axvspan(t[sel[0]], t[-1], color="0.9", zorder=0, label="averaged")
    if omega > 0:
        for m in np.arange(2 * np.pi / omega, t[-1], 2 * np.pi / omega):
            ax[0].axvline(m, color="0.75", lw=0.6, ls=":")
        ax[0].set_title("dotted lines: rotation periods")
    ax[0].set_ylim(0, 1); ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("largest-cluster fraction"); ax[0].legend(loc="upper left")
    nb = max(int(box // (cell_factor * diameter)), 1)
    q = np.pi * (diameter / 2) ** 2 / (box / nb) ** 2            # phi per disk in a cell
    bins = (np.arange(int(round(phi_loc.max() / q)) + 2) - 0.5) * q
    ax[1].hist(phi_loc, bins=bins, density=True, color="C1")
    ax[1].axvline(float(z["phi"]), color="k", ls="--", lw=1, label="mean φ")
    ax[1].set_xlabel(f"local area fraction ({cell_factor:.0f}d cells)")
    ax[1].set_ylabel("probability density"); ax[1].legend()
    ax[1].set_title("one peak: homogeneous   two peaks: phase separation")
    if linear:
        nb = max(int(box // diameter), 1)
        ang = z["angles"]
        edges = np.linspace(0, box, nb + 1)
        imb = np.zeros((len(t), nb)); dens_x = np.zeros((len(t), nb))
        for k in range(len(t)):
            p = np.mod(centers[k], box); s = np.cos(ang[k]) > 0
            npos, _ = np.histogram(p[s, 1], edges); nneg, _ = np.histogram(p[~s, 1], edges)
            imb[k] = (npos - nneg) / np.maximum(npos + nneg, 1)
            dens_x[k], _ = np.histogram(p[:, 0], edges)
        ext = [0, t[-1], 0, box * 1e6]
        im = axs[1, 0].imshow(imb.T, origin="lower", aspect="auto", cmap="coolwarm",
                              vmin=-1, vmax=1, extent=ext, interpolation="nearest")
        axs[1, 0].set_title("lanes: direction imbalance along y\n(horizontal stripes = lanes)")
        axs[1, 0].set_xlabel("time (s)"); axs[1, 0].set_ylabel("y (µm)")
        fig.colorbar(im, ax=axs[1, 0], label="(n₊ − n₋)/(n₊ + n₋)")
        im2 = axs[1, 1].imshow(dens_x.T, origin="lower", aspect="auto", cmap="magma_r",
                               extent=ext, interpolation="nearest")
        axs[1, 1].set_title("jams: particle count along x\n(dark stripes = bands across the drive)")
        axs[1, 1].set_xlabel("time (s)"); axs[1, 1].set_ylabel("x (µm)")
        fig.colorbar(im2, ax=axs[1, 1], label="particles per strip")
    fig.suptitle(_run_label(z) + f", φ = {float(z['phi']):.2f}, N = {int(z['N'])}")
    fig.tight_layout()
    out_path = out_path or str(npz_path).replace(".npz", "_summary.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def _run_label(z):
    if "drive" in z.files and str(z["drive"]) == "linear":
        return "linear polarization"
    return f"$R_T$ = {float(z['R_T'])*1e6:.0f} µm"


# =============================================================================
# running
# =============================================================================
def run_mips_point(R_T, phi, box, T_total, seeds=(0, 1), t_sample=1.0, dt=5e-4,
                   F=F_PROP, diameter=DIAMETER, resistance_body=None,
                   kT=None, epsilon=None, capacity_multiplier=4.0,
                   out_dir="mips_runs", verbose=True, video=True, summary_plot=True,
                   video_max_frames=600, initial_extra_capacity=16,
                   drive="circular", lin_coeffs=(LIN_FX, LIN_FY, LIN_TZ),
                   lin_power=LIN_POWER):
    """Simulate one (R_T, phi) point for several seeds and return the observables.

    Returns a dict with the parameters, seed-averaged observables (``*_mean``
    keys from analyze_run averaged over seeds, plus ``*_seed_std``) and the
    per-seed results under ``"per_seed"``.
    """
    import importlib
    sim = importlib.import_module(SIM_MODULE)
    from trajectory import DEFAULT_RESISTANCE_BODY, KT_300K

    rb = np.asarray(DEFAULT_RESISTANCE_BODY if resistance_body is None else resistance_body)
    kT = KT_300K if kT is None else kT
    eps = 5e-17 * (diameter / 1e-6) ** 2 if epsilon is None else epsilon

    N = n_for_phi(phi, box, diameter)
    phi_real = N * np.pi * (diameter / 2) ** 2 / box**2
    if drive == "linear":
        FX, FY, TZ = lin_coeffs
        drive_fn = sim.make_active_optical_drive(list(FX), list(FY), list(TZ),
                                                 phi_pol=0.0, power=lin_power)
        v0 = FX[0] * 1e-12 * lin_power / rb[0, 0]          # speed when aligned
        tau, omega, period, R_T = 0.0, 0.0, np.nan, np.nan
        k_align = TZ[2] * 1e-21 * lin_power / 2 / rb[2, 2]  # tau = (TZ/2) sin 2a
        if verbose:
            print(f"  linear: v0 = {v0*1e6:.2f} um/s, alignment time "
                  f"1/(2k) = {1/(2*k_align):.1f} s")
    else:
        tau = torque_for_radius(R_T, F, rb)
        v0 = F / rb[0, 0]
        omega = tau / rb[2, 2]
        period = 2 * np.pi / omega
        drive_fn = sim.make_drive(F_body=(F, 0.0), tau=tau)
        if box < 3 * R_T:
            print(f"  WARNING: box {box*1e6:.0f} um < 3 R_T = {3*R_T*1e6:.0f} um")
        if T_total < 5 * period:
            print(f"  WARNING: run is only {T_total/period:.1f} rotation periods")

    steps_per_sample = int(round(t_sample / dt))
    u = np.ones(int(round(T_total / t_sample)))

    runner_kw = dict(
        n_particles=N, sigma=diameter, box=box,
        shape=sim.single_bead_shape(),
        drive_fn=drive_fn,
        init=make_disk_mips_init(diameter),
        epsilon=eps, dt=dt, steps_per_sample=steps_per_sample, kT=kT,
        resistance_body=rb, capacity_multiplier=capacity_multiplier,
        progress=False)
    # The lattice start has (almost) no neighbours, so capacity_multiplier alone
    # gives a tiny buffer that overflows once clusters form -> a full rerun.
    # Reserve slots up front if make_runner supports it (6 suffice for disks).
    import inspect
    if "initial_extra_capacity" in inspect.signature(sim.make_runner).parameters:
        runner_kw["initial_extra_capacity"] = initial_extra_capacity
    runner = sim.make_runner(**runner_kw)

    os.makedirs(out_dir, exist_ok=True)
    per_seed = []
    for seed in seeds:
        t0 = time.time()
        centers, angles, _ = runner(u, seed=seed)
        res = analyze_run(centers, box, t_sample, v0, diameter, angles=angles)
        res["seed"] = seed
        res["wall_s"] = time.time() - t0
        per_seed.append(res)
        head = "LIN" if drive == "linear" else f"RT{R_T*1e6:.0f}"
        tag = f"{head}_phi{phi:.2f}_box{box*1e6:.0f}_s{seed}"
        npz = os.path.join(out_dir, tag + ".npz")
        np.savez_compressed(npz, centers=centers, angles=angles, box=box,
                            t_sample=t_sample, R_T=R_T, phi=phi_real, N=N,
                            tau=tau, F=F, v0=v0, omega=omega, lcf_t=res["lcf_t"],
                            drive=drive)
        if verbose:
            extra = (f"  drive_mob={res['drive_mob']:.2f}  lanes={res['lane_seg']:.2f}"
                     if drive == "linear" else "")
            print(f"    seed {seed}: lcf={res['lcf_mean']:.2f}  "
                  f"bimod={res['bimodality']:.2f}  speed={res['speed_rel']:.2f}"
                  f"{extra}  ({res['wall_s']:.0f} s)")
        if summary_plot:
            plot_run_summary(npz, diameter=diameter)
        if video:
            t1 = time.time()
            path = render_run(npz, max_frames=video_max_frames, diameter=diameter)
            if verbose:
                print(f"    video -> {path} ({time.time() - t1:.0f} s)")

    out = {"drive": drive, "R_T_um": R_T * 1e6, "R_T_over_d": R_T / diameter, "phi": phi_real,
           "N": N, "box_um": box * 1e6, "tau_pNnm": tau * 1e21,
           "period_s": period, "T_total_s": T_total, "per_seed": per_seed}
    for k in SUMMARY_KEYS:
        vals = np.array([r[k] for r in per_seed])
        out[k] = float(np.nanmean(vals))
        out[k + "_seed_std"] = float(np.nanstd(vals))
    return out


def print_table(rows):
    """Summary table. No automatic verdict: read the numbers and the plots."""
    circ = [r for r in rows if r.get("drive", "circular") != "linear"]
    lin = [r for r in rows if r.get("drive") == "linear"]
    if circ:
        head = (f"{'R_T/d':>6} {'phi':>5} {'N':>5} | {'lcf':>11} {'lcf_max':>7} "
                f"{'bimod':>6} {'sd_phi':>6} {'speed':>6} {'blocked':>7}")
        print(head); print("-" * len(head))
        for r in circ:
            print(f"{r['R_T_over_d']:6.1f} {r['phi']:5.2f} {r['N']:5d} | "
                  f"{r['lcf_mean']:.2f}±{r['lcf_mean_seed_std']:.2f}  {r['lcf_max']:7.2f} "
                  f"{r['bimodality']:6.2f} {r['phi_loc_std']:6.2f} {r['speed_rel']:6.2f} "
                  f"{r['blocked_frac']:7.2f}")
    if lin:
        if circ:
            print()
        head = (f"linear  {'phi':>5} {'N':>5} | {'drive_mob':>9} {'lanes':>5} "
                f"{'S':>5} {'band_x':>6} {'band_y':>6} | {'lcf':>5} {'bimod':>6} "
                f"{'sd_phi':>6} {'blocked':>7}")
        print(head); print("-" * len(head))
        for r in lin:
            print(f"{'':6}  {r['phi']:5.2f} {r['N']:5d} | {r['drive_mob']:9.2f} "
                  f"{r['lane_seg']:5.2f} {r['nematic_S']:5.2f} {r['band_x']:6.2f} "
                  f"{r['band_y']:6.2f} | {r['lcf_mean']:5.2f} {r['bimodality']:6.2f} "
                  f"{r['phi_loc_std']:6.2f} {r['blocked_frac']:7.2f}")
        print("  drive_mob: speed along own direction / v0 (1 = free flow, ~0 = jammed)\n"
              "  lanes: same-direction share of y-neighbours (0.5 = mixed, 1 = lanes)\n"
              "  S: alignment with the polarization axis; band_x / band_y: density\n"
              "  stripes along x (jams across the drive) / along y (1 = random)")


def run_mips_scan(plan=DEFAULT_PLAN, phis=DEFAULT_PHIS, seeds=(0, 1),
                  out_dir="mips_runs", **kw):
    """Run every (R_T, box, T_total) in ``plan`` at every phi; print a summary."""
    rows = []
    for R_T, box, T_total in plan:
        for phi in phis:
            what = ("linear polarization" if kw.get("drive") == "linear"
                    else f"R_T = {R_T*1e6:.0f} um")
            print(f"{what}, phi = {phi:.2f}, box = {box*1e6:.0f} um, T = {T_total:.0f} s")
            rows.append(run_mips_point(R_T, phi, box, T_total, seeds=seeds,
                                       out_dir=out_dir, **kw))
    print()
    print_table(rows)

    keys = ["R_T_um", "R_T_over_d", "phi", "N", "box_um", "tau_pNnm", "period_s",
            "T_total_s"] + SUMMARY_KEYS + [k + "_seed_std" for k in SUMMARY_KEYS]
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, "summary.csv"),
               np.array([[r[k] for k in keys] for r in rows]),
               delimiter=",", header=",".join(keys), comments="")
    return rows


# =============================================================================
# command line
# =============================================================================
def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="MIPS scan for chiral disks (circular polarization).",
        epilog="examples:\n"
               "  python mips_scan.py                       # full default plan\n"
               "  python mips_scan.py --quick               # one short test point\n"
               "  python mips_scan.py --rt 44 --phi 0.4 0.6 --box 150 --time 1200\n"
               "  python mips_scan.py --render mips_runs/*.npz   # videos for saved runs\n"
               "  python mips_scan.py --drive linear --phi 0.2 0.4 0.5 --box 300\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drive", choices=["circular", "linear"], default="circular",
                   help="circular: chiral disks (uses --rt); linear: FDTD linear-polarization "
                        "drive, disks align to +x/-x (ignores --rt)")
    p.add_argument("--power", type=float, default=LIN_POWER,
                   help="linear drive: power factor on the FDTD coefficients (default 3)")
    p.add_argument("--quick", action="store_true",
                   help="one short point (R_T=44 um, phi=0.4, 1 seed, 300 s) to check runtime")
    p.add_argument("--rt", type=float, nargs="+",
                   help="turning radii in um (default: plan 20, 44, 80)")
    p.add_argument("--phi", type=float, nargs="+", default=list(DEFAULT_PHIS),
                   help="area fractions (default: 0.2 0.4 0.6)")
    p.add_argument("--box", type=float, default=None,
                   help="box side in um (default: max(150, 3 R_T) rounded up to 50)")
    p.add_argument("--time", type=float, default=None,
                   help="simulated time in s (default: 10 rotation periods)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--dt", type=float, default=5e-4,
                   help="time step in s (1e-3 is safe for disks; check overlaps)")
    p.add_argument("--capacity", type=float, default=4.0,
                   help="neighbor-list capacity multiplier (higher = fewer overflow reruns)")
    p.add_argument("--out", default="mips_runs", help="output folder")
    p.add_argument("--no-video", action="store_true",
                   help="skip the videos (summary plots are still made)")
    p.add_argument("--render", nargs="+", metavar="NPZ",
                   help="only make videos + summary plots for saved runs, e.g. mips_runs/*.npz")
    p.add_argument("--module", default=None,
                   help=f"file (without .py) defining make_runner (default: {SIM_MODULE})")
    return p.parse_args()


def _default_box(R_T):
    return max(150e-6, np.ceil(3 * R_T / 50e-6) * 50e-6)


def _default_time(R_T, F=F_PROP, diameter=DIAMETER):
    from trajectory import DEFAULT_RESISTANCE_BODY
    rb = np.asarray(DEFAULT_RESISTANCE_BODY)
    omega = torque_for_radius(R_T, F, rb) / rb[2, 2]
    return float(np.ceil(10 * 2 * np.pi / omega / 100) * 100)     # 10 periods, rounded


if __name__ == "__main__":
    args = _parse_args()
    if args.module:
        SIM_MODULE = args.module

    if args.render:
        for f in args.render:
            print(f"{f}: {plot_run_summary(f)}, {render_run(f)}")
        raise SystemExit

    extra = {}
    if args.drive == "linear":
        box = (args.box or 300.0) * 1e-6
        T = args.time or 600.0
        plan, phis, seeds = [(np.nan, box, T)], args.phi, args.seeds
        extra = {"drive": "linear", "lin_power": args.power}
        if args.quick:
            plan, phis, seeds = [(np.nan, 150e-6, 200.0)], [0.4], [0]
    elif args.quick:
        plan, phis, seeds = [(44e-6, 150e-6, 300.0)], [0.4], [0]
    elif args.rt is None and args.box is None and args.time is None:
        plan, phis, seeds = DEFAULT_PLAN, args.phi, args.seeds
    else:
        rts = [r * 1e-6 for r in (args.rt or [p[0] * 1e6 for p in DEFAULT_PLAN])]
        plan = [(R, args.box * 1e-6 if args.box else _default_box(R),
                 args.time if args.time else _default_time(R)) for R in rts]
        phis, seeds = args.phi, args.seeds

    t0 = time.time()
    run_mips_scan(plan=plan, phis=phis, seeds=tuple(seeds), out_dir=args.out, dt=args.dt,
                  video=not args.no_video, capacity_multiplier=args.capacity, **extra)
    print(f"\ntotal wall time {time.time() - t0:.0f} s; results in {args.out}/")
