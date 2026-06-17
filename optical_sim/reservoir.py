"""Reservoir-computing gating tests for the light-driven metavehicle swarm.

Treats the swarm as a physical reservoir: a scalar input series u(n) is injected
into a fixed *random spatial light mask*, the swarm evolves for a fixed number of
MD steps per input symbol, and a feature vector (coarse density + velocity +
orientation order) is read out each input step. Only a linear (ridge) readout
would be trained downstream.

Before building a full RC pipeline, two cheap gating tests decide whether this
reservoir is viable AT ALL -- especially important because the Brownian (kT)
noise is physical and cannot be removed:

  1. Consistency / Echo State Property: drive with the SAME input under DIFFERENT
     noise seeds. If the reservoir states converge (high cross-run correlation),
     the input dominates and a deterministic readout is possible. If not, noise
     sets a hard performance floor.
  2. Linear memory capacity: how many past inputs a linear readout can recover
     from the current state, MC_k = corr^2(u[n-k], readout). Sum = total MC.

Run:  python reservoir.py
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random, lax, jit, grad
from jax_md import space, rigid_body, energy

from brownian_generalized_2d import (brownian_generalized_rigid_2d,
                                      brownian_generalized_2d)
from simulate_swarm import (square_shape, _build_interaction,
                            grid_initial_condition, DEFAULT_F_BODY, DEFAULT_TAU)
from simulate_trajectory import KT_300K, DEFAULT_RESISTANCE_BODY


# --------------------------------------------------------------------------
# build the reservoir once (single force_fn -> single canonicalize, reused
# across seeds; also avoids the cross-run jax_md pytree bug)
# --------------------------------------------------------------------------
def build_reservoir(N=100, side=10e-6, n_grid=4, area_fraction=0.15,
                    epsilon=5e-17, F_body=DEFAULT_F_BODY, tau=DEFAULT_TAU,
                    dt=1e-4, kT=KT_300K, resistance=DEFAULT_RESISTANCE_BODY,
                    mask_grid=4, I_base=20.0, gain=0.9, steps_per_input=200,
                    mask_seed=0):
    shape, sigma = square_shape(side, n_grid)
    box = float(side * np.sqrt(N / area_fraction))
    disp, shift = space.periodic(box)
    # large Verlet skin: particles move fast under bright light, so a small skin
    # would force a neighbor-list rebuild almost every step (catastrophically slow)
    inter_force, nbr_fns = _build_interaction(
        shape, sigma, epsilon, disp, box, True,
        dr_threshold=3.0 * sigma, capacity_multiplier=2.5)

    F_body = jnp.asarray(F_body, dtype=jnp.float64)
    tau = jnp.float64(tau)
    # fixed random input mask: a weight in [0,1] per coarse cell
    mw = random.uniform(random.PRNGKey(mask_seed), (mask_grid, mask_grid),
                        dtype=jnp.float64)

    def region_weight(centers):
        ix = jnp.clip((centers[:, 0] / box * mask_grid).astype(jnp.int32),
                      0, mask_grid - 1)
        iy = jnp.clip((centers[:, 1] / box * mask_grid).astype(jnp.int32),
                      0, mask_grid - 1)
        return mw[ix, iy]

    def force_fn(body, light_u=0.0, neighbor=None):
        f_int = inter_force(body, neighbor=neighbor)
        w = region_weight(body.center)                          # (N,) in [0,1]
        # input modulates the local intensity; some regions brighten, others dim
        s = jnp.maximum(I_base * (1.0 + gain * light_u * (2.0 * w - 1.0)), 0.0)
        R = rigid_body.rotation2d(body.orientation)
        f_lab = jnp.einsum("nij,j->ni", R, F_body) * s[:, None]
        return rigid_body.RigidBody(f_int.center + f_lab,
                                    f_int.orientation + s * tau)

    init_fn, step_fn = brownian_generalized_rigid_2d(
        force_fn, shift, dt, kT, resistance_body=resistance)

    def rollout(state, nbrs, u_seq):
        def input_step(carry, u):
            s, nb = carry
            def md(c, _):
                s, nb = c
                nb = nb.update(s.position)
                s = step_fn(s, light_u=u, neighbor=nb)
                return (s, nb), None
            (s, nb), _ = lax.scan(md, (s, nb), xs=None, length=steps_per_input)
            return (s, nb), (s.position.center, s.position.orientation)
        return lax.scan(input_step, (state, nbrs), xs=u_seq)

    return {"box": box, "N": N, "init_fn": init_fn, "nbr_fns": nbr_fns,
            "rollout": jit(rollout)}


def run_reservoir(res, u_seq, seed=0):
    """Drive the reservoir with input ``u_seq``; return per-input-step state."""
    body0, key = grid_initial_condition(res["N"], res["box"], seed)
    state = res["init_fn"](key, body0)
    nbrs = res["nbr_fns"].allocate(body0, extra_capacity=128)
    (state, nbrs), (cen, ang) = res["rollout"](state, nbrs, jnp.asarray(u_seq))
    jax.block_until_ready(cen)
    return np.asarray(cen), np.asarray(ang), bool(nbrs.did_buffer_overflow)


# --------------------------------------------------------------------------
# FAST point-particle reservoir (no rigid bodies -> ~16x fewer entities).
# Same active physics: body-frame drive + steric repulsion + Brownian noise +
# light coupling. The initial-condition seed is SEPARATE from the noise seed so
# we can isolate noise- vs initial-condition-driven divergence (ESP diagnostic).
# --------------------------------------------------------------------------
def _rot2(theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.stack([jnp.stack([c, -s], -1), jnp.stack([s, c], -1)], -2)  # (N,2,2)


def build_point_reservoir(N=200, box_um=60.0, sigma=2e-6, epsilon=5e-17,
                          F_body=DEFAULT_F_BODY, tau=DEFAULT_TAU, dt=1e-4,
                          kT=KT_300K, resistance=DEFAULT_RESISTANCE_BODY,
                          mask_grid=4, I_base=20.0, gain=0.9, steps_per_input=300,
                          dr_threshold_factor=1.0, mask_seed=0):
    box = box_um * 1e-6
    disp, shift_xy = space.periodic(box)
    neighbor_fn, ss_energy = energy.soft_sphere_neighbor_list(
        disp, box, sigma=sigma, epsilon=epsilon, alpha=2.0,
        dr_threshold=dr_threshold_factor * sigma, capacity_multiplier=2.0)
    energy_grad = grad(lambda R, nbr: ss_energy(R, neighbor=nbr))

    F_body = jnp.asarray(F_body, dtype=jnp.float64)
    tau = jnp.float64(tau)
    mw = random.uniform(random.PRNGKey(mask_seed), (mask_grid, mask_grid),
                        dtype=jnp.float64)

    def region_weight(R):
        ix = jnp.clip((R[:, 0] / box * mask_grid).astype(jnp.int32), 0, mask_grid - 1)
        iy = jnp.clip((R[:, 1] / box * mask_grid).astype(jnp.int32), 0, mask_grid - 1)
        return mw[ix, iy]

    def shift_fn(q, dq, **kw):
        xy = shift_xy(q[:, :2], dq[:, :2])                      # periodic wrap
        th = (q[:, 2] + dq[:, 2] + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return jnp.concatenate([xy, th[:, None]], axis=1)

    def force_fn_body(q, light_u=0.0, neighbor=None):
        R, th = q[:, :2], q[:, 2]
        F_lab = -energy_grad(R, neighbor)                       # steric, lab frame
        F_inter_body = jnp.einsum("nji,nj->ni", _rot2(th), F_lab)   # lab -> body
        s = jnp.maximum(I_base * (1.0 + gain * light_u * (2.0 * region_weight(R) - 1.0)), 0.0)
        F_act_body = s[:, None] * F_body
        return jnp.concatenate([F_inter_body + F_act_body, (s * tau)[:, None]], axis=1)

    init_fn, step_fn = brownian_generalized_2d(
        force_fn_body, shift_fn, dt, kT, resistance_body=resistance)

    def make_init(ic_seed, noise_seed):
        m = int(np.ceil(np.sqrt(N)))
        coords = (np.arange(m) + 0.5) * box / m
        gx, gy = np.meshgrid(coords, coords)
        centers = np.stack([gx.ravel(), gy.ravel()], 1)[:N]
        th = np.asarray(random.uniform(random.PRNGKey(ic_seed), (N,),
                                       minval=-np.pi, maxval=np.pi))
        q0 = jnp.asarray(np.concatenate([centers, th[:, None]], 1), dtype=jnp.float64)
        state = init_fn(random.PRNGKey(noise_seed), q0)
        nbrs = neighbor_fn.allocate(q0[:, :2], extra_capacity=64)
        return state, nbrs

    def rollout(state, nbrs, u_seq):
        def input_step(carry, u):
            s, nb = carry
            def md(c, _):
                s, nb = c
                nb = nb.update(s.position[:, :2])
                s = step_fn(s, light_u=u, neighbor=nb)
                return (s, nb), None
            (s, nb), _ = lax.scan(md, (s, nb), xs=None, length=steps_per_input)
            return (s, nb), (s.position[:, :2], s.position[:, 2])
        return lax.scan(input_step, (state, nbrs), xs=u_seq)

    return {"box": box, "N": N, "make_init": make_init, "rollout": jit(rollout)}


def run_point_reservoir(res, u_seq, ic_seed=0, noise_seed=0):
    state, nbrs = res["make_init"](ic_seed, noise_seed)
    (state, nbrs), (cen, ang) = res["rollout"](state, nbrs, jnp.asarray(u_seq))
    jax.block_until_ready(cen)
    return np.asarray(cen), np.asarray(ang), bool(nbrs.did_buffer_overflow)


# --------------------------------------------------------------------------
# readout features: coarse density + velocity + orientation order on a grid
# --------------------------------------------------------------------------
def features(cen, ang, box, G=6):
    T, N, _ = cen.shape
    ix = np.clip((cen[:, :, 0] / box * G).astype(int), 0, G - 1)
    iy = np.clip((cen[:, :, 1] / box * G).astype(int), 0, G - 1)
    idx = ix * G + iy                                          # (T, N) cell index

    d = np.diff(cen, axis=0, prepend=cen[:1])                  # per-step displacement
    d -= box * np.round(d / box)                               # minimum image

    dens = np.zeros((T, G * G)); vx = np.zeros((T, G * G))
    vy = np.zeros((T, G * G)); cnt = np.zeros((T, G * G))
    cth = np.zeros((T, G * G)); sth = np.zeros((T, G * G))
    for t in range(T):
        np.add.at(dens[t], idx[t], 1.0)
        np.add.at(vx[t], idx[t], d[t, :, 0]); np.add.at(vy[t], idx[t], d[t, :, 1])
        np.add.at(cnt[t], idx[t], 1.0)
        np.add.at(cth[t], idx[t], np.cos(ang[t])); np.add.at(sth[t], idx[t], np.sin(ang[t]))
    dens /= N; cth /= N; sth /= N
    cnt = np.maximum(cnt, 1.0); vx /= cnt; vy /= cnt
    return np.concatenate([dens, vx, vy, cth, sth], axis=1)    # (T, 5*G^2)


# --------------------------------------------------------------------------
# gating metrics
# --------------------------------------------------------------------------
def memory_capacity(X, u, max_delay=40, washout=150, ridge=1e-4):
    """Linear memory capacity MC_k = corr^2(u[n-k], ridge-readout(X[n]))."""
    T = len(X)
    idx = np.arange(washout, T)
    ntr = int(0.7 * len(idx))
    tr, te = idx[:ntr], idx[ntr:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9

    def design(rows):
        return np.hstack([(X[rows] - mu) / sd, np.ones((len(rows), 1))])

    Atr, Ate = design(tr), design(te)
    Ginv = np.linalg.inv(Atr.T @ Atr + ridge * np.eye(Atr.shape[1]))
    MC = np.zeros(max_delay)
    for k in range(1, max_delay + 1):
        W = Ginv @ (Atr.T @ u[tr - k])
        pred, yte = Ate @ W, u[te - k]
        if pred.std() > 1e-12 and yte.std() > 1e-12:
            MC[k - 1] = np.corrcoef(pred, yte)[0, 1] ** 2
    return MC, float(MC.sum())


def consistency(X1, X2, washout=150):
    """ESP test: cross-run agreement of states under identical input, diff noise.

    Returns (per-timestep state cosine over time, mean per-feature correlation).
    """
    A, B = X1[washout:], X2[washout:]
    mu, sd = A.mean(0), A.std(0) + 1e-9
    Az, Bz = (A - mu) / sd, (B - mu) / sd
    # per-timestep cosine similarity between the two standardized state vectors
    num = np.sum(Az * Bz, axis=1)
    den = np.linalg.norm(Az, axis=1) * np.linalg.norm(Bz, axis=1) + 1e-12
    cos_t = num / den
    # per-feature temporal correlation (the "consistency capacity")
    cors = []
    for j in range(A.shape[1]):
        if A[:, j].std() > 1e-9 and B[:, j].std() > 1e-9:
            cors.append(np.corrcoef(A[:, j], B[:, j])[0, 1])
    return cos_t, float(np.mean(cors))


def main(T=400, seed_input=0, washout=100, max_delay=30, out_path=None):
    from pathlib import Path
    from datetime import datetime
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed_input)
    u = rng.uniform(-1.0, 1.0, size=T)                         # i.i.d. RC probe

    print("building reservoir...")
    res = build_reservoir()
    print(f"box = {res['box']*1e6:.0f} um, N = {res['N']}")

    print("run 1 (noise seed 1)..."); c1, a1, o1 = run_reservoir(res, u, seed=1)
    print("run 2 (noise seed 2)..."); c2, a2, o2 = run_reservoir(res, u, seed=2)
    if o1 or o2:
        print("WARNING: neighbor buffer overflowed -> results suspect")
    X1 = features(c1, a1, res["box"]); X2 = features(c2, a2, res["box"])

    cos_t, cons = consistency(X1, X2, washout=washout)
    MC, MC_tot = memory_capacity(X1, u, max_delay=max_delay, washout=washout)
    print(f"\nESP / consistency: mean per-feature corr = {cons:.3f} "
          f"(late-time state cosine = {cos_t[-50:].mean():.3f})")
    print(f"memory capacity: total MC = {MC_tot:.2f}  "
          f"(MC_1={MC[0]:.2f}, half-life ~ {np.argmax(MC<MC[0]/2)+1} steps)")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(cos_t, lw=1)
    ax[0].axhline(1.0, color="0.7", ls="--", lw=1)
    ax[0].set(xlabel="input step (after washout)", ylabel="cross-run state cosine",
              title=f"ESP / consistency (mean feat.corr={cons:.2f})", ylim=(-0.1, 1.05))
    ax[1].bar(np.arange(1, len(MC) + 1), MC, width=0.9)
    ax[1].set(xlabel="delay k (input steps)", ylabel="MC_k = corr$^2$",
              title=f"memory capacity (total={MC_tot:.1f})")
    # forgetting-of-IC check: does cross-run state difference decay early on?
    early = consistency(X1, X2, washout=0)[0]
    ax[2].plot(early[:200], lw=1)
    ax[2].set(xlabel="input step from start", ylabel="cross-run state cosine",
              title="initial-condition washout", ylim=(-0.1, 1.05))
    fig.tight_layout()

    out_path = out_path or (Path(__file__).resolve().parent / "results"
                            / f"reservoir_gating_{datetime.now():%Y%m%d-%H%M%S}.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return {"consistency": cons, "MC_total": MC_tot, "MC": MC}


def narma10(u):
    """Standard NARMA-10 target: nonlinear, 10-step memory. u ~ Uniform[0,0.5]."""
    y = np.zeros(len(u))
    for n in range(9, len(u) - 1):
        y[n + 1] = (0.3 * y[n] + 0.05 * y[n] * np.sum(y[n - 9:n + 1])
                    + 1.5 * u[n - 9] * u[n] + 0.1)
    return y


def _windowed(X, rows, n_hist):
    """Sliding window of the last n_hist state-vectors for each row (the readout)."""
    T = len(X)
    return np.hstack([X[np.clip(rows - k, 0, T - 1)] for k in range(n_hist)])


def fit_readout(X, y, n_hist=1, washout=150,
                ridges=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5),
                train_frac=0.6, val_frac=0.2):
    """Ridge readout over a history window, with ridge chosen on a validation
    split (so the many-features/few-samples case is handled fairly). Returns the
    held-out test NRMSE. Identical procedure is used for reservoir and control,
    so the comparison is fair regardless of feature count.
    """
    T = len(X)
    rows = np.arange(washout, T)
    A = _windowed(X, rows, n_hist)
    yv = y[rows]
    n = len(rows)
    ntr, nval = int(train_frac * n), int(val_frac * n)
    tr = np.arange(ntr)
    va = np.arange(ntr, ntr + nval)
    te = np.arange(ntr + nval, n)
    mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-9
    B = np.hstack([(A - mu) / sd, np.ones((n, 1))])

    def fit(rows_, lam):
        G = B[rows_].T @ B[rows_] + lam * np.eye(B.shape[1])
        return np.linalg.solve(G, B[rows_].T @ yv[rows_])

    def nrmse(rows_, W):
        p = B[rows_] @ W
        return float(np.sqrt(np.mean((p - yv[rows_]) ** 2) / (np.var(yv[rows_]) + 1e-12)))

    best = min(ridges, key=lambda lam: nrmse(va, fit(tr, lam)))   # pick lam on val
    W = fit(np.concatenate([tr, va]), best)                       # refit on train+val
    return nrmse(te, W), B[te] @ W, yv[te]


def main_narma(T=1500, N=400, box_um=85.0, steps_per_input=150,
               n_hist_list=(1, 20, 60), seed_input=0, out_path=None):
    """NARMA-10 with a historical-state readout, vs a no-reservoir control."""
    from pathlib import Path
    from datetime import datetime
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed_input)
    u = rng.uniform(0.0, 0.5, size=T)                  # NARMA driving input
    y = narma10(u)
    light = 4.0 * u - 1.0                               # map [0,0.5] -> [-1,1]

    print(f"building point reservoir N={N}...")
    res = build_point_reservoir(N=N, box_um=box_um, steps_per_input=steps_per_input)
    cen, ang, ov = run_point_reservoir(res, light, ic_seed=1, noise_seed=1)
    if ov:
        print("WARNING: neighbor buffer overflow")
    X = features(cen, ang, res["box"])
    U = light[:, None]                                 # raw input as a 1-D "state"

    print("NARMA-10 test NRMSE (lower=better; 1.0 = predicting the mean):")
    res_nrmse, ctrl_nrmse = {}, {}
    best = (1e9, None, None, None)
    for nh in n_hist_list:
        nr, pred, yte = fit_readout(X, y, n_hist=nh)
        nc, _, _ = fit_readout(U, y, n_hist=nh)
        res_nrmse[nh], ctrl_nrmse[nh] = nr, nc
        print(f"  n_hist={nh:>3}:  reservoir={nr:.3f}   raw-input control={nc:.3f}")
        if nr < best[0]:
            best = (nr, nh, pred, yte)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    nhs = list(n_hist_list)
    ax[0].plot(nhs, [res_nrmse[k] for k in nhs], "o-", label="reservoir + history")
    ax[0].plot(nhs, [ctrl_nrmse[k] for k in nhs], "s--", label="raw-input control")
    ax[0].axhline(1.0, color="0.7", ls=":", lw=1, label="predict-mean baseline")
    ax[0].set(xlabel="n_hist (history length)", ylabel="NARMA-10 test NRMSE",
              title="does the reservoir compute?"); ax[0].legend()
    _, nh, pred, yte = best
    ax[1].plot(yte[:200], lw=1.5, label="NARMA-10 target")
    ax[1].plot(pred[:200], lw=1.0, label=f"reservoir pred (n_hist={nh})")
    ax[1].set(xlabel="step", ylabel="y", title=f"best fit (NRMSE={best[0]:.2f})")
    ax[1].legend()
    fig.tight_layout()
    out_path = out_path or (Path(__file__).resolve().parent / "results"
                            / f"reservoir_narma_{datetime.now():%Y%m%d-%H%M%S}.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return res_nrmse, ctrl_nrmse


def main_point(T=500, seed_input=0, washout=120, max_delay=30, out_path=None):
    """Gating test on the FAST point reservoir, with the noise-vs-IC diagnostic."""
    from pathlib import Path
    from datetime import datetime
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u = np.random.default_rng(seed_input).uniform(-1.0, 1.0, size=T)
    print("building point reservoir...")
    res = build_point_reservoir()
    print(f"box = {res['box']*1e6:.0f} um, N = {res['N']} point particles")

    # strict ESP: different initial condition AND different noise
    cA, aA, _ = run_point_reservoir(res, u, ic_seed=1, noise_seed=1)
    cB, aB, _ = run_point_reservoir(res, u, ic_seed=2, noise_seed=2)
    # noise-only: SAME initial condition, different noise (isolates thermal noise)
    cC, aC, _ = run_point_reservoir(res, u, ic_seed=7, noise_seed=11)
    cD, aD, ov = run_point_reservoir(res, u, ic_seed=7, noise_seed=22)
    if ov:
        print("WARNING: neighbor buffer overflow")

    box = res["box"]
    XA, XB = features(cA, aA, box), features(cB, aB, box)
    XC, XD = features(cC, aC, box), features(cD, aD, box)

    cos_strict, cons_strict = consistency(XA, XB, washout=washout)
    cos_noise, cons_noise = consistency(XC, XD, washout=washout)
    MC, MC_tot = memory_capacity(XA, u, max_delay=max_delay, washout=washout)

    print(f"\nESP strict (diff IC + diff noise): mean feat.corr = {cons_strict:.3f}")
    print(f"ESP noise-only (same IC, diff noise): mean feat.corr = {cons_noise:.3f}")
    print(f"memory capacity: total MC = {MC_tot:.2f} (MC_1={MC[0]:.2f})")
    if cons_noise > 0.7 and cons_strict < 0.4:
        verdict = "IC-sensitivity/chaos limits it (noise alone is OK) -> fixable with fading memory/washout"
    elif cons_noise < 0.4:
        verdict = "thermal noise alone destroys reproducibility -> hard limit for single-shot RC"
    else:
        verdict = "partial reproducibility"
    print("diagnosis:", verdict)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(cos_strict, lw=1, label=f"diff IC+noise ({cons_strict:.2f})")
    ax[0].plot(cos_noise, lw=1, label=f"same IC, diff noise ({cons_noise:.2f})")
    ax[0].axhline(1.0, color="0.7", ls="--", lw=1)
    ax[0].set(xlabel="input step (after washout)", ylabel="cross-run state cosine",
              title="ESP: noise vs IC", ylim=(-0.1, 1.05)); ax[0].legend()
    ax[1].bar(np.arange(1, len(MC) + 1), MC, width=0.9)
    ax[1].set(xlabel="delay k", ylabel="MC_k = corr$^2$",
              title=f"memory capacity (total={MC_tot:.1f})")
    early_s = consistency(XA, XB, washout=0)[0]
    early_n = consistency(XC, XD, washout=0)[0]
    ax[2].plot(early_s[:200], lw=1, label="diff IC+noise")
    ax[2].plot(early_n[:200], lw=1, label="same IC, diff noise")
    ax[2].set(xlabel="input step from start", ylabel="cross-run state cosine",
              title="initial-condition washout", ylim=(-0.1, 1.05)); ax[2].legend()
    fig.tight_layout()

    out_path = out_path or (Path(__file__).resolve().parent / "results"
                            / f"reservoir_point_{datetime.now():%Y%m%d-%H%M%S}.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return {"cons_strict": cons_strict, "cons_noise": cons_noise, "MC_total": MC_tot}


if __name__ == "__main__":
    main_point()
