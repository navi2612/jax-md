"""Validation of the 2D Brownian integrator for an ANISOTROPIC, chirally
coupled mobility (non-diagonal resistance matrix).

A diagonal resistance leaves three code paths dormant. This script exercises
them with a test resistance that is anisotropic (gamma_x != gamma_y) and has a
translation-rotation coupling (R[0,2] != 0):

  1. Increment covariance  : Cov(dq_lab) = 2 kT dt B(theta) mu B(theta)^T.
       At theta=0 this recovers the full mobility tensor mu = R^-1 (including the
       off-diagonal coupling); at theta=90 deg it checks the lab rotation B.
       This is exact (single step, no time-integration error).

  2. Trap equilibrium       : in an isotropic harmonic trap the integrator must
       sample Boltzmann, Var(x) = Var(y) = kT/k, regardless of the coupling.

  3. Drift-term effect      : re-run check 2 with include_drift=False to quantify
       how much the thermal-drift term moves this observable. It is small here:
       an isotropic trap with a freely diffusing orientation averages the
       coupling out by symmetry. The term is still required in general (it equals
       kT*grad.M_lab analytically, the correction that makes the Ito scheme
       sample Boltzmann for a configuration-dependent mobility); this check just
       reports its size for the present setup rather than asserting PASS/FAIL.

Run directly:  python validate_anisotropic.py
"""

import numpy as np
import jax.numpy as jnp
from jax import vmap

from brownian_generalized_2d import rotation_2d_3x3
from simulate_trajectory import (
    run_brownian_2d, make_constant_force, lab_to_body_force, KT_300K,
)

PASS_TOL = 0.05

# Anisotropic, chirally coupled resistance (symmetric positive-definite):
#   x and y have different drag; x couples to the rotation via R[0,2].
# PD check: gx*grot - c^2 = 3e-7*8e-18 - (6e-13)^2 = 2.04e-24 > 0.
R_ANISO = jnp.array([
    [3.0e-7, 0.0,    6.0e-13],
    [0.0,    7.0e-7, 0.0],
    [6.0e-13, 0.0,   8.0e-18],
], dtype=jnp.float64)


def _mobility(R):
    return np.linalg.inv(np.asarray(R, dtype=np.float64))


def check_increment_covariance(R=R_ANISO, kT=KT_300K, dt=1e-3, N=400_000,
                               theta0=0.0, seed=0):
    """Single-step Cov(dq_lab) vs 2 kT dt B mu B^T (recovers the mobility)."""
    mu = _mobility(R)
    q0 = np.zeros((N, 3))
    q0[:, 2] = theta0
    traj, _ = run_brownian_2d(make_constant_force(0.0, 0.0, 0.0),
                              jnp.asarray(q0), n_steps=1, dt=dt, kT=kT,
                              resistance_body=R, seed=seed)
    dq = np.asarray(traj)[0] - q0                  # (N, 3) lab-frame increments
    cov_meas = np.cov(dq.T)                         # 3x3 (mean removed)
    B = np.asarray(rotation_2d_3x3(jnp.asarray(theta0)))
    cov_theory = 2.0 * float(kT) * dt * (B @ mu @ B.T)
    return cov_meas, cov_theory


def _trap_force(k):
    def force(q, **kwargs):
        F_lab = -k * q[:, :2]                       # isotropic, per particle
        F_body = vmap(lab_to_body_force)(q[:, 2], F_lab)
        tau = jnp.zeros((q.shape[0],), dtype=q.dtype)
        return jnp.concatenate([F_body, tau[:, None]], axis=1)
    return force


def check_trap_equilibrium(R=R_ANISO, kT=KT_300K, k=2e-6, n_steps=4000,
                           dt=1e-3, N=2000, seed=1, include_drift=True):
    """Equilibrium Var(x), Var(y) in an isotropic trap; expect kT/k."""
    var_theory = float(kT) / k
    q0 = jnp.zeros((N, 3), dtype=jnp.float64)
    traj, _ = run_brownian_2d(_trap_force(k), q0, n_steps=n_steps, dt=dt, kT=kT,
                              resistance_body=R, seed=seed,
                              include_drift=include_drift)
    tail = np.asarray(traj)[n_steps // 2:, :, :2]   # drop burn-in
    var_xy = tail.reshape(-1, 2).var(axis=0)
    return float(var_xy[0]), float(var_xy[1]), var_theory


def _fmt_mat(m):
    return "\n".join("  [" + "  ".join(f"{v: .3e}" for v in row) + "]" for row in m)


def _cov_report(cov_m, cov_t):
    """Compare covariance entries fairly despite huge magnitude spread.

    Entries are judged significant by the *theoretical correlation* (so the
    diagonal variances and the x-theta coupling all count, even though mu_theta
    dwarfs mu_x). Returns (ratio matrix, significant mask, worst rel.err).
    """
    d_t = np.sqrt(np.diag(cov_t))
    corr_t = cov_t / np.outer(d_t, d_t)
    sig = np.abs(corr_t) > 0.05                     # correlated (diagonal = 1)
    ratio = cov_m / cov_t
    worst = float(np.abs(ratio - 1.0)[sig].max())
    return ratio, sig, worst


def main(save_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime

    print("Validating the anisotropic / chirally coupled integrator...\n")
    mu = _mobility(R_ANISO)
    print("test resistance R (N.s/m, mixed, N.m.s):")
    print(_fmt_mat(np.asarray(R_ANISO)))
    print("=> mobility mu = R^-1 has coupling mu[x,theta] = "
          f"{mu[0,2]: .3e} (dimensionless coupling "
          f"{abs(mu[0,2])/np.sqrt(mu[0,0]*mu[2,2]):.2f})\n")

    all_pass = True

    # --- 1. increment covariance at theta = 0 and 90 deg ---
    for theta0, tag in ((0.0, "theta=0"), (np.pi / 2, "theta=90deg")):
        cov_m, cov_t = check_increment_covariance(theta0=theta0)
        _, _, worst = _cov_report(cov_m, cov_t)
        ok = worst < PASS_TOL
        all_pass &= ok
        print(f"1. increment covariance ({tag}): worst rel.err over the "
              f"variances + x-theta coupling = {worst:.2%}  {'PASS' if ok else 'FAIL'}")
    # show the recovered tensor at theta=0 vs theory (anisotropy + coupling)
    cov_m, cov_t = check_increment_covariance(theta0=0.0)
    print(f"   anisotropy mu_xx/mu_yy: measured {cov_m[0,0]/cov_m[1,1]:.3f}, "
          f"theory {cov_t[0,0]/cov_t[1,1]:.3f}  (!= gamma_y/gamma_x: coupling lifts mu_xx)")
    rho_m = cov_m[0, 2] / np.sqrt(cov_m[0, 0] * cov_m[2, 2])
    rho_t = cov_t[0, 2] / np.sqrt(cov_t[0, 0] * cov_t[2, 2])
    print(f"   x-theta coupling rho:   measured {rho_m:.3f}, theory {rho_t:.3f}\n")

    # --- 2. trap equilibrium WITH drift ---
    vx, vy, vth = check_trap_equilibrium(include_drift=True)
    ex, ey = abs(vx - vth) / vth, abs(vy - vth) / vth
    ok = ex < PASS_TOL and ey < PASS_TOL
    all_pass &= ok
    print(f"2. trap equilibrium (drift ON):  Var(x)={vx:.3e} ({ex:.1%}), "
          f"Var(y)={vy:.3e} ({ey:.1%}), expected kT/k={vth:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")

    # --- 3. trap equilibrium WITHOUT drift (quantify the drift term's effect) ---
    vx0, vy0, _ = check_trap_equilibrium(include_drift=False)
    dxe = abs(vx - vx0) / vth                        # shift caused by the drift term
    print(f"3. drift-term effect on Var(x): drift ON {vx:.3e} vs OFF {vx0:.3e} "
          f"-> {dxe:.1%} of kT/k (within the ~{max(ex,ey):.0%} statistical noise; "
          f"averaged out by symmetry here)")
    print("-" * 80)
    print("ALL CHECKS PASSED\n" if all_pass else "SOME CHECKS FAILED\n")

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cov_m, cov_t = check_increment_covariance(theta0=0.0)
    ratio, sig, _ = _cov_report(cov_m, cov_t)
    labels = ["x", "y", "theta"]
    ax = axes[0]
    im = ax.imshow(np.where(sig, ratio, np.nan), vmin=0.9, vmax=1.1, cmap="RdBu_r")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{ratio[i,j]:.2f}" if sig[i, j] else "~0",
                    ha="center", va="center", fontsize=9)
    ax.set_title("1. Cov(dq) / theory  (theta=0)\n(1.00 = exact; ~0 = uncoupled)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    ax.bar([0, 1, 2, 3], [vx/vth, vy/vth, vx0/vth, vy0/vth],
           color=["#1b7837", "#1b7837", "#b2182b", "#b2182b"])
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Var x\n(drift)", "Var y\n(drift)",
                        "Var x\n(no drift)", "Var y\n(no drift)"])
    ax.set_ylabel("Var / (kT/k)")
    ax.set_title("2-3. Trap equilibrium (FDT)\n1.00 = correct Boltzmann")
    fig.tight_layout()

    save_dir = Path(save_dir) if save_dir else (
        Path(__file__).resolve().parent / "results"
        / f"validation_aniso_{datetime.now():%Y%m%d-%H%M%S}")
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "validation_anisotropic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return all_pass


if __name__ == "__main__":
    main()
