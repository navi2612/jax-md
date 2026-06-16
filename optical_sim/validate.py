"""Physical validation of the 2D Brownian integrator for a single point particle.

Runs four independent checks against analytic predictions and reports the
measured value vs the expected one with a relative error and PASS/FAIL:

  1. Translational diffusion  : MSD(t) = 4 D_t t,     D_t = kT * mu_t   (Einstein)
  2. Rotational diffusion      : <cos d_theta> = e^{-D_r t}, D_r = kT * mu_r
  3. Harmonic-trap equilibrium : Var(x) = kT / k        (fluctuation-dissipation)
  4. Deterministic drift (kT=0): x(t) = mu_t F t,  theta(t) = mu_r tau t

Checks 1 & 3 together are the strong one: 1 measures the diffusion produced by
the *noise*, 3 measures the equilibrium set by the balance of *noise and drag*.
If the sqrt(2 kT dt) amplitude or the mobility were wrong, 3 would fail even if
1 looked plausible. Check 4 isolates the deterministic force/mobility path.

The checks use an ensemble of independent walkers (one simulation, N particles,
no interaction force), so the ensemble MSD/variance match the single-particle
theory directly. For a force-free particle the Euler-Maruyama step draws exact
Gaussian increments, so a larger dt is used for the stochastic checks without
loss of accuracy; the trap check keeps k*mu*dt << 1 so its discretization bias
(~k*mu*dt/2) stays negligible.

Run directly:  python validate.py
"""

import numpy as np
import jax.numpy as jnp
from jax import vmap

from simulate_trajectory import (
    run_brownian_2d, make_constant_force, lab_to_body_force,
    DEFAULT_RESISTANCE_BODY, KT_300K, K_B,
)

N_WALKERS = 2000
PASS_TOL = 0.05 # 5% relative error -> PASS


def _mobility(resistance_body):
    return np.linalg.inv(np.asarray(resistance_body, dtype=np.float64))


def check_translational_diffusion(kT=KT_300K, resistance_body=DEFAULT_RESISTANCE_BODY,
                                  n_steps=2000, dt=1e-3, N=N_WALKERS, seed=0):
    """MSD(t) = 4 D_t t for free 2D diffusion."""
    mu = _mobility(resistance_body)
    D_theory = float(kT) * mu[0, 0]

    q0 = jnp.zeros((N, 3), dtype=jnp.float64)
    traj, _ = run_brownian_2d(make_constant_force(0.0, 0.0, 0.0), q0,
                              n_steps=n_steps, dt=dt, kT=kT,
                              resistance_body=resistance_body, seed=seed)
    traj = np.asarray(traj)                       # (n_steps, N, 3)
    t = dt * np.arange(1, n_steps + 1)
    msd = (traj[:, :, :2] ** 2).sum(axis=2).mean(axis=1)   # 2D MSD, <|dr|^2>
    slope = np.polyfit(t, msd, 1)[0]
    D_meas = slope / 4.0                           # MSD_2D = 4 D t
    return dict(name="translational diffusion D_t [m^2/s]",
                measured=D_meas, expected=D_theory, t=t, msd=msd)


def check_rotational_diffusion(kT=KT_300K, resistance_body=DEFAULT_RESISTANCE_BODY,
                               n_steps=2000, dt=1e-3, N=N_WALKERS, seed=0):
    """Orientational autocorrelation <cos d_theta> = exp(-D_r t)."""
    mu = _mobility(resistance_body)
    D_theory = float(kT) * mu[2, 2]

    q0 = jnp.zeros((N, 3), dtype=jnp.float64)
    traj, _ = run_brownian_2d(make_constant_force(0.0, 0.0, 0.0), q0,
                              n_steps=n_steps, dt=dt, kT=kT,
                              resistance_body=resistance_body, seed=seed)
    traj = np.asarray(traj)
    t = dt * np.arange(1, n_steps + 1)
    C = np.cos(traj[:, :, 2]).mean(axis=1)         # theta(0)=0; cos is wrap-safe
    m = C > 0.1                                    # fit only the clean part
    D_meas = np.polyfit(t[m], -np.log(C[m]), 1)[0]
    return dict(name="rotational diffusion D_r [rad^2/s]",
                measured=D_meas, expected=D_theory, t=t, C=C)


def check_harmonic_equilibrium(kT=KT_300K, resistance_body=DEFAULT_RESISTANCE_BODY,
                               k=2e-6, n_steps=4000, dt=1e-3, N=N_WALKERS, seed=1):
    """Equilibrium Var(x) = kT/k in an isotropic harmonic trap (FDT check)."""
    var_theory = float(kT) / k

    def trap_force(q, **kwargs):
        R = q[:, :2]
        F_lab = -k * R                             # independent per particle
        theta = q[:, 2]
        F_body = vmap(lab_to_body_force)(theta, F_lab)
        tau = jnp.zeros((q.shape[0],), dtype=q.dtype)
        return jnp.concatenate([F_body, tau[:, None]], axis=1)

    q0 = jnp.zeros((N, 3), dtype=jnp.float64)
    traj, _ = run_brownian_2d(trap_force, q0, n_steps=n_steps, dt=dt, kT=kT,
                              resistance_body=resistance_body, seed=seed)
    traj = np.asarray(traj)
    tail = traj[n_steps // 2:, :, :2]              # drop burn-in, then pool
    var_meas = float(tail.reshape(-1, 2).var(axis=0).mean())
    samples = tail[:, :, 0].ravel()
    return dict(name="trap equilibrium Var(x) [m^2]",
                measured=var_meas, expected=var_theory, samples=samples,
                std_theory=np.sqrt(var_theory))


def check_deterministic_drift(resistance_body=DEFAULT_RESISTANCE_BODY,
                              Fx=1e-12, tau=1e-18, dt=1e-4):
    """Noise-free (kT=0): x(t) = mu_t Fx t and theta(t) = mu_r tau t."""
    mu = _mobility(resistance_body)

    # translation: torque-free so theta stays 0 and body x == lab x
    traj_x, _ = run_brownian_2d(make_constant_force(Fx, 0.0, 0.0),
                                jnp.zeros((1, 3)), n_steps=1000, dt=dt, kT=0.0,
                                resistance_body=resistance_body)
    traj_x = np.asarray(traj_x)
    t_x = dt * np.arange(1, 1001)
    x = traj_x[:, 0, 0]
    x_theory = mu[0, 0] * Fx * t_x
    v_meas = np.polyfit(t_x, x, 1)[0]
    v_theory = mu[0, 0] * Fx

    # rotation: small torque, short time so theta does not wrap
    traj_th, _ = run_brownian_2d(make_constant_force(0.0, 0.0, tau),
                                 jnp.zeros((1, 3)), n_steps=100, dt=dt, kT=0.0,
                                 resistance_body=resistance_body)
    traj_th = np.asarray(traj_th)
    t_th = dt * np.arange(1, 101)
    w_meas = np.polyfit(t_th, traj_th[:, 0, 2], 1)[0]
    w_theory = mu[2, 2] * tau

    return dict(name="deterministic drift speed [m/s]",
                measured=v_meas, expected=v_theory, t=t_x, x=x, x_theory=x_theory,
                w_meas=w_meas, w_theory=w_theory)


def _rel_err(meas, exp):
    return abs(meas - exp) / abs(exp)


def main(save_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime

    print("Validating brownian_generalized_2d against analytic predictions...\n")
    trans = check_translational_diffusion()
    rot = check_rotational_diffusion()
    trap = check_harmonic_equilibrium()
    drift = check_deterministic_drift()

    rows = [trans, rot, trap, drift]
    print(f"{'check':<36}{'measured':>14}{'expected':>14}{'rel.err':>10}  result")
    print("-" * 88)
    all_pass = True
    for r in rows:
        e = _rel_err(r["measured"], r["expected"])
        ok = e < PASS_TOL
        all_pass &= ok
        print(f"{r['name']:<36}{r['measured']:>14.4e}{r['expected']:>14.4e}"
              f"{e:>9.2%}  {'PASS' if ok else 'FAIL'}")
    # the rotation sub-check inside the drift test
    e_w = _rel_err(drift["w_meas"], drift["w_theory"])
    all_pass &= e_w < PASS_TOL
    print(f"{'deterministic spin rate [rad/s]':<36}{drift['w_meas']:>14.4e}"
          f"{drift['w_theory']:>14.4e}{e_w:>9.2%}  {'PASS' if e_w < PASS_TOL else 'FAIL'}")
    print("-" * 88)
    print("ALL CHECKS PASSED\n" if all_pass else "SOME CHECKS FAILED\n")

    # ---- figure: one panel per check ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.plot(trans["t"], trans["msd"], lw=1.2, label="simulated")
    ax.plot(trans["t"], 4 * trans["expected"] * trans["t"], "k--",
            label="4 D_t t (theory)")
    ax.set(xlabel="t [s]", ylabel="MSD [m$^2$]", title="1. Translational diffusion")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(rot["t"], rot["C"], lw=1.2, label="simulated")
    ax.plot(rot["t"], np.exp(-rot["expected"] * rot["t"]), "k--",
            label=r"$e^{-D_r t}$ (theory)")
    ax.set(xlabel="t [s]", ylabel=r"$\langle\cos\Delta\theta\rangle$",
           title="2. Rotational diffusion")
    ax.legend()

    ax = axes[1, 0]
    ax.hist(trap["samples"] * 1e9, bins=80, density=True, alpha=0.6,
            label="simulated")
    xs = np.linspace(trap["samples"].min(), trap["samples"].max(), 200)
    g = np.exp(-xs ** 2 / (2 * trap["expected"])) / np.sqrt(2 * np.pi * trap["expected"])
    ax.plot(xs * 1e9, g / 1e9, "k--", label="Gaussian, Var=kT/k")
    ax.set(xlabel="x [nm]", ylabel="pdf", title="3. Harmonic-trap equilibrium")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(drift["t"], drift["x"] * 1e9, lw=1.5, label="simulated")
    ax.plot(drift["t"], drift["x_theory"] * 1e9, "k--", label=r"$\mu_t F t$ (theory)")
    ax.set(xlabel="t [s]", ylabel="x [nm]", title="4. Deterministic drift (kT=0)")
    ax.legend()

    fig.tight_layout()
    save_dir = Path(save_dir) if save_dir else (
        Path(__file__).resolve().parent / "results"
        / f"validation_{datetime.now():%Y%m%d-%H%M%S}")
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "validation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return all_pass


if __name__ == "__main__":
    main()
