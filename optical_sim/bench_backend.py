"""Benchmark the Phase-2 workload on this machine's backend.

Builds exactly what ``collective.py phase2`` runs (triangles, straight
swimmers, periodic 400 um box) without saving anything, runs it once to
compile and once timed, and prints steps/s plus the wall time one full Phase-2
run (1000 s simulated, dt = 4e-4 s) would take.

Precision and device come from the environment, so compare them as separate
processes:

    JAX_PLATFORMS=cpu  python bench_backend.py               # CPU, float64
    JAX_PLATFORMS=cuda python bench_backend.py               # GPU, float64
    JAX_PLATFORMS=cuda OPTSIM_X64=0 python bench_backend.py  # GPU, float32

    python bench_backend.py --parallel 4    # 4 copies at once: total throughput
    python bench_backend.py --phi 0.45      # the 720-particle case

Reference (2026-10-22, MacBook Pro M2 Max, CPU float64, 2 runs in parallel):
~525 steps/s per run for phi=0.25 (400 triangles) -> ~80 min per run.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np


def single(phi, samples, t_sample, dt):
    import jax
    import collective as c
    from jax_md import rigid_body
    from swarm import make_runner

    L = 400e-6
    n = c.counts_for_phi_box(phi, L, [1.0])[0]
    types = [c.PType("triangle", n, "tip")]
    drive, species = c.make_species_drive(types)
    shape = c.shape_of("triangle")
    inter = c.make_interaction(None, (L / 2, L / 2), None)
    runner = make_runner(n_particles=n, sigma=c.SIGMA, box=L, shape=shape,
                         drive_fn=drive, interaction_fn=inter,
                         init=c.make_periodic_init(types, species, L), dt=dt,
                         steps_per_sample=int(round(t_sample / dt)),
                         initial_extra_capacity=32, progress=False)
    u = np.ones(samples)
    t0 = time.time()
    C, A, _ = runner(u, seed=0)                 # includes compilation
    t_first = time.time() - t0
    t0 = time.time()
    C, A, _ = runner(u, seed=1)                 # timed, already compiled
    t_run = time.time() - t0

    steps = samples * int(round(t_sample / dt))
    d = np.diff(C, axis=0)
    d -= L * np.round(d / L)
    speed = np.linalg.norm(d, axis=-1).mean() / t_sample / c.V0
    return {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "x64": bool(jax.config.read("jax_enable_x64")),
        "dtype": str(C.dtype),
        "n_particles": int(n),
        "steps_per_s": steps / t_run,
        "compile_s": t_first - t_run,
        "finite": bool(np.isfinite(C).all() and np.isfinite(A).all()),
        "mean_speed_over_v0": float(speed),
        "hours_per_phase2_run": 1000.0 / dt / (steps / t_run) / 3600,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi", type=float, default=0.25)
    ap.add_argument("--samples", type=int, default=20, help="1-s samples per timed run")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    t_sample, dt = 1.0, 4e-4

    if a.parallel == 1:
        r = single(a.phi, a.samples, t_sample, dt)
        if a.json:
            print("RESULT " + json.dumps(r))
        else:
            for k, v in r.items():
                print(f"{k:22s} {v:.3g}" if isinstance(v, float) else f"{k:22s} {v}")
        return

    cmd = [sys.executable, os.path.abspath(__file__), "--phi", str(a.phi),
           "--samples", str(a.samples), "--json"]
    procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              text=True) for _ in range(a.parallel)]
    res = []
    for p in procs:
        out, _ = p.communicate()
        for line in out.splitlines():
            if line.startswith("RESULT "):
                res.append(json.loads(line[7:]))
    if not res:
        raise SystemExit("no child finished; run without --parallel to see the error")
    sps = [r["steps_per_s"] for r in res]
    print(f"backend {res[0]['backend']}  x64 {res[0]['x64']}  n {res[0]['n_particles']}  "
          f"{len(res)} parallel runs")
    print(f"per run   {np.mean(sps):8.0f} steps/s  -> "
          f"{1000 / dt / np.mean(sps) / 3600:.2f} h per Phase-2 run")
    print(f"total     {np.sum(sps):8.0f} steps/s  -> "
          f"{np.sum(sps) / (1000 / dt) * 24:.1f} Phase-2 runs per day")


if __name__ == "__main__":
    main()
