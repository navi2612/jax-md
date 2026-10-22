# Collective-behaviour campaign — handoff

State as of 2026-10-22, written when moving from the MacBook (M2 Max) to the
Ryzen 9950X3D / RTX 5080 PC. Everything needed to continue is here, in the code
and in `optical_sim/results/` (copied separately; results are git-ignored).

## Goal

Find simulations of optically driven micro-particles that show **collective
behaviour an experiment could reproduce**. Shapes: disks, squares, triangles,
and **mixtures** of them in one arena (not composite / glued particles). There
is no experimental data yet; the simulations are meant to guide the design.

Freedoms agreed with Ivan:
- Force and torque may be chosen freely as long as they are realistic
  (metasurfaces make them easy to engineer). Measured design: F ≈ 1.135 pN at
  30 µW/µm² on a 10×10×1 µm particle → v0 = F/γ_T ≈ 2.4 µm/s, propulsion
  efficiency η ≈ 0.085 of the 13 pN optical momentum flux.
- More particles and larger areas are fine.
- A wall is fine for screening; the experiment-like boundary is a light spot.

Working rules (from earlier sessions): Ivan handles all deletes and commits;
use multi-bead bodies, never single-bead disks to save time (no collision
torque); one run is not a measurement (replicate seeds, give intervals, check
the run is long enough before believing a null); if a file was deleted, ask
rather than digging it out of git history.

## Physics that shapes everything

- Drag (all shapes, `trajectory.DEFAULT_RESISTANCE_BODY`): γ_T = 470e-9 N s/m,
  γ_R = 8000e-21 N m s. Rotational diffusion D_R = kT/γ_R ≈ 5·10⁻⁴ s⁻¹, i.e. a
  heading persists ~30 min. **A straight swimmer travels millimetres before it
  turns**, so any boundary closer than ~1 mm captures it.
- Translational diffusion kT/γ_T ≈ 0.009 µm²/s: a particle in the dark is frozen.
- Circle swimmers: turning radius R_T = F γ_R / (γ_T τ) (`collective.tau_for_RT`).
  Orbit period 2πR_T/v0 (≈ 105 s for R_T = 40 µm).
- Force and torque both scale linearly with intensity, so R_T does not change
  in dim light; particles just slow down.
- The optical **gradient force is not modelled**. It is exactly zero in uniform
  light, but at a beam edge it may matter: slab estimate F_grad ≈ Δε·V·I/(2 n c L)
  gives ~4 pN (3.6× the propulsion work over one particle length) for a
  silicon-like particle, ~0.1–0.3 pN for polymer/glass. If the particles are
  mostly high-index, a light-only boundary could reflect rather than absorb.
  Cheap model if needed: U = −κ Σ_beads I(bead), κ from one MEEP calibration run.

## Code map (optical_sim/)

| file | role |
|---|---|
| `swarm.py` | base: `make_runner`, shapes, drives, neighbour-list soft spheres. Precision switch `OPTSIM_X64` (default 1 = float64). |
| `arena.py` | single-shape light spot + circular wall (the first experiments, below). |
| `collective.py` | the campaign: equal-area disk/square/triangle from beads of one size (σ = 10/3 µm, smooth outlines, spacing 0.5 σ, + centre bead), mixtures via jax-md `shape_species`, per-type drive (F, chiral τ, linear-pol aligning torque K sin 2(φ_pol−θ)), circular wall as an energy on every bead **or** periodic box, controls, analysis, ledgers, videos. |
| `runs.py` | `save_run`: one folder per run (`trajectory.npz`, `metadata.json`), now also `code/` = all imported optical_sim modules as of process start + sha256 in metadata. |
| `bench_backend.py` | CPU vs GPU / float64 vs float32 benchmark of the Phase-2 workload. |
| `environment.yml` | minimal env (`optsim`); GPU notes inside. |

Body frame: drive along body +x; triangles point a vertex (`heading="tip"`) or
the base (`"base"`) along +x. All shapes have area 100 µm² (triangle side
15.2 µm, disk radius 5.64 µm); bead-rounded corners make the collision outline
94–98 µm².

Analysis references (important — earlier versions got this wrong):
- **Structure** (largest cluster, density fluctuations): random sequential
  placement (RSA) at the same composition. *Not* the start configuration (it is
  pushed apart until just touching → contact-rich) and *not* the
  interactions-off control (bodies overlap freely).
- **Dynamics** (rotation, polar order, alignment, flocking, lanes): the same run
  with interactions switched off (`*_ctrl` folder).
- Walled arenas: always split wall accumulation from bulk (`collective.wall_split`).
- Rotation counts only if it *increases* vs control (`|ω| − |ω_ctrl| ≥ 0.15`);
  interactions usually suppress it.

## Results so far

### 1. Light spot, single shape (`arena.py`, 2026-10-21; results/arena_2026102*)

Homogeneous top-hat spot (2 µm tanh edge), wall far outside, 3×3-bead squares.
Outside the light particles freeze, nothing returns: the spot drains.

| spot | φ | drive | half gone after | inside at 1200 s | free-swimmer baseline |
|---|---|---|---|---|---|
| 110 µm | 0.3 | R_T 40 µm | 100 s | 0.05 | 0.40 |
| 110 µm | 0.3 | R_T 20 µm | 108 s | 0.15 | 0.67 |
| 110 µm | 0.3 | linear pol | 46 s | 0.03 | – |
| 220 µm | 0.3 | R_T 20 µm | 572 s | 0.42 | 0.83 |
| 220 µm | 0.5 | R_T 20 µm | 340 s | 0.27 | 0.83 |

Collisions, not orbit geometry, eject particles (compare with the
non-interacting baseline); higher density drains faster. No edge current, no
collective state inside. Linear-pol swimmers leave along ±x.

### 2. Phase 1 — walled 150 µm arena, light everywhere (results/collective/LEDGER.md)

3 shapes × {R_T 20, R_T 60, straight, linear pol K = 3e-19 N m} × φ {0.25, 0.45},
900 s. 20 of 24 done (triangle φ = 0.45 dropped: not informative).
The ledger's "hits" column used an old rotation criterion; the re-score is:

- **Wall dominates:** 18/20 runs have 73–99 % of particles in wall-attached
  clusters, largest bulk cluster 1–9 %. Only disks with R_T = 20 µm stay in the
  bulk (interior 77–83 %), but form no clusters there (13–16 %).
- **Disks never align** (no collision torque) — as expected.
- **Triangles align — the one real signal:** polar order 0.40–0.47 vs control
  0.03–0.07 for R_T 20, R_T 60 and linear pol (straight: 0.26 vs 0.09). For
  linear pol the control is half +x / half −x; interactions make a majority pick
  one direction = symmetry breaking. Squares weaker (0.18–0.28 in the interior).
  Caveat: most triangles sit at the wall pointing outward, so this needs a
  wall-free test → Phase 2.

### 3. Phase 2 — periodic box (bulk), NOT YET RUN

Started on the Mac and stopped half-way through the first two runs for the
move; no Phase-2 result exists (`results/collective/LEDGER_periodic.md` absent).

Grid (`collective.phase2_configs`): 400 µm periodic box, 1000 s, φ 0.25
(400 particles) and 0.45 (720):
straight swimmers {triangle tip, triangle base, square, disk},
linear pol {triangle, square}, chiral R_T 60 µm {triangle}.
New observables: polar order + its time series, flock = |⟨v⟩|/v0, lane index
(+x/−x mover segregation), f_dense (fraction in cells with local φ > 0.55),
largest cluster. Verdicts: `collective.THRESH_P`.

Cost on the Mac (CPU float64): ~525–614 steps/s for 400 particles → ~1.1 h
per φ = 0.25 run, ~2 h per φ = 0.45 run, 2 in parallel.

## Next steps

0. **Benchmark the new PC** (≈ 15 min), then decide how to run:
   ```bash
   cd optical_sim
   JAX_PLATFORMS=cpu  python bench_backend.py
   JAX_PLATFORMS=cpu  python bench_backend.py --parallel 4     # also try 8
   JAX_PLATFORMS=cuda python bench_backend.py                  # needs Linux/WSL2
   JAX_PLATFORMS=cuda OPTSIM_X64=0 python bench_backend.py
   ```
   Consumer GPUs run float64 at ~1/64 speed, so the GPU likely only pays off in
   float32. Float32 looks safe on paper (positions ~1e-4 m → resolution ~1e-11 m
   ≪ step ~1e-9 m; forces/energies far from float32 limits) and gave the same
   mean speed on the Mac CPU (0.881 vs 0.880 v0), but must be validated: one
   Phase-2 config in both precisions, same seed set, statistics must agree
   within their seed-to-seed spread. If the GPU wins, the next big gain is
   batching seeds/configs with `vmap` in `make_runner`.
1. **Phase 2** on whatever is fastest. Workers resume by skipping configs
   already in the ledger:
   ```bash
   python collective.py phase2 --worker 0 --nworkers N   # one per worker k < N
   ```
   (CPU: one process uses ~3 cores; on 16 cores try N = 4–6.)
2. **Phase 3:** confirm hits with 3 seeds (block-bootstrap CIs) and a 2× larger
   box (flocks/lanes can be finite-size artefacts).
3. **Phase 4:** experiment-like boundary — large light spot (300–500 µm,
   ≫ flock size), light-only edge or distant wall. Key question: what a flock
   does at the edge (turn, break up, escape). Consider the gradient force here.
4. **Phase 5:** mixtures that are easy to fabricate: active + passive
   (no metasurface), left- + right-handed circle swimmers (segregation?),
   triangles + squares. `collective.PType` lists already support this.

## Housekeeping

- Test folders `results/collective/coll_*_test_*` can be deleted.
- Phase-1/2 runs before 2026-10-22 11:00 have no `code/` snapshot and record
  `collective.py` as untracked; Phase 1 used an older `collective.py` (old
  rotation criterion). Raw trajectories are intact, so re-analysis is safe.
- To continue with Claude Code on the new PC: point it at this file. The
  Mac-side memory notes live in
  `~/.claude/projects/-Users-ivan-Documents-Promotion-Simulation-jax-md-jax-md/memory/`
  and can be copied into the new machine's project memory folder.
