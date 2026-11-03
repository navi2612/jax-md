# Speedup notes

Status (2026-11-03): change 1 below is already in `run.py`
(`initial_extra_capacity=4`). **Changes 2-5 are not applied.** These notes exist so
the work can be picked up later.

All numbers: 1131 triangles, phi 0.4, R = 300 um, wall, float32, dt = 0.4 ms,
2500 steps per sample, RTX 5080, unless stated otherwise. "Today" below means the
code as benchmarked, with `initial_extra_capacity=32`.

## Where the time goes

| Per step | Time |
|---|---:|
| Empty loop (fixed GPU cost) | 0.014 ms |
| Integrator + drive | 0.015 ms |
| + wall | 0.041 ms |
| + contacts, 48 neighbor slots, list never checked | 0.378 ms |
| + contacts, 16 neighbor slots, list never checked | 0.104 ms |
| Neighbor list check alone | 0.132 ms |
| **Full step as benchmarked (48 slots)** | **0.487 ms** |

Two causes:

1. **Too many slots.** `initial_extra_capacity=32` gave 48 slots per body, and
   empty slots are still computed. No body in the 1500 s triangle runs
   ever had more than 12 neighbors.
2. **The list is checked every step.** It only rebuilds about every 670 steps, so
   nearly all checks do nothing, yet each one costs more than the forces at 16 slots.

## Measured gain

Prototype: tight slots and a check every **50** steps. Not the recommended 25.

| Run | Today | Prototype | Speedup |
|---|---:|---:|---:|
| Triangle, phi 0.4 | 1.21 s/sample | 0.27 s/sample | 4.4x |
| Square, phi 0.4 | 0.91 s/sample | 0.24 s/sample | 3.8x |
| Disk, phi 0.4 | 0.85 s/sample | 0.15 s/sample | 5.6x |
| Triangle, phi 0.6 | 1.63 s/sample | 0.45 s/sample | 3.7x |
| Triangle, phi 0.2 | 0.92 s/sample | 0.21 s/sample | 4.4x |

Per step, triangles, by check interval at 16 slots: every step 0.21-0.24 ms,
every 10 steps 0.121 ms, every 25 steps 0.112 ms, every 50 steps 0.108 ms.

## The changes, safest first

### 1. Fewer spare slots (setting only)

Already set in `run.py`: `initial_extra_capacity=4` instead of `32`. This gives about
20 slots for triangles at phi 0.4; 20 slots measured about 1.7x faster. The value 4
itself was not timed.

A full list is detected, never silent: `make_runner` grows the list and restarts
the run, which costs time but not correctness. Avoid 0, because the list is sized
from the crowding at the start, so a sparse start could overflow once clusters form.

### 2. Slots based on shape size

In `spheropolygon.make_interaction`, use a fixed slot count instead of one based on
the start. The prototype gave 16 slots for triangles, 12 for squares, 9 for disks:

```python
cutoff = 2 * shapes.reach + list_margin            # list_margin = 0.4e-6
neighbor_fn = partition.neighbor_list(displacement_fn, box, cutoff,
                                      dr_threshold=dr_threshold,   # 1.6e-6
                                      capacity_multiplier=1.0)
r_list = cutoff + dr_threshold
n_slots = int(np.ceil(1.6 * np.pi * r_list ** 2 / AREA))

def allocate(body, extra_capacity=0, **kw):
    used = neighbor_fn.allocate(wrap(body.center), **kw).idx.shape[1]
    nbrs = neighbor_fn.allocate(wrap(body.center),
                                extra_capacity=extra_capacity + max(0, n_slots - used), **kw)
    return dataclasses.replace(nbrs, update_fn=update)
```

The list radius stays 17.5 um for triangles, but it rebuilds at 0.8 um of movement
instead of 1 um. That spare 0.4 um allows each body 200 nm between checks. The
margin only matters once checks are less frequent (change 3).

### 3. Check every 25 steps

In `make_runner`, check once per chunk of steps and keep the list fixed inside it.
The prototype added an `update_every` argument:

```python
def chunk(c, n_steps):
    state, nbrs, ovf = c
    nbrs = nbrs.update(state.position)
    ovf = ovf | (nbrs.did_buffer_overflow != 0)
    state, _ = lax.scan(lambda s, _: (step_fn(s, neighbor=nbrs, light=light_t), None),
                        state, xs=None, length=n_steps)
    return state, nbrs, ovf
```

Run `steps_per_sample // update_every` chunks, plus one for any remainder. Handle
the case without a neighbor list separately.

**A check interval is only safe for a limited force.** A contact can be missed with
no warning if bodies move more than the margin between checks. Estimated largest
move per body (drift + 5-sigma thermal kick, ignoring bodies pushing each other),
against the 200 nm allowance:

| Check every | Force x1 | Force x3 (`power=3`) | Force x10 |
|---:|---:|---:|---:|
| 50 steps | 142 nm | **239 nm, unsafe** | **576 nm, unsafe** |
| 25 steps | 90 nm | 138 nm | **307 nm, unsafe** |
| 10 steps | 52 nm | 71 nm | 138 nm |

Measured at 50 steps and force x1: up to 127 nm for triangles and squares, 157 nm
for disks, so the estimates can run a little low. A larger `dt` also shrinks the
safe interval.

**Use 25 steps**: about 4% slower than 50, and it covers `power=3`.

### 4. Movement guard

**Never built.** At each check, record the largest distance any body moved since
the previous check and keep the maximum across the run. If it exceeds
`list_margin / 2`, print a warning and save it in the run metadata. Measure with the
periodic displacement, or a body crossing the box edge looks like a huge jump. Time
it too; it should be cheap, since it only runs once per check.

### 5. Force rewrite (optional, higher risk)

Computes each body's force from its own energy with its neighbors held fixed, which
avoids adding into shared slots on the GPU. Measured about 2x faster (0.7 vs
1.42 s/sample, 48 slots, one rough comparison with another job running):

```python
def force(body, neighbor, **kw):
    C, T = body.center, body.orientation
    Vall = _lab_vertices(P, T); Eall = jnp.take(Vall, nx, axis=1)
    def energy_i(ci, ti, idx):
        valid = idx < N; j = jnp.where(valid, idx, 0)
        Vi = _lab_vertices(P, ti); Ei = jnp.take(Vi, nx, axis=0)
        dR = C[j] - ci; dR = dR - box * jnp.round(dR / box)            # R_j - R_i
        Aj, Bj = Vall[j] + dR[:, None], Eall[j] + dR[:, None]
        d1 = _point_segment(Vi[None, :, None, :], Aj[:, None, :, :], Bj[:, None, :, :])    # my vertices, their edges
        d2 = _point_segment(Aj[:, :, None, :], Vi[None, None, :, :], Ei[None, None, :, :]) # their vertices, my edges
        m = valid[:, None, None]
        return 0.5 * k * (jnp.sum(jnp.where(m, jnp.maximum(2 * rad - d1, 0.0) ** 2, 0.0))
                          + jnp.sum(jnp.where(m, jnp.maximum(2 * rad - d2, 0.0) ** 2, 0.0)))
    Fc, Ft = vmap(grad(energy_i, (0, 1)))(C, T, neighbor.idx)
    return rigid_body.RigidBody(-Fc, -Ft)
```

Known problems, all to resolve before using it:

- **Triangles only so far.** It has one shape per run and lacks the 1/2 weight that
  `make_interaction` applies to disk-disk contacts, so **as written it would double
  disk contact forces**.
- **Newton's 3rd law is no longer exact.** Each contact is computed twice, so the
  force pairs match only up to rounding.
- **Forces differ from the current code** by up to 0.07% of the largest force
  (1.66e-15 N of 2.36e-12 N). That is about what float32 position rounding should
  give, but this was estimated, not confirmed. Compare both in float64 first.
- **Not needed for correct physics.** It is only worth it for the speed, or for
  exact repeats.

## Evidence so far

From the 50-step prototype of changes 2 and 3, not the recommended 25 steps:

| Run, 200 s then 2000 checks | Slots | Most neighbors | Close pairs checked | Missed |
|---|---:|---:|---:|---:|
| Triangle, phi 0.6 | 16 | 12 | 8.7 million | 0 |
| Square, phi 0.6 | 12 | 8 | 6.8 million | 0 |
| Disk, phi 0.6 | 9 | 6 | 6.4 million | 0 |
| Triangle, phi 0.2 | 16 | 11 | 1.8 million | 0 |

Second half of a 150 s run, triangles, phi 0.4, seeds 0 / 1:

| | Today | Prototype |
|---|---:|---:|
| Speed (um/s) | 0.759 / 0.826 | 0.776 / 0.802 |
| Fraction near the wall | 0.147 / 0.144 | 0.148 / 0.147 |

This would catch a real error but not a 1% shift, because it only has two seeds.

**Not tested:** 25-step checks, triple force, the guard, periodic runs, and anything
with the changes combined in the real code.

## Rechecking after any change

Same-seed GPU runs never match exactly (rounding in the GPU's force sum, plus XLA
auto-tuning), so **exact path differences are expected and are not a failure.**
Differences of tens of um per particle after 150 s also appear from a 0.03 nm nudge
to a single particle. Compare averages across seeds instead.

1. `python validate.py` and `python validate_spheropolygon.py`.
2. Missed contacts: from jammed states, at every check, confirm that all pairs close
   enough to touch are in the current list (should be 0 missing). Also record the
   largest move between checks. Run it at the forces you actually use, including
   `power=3`.
3. Averages against the old code, over more than two seeds: speed, wall fraction,
   cluster sizes. The old code is in git.
4. Time it on the GPU with no other job running.

## Tried, not worth it

- **Compiler flags:** none gave a clear gain. Turning off command buffers
  (`--xla_gpu_enable_command_buffer=`) made it about 2x slower.
- **`--xla_gpu_deterministic_ops=true`:** far too slow to use; the second sample
  alone took over 2.5 minutes.
- **Sparse neighbor list:** about 5.5x at a safe size against 4.5x for a dense list
  with 16 slots and 50-step checks. It needs the contact code rewritten and can
  overflow more easily.
- **`dt` = 1 ms:** 2.5x faster, but jammed particles move about 20% too fast.
