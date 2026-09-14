"""Minimal reservoir-computing check for the light-driven swarm: memory capacity.

One quantity is tracked, the linear *memory capacity* MC (Jaeger 2002):

    1. drive the swarm with an i.i.d. random input u(t) in [0, 1]
    2. turn each recorded frame into a camera-like feature vector x(t)
    3. train a linear (ridge) readout to reconstruct u(t - k) from x(t)
    4. MC_k = corr^2(prediction, truth) on held-out data, MC = sum_k MC_k

MC_k as a function of k is the "forgetting curve". MC is bounded by the number
of features, and a shuffled-target control shows the chance level.

Only the pieces below the parameter block need to change to try another
reservoir: `make_light_pattern` (how the input is encoded in the light) and
`features` (what is read out).

Run:  /opt/miniconda3/envs/pmeep/bin/python optical_sim/reservoir_memory.py
"""
import numpy as np
import jax.numpy as jnp

import swarm
from trajectory import DEFAULT_RESISTANCE_BODY

# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
N_PARTICLES = 20
SIGMA = 1e-6                    # bead diameter [m]
BOX = 40e-6                     # must stay above the turning radius (printed below)
DT = 1e-3                       # integration step [s]
STEPS_PER_SAMPLE = 200          # one input sample is held this long -> 0.2 s
N_SAMPLES = 600                 # length of the input series (frames recorded)
N_WASHOUT = 50                  # first frames dropped (initial-condition transient)
TRAIN_FRACTION = 0.6            # of the remaining frames; the rest is the test set
MAX_DELAY = 20                  # MC_k computed for k = 1 .. MAX_DELAY
RIDGE = 1e-2                    # readout regularisation (features are standardised)
N_GRID = 8                      # camera: N_GRID x N_GRID occupancy image
SEEDS = (0, 1, 2)               # replicas: initial condition + Brownian noise

# how the scalar input is written into the light
I0 = 1.0                        # mean intensity (1.0 = the default drive)
MASK_STRENGTH = 0.8             # 0 -> spatially uniform light; >0 -> random tile mask
MASK_TILE = 5e-6                # tile size of the random mask [m]
MASK_SEED = 123                 # the mask is fixed; only the swarm seeds vary


# --------------------------------------------------------------------------
# input signal
# --------------------------------------------------------------------------
def make_input(n_samples, seed):
    """i.i.d. uniform input in [0, 1] -- the standard signal for memory capacity."""
    return np.random.default_rng(seed).uniform(0.0, 1.0, n_samples)


# --------------------------------------------------------------------------
# light pattern: how u(t) is encoded in the intensity seen by each particle
# --------------------------------------------------------------------------
def make_light_pattern(box):
    """Return intensity_fn(centers, u) -> per-particle intensity (N,).

    I(x, u) = I0 * (1 + MASK_STRENGTH * m(x) * (2u - 1)),  m(x) in [-1, 1]

    m is a fixed random tile pattern (an SLM mask). With MASK_STRENGTH = 0 every
    particle sees the same intensity, and for non-interacting particles the
    input then only re-times the light-off trajectory (no memory to read out).
    The mask makes particles in different tiles respond differently to the
    same input, the analogue of the input weights of an echo state network.
    """
    n_tiles = int(np.ceil(box / MASK_TILE))
    mask = jnp.asarray(np.random.default_rng(MASK_SEED).uniform(-1, 1, (n_tiles, n_tiles)))

    def intensity_fn(centers, u):
        ij = jnp.floor(centers / MASK_TILE).astype(int) % n_tiles
        m = mask[ij[:, 0], ij[:, 1]]
        return I0 * (1.0 + MASK_STRENGTH * m * (2.0 * u - 1.0))

    return intensity_fn


def make_drive(box):
    """swarm drive_fn(body, light): the default circular-polarisation drive
    (force + torque) scaled by the local intensity of the pattern."""
    intensity_fn = make_light_pattern(box)
    uniform = swarm.make_drive()               # force/torque per unit intensity

    def drive_fn(body, light):
        f_lab, tau = uniform(body, 1.0)
        s = intensity_fn(body.center, light)
        return f_lab * s[:, None], tau * s

    drive_fn.spec = {"type": "masked", "I0": I0, "mask_strength": MASK_STRENGTH,
                     "mask_tile_m": MASK_TILE, "mask_seed": MASK_SEED}
    return drive_fn


# --------------------------------------------------------------------------
# readout features: what a camera would give us
# --------------------------------------------------------------------------
def features(centers, box):
    """Occupancy image per frame: (n_frames, N_GRID*N_GRID) particle counts."""
    ij = np.floor(centers % box / box * N_GRID).astype(int).clip(0, N_GRID - 1)
    flat = ij[..., 0] * N_GRID + ij[..., 1]                      # (n_frames, N)
    n_frames = centers.shape[0]
    x = np.zeros((n_frames, N_GRID * N_GRID))
    for t in range(n_frames):
        np.add.at(x[t], flat[t], 1.0)
    return x


# --------------------------------------------------------------------------
# linear readout and memory capacity
# --------------------------------------------------------------------------
def ridge_fit(x, y, ridge):
    """Closed-form ridge regression with a bias column; returns weights."""
    xb = np.hstack([x, np.ones((x.shape[0], 1))])
    a = xb.T @ xb + ridge * np.eye(xb.shape[1])
    return np.linalg.solve(a, xb.T @ y)


def ridge_predict(x, w):
    return np.hstack([x, np.ones((x.shape[0], 1))]) @ w


def memory_capacity(x, u, shuffle=False):
    """Forgetting curve MC_k, k = 1..MAX_DELAY, from features x(t) and input u(t).

    With shuffle=True the target is permuted in time: what MC_k a readout finds
    when there is nothing to find (chance level for this test-set size)."""
    rng = np.random.default_rng(0)
    mc = np.zeros(MAX_DELAY)
    for k in range(1, MAX_DELAY + 1):
        t = np.arange(max(N_WASHOUT, k), len(u))     # frames with a valid target
        xs, ys = x[t], u[t - k]
        if shuffle:
            ys = rng.permutation(ys)
        n_train = int(TRAIN_FRACTION * len(t))
        mu, sd = xs[:n_train].mean(0), xs[:n_train].std(0) + 1e-12
        xs = (xs - mu) / sd
        w = ridge_fit(xs[:n_train], ys[:n_train], RIDGE)
        pred = ridge_predict(xs[n_train:], w)
        r = np.corrcoef(pred, ys[n_train:])[0, 1]
        mc[k - 1] = 0.0 if np.isnan(r) else r ** 2
    return mc


# --------------------------------------------------------------------------
def print_timescales():
    gamma_tr, gamma_rot = float(DEFAULT_RESISTANCE_BODY[0, 0]), float(DEFAULT_RESISTANCE_BODY[2, 2])
    v = swarm.DEFAULT_F_BODY[0] / gamma_tr * I0
    omega = swarm.DEFAULT_TAU / gamma_rot * I0
    hold = STEPS_PER_SAMPLE * DT
    print(f"speed {v*1e6:.2f} um/s, turning radius {v/omega*1e6:.1f} um (box {BOX*1e6:.0f} um)")
    print(f"hold time per sample {hold:.2f} s, rotation period {2*np.pi/omega:.1f} s "
          f"= {2*np.pi/omega/hold:.0f} samples, box crossing {BOX/v/hold:.0f} samples")
    print(f"total simulated time {N_SAMPLES*hold:.0f} s")


def main():
    print_timescales()
    runner = swarm.make_runner(N_PARTICLES, sigma=SIGMA, box=BOX, dt=DT,
                               steps_per_sample=STEPS_PER_SAMPLE,
                               drive_fn=make_drive(BOX))

    curves = []
    for seed in SEEDS:
        u = make_input(N_SAMPLES, seed)
        centers, angles, box = runner(u, seed=seed)
        x = features(centers, box)
        mc = memory_capacity(x, u)
        chance = memory_capacity(x, u, shuffle=True)
        curves.append(mc)
        print(f"seed {seed}: MC = {mc.sum():.2f}  (chance {chance.sum():.2f}, "
              f"bound {x.shape[1]})   MC_1..5 = {np.round(mc[:5], 2)}")

    curves = np.array(curves)
    print(f"\nMC = {curves.sum(1).mean():.2f} +- {curves.sum(1).std():.2f} over {len(SEEDS)} seeds")
    print("forgetting curve (mean over seeds):")
    for k, m in enumerate(curves.mean(0), start=1):
        print(f"  k={k:2d}  MC_k={m:.3f}  " + "#" * int(40 * m))
    return curves


if __name__ == "__main__":
    main()
