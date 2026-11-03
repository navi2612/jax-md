import os
# GPU in float32 unless set otherwise; must come before anything imports jax
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("OPTSIM_X64", "0")

import numpy as np

import spheropolygon as S
from render_swarm import make_video
from runs import save_run
from swarm import make_active_optical_drive, make_drive, make_runner
from trajectory import DEFAULT_RESISTANCE_BODY

# --- shape and boundary ----------------------------------------------------
KIND = "triangle"                # triangle | square | disk
HEADING = "tip"                  # triangle only: tip | base first
BOUNDARY = "wall"                # wall | periodic
PHI = 0.4                        # area fraction
R = 300e-6                       # arena radius [m]; periodic uses a box of the same area

# --- force -----------------------------------------------------------------
drive = make_drive(F_body=(1.135e-12, 0.0), tau=0.0)

F = 1.135e-12                    # propulsion force [N]
R_T = 80e-6                      # turning radius [m]; sign of tau sets the direction
G = DEFAULT_RESISTANCE_BODY
tau = F * float(G[2, 2]) / (float(G[0, 0]) * R_T)
drive = make_drive(F_body=(F, 0.0), tau=tau)

# fitted forces from active_optical.py; power = 3 scales 10 uW fits to 30 uW,
# use 1 if the coefficients are already at the power you want
# FX = [1.24171, 0.443034, 0.0]        # desing1_res60_linear_30uW
# FY = [0.0, 0.0, 0.066536]
# TZ = [0, 0, 244.48]
# drive = make_active_optical_drive(FX, FY, TZ, phi_pol=0.0, power=1)

# --- time ------------------------------------------------------------------
T_TOTAL = 500.0                 # simulated time [s]
T_SAMPLE = 1.0                   # time between saved frames [s]
DT = 4e-4                        # time step [s]
SEED = 0

# --- movie -----------------------------------------------------------------
FPS = 24
TARGET_FRAMES = 300


if __name__ == "__main__":
    N = int(round(PHI * np.pi * R ** 2 / S.AREA))
    core, r = S.core_of(KIND, HEADING)
    shapes = S.Shapes([core], [r])
    sigma = S.SIDE / 3

    if BOUNDARY == "wall":
        box = 2 * R + 20e-6
        center = (box / 2, box / 2)
        wall_fn = S.make_circular_wall(shapes, center, R)
        init = S.make_init(core, r, center, R)
    elif BOUNDARY == "periodic":
        box = np.sqrt(np.pi) * R     # same area as the arena: same N at the same PHI
        wall_fn = None
        init = S.make_init(core, r)
    else:
        raise ValueError(BOUNDARY)

    runner = make_runner(n_particles=N, sigma=sigma, box=box, drive_fn=drive,
                         interaction_fn=S.make_interaction(shapes, None),
                         wall_fn=wall_fn, init=init, dt=DT,
                         steps_per_sample=int(round(T_SAMPLE / DT)),
                         initial_extra_capacity=4, progress=True)

    u = np.ones(int(round(T_TOTAL / T_SAMPLE)))
    centers, angles, box = runner(u, seed=SEED)

    arrays = {"centers": centers, "angles": angles, "light": u, "box": box,
              "sigma": sigma, "body_outline": S.rounded_outline(core, r)}
    if BOUNDARY == "wall":
        arrays["wall"] = np.array([*center, R])      # make_video draws it

    run_dir = save_run(
        arrays=arrays,
        meta={**runner.meta, "n_samples": len(u), "seed": SEED, "kind": KIND,
              "heading": HEADING, "boundary": BOUNDARY, "phi": PHI, "R_m": R,
              "x64": os.environ["OPTSIM_X64"] != "0"},
        label=f"{KIND}_{BOUNDARY}_phi{PHI:.2f}", prefix="swarm", source=__file__)

    make_video(run_dir, out_path=str(run_dir / "movie.mp4"), color_by="orientation",
               target_frames=TARGET_FRAMES, fps=FPS)
