import os
# GPU in float32 unless set otherwise; must come before anything imports jax
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("OPTSIM_X64", "0")

import numpy as np

import spheropolygon as S
from render_swarm import make_video
from runs import save_run
from swarm import make_active_optical_drive, make_drive, make_runner

# --- shape and arena -------------------------------------------------------
KIND = "triangle"                # triangle | square | disk
HEADING = "tip"                  # triangle only: tip | base first
PHI = 0.4                        # area fraction
R = 300e-6                       # arena radius [m]

# --- force -----------------------------------------------------------------
drive = make_drive(F_body=(1.135e-12, 0.0), tau=0.0)

# fitted forces from active_optical.py; power = 3 scales 10 uW fits to 30 uW,
# use 1 if the coefficients are already at the power you want
# FX = [1.24171, 0.443034, 0.0]        # desing1_res60_linear_30uW
# FY = [0.0, 0.0, 0.066536]
# TZ = [0, 0, 244.48]
# drive = make_active_optical_drive(FX, FY, TZ, phi_pol=0.0, power=1)

# --- time ------------------------------------------------------------------
T_TOTAL = 1500.0                 # simulated time [s]
T_SAMPLE = 1.0                   # time between saved frames [s]
DT = 4e-4                        # time step [s]
SEED = 0

# --- movie -----------------------------------------------------------------
FPS = 24
TARGET_FRAMES = 300


if __name__ == "__main__":
    box = 2 * R + 20e-6
    center = (box / 2, box / 2)
    N = int(round(PHI * np.pi * R ** 2 / S.AREA))
    core, r = S.core_of(KIND, HEADING)
    sigma = S.SIDE / 3

    runner = make_runner(n_particles=N, sigma=sigma, box=box, drive_fn=drive,
                         interaction_fn=S.make_interaction(S.Shapes([core], [r]), None,
                                                           center=center, r_wall=R),
                         init=S.make_arena_init(core, r, center, R), dt=DT,
                         steps_per_sample=int(round(T_SAMPLE / DT)),
                         initial_extra_capacity=32, progress=True)

    u = np.ones(int(round(T_TOTAL / T_SAMPLE)))
    centers, angles, box = runner(u, seed=SEED)

    run_dir = save_run(
        arrays={"centers": centers, "angles": angles, "light": u, "box": box,
                "sigma": sigma, "body_outline": S.rounded_outline(core, r),
                "wall": np.array([*center, R])},
        meta={**runner.meta, "n_samples": len(u), "seed": SEED, "kind": KIND,
              "heading": HEADING, "phi": PHI, "r_arena_m": R,
              "x64": os.environ["OPTSIM_X64"] != "0"},
        label=f"{KIND}_phi{PHI:.2f}", prefix="swarm", source=__file__)

    make_video(run_dir, out_path=str(run_dir / "movie.mp4"), color_by="orientation",
               target_frames=TARGET_FRAMES, fps=FPS)
