"""
Two-panel poster figure for metavehicle trajectories:

    left  = linear polarization   (straight, self-aligned)
    right = circular polarization (orbit, continuously turning)

Design choices (see poster discussion):
- Both panels share ONE physical scale (equal aspect, same window size), so the
  straight run and the circle are directly comparable.
- The particle is drawn as a small *to-scale* oriented footprint at a few points
  along each path -- big enough to read orientation, not detailed. On the left
  the markers stay aligned; on the right they rotate around the loop.
- A single large `particle.png` "hero" inset (with its own scale bar) shows what
  the particle actually looks like, decoupled from the trajectory scale, with a
  zoom line to one point on the orbit. This turns the size disparity (10 um
  particle, ~100 um orbit) into an explicit feature instead of a problem.

Trajectory input format
-----------------------
Each trajectory is an array of shape (N, 2) or (N, 3):
    columns [x, y]        -> orientation inferred from the velocity tangent
    columns [x, y, theta] -> orientation taken from theta (radians)
Positions are in `pos_units` ("m" for your jax-md output, or "um").
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Arc, FancyArrowPatch, ConnectionPatch
import matplotlib.image as mpimg


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _to_um(traj, pos_units, particle=0):
    traj = np.asarray(traj, dtype=float)
    # jax-md scan output is (N_steps, N_particles, 3): collapse the particle axis
    if traj.ndim == 3:
        traj = traj[:, particle, :]
    traj = np.atleast_2d(traj)
    xy = traj[:, :2].copy()
    if pos_units == "m":
        xy *= 1e6
    elif pos_units != "um":
        raise ValueError("pos_units must be 'm' or 'um'")
    theta = traj[:, 2] if traj.shape[1] >= 3 else None
    return xy, theta


def _orientations(xy, theta, idxs):
    """Orientation (rad) at the given sample indices."""
    if theta is not None:
        return theta[idxs]
    # infer from local tangent (central difference)
    ang = np.zeros(len(xy))
    d = np.gradient(xy, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    return ang[idxs]


def _solid_line(ax, x, y, color, lw, zorder=2):
    ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", zorder=zorder)


def _particle_footprint(size_xy):
    """Rectangle (w x h) for the particle, centered at origin, long axis = +x.

    A short heading tick (drawn in _draw_markers) marks the front so the
    orientation is readable even when the footprint is square.
    """
    w, h = size_xy
    return np.array([
        [-w / 2, -h / 2],
        [+w / 2, -h / 2],
        [+w / 2, +h / 2],
        [-w / 2, +h / 2],
    ])


def _draw_markers(ax, xy, theta, idxs, size_xy, color, style="rectangle"):
    base = _particle_footprint(size_xy)
    w = size_xy[0]
    for k, i in enumerate(idxs):
        a = theta[k]
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, -s], [s, c]])
        if style == "rectangle":
            verts = (base @ R.T) + xy[i]
            ax.add_patch(Polygon(verts, closed=True, facecolor=color,
                                 edgecolor="black", linewidth=0.8,
                                 alpha=0.85, zorder=5))
            front = xy[i] + R @ np.array([w / 2, 0.0])   # heading tick
            ax.plot([xy[i, 0], front[0]], [xy[i, 1], front[1]],
                    color="black", lw=1.2, zorder=6)
        elif style == "arrow":
            L = size_xy[0]
            ax.add_patch(FancyArrowPatch(
                xy[i] - 0.5 * L * np.array([c, s]),
                xy[i] + 0.5 * L * np.array([c, s]),
                arrowstyle="-|>", mutation_scale=12,
                color=color, lw=2.0, zorder=5))
        else:
            raise ValueError("style must be 'rectangle' or 'arrow'")


def _scale_bar(ax, length_um, window_um, center, label=None, pad_frac=0.08):
    x0 = center[0] - window_um / 2 + pad_frac * window_um
    y0 = center[1] - window_um / 2 + pad_frac * window_um
    ax.plot([x0, x0 + length_um], [y0, y0], color="black", lw=3,
            solid_capstyle="butt", zorder=8)
    ax.text(x0 + length_um / 2, y0 + 0.02 * window_um,
            label or f"{length_um:g} um", ha="center", va="bottom",
            fontsize=12, zorder=8)


def _linear_glyph(ax, window_um, center, angle_deg=0.0, fontsize=12):
    """Double-headed arrow for linear polarization, top-left."""
    cx = center[0] - 0.30 * window_um
    cy = center[1] + 0.38 * window_um
    L = 0.10 * window_um
    a = np.radians(angle_deg)
    d = L * np.array([np.cos(a), np.sin(a)])
    ax.add_patch(FancyArrowPatch((cx - d[0], cy - d[1]), (cx + d[0], cy + d[1]),
                                 arrowstyle="<|-|>", mutation_scale=14,
                                 color="crimson", lw=2.2, zorder=8))
    ax.text(cx, cy + 0.06 * window_um, r"$E_{pol}$", color="crimson",
            ha="center", va="bottom", fontsize=fontsize, zorder=8)


def _circular_glyph(ax, window_um, center, handedness=+1, fontsize=12):
    """Circular arrow for circular polarization, top-left."""
    cx = center[0] - 0.30 * window_um
    cy = center[1] + 0.38 * window_um
    r = 0.06 * window_um
    t1, t2 = (20, 300) if handedness > 0 else (240, -40)
    ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=min(t1, t2),
                     theta2=max(t1, t2), color="crimson", lw=2.2, zorder=8))
    # arrowhead at the leading end
    aend = np.radians(t2)
    tip = np.array([cx + r * np.cos(aend), cy + r * np.sin(aend)])
    tang = handedness * np.array([-np.sin(aend), np.cos(aend)])
    ax.add_patch(FancyArrowPatch(tip - 1e-3 * tang, tip + 1e-3 * tang,
                                 arrowstyle="-|>", mutation_scale=14,
                                 color="crimson", lw=2.2, zorder=8))
    lbl = r"$E_{pol}$" if handedness > 0 else r"$E_{pol}$"
    ax.text(cx, cy + r + 0.01 * window_um, lbl, color="crimson",
            ha="center", va="bottom", fontsize=fontsize, zorder=8)


def _radius_annotation(ax, center, radius, angle_deg=-58.0, color="black",
                       fontsize=13):
    """Arrow from the orbit centre to the rim, labelled R_T (fills the empty
    interior of the circular orbit with the key result)."""
    a = np.radians(angle_deg)
    tip = (center[0] + radius * np.cos(a), center[1] + radius * np.sin(a))
    ax.add_patch(FancyArrowPatch(tuple(center), tip, arrowstyle="-|>",
                                 mutation_scale=14, color=color, lw=1.8,
                                 zorder=7))
    ax.scatter([center[0]], [center[1]], s=14, color=color, zorder=7)
    mid = np.array([center[0] + 0.5 * radius * np.cos(a),
                    center[1] + 0.5 * radius * np.sin(a)])
    perp = np.array([-np.sin(a), np.cos(a)])
    lab = mid + 0.16 * radius * perp
    ax.text(lab[0], lab[1], rf"$R_T \approx {radius:.0f}\,\mu$m",
            color=color, ha="center", va="center", fontsize=fontsize, zorder=8)


def _hero_inset(parent_ax, image_path, bounds=(0.55, 0.55, 0.42, 0.42),
                caption=None, band_frac=0.16):
    """Big particle render placed *inside* parent_ax. No scale bar -- the
    to-scale markers + connector already carry the size. An optional one-line
    `caption` (e.g. "10 x 10 um") sits in a white strip below the image.
    bounds are in parent-axes fraction. Returns the inset axes."""
    ax = parent_ax.inset_axes(bounds)
    ax.set_facecolor("white")
    try:
        img = mpimg.imread(image_path)
        h, w = img.shape[0], img.shape[1]
        ax.imshow(img, extent=(0, w, h, 0), zorder=1)
        band = (band_frac * h) if caption else 0.0
        ax.set_xlim(0, w)
        ax.set_ylim(h + band, 0)
        if caption:
            ax.text(0.5 * w, h + 0.55 * band, caption, color="black",
                    ha="center", va="center", fontsize=10, zorder=3)
    except FileNotFoundError:
        ax.text(0.5, 0.5, "particle.png\nnot found", ha="center", va="center")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
def _decimate(xy, th, max_points):
    """Subsample a long trajectory to at most max_points for fast rendering."""
    n = len(xy)
    if n <= max_points:
        return xy, th
    step = int(np.ceil(n / max_points))
    sel = np.arange(0, n, step)
    if sel[-1] != n - 1:
        sel = np.append(sel, n - 1)
    return xy[sel], (th[sel] if th is not None else None)


def _markers_in_window(xy, cen, win, n):
    """Pick n marker indices among the points lying inside the window."""
    half = win / 2
    inside = np.where((np.abs(xy[:, 0] - cen[0]) <= half) &
                      (np.abs(xy[:, 1] - cen[1]) <= half))[0]
    if len(inside) == 0:
        return np.array([], dtype=int)
    pick = np.unique(np.linspace(0, len(inside) - 1, n).astype(int))
    return inside[pick]


def plot_polarization_trajectories(
    traj_linear,
    traj_circular,
    pos_units="m",
    particle_size_um=(10.0, 10.0),
    marker_style="rectangle",
    n_markers=6,
    window_um=None,            # shared window size; None -> auto from circular
    linear_pol_angle_deg=0.0,
    handedness=+1,             # +1 LCP, -1 RCP (sets circular glyph + sense)
    particle_image="particle.png",
    image_span_um=14.0,        # physical width the hero image spans, for its bar
    line_color="#0f54c4",      # solid colour of the trajectory line (red)
    marker_color="0.6",        # particle footprint colour (grey)
    window_pad=1.1,            # window = window_pad * orbit diameter (+ margin)
    show_radius=True,          # draw the R_T arrow + value in the circular panel
    note_linear=None,          # optional caption text, bottom of linear panel
    note_circular=None,        # optional caption text, bottom of circular panel
    max_points=4000,           # decimation cap -> huge speedup for long runs
    out_path="metavehicle_trajectories.png",
    dpi=300,
    show=False,
):
    if np.isscalar(particle_size_um):
        particle_size_um = (float(particle_size_um), float(particle_size_um))

    xy_lin, th_lin = _decimate(*_to_um(traj_linear, pos_units), max_points)
    xy_cir, th_cir = _decimate(*_to_um(traj_circular, pos_units), max_points)

    ext_lin = (np.ptp(xy_lin[:, 0]), np.ptp(xy_lin[:, 1]))
    ext_cir = (np.ptp(xy_cir[:, 0]), np.ptp(xy_cir[:, 1]))
    print(f"linear   extent: {ext_lin[0]:.1f} x {ext_lin[1]:.1f} um, "
          f"{len(xy_lin)} pts")
    print(f"circular extent: {ext_cir[0]:.1f} x {ext_cir[1]:.1f} um, "
          f"{len(xy_cir)} pts")

    # shared window sized tightly to the circular orbit
    if window_um is None:
        window_um = window_pad * max(ext_cir) + 2 * max(particle_size_um)

    # circular centered on the orbit; linear centered near its start so a
    # window-sized chunk is shown (the rest is clipped to the same axis size)
    cen_cir = xy_cir.mean(axis=0)
    d = xy_lin[-1] - xy_lin[0]
    u = d / np.hypot(*d) if np.hypot(*d) > 0 else np.array([1.0, 0.0])
    cen_lin = xy_lin[0] + 0.35 * window_um * u

    plt.rcParams.update({"font.size": 13})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6.2), dpi=dpi)

    for ax, xy, th, cen, kind in (
        (ax0, xy_lin, th_lin, cen_lin, "linear"),
        (ax1, xy_cir, th_cir, cen_cir, "circular"),
    ):
        _solid_line(ax, xy[:, 0], xy[:, 1], line_color, lw=2.5)

        idxs = _markers_in_window(xy, cen, window_um, n_markers)
        if len(idxs):
            ang = _orientations(xy, th, idxs)
            _draw_markers(ax, xy, ang, idxs, particle_size_um, marker_color,
                          marker_style)

        # start marker only if it falls inside the shown window
        if (abs(xy[0, 0] - cen[0]) <= window_um / 2 and
                abs(xy[0, 1] - cen[1]) <= window_um / 2):
            ax.scatter(xy[0, 0], xy[0, 1], s=70, facecolor="white",
                       edgecolor="black", linewidth=1.5, zorder=7)

        ax.set_xlim(cen[0] - window_um / 2, cen[0] + window_um / 2)
        ax.set_ylim(cen[1] - window_um / 2, cen[1] + window_um / 2)
        ax.set_aspect("equal")
        ax.set_xlabel("x  (µm)")
        if kind =="linear":
            ax.set_ylabel("y  (µm)")
        ax.tick_params(labelsize=11)

        if kind == "linear":
            _linear_glyph(ax, window_um, cen, linear_pol_angle_deg)
            note = note_linear
        else:
            _circular_glyph(ax, window_um, cen, handedness)
            if show_radius:
                r_T = float(np.mean(np.hypot(xy[:, 0] - cen[0],
                                             xy[:, 1] - cen[1])))
                print(f"orbit radius R_T ~ {r_T:.1f} um")
                _radius_annotation(ax, cen, r_T)
            note = note_circular

        if note:
            ax.text(0.5, 0.03, note, transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=12, color="0.25")

    ax0.set_title("Linear polarization", fontsize=16, pad=10)
    ax1.set_title("Circular polarization", fontsize=16, pad=10)

    # hero inset placed INSIDE the linear panel (upper-right, where it's empty),
    # with a connector to a marker on the linear trajectory
    caption = f"{particle_size_um[0]:g} × {particle_size_um[1]:g} µm"
    hero = _hero_inset(ax0, particle_image, bounds=(0.55, 0.6, 0.33, 0.33),
                       caption=caption)
    try:
        idxs_l = _markers_in_window(xy_lin, cen_lin, window_um, n_markers)
        if len(idxs_l):
            i = int(idxs_l[len(idxs_l) // 2])
            con = ConnectionPatch(
                xyA=(0.5, 0.0), coordsA=hero.transAxes,
                xyB=(xy_lin[i, 0], xy_lin[i, 1]), coordsB=ax0.transData,
                color="0.4", lw=1.0, linestyle="--")
            fig.add_artist(con)
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"saved {out_path}")
    if show:
        return fig
    plt.close(fig)
    return None


def _demo_force_curve():
    """Demo orientation sweep (linear pol): restoring torque ~ sin 2phi and
    propulsion force ~ cos^2 phi, phi = 0..180 deg. Replace with your MEEP sweep."""
    phi = np.linspace(0.0, 180.0, 181)
    tau = np.sin(np.radians(2.0 * phi))           # tau_z / tau_0
    force = np.cos(np.radians(phi)) ** 2          # F_x / F_max
    return phi, tau, force


def plot_with_force_curve(
    traj_linear,
    traj_circular,
    phi_deg=None,              # orientation sweep (deg); None -> demo data
    tau_curve=None,            # tau_z(phi), normalized; None -> demo
    force_curve=None,          # F_x(phi),  normalized; None -> demo
    pos_units="m",
    particle_size_um=(10.0, 10.0),
    marker_style="rectangle",
    n_markers=6,
    handedness=+1,
    linear_pol_angle_deg=0.0,
    window_pad=1.1,
    lin_height_frac=0.28,      # height of the linear strip as fraction of column
    line_color="#0f54c4",
    marker_color="0.6",
    tau_color="#1b4965",
    force_color="#e8913a",
    note_linear=None,
    note_circular=None,
    font_scale=1.0,            # >1 enlarges all ticks/labels/titles for posters
    max_points=4000,
    fig_size=(15.5, 7.0),
    out_path="metavehicle_force_curve.png",
    dpi=300,
    show=False,
):
    """Three-panel poster figure:
        top-left    : optical response vs orientation (linear pol)
        bottom-left : linear-pol trajectory (straight, self-aligned)
        right       : circular-pol trajectory (orbit + R_T), spans both rows
    """
    if np.isscalar(particle_size_um):
        particle_size_um = (float(particle_size_um), float(particle_size_um))
    if phi_deg is None or tau_curve is None or force_curve is None:
        phi_deg, tau_curve, force_curve = _demo_force_curve()

    fs = font_scale
    plt.rcParams.update({
        "font.size": 15 * fs,
        "axes.titlesize": 21 * fs,
        "axes.labelsize": 19 * fs,
        "xtick.labelsize": 15 * fs,
        "ytick.labelsize": 15 * fs,
        "legend.fontsize": 14 * fs,
    })

    xy_lin, th_lin = _decimate(*_to_um(traj_linear, pos_units), max_points)
    xy_cir, th_cir = _decimate(*_to_um(traj_circular, pos_units), max_points)

    cen_cir = xy_cir.mean(axis=0)
    ext_cir = max(np.ptp(xy_cir[:, 0]), np.ptp(xy_cir[:, 1]))
    W = window_pad * ext_cir + 2 * max(particle_size_um)
    r_T = float(np.mean(np.hypot(xy_cir[:, 0] - cen_cir[0],
                                 xy_cir[:, 1] - cen_cir[1])))
    print(f"orbit radius R_T ~ {r_T:.1f} um")

    fig_w, fig_h = fig_size
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    # --- manual layout (figure fractions): the two left panels share ONE width
    #     (so their y-axes align) and their combined height == the right panel ---
    bottom, top = 0.10, 0.92
    H = top - bottom                          # shared column height
    circ_w = H * fig_h / fig_w                # square circular box (in inches)
    circ_left = 0.985 - circ_w
    ax_cir = fig.add_axes([circ_left, bottom, circ_w, H])

    left1, gap_x = 0.085, 0.13      # wider gap so the F_x twin axis clears the
    col_w = circ_left - gap_x - left1         # circular panel's y-axis labels
    h_lin = lin_height_frac * H               # short linear strip
    gap_y = 0.15
    h_curve = H - gap_y - h_lin
    ax_curve = fig.add_axes([left1, bottom + h_lin + gap_y, col_w, h_curve])
    ax_lin = fig.add_axes([left1, bottom, col_w, h_lin])

    # shared physical scale (inches per um), fixed by the square circular box
    s = (circ_w * fig_w) / W

    # ---- top-left: optical response vs orientation ----
    ax_curve.axhline(0.0, color="0.7", lw=1.0, zorder=1)
    l1, = ax_curve.plot(phi_deg, tau_curve, color=tau_color, lw=2.6,
                        label=r"$\tau_z/\tau_0$", zorder=3)
    ax_curve.set_xlabel(r"orientation $\varphi$ (deg)")
    ax_curve.set_ylabel(r"$\tau_z/\tau_0$", color=tau_color)
    ax_curve.tick_params(axis="y", colors=tau_color)
    ax_curve.set_xlim(phi_deg.min(), phi_deg.max())
    ax_curve.set_xticks([0, 45, 90, 135, 180])

    ax_f = ax_curve.twinx()
    l2, = ax_f.plot(phi_deg, force_curve, color=force_color, lw=2.6, ls="--",
                    label=r"$F_x/F_{max}$", zorder=3)
    ax_f.set_ylabel(r"$F_x/F_{max}$", color=force_color)
    ax_f.tick_params(axis="y", colors=force_color)

    # mark the stable, self-aligned operating point (phi = 0)
    ax_curve.scatter([0], [0], s=60, color=tau_color, zorder=5)
    ax_curve.annotate("aligned\n(stable)", xy=(0, 0), xytext=(22, 0.32),
                      textcoords="data", fontsize=13 * fs, color="0.25",
                      arrowprops=dict(arrowstyle="->", color="0.5"))
    ax_curve.legend(handles=[l1, l2], loc="lower center", ncol=2, frameon=False)
    ax_curve.set_title("Optical response vs orientation")

    # ---- bottom-left: linear trajectory (short strip, same scale as orbit) ----
    xr = (col_w * fig_w) / s                   # x-range at shared scale
    yr = (h_lin * fig_h) / s                   # y-range (tight) at shared scale
    d = xy_lin[-1] - xy_lin[0]
    u = d / np.hypot(*d) if np.hypot(*d) > 0 else np.array([1.0, 0.0])
    cen_lin = xy_lin[0] + 0.40 * xr * u
    cen_lin[1] = xy_lin[0, 1]
    _solid_line(ax_lin, xy_lin[:, 0], xy_lin[:, 1], line_color, lw=2.5)
    hx, hy = xr / 2, yr / 2
    inside = np.where((np.abs(xy_lin[:, 0] - cen_lin[0]) <= hx) &
                      (np.abs(xy_lin[:, 1] - cen_lin[1]) <= hy))[0]
    if len(inside):
        pick = inside[np.unique(np.linspace(0, len(inside) - 1,
                                            n_markers).astype(int))]
        ang = _orientations(xy_lin, th_lin, pick)
        _draw_markers(ax_lin, xy_lin, ang, pick, particle_size_um,
                      marker_color, marker_style)
    if abs(xy_lin[0, 0] - cen_lin[0]) <= hx:
        ax_lin.scatter(xy_lin[0, 0], xy_lin[0, 1], s=70, facecolor="white",
                       edgecolor="black", linewidth=1.5, zorder=7)
    ax_lin.set_xlim(cen_lin[0] - hx, cen_lin[0] + hx)
    ax_lin.set_ylim(cen_lin[1] - hy, cen_lin[1] + hy)
    ax_lin.set_xlabel("x  (µm)"); ax_lin.set_ylabel("y  (µm)")
    # E_pol double arrow, top-left of the strip
    gx, gy, Lg = cen_lin[0] - 0.42 * xr, cen_lin[1] + 0.28 * yr, 0.06 * xr
    aa = np.radians(linear_pol_angle_deg)
    dd = Lg * np.array([np.cos(aa), np.sin(aa)])
    ax_lin.add_patch(FancyArrowPatch((gx - dd[0], gy - dd[1]),
                                     (gx + dd[0], gy + dd[1]),
                                     arrowstyle="<|-|>", mutation_scale=14,
                                     color="crimson", lw=2.2, zorder=8))
    ax_lin.text(gx, gy + 0.12 * yr, r"$E_{pol}$", color="crimson",
                ha="center", va="bottom", fontsize=15 * fs, zorder=8)
    ax_lin.set_title("Linear polarization")
    if note_linear:
        ax_lin.text(0.5, 0.04, note_linear, transform=ax_lin.transAxes,
                    ha="center", va="bottom", fontsize=13 * fs, color="0.25")

    # ---- right: circular trajectory ----
    _solid_line(ax_cir, xy_cir[:, 0], xy_cir[:, 1], line_color, lw=2.5)
    idxs = _markers_in_window(xy_cir, cen_cir, W, n_markers)
    if len(idxs):
        ang = _orientations(xy_cir, th_cir, idxs)
        _draw_markers(ax_cir, xy_cir, ang, idxs, particle_size_um,
                      marker_color, marker_style)
    ax_cir.scatter(xy_cir[0, 0], xy_cir[0, 1], s=70, facecolor="white",
                   edgecolor="black", linewidth=1.5, zorder=7)
    ax_cir.set_xlim(cen_cir[0] - W / 2, cen_cir[0] + W / 2)
    ax_cir.set_ylim(cen_cir[1] - W / 2, cen_cir[1] + W / 2)
    ax_cir.set_xlabel("x  (µm)"); ax_cir.set_ylabel("y  (µm)")
    _circular_glyph(ax_cir, W, cen_cir, handedness, fontsize=15 * fs)
    _radius_annotation(ax_cir, cen_cir, r_T, fontsize=15 * fs)
    ax_cir.set_title("Circular polarization")
    if note_circular:
        ax_cir.text(0.5, 0.03, note_circular, transform=ax_cir.transAxes,
                    ha="center", va="bottom", fontsize=13 * fs, color="0.25")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"saved {out_path}")
    if show:
        return fig
    plt.close(fig)
    return None


def plot_polarization_trajectories_stacked(
    traj_linear,
    traj_circular,
    pos_units="m",
    particle_size_um=(10.0, 10.0),
    marker_style="rectangle",
    n_markers=6,
    handedness=+1,
    linear_pol_angle_deg=0.0,
    particle_image="particle.png",
    line_color="#0f54c4",
    marker_color="0.6",
    window_pad=1.1,
    note_linear=None,
    note_circular=None,
    speed_um_s=None,           # optional, shown in the info box
    max_points=4000,
    fig_size=(11.0, 8.5),
    out_path="metavehicle_trajectories_stacked.png",
    dpi=300,
    show=False,
):
    """Stacked layout: wide linear strip on top; circular orbit (square) +
    metavehicle info box on the bottom row. Both trajectory panels share the
    SAME physical scale (markers are to-scale and identical across panels)."""
    if np.isscalar(particle_size_um):
        particle_size_um = (float(particle_size_um), float(particle_size_um))

    xy_lin, th_lin = _decimate(*_to_um(traj_linear, pos_units), max_points)
    xy_cir, th_cir = _decimate(*_to_um(traj_circular, pos_units), max_points)

    cen_cir = xy_cir.mean(axis=0)
    ext_cir = max(np.ptp(xy_cir[:, 0]), np.ptp(xy_cir[:, 1]))
    W = window_pad * ext_cir + 2 * max(particle_size_um)        # square window
    r_T = float(np.mean(np.hypot(xy_cir[:, 0] - cen_cir[0],
                                 xy_cir[:, 1] - cen_cir[1])))
    print(f"orbit radius R_T ~ {r_T:.1f} um")

    fig_w, fig_h = fig_size
    plt.rcParams.update({"font.size": 13})
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    # ---- box rectangles in figure fractions ----
    left, bottom = 0.09, 0.07
    circ_w = 0.40
    circ_h = circ_w * fig_w / fig_h          # square box in inches
    gap_x = 0.04
    info_w = (left + 0.87) - (left + circ_w + gap_x)
    gap_y = 0.10
    lin_w, lin_h = 0.87, 0.20
    lin_bottom = bottom + circ_h + gap_y

    ax_cir = fig.add_axes([left, bottom, circ_w, circ_h])
    ax_info = fig.add_axes([left + circ_w + gap_x, bottom, info_w, circ_h])
    ax_lin = fig.add_axes([left, lin_bottom, lin_w, lin_h])

    # scale (inches per um), fixed by the square circular box -> shared by all
    s = (circ_w * fig_w) / W

    # ---- circular panel ----
    _solid_line(ax_cir, xy_cir[:, 0], xy_cir[:, 1], line_color, lw=2.5)
    idxs = _markers_in_window(xy_cir, cen_cir, W, n_markers)
    if len(idxs):
        ang = _orientations(xy_cir, th_cir, idxs)
        _draw_markers(ax_cir, xy_cir, ang, idxs, particle_size_um,
                      marker_color, marker_style)
    ax_cir.scatter(xy_cir[0, 0], xy_cir[0, 1], s=70, facecolor="white",
                   edgecolor="black", linewidth=1.5, zorder=7)
    ax_cir.set_xlim(cen_cir[0] - W / 2, cen_cir[0] + W / 2)
    ax_cir.set_ylim(cen_cir[1] - W / 2, cen_cir[1] + W / 2)
    _circular_glyph(ax_cir, W, cen_cir, handedness)
    _radius_annotation(ax_cir, cen_cir, r_T)
    ax_cir.set_xlabel("x  (µm)"); ax_cir.set_ylabel("y  (µm)")
    ax_cir.set_title("Circular polarization", fontsize=16, pad=8)
    if note_circular:
        ax_cir.text(0.5, 0.03, note_circular, transform=ax_cir.transAxes,
                    ha="center", va="bottom", fontsize=12, color="0.25")

    # ---- linear strip (same scale; size limits from the box) ----
    xr = (lin_w * fig_w) / s          # x-range at shared scale
    yr = (lin_h * fig_h) / s          # y-range at shared scale
    d = xy_lin[-1] - xy_lin[0]
    u = d / np.hypot(*d) if np.hypot(*d) > 0 else np.array([1.0, 0.0])
    cen_lin = xy_lin[0] + 0.40 * xr * u
    cen_lin[1] = xy_lin[0, 1]

    _solid_line(ax_lin, xy_lin[:, 0], xy_lin[:, 1], line_color, lw=2.5)
    hx, hy = xr / 2, yr / 2
    inside = np.where((np.abs(xy_lin[:, 0] - cen_lin[0]) <= hx) &
                      (np.abs(xy_lin[:, 1] - cen_lin[1]) <= hy))[0]
    if len(inside):
        pick = inside[np.unique(np.linspace(0, len(inside) - 1,
                                            n_markers).astype(int))]
        ang = _orientations(xy_lin, th_lin, pick)
        _draw_markers(ax_lin, xy_lin, ang, pick, particle_size_um,
                      marker_color, marker_style)
    if abs(xy_lin[0, 0] - cen_lin[0]) <= hx:
        ax_lin.scatter(xy_lin[0, 0], xy_lin[0, 1], s=70, facecolor="white",
                       edgecolor="black", linewidth=1.5, zorder=7)
    ax_lin.set_xlim(cen_lin[0] - hx, cen_lin[0] + hx)
    ax_lin.set_ylim(cen_lin[1] - hy, cen_lin[1] + hy)

    # E_pol double arrow, left side of the strip
    gx, gy, Lg = cen_lin[0] - 0.42 * xr, cen_lin[1] + 0.28 * yr, 0.06 * xr
    a = np.radians(linear_pol_angle_deg)
    dd = Lg * np.array([np.cos(a), np.sin(a)])
    ax_lin.add_patch(FancyArrowPatch((gx - dd[0], gy - dd[1]),
                                     (gx + dd[0], gy + dd[1]),
                                     arrowstyle="<|-|>", mutation_scale=14,
                                     color="crimson", lw=2.2, zorder=8))
    ax_lin.text(gx, gy + 0.10 * yr, r"$E_{pol}$", color="crimson",
                ha="center", va="bottom", fontsize=12, zorder=8)
    ax_lin.set_xlabel("x  (µm)"); ax_lin.set_ylabel("y  (µm)")
    ax_lin.set_title("Linear polarization", fontsize=16, pad=8)
    if note_linear:
        ax_lin.text(0.5, 0.05, note_linear, transform=ax_lin.transAxes,
                    ha="center", va="bottom", fontsize=12, color="0.25")

    # ---- info box: metavehicle image + key numbers ----
    ax_info.axis("off")
    cap = f"{particle_size_um[0]:g} × {particle_size_um[1]:g} µm"
    _hero_inset(ax_info, particle_image, bounds=(0.10, 0.46, 0.80, 0.52),
                caption=cap)
    lines = [rf"$R_T \approx {r_T:.0f}\,\mu$m"]
    if speed_um_s is not None:
        lines.append(rf"$v \approx {speed_um_s:.1f}\,\mu$m/s")
    ax_info.text(0.5, 0.32, "\n".join(lines), transform=ax_info.transAxes,
                 ha="center", va="top", fontsize=14)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"saved {out_path}")
    if show:
        return fig
    plt.close(fig)
    return None


# --------------------------------------------------------------------------
# standalone demo with synthetic trajectories
# --------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # linear: straight run along +x with small jitter, orientation fixed
    t = np.linspace(0, 1, 2000)
    x = 200.0 * t + rng.normal(0, 1.0, t.size).cumsum() * 0.05
    y = rng.normal(0, 1.0, t.size).cumsum() * 0.05
    theta = np.full_like(t, 0.0)
    lin = np.column_stack([x * 1e-6, y * 1e-6, theta])   # in metres

    # circular: orbit of radius R with the body pointing tangent to the path
    R = 90.0
    phi = np.linspace(0, 2.2 * 2 * np.pi, 4000)
    xc = R * np.cos(phi)
    yc = R * np.sin(phi)
    thc = phi + np.pi / 2          # tangent direction
    cir = np.column_stack([xc * 1e-6, yc * 1e-6, thc])   # in metres

    plot_polarization_trajectories(
        lin, cir,
        pos_units="m",
        particle_size_um=(12.0, 10.0),
        marker_style="rectangle",
        n_markers=6,
        handedness=+1,
        particle_image="particle.png",
        out_path="metavehicle_trajectories.png",
    )

    plot_polarization_trajectories_stacked(
        lin, cir,
        pos_units="m",
        particle_size_um=(12.0, 10.0),
        n_markers=6,
        handedness=+1,
        particle_image="particle.png",
        note_linear="aligns to E-field → straight path",
        note_circular="continuous turning under CP",
        speed_um_s=2.5,
        out_path="metavehicle_trajectories_stacked.png",
    )

    plot_with_force_curve(
        lin, cir,
        pos_units="m",
        particle_size_um=(10.0, 10.0),
        n_markers=6,
        handedness=+1,
        font_scale=1.2,
        note_linear="self-aligns → straight path",
        note_circular="continuous turning under CP",
        out_path="metavehicle_force_curve.png",
    )
