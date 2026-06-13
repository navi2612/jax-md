"""
Compute area-weighted centroids of a metavehicle GDS, per layer, and
(optionally) sweep the torque reference point to see its effect on the
predicted active-swimmer turning radius.

Requires: gdstk  (pip install gdstk)
"""

import numpy as np

try:
    import gdstk
except ImportError as exc:
    raise SystemExit("This helper needs gdstk:  pip install gdstk") from exc


HOST_LAYER = 1
ANTENNA_LAYER = 0


def _polygon_area_centroid(points: np.ndarray):
    """Signed area and centroid of one polygon via the shoelace formula."""
    x, y = points[:, 0], points[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    A = 0.5 * cross.sum()
    if abs(A) < 1e-30:
        cx, cy = points.mean(axis=0)
        return 0.0, cx, cy
    cx = ((x + x1) * cross).sum() / (6.0 * A)
    cy = ((y + y1) * cross).sum() / (6.0 * A)
    return A, cx, cy


def _layer_centroid(polys):
    """Area-weighted centroid over a list of gdstk polygons (uses |area|)."""
    tot_A = 0.0
    sx = sy = 0.0
    for p in polys:
        A, cx, cy = _polygon_area_centroid(np.asarray(p.points, dtype=float))
        A = abs(A)
        tot_A += A
        sx += A * cx
        sy += A * cy
    if tot_A == 0.0:
        raise ValueError("No polygon area found on this layer.")
    return tot_A, sx / tot_A, sy / tot_A


def read_polygons(gds_path: str):
    """Return {layer: [gdstk.Polygon, ...]} for the top-level cell."""
    lib = gdstk.read_gds(gds_path)
    top = lib.top_level()
    if not top:
        raise ValueError(f"No top-level cell found in {gds_path}")
    cell = top[0]
    by_layer: dict[int, list] = {}
    for poly in cell.polygons:
        by_layer.setdefault(poly.layer, []).append(poly)
    return by_layer


def gds_centroids(gds_path: str,
                  host_layer: int = HOST_LAYER,
                  antenna_layer: int = ANTENNA_LAYER):
    """Compute candidate centers (all in GDS/um coordinates).

    host    : centroid of the host footprint  (best drag-center proxy)
    antenna : centroid of the antennas        (where the asymmetry is)
    full    : area-weighted centroid of host+antennas (material centroid)
    """
    polys = read_polygons(gds_path)
    if host_layer not in polys:
        raise ValueError(f"Host layer {host_layer} not in GDS layers {sorted(polys)}")

    A_host, hx, hy = _layer_centroid(polys[host_layer])
    result = {
        "host": (hx, hy),
        "host_area_um2": A_host,
        "layers_present": sorted(polys),
    }

    if antenna_layer in polys:
        A_ant, ax, ay = _layer_centroid(polys[antenna_layer])
        A_all = A_host + A_ant
        fx = (A_host * hx + A_ant * ax) / A_all
        fy = (A_host * hy + A_ant * ay) / A_all
        result.update({
            "antenna": (ax, ay),
            "antenna_area_um2": A_ant,
            "full": (fx, fy),
            "full_area_um2": A_all,
        })
    else:
        result.update({"antenna": None, "full": (hx, hy)})

    return result


def turning_radius_um(F_xy_N, tau_z0_Nm, center_um,
                      gamma_T=4.7e-7, gamma_R=8.0e-18):
    """Predicted turning radius (um) if the torque is referenced to center_um.

    F_xy_N    : in-plane optical force (Fx, Fy) in N, about the box center (0,0)
    tau_z0_Nm : z-torque in N.m, about the box center (0,0)
    center_um : (cx, cy) candidate reference point in um
    """
    Fx, Fy = F_xy_N
    cx, cy = np.asarray(center_um, dtype=float) * 1e-6
    tau_z = tau_z0_Nm - (cx * Fy - cy * Fx)      # transport torque to new center
    v = np.hypot(Fx, Fy) / gamma_T               # m/s, center-independent
    omega = tau_z / gamma_R                       # rad/s
    if omega == 0.0:
        return np.inf
    return abs(v / omega) * 1e6


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gds", help="path to the metavehicle GDS file")
    ap.add_argument("--host-layer", type=int, default=HOST_LAYER)
    ap.add_argument("--antenna-layer", type=int, default=ANTENNA_LAYER)
    ap.add_argument("--Fx", type=float, default=-0.002068171305543345e-12)
    ap.add_argument("--Fy", type=float, default=1.1471194924393142e-12)
    ap.add_argument("--Tauz", type=float, default=71.72438171368314e-21)
    ap.add_argument("--gamma-T", type=float, default=4.7e-7)
    ap.add_argument("--gamma-R", type=float, default=8.0e-18)
    args = ap.parse_args()

    c = gds_centroids(args.gds, args.host_layer, args.antenna_layer)
    F = (args.Fx, args.Fy)

    print(f"\nGDS layers present: {c['layers_present']}")
    print(f"host footprint   : ({c['host'][0]:+.4f}, {c['host'][1]:+.4f}) um"
          f"   area={c['host_area_um2']:.2f} um^2   <-- drag-center proxy")
    if c.get("antenna"):
        print(f"antennas         : ({c['antenna'][0]:+.4f}, {c['antenna'][1]:+.4f}) um"
              f"   area={c['antenna_area_um2']:.2f} um^2")
        print(f"full (host+ant)  : ({c['full'][0]:+.4f}, {c['full'][1]:+.4f}) um"
              f"   area={c['full_area_um2']:.2f} um^2")

    print("\nturning radius for each candidate torque center:")
    for label in ("host", "antenna", "full"):
        ctr = c.get(label)
        if ctr is None:
            continue
        R = turning_radius_um(F, args.Tauz, ctr, args.gamma_T, args.gamma_R)
        print(f"  {label:8s} center=({ctr[0]:+.3f},{ctr[1]:+.3f}) um  ->  R = {R:8.1f} um")

    print("\n1-D sweep of torque center along x (cy=0):")
    for cx in np.linspace(-3.0, 3.0, 13):
        R = turning_radius_um(F, args.Tauz, (cx, 0.0), args.gamma_T, args.gamma_R)
        print(f"  cx = {cx:+.2f} um   ->  R = {R:8.1f} um")
