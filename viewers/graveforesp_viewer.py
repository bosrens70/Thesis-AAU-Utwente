# -*- coding: utf-8 -*-
"""
Graveforesp Viewer — All Point Clouds + All Utilities + Toggleable Layers
==========================================================================
Refactored to use core/ for shared configuration.

Note: this viewer does NOT use init_site() because it loads multiple PLYs
via polygon overlap rather than a single PLY_FILE.  Constants are still
imported from core/config.

Usage: python viewers/graveforesp_viewer.py
"""

import sys
from pathlib import Path

# Ensure the project root is on the path so `core` is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
import geopandas as gpd
import numpy as np
import matplotlib.path as mpath
from shapely.geometry import Polygon
from shapely.ops import unary_union
import re
import time
from core.config import (
    GML_PATH, AREA_REF_GEOJSON, PLY_BASE_DIR, CROP_RADIUS,
    CLASS_LABELS, DEFAULT_CLASS_COLOR,
    LINE_LAYERS, COMPONENT_LAYERS, COMP_TO_LINE,
    COMPONENT_SPHERE_RADIUS,
    DepthSource, DepthConfig, PIPE_DEPTH_CONFIG, COMPONENT_DEPTH_CONFIG,
    forsyningsart_color,
)
from core.gui_helpers import make_legend_row
from core.geometry import fit_plane_z, segment_to_plane, srgb_to_linear
from core.ledningstrace import get_ledningstrace_display_info, get_storage_key, get_bredde_width

# Buffer (metres) around the Graveforesp polygon
BUFFER = 2.0

# ── Ground-plane fit settings ────────────────────────────────────────────────
# The ground level is a single best-fit plane through the top surface of all
# point clouds. We sample that top surface by binning points into XY cells and
# taking a high percentile of Z per cell (top of trench = street surface),
# then fit z = a*x + b*y + c to those samples.
GROUND_CELL_M  = 2.0    # XY cell size (m) for top-surface sampling
GROUND_PCTILE  = 95.0   # Z percentile per cell (top surface)
GROUND_TRENCH_CLASS = 2  # class id whose top is the street surface ("Trench")

# Alias for backward compat
_DEFAULT_CLASS_COLOR = DEFAULT_CLASS_COLOR


def _cell_top_samples(pts, cell=GROUND_CELL_M, pctile=GROUND_PCTILE):
    """
    Bin points into XY cells of size `cell` and return one top-surface sample
    per populated cell.

    Returns an (M, 3) array of (cell_center_x, cell_center_y, pctile-Z). Using
    a per-cell percentile (rather than every point) means the plane is fit to
    the upper surface of the cloud, not its full thickness.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return np.empty((0, 3), dtype=float)
    cells = np.floor(pts[:, :2] / cell).astype(np.int64)
    uniq, inv = np.unique(cells, axis=0, return_inverse=True)
    samples = np.empty((len(uniq), 3), dtype=float)
    for k in range(len(uniq)):
        zc = pts[inv == k, 2]
        samples[k, 0] = (uniq[k, 0] + 0.5) * cell
        samples[k, 1] = (uniq[k, 1] + 0.5) * cell
        samples[k, 2] = np.percentile(zc, pctile)
    return samples

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
_required = {
    "AREA_REF_GEOJSON": AREA_REF_GEOJSON,
    "GML_PATH": GML_PATH,
    "PLY_BASE_DIR": PLY_BASE_DIR,
}
_missing = [(n, p) for n, p in _required.items() if not Path(p).exists()]
if _missing:
    print("\n[CONFIG ERROR] Missing paths:")
    for n, p in _missing:
        print(f"  {n:<20} = {p}")
    raise SystemExit(1)
print("Config paths OK.\n")

# ─────────────────────────────────────────────────────────────────────────────
# VIEWER-SPECIFIC CODE — OPTIMIZED DATA LOADING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
_t0 = time.perf_counter()

# Cylinder resolution: lower = fewer triangles (6 vs 12 = half the geometry)
_CYL_RESOLUTION = 6

# Read only every Nth point from each PLY file.
# 10 = keep 1 in 10 points → ~10× faster I/O, ~10× fewer points.
# Set to 1 to read all points (full resolution).
SUBSAMPLE_EVERY = 60


# ─────────────────────────────────────────────────────────────────────────────
# FAST PLY READER — reads only every Nth point, never loads full cloud
# ─────────────────────────────────────────────────────────────────────────────
import io

def _fast_read_ply_subsampled(ply_path, stride=SUBSAMPLE_EVERY):
    """
    Read an ASCII PLY, keeping only every `stride`-th point.
    Never loads the full point cloud into memory.
    Returns (xyz, rgb, class_labels_or_None, n_total_points).
    """
    ply_path = str(ply_path)

    # ── Parse header ──────────────────────────────────────────────────
    header_lines = 0
    n_verts = 0
    prop_names = []
    with open(ply_path, 'r', errors='replace') as f:
        for line in f:
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                n_verts = int(stripped.split()[-1])
            elif stripped.startswith("property "):
                prop_names.append(stripped.split()[-1])
            elif stripped == "end_header":
                break

    x_col = prop_names.index("x")
    y_col = prop_names.index("y")
    z_col = prop_names.index("z")
    has_rgb = all(c in prop_names for c in ("red", "green", "blue"))
    has_class = "class" in prop_names
    r_col = prop_names.index("red") if has_rgb else None
    g_col = prop_names.index("green") if has_rgb else None
    b_col = prop_names.index("blue") if has_rgb else None
    cls_col = prop_names.index("class") if has_class else None

    # ── Read only every stride-th line (never loads full file) ────────
    sampled_lines = []
    with open(ply_path, 'r', errors='replace') as f:
        # Skip header
        for _ in range(header_lines):
            next(f)
        # Read every stride-th data line
        for i, line in enumerate(f):
            if i >= n_verts:
                break
            if i % stride == 0:
                sampled_lines.append(line)

    if not sampled_lines:
        return np.empty((0, 3)), None, None, n_verts

    # ── Parse subsampled lines with numpy (fast C code) ───────────────
    data = np.loadtxt(io.StringIO(''.join(sampled_lines)))
    if data.ndim == 1:
        data = data.reshape(1, -1)

    xyz = data[:, [x_col, y_col, z_col]]
    rgb = data[:, [r_col, g_col, b_col]].astype(np.uint8) if has_rgb else None
    cls = data[:, cls_col].astype(int) if has_class else None

    return xyz, rgb, cls, n_verts

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load Graveforesp polygon
# ─────────────────────────────────────────────────────────────────────────────
_t_step = time.perf_counter()
print("Loading Graveforesp polygon from GML ...")
gdf_grave  = gpd.read_file(GML_PATH, layer="Graveforesp")
grave_geom = gdf_grave.geometry.iloc[0]

# Buffer the polygon by BUFFER metres (in UTM coordinates)
grave_buffered = grave_geom.buffer(BUFFER)

grave_xy_utm   = np.array(grave_geom.exterior.coords)[:, :2]
buf_xy_utm     = np.array(grave_buffered.exterior.coords)[:, :2]

print(f"  Original polygon vertices: {len(grave_xy_utm)}")
print(f"  Buffered polygon vertices: {len(buf_xy_utm)}")

# Determine which area the Graveforesp falls in
ref = gpd.read_file(AREA_REF_GEOJSON)

grave_cx, grave_cy = grave_geom.centroid.x, grave_geom.centroid.y
print(f"  Graveforesp centroid (UTM): {grave_cx:.1f}, {grave_cy:.1f}")

# Find closest area reference point
dists = []
for _, r in ref.iterrows():
    d = np.sqrt((r.geometry.x - grave_cx)**2 + (r.geometry.y - grave_cy)**2)
    dists.append((r["name"], d, r.geometry.x, r.geometry.y, r.geometry.z))
dists.sort(key=lambda x: x[1])

AREA_NAME = dists[0][0]
TX, TY, TZ = dists[0][2], dists[0][3], dists[0][4]
print(f"  Closest area: {AREA_NAME}  (dist={dists[0][1]:.1f} m)")
print(f"  Origin -> TX={TX:.3f}  TY={TY:.3f}  TZ={TZ:.3f}")

# Convert polygons to local coordinates
grave_xy_local = grave_xy_utm - np.array([TX, TY])
buf_xy_local   = buf_xy_utm - np.array([TX, TY])
buf_path       = mpath.Path(buf_xy_local)

# Local buffered bbox for fast screening
gx_min = buf_xy_local[:, 0].min()
gx_max = buf_xy_local[:, 0].max()
gy_min = buf_xy_local[:, 1].min()
gy_max = buf_xy_local[:, 1].max()

# UTM buffered bbox for utility filtering
buf_min_x = buf_xy_utm[:, 0].min()
buf_max_x = buf_xy_utm[:, 0].max()
buf_min_y = buf_xy_utm[:, 1].min()
buf_max_y = buf_xy_utm[:, 1].max()

print(f"  Buffered bbox (local): X[{gx_min:.1f}, {gx_max:.1f}]  Y[{gy_min:.1f}, {gy_max:.1f}]")
print(f"  [timer] Polygon loading: {time.perf_counter() - _t_step:.2f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Find PLY files + SINGLE-PASS load, screen, crop (was two passes)
# ─────────────────────────────────────────────────────────────────────────────
_t_step = time.perf_counter()

area_num = re.search(r"\d+", AREA_NAME).group()
ply_dirs = sorted(Path(PLY_BASE_DIR).glob(f"*Area_{area_num}*"))
if not ply_dirs:
    ply_dirs = sorted(Path(PLY_BASE_DIR).glob(f"*Area{area_num}*"))

all_ply_files = []
for d in ply_dirs:
    if "Finetuning" in d.name:
        continue
    plys = sorted(d.glob("Area_*_Site_*.ply"))
    plys = [p for p in plys if "_utm" not in p.stem.lower()]
    all_ply_files.extend(plys)
    if plys:
        print(f"  Found {len(plys)} PLY files in {d.name}")

if not all_ply_files:
    print("[ERROR] No PLY files found for area. Check PLY_BASE_DIR.")
    raise SystemExit(1)

# ── FAST SUBSAMPLED READ: never loads full cloud into memory ──────────────
print(f"\nSubsampled load (every {SUBSAMPLE_EVERY}th point) + crop for {len(all_ply_files)} sites ...")
all_pcd_filtered = []
all_class_labels = []
site_names       = []   # PLY stem per kept site (parallel to all_pcd_filtered)
total_pts_raw    = 0
total_pts_read   = 0
total_pts_filt   = 0

# Top-surface samples (one (x, y, z) per populated XY cell, across all clouds)
# used to fit a single best-fit ground plane below.
_ground_samples = []

print(f"\n{'Site':>30}  {'Total':>10}  {'Read':>10}  {'In poly':>10}  {'Time':>6}")
print("-" * 76)

for ply_path in all_ply_files:
    _t_ply = time.perf_counter()

    # ── Read only every Nth point (skips Open3D entirely) ─────────────
    xyz, rgb, cls, n_total = _fast_read_ply_subsampled(ply_path, stride=SUBSAMPLE_EVERY)
    total_pts_raw += n_total
    n_read = len(xyz)
    total_pts_read += n_read

    if n_read == 0:
        _dt = time.perf_counter() - _t_ply
        print(f"  {ply_path.stem:>28}  {n_total:>10,}  {n_read:>10,}  {'skip':>10}  {_dt:.2f}s")
        continue

    # ── Fast bbox pre-filter (vectorised) ─────────────────────────────
    bbox_mask = (
        (xyz[:, 0] >= gx_min) & (xyz[:, 0] <= gx_max) &
        (xyz[:, 1] >= gy_min) & (xyz[:, 1] <= gy_max)
    )

    if not bbox_mask.any():
        _dt = time.perf_counter() - _t_ply
        print(f"  {ply_path.stem:>28}  {n_total:>10,}  {n_read:>10,}  {'skip':>10}  {_dt:.2f}s")
        continue

    # ── Polygon crop on bbox-filtered candidates ──────────────────────
    candidates = xyz[bbox_mask]
    poly_mask = buf_path.contains_points(candidates[:, :2])
    final_mask = np.zeros(n_read, dtype=bool)
    final_mask[np.where(bbox_mask)[0][poly_mask]] = True

    n_filt = final_mask.sum()
    if n_filt == 0:
        _dt = time.perf_counter() - _t_ply
        print(f"  {ply_path.stem:>28}  {n_total:>10,}  {n_read:>10,}  {'0':>10}  {_dt:.2f}s")
        continue

    total_pts_filt += n_filt

    # ── Build Open3D PointCloud from subsampled + cropped arrays ──────
    pcd_filt = o3d.geometry.PointCloud()
    pcd_filt.points = o3d.utility.Vector3dVector(xyz[final_mask])
    if rgb is not None:
        pcd_filt.colors = o3d.utility.Vector3dVector(rgb[final_mask].astype(float) / 255.0)
    all_pcd_filtered.append(pcd_filt)
    site_names.append(ply_path.stem)

    # ── Class labels come for free from the subsampled read ───────────
    if cls is not None:
        all_class_labels.append(cls[final_mask])
    else:
        all_class_labels.append(None)

    # ── Collect top-surface samples for the ground-plane fit ─────────
    # Prefer class-2 ("Trench") points — the top of the trench is the
    # street surface. Fall back to all points if the class is absent.
    _site_pts = xyz[final_mask]
    if cls is not None:
        _trench_mask = cls[final_mask] == GROUND_TRENCH_CLASS
        _top_src = _site_pts[_trench_mask] if _trench_mask.sum() > 0 else _site_pts
    else:
        _top_src = _site_pts
    _ground_samples.append(_cell_top_samples(_top_src))

    _dt = time.perf_counter() - _t_ply
    print(f"  {ply_path.stem:>28}  {n_total:>10,}  {n_read:>10,}  {n_filt:>10,}  {_dt:.2f}s")

print("-" * 76)
print(f"  {'Total':>28}  {total_pts_raw:>10,}  {total_pts_read:>10,}  {total_pts_filt:>10,}  "
      f"({len(all_pcd_filtered)} sites)")
print(f"  [timer] PLY subsampled load + crop: {time.perf_counter() - _t_step:.2f}s\n")

if not all_pcd_filtered:
    print("[ERROR] No points fell within the buffered Graveforesp polygon.")
    raise SystemExit(1)

# Merge all filtered point clouds into one
merged_pcd = all_pcd_filtered[0]
for p in all_pcd_filtered[1:]:
    merged_pcd += p

all_pts = np.asarray(merged_pcd.points)
cloud_centroid = all_pts.mean(axis=0)

# Build merged class labels array
original_colors = np.asarray(merged_pcd.colors).copy()
has_any_labels = any(cl is not None for cl in all_class_labels)

if has_any_labels:
    merged_labels = []
    for i, cl in enumerate(all_class_labels):
        n = len(np.asarray(all_pcd_filtered[i].points))
        if cl is not None:
            merged_labels.append(cl)
        else:
            merged_labels.append(np.full(n, -1, dtype=int))
    merged_class_labels = np.concatenate(merged_labels)

    class_colors = np.zeros_like(original_colors)
    for cls_id, cfg in CLASS_LABELS.items():
        mask = merged_class_labels == cls_id
        class_colors[mask] = cfg["color"]
    unknown = ~np.isin(merged_class_labels, list(CLASS_LABELS.keys()))
    class_colors[unknown] = original_colors[unknown]
else:
    merged_class_labels = None
    class_colors = None

print(f"  Merged point cloud: {len(all_pts):,} points")
print(f"  Cloud centroid (local): [{cloud_centroid[0]:.2f}, {cloud_centroid[1]:.2f}, {cloud_centroid[2]:.2f}]")

# Z range of merged point cloud — used to clamp utility depths
PC_Z_MIN = float(all_pts[:, 2].min())
PC_Z_MAX = float(all_pts[:, 2].max())
print(f"  Point cloud Z range: [{PC_Z_MIN:.2f}, {PC_Z_MAX:.2f}]")

# Per-point → site mapping (same order as merged_pcd) + per-site point counts,
# so a clicked point can be traced back to its source PLY (site).
_site_point_counts = [len(np.asarray(p.points)) for p in all_pcd_filtered]
_site_index_per_point = np.concatenate([
    np.full(_site_point_counts[i], i, dtype=np.int32)
    for i in range(len(all_pcd_filtered))
]) if all_pcd_filtered else np.empty(0, dtype=np.int32)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Ground level = locally-adaptive surface following the road
# ─────────────────────────────────────────────────────────────────────────────
# A single plane cannot follow a long road with real elevation change: some
# clouds end up well above/below it. Instead we keep the per-cell top-surface
# samples and look up ground Z by inverse-distance weighting (IDW) over the
# nearest samples, so the ground hugs the local road height. A global best-fit
# plane is kept only as a smooth fallback for query points with no nearby
# samples (and a scalar GROUND_Z as a last resort / label value).

_GROUND_IDW_K     = 8       # nearest top-surface samples to blend
_GROUND_IDW_POWER = 2.0     # IDW exponent (higher = more local)

_ground_samples = [s for s in _ground_samples if len(s)]
_ground_pts = np.vstack(_ground_samples) if _ground_samples else np.empty((0, 3))

# Global plane (fallback + diagnostics)
GROUND_PLANE, _plane_inliers = fit_plane_z(_ground_pts)

# KD-tree over sample XY for fast local lookup
_ground_xy = _ground_pts[:, :2].copy() if len(_ground_pts) else None
_ground_z  = _ground_pts[:, 2].copy() if len(_ground_pts) else None
_ground_tree = None
if _ground_xy is not None and len(_ground_xy) >= 1:
    try:
        from scipy.spatial import cKDTree
        _ground_tree = cKDTree(_ground_xy)
    except Exception as _e:
        print(f"  [warn] scipy KD-tree unavailable ({_e}); using brute-force IDW")

if len(_ground_pts):
    _n_tot = len(_ground_pts)
    GROUND_Z = float(np.median(_ground_pts[:, 2]))
    if GROUND_PLANE is not None:
        _a, _b, _c = GROUND_PLANE
        _pred = _ground_pts[:, 0] * _a + _ground_pts[:, 1] * _b + _c
        _resid = _ground_pts[:, 2] - _pred
        _plane_rms = float(np.sqrt(np.mean(_resid[_plane_inliers] ** 2)))
        _slope_pct = float(np.hypot(_a, _b)) * 100.0
        _z_spread  = float(_ground_pts[:, 2].max() - _ground_pts[:, 2].min())
        print(f"\n  Ground model: local IDW over {_n_tot} top-surface samples")
        print(f"    sample Z spread = {_z_spread:.2f} m  "
              f"(plane slope {_slope_pct:.2f}%, plane RMS {_plane_rms*1000:.0f} mm)")
        _pick_method_tag = (f"local IDW ({_n_tot} samples, "
                            f"ΔZ {_z_spread:.2f} m)")
    else:
        _pick_method_tag = f"local IDW ({_n_tot} samples)"
        print(f"\n  Ground model: local IDW over {_n_tot} top-surface samples")
else:
    GROUND_Z = float(np.percentile(all_pts[:, 2], GROUND_PCTILE))
    _pick_method_tag = "global P95 (no ground samples)"
    print(f"\n  [warn] No ground samples — using global P{GROUND_PCTILE:.0f} "
          f"= {GROUND_Z:.3f} m")


def _ground_z_at(x_local, y_local):
    """
    Local ground Z that follows the road surface, via IDW over the nearest
    top-surface samples. Falls back to the best-fit plane (then scalar
    GROUND_Z) where no samples exist.
    """
    if _ground_z is None or len(_ground_z) == 0:
        if GROUND_PLANE is not None:
            a, b, c = GROUND_PLANE
            return float(a * x_local + b * y_local + c)
        return GROUND_Z

    k = min(_GROUND_IDW_K, len(_ground_z))
    if _ground_tree is not None:
        d, idx = _ground_tree.query([x_local, y_local], k=k)
        d   = np.atleast_1d(d)
        idx = np.atleast_1d(idx)
    else:
        diff = _ground_xy - np.array([x_local, y_local])
        dist = np.sqrt((diff * diff).sum(axis=1))
        idx  = np.argpartition(dist, k - 1)[:k]
        d    = dist[idx]

    # Exact / near-exact hit → return that sample directly
    if np.any(d < 1e-9):
        return float(_ground_z[idx[int(np.argmin(d))]])

    w = 1.0 / (d ** _GROUND_IDW_POWER)
    return float(np.sum(w * _ground_z[idx]) / np.sum(w))

# ── Depth-source colour map (sRGB — used for GUI labels and depth meshes) ────
# Matches base_viewer so the two viewers read identically.
_DSRC_COLOR_SRGB = {
    DepthSource.REGISTERED:   [0.4, 1.0, 0.4],   # green
    DepthSource.VEJLEDENDE:   [0.4, 0.8, 1.0],   # light blue
    DepthSource.FEATURE_MEAN: [1.0, 0.7, 0.3],   # orange
    DepthSource.LAYER_MEAN:   [1.0, 0.7, 0.3],   # orange
    DepthSource.GROUND_PLANE: [1.0, 0.4, 0.4],   # red
    DepthSource.NONE:         [0.5, 0.5, 0.5],   # grey
}


def _dsrc_linear(src):
    """Depth-source colour converted from sRGB to linear for Open3D meshes."""
    s = _DSRC_COLOR_SRGB.get(src, [0.5, 0.5, 0.5])
    return [srgb_to_linear(c) for c in s]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def segment_to_cylinder(p1, p2, radius, color, resolution=_CYL_RESOLUTION):
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return None

    cyl = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=length, resolution=resolution, split=1
    )
    z_axis = np.array([0.0, 0.0, 1.0])
    direction = vec / length
    cross = np.cross(z_axis, direction)
    cross_norm = np.linalg.norm(cross)
    dot = np.dot(z_axis, direction)

    if cross_norm > 1e-6:
        axis = cross / cross_norm
        angle = np.arctan2(cross_norm, dot)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cyl.rotate(R, center=[0.0, 0.0, 0.0])
    elif dot < 0:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([1.0, 0.0, 0.0]) * np.pi
        )
        cyl.rotate(R, center=[0.0, 0.0, 0.0])

    cyl.translate((p1 + p2) / 2.0)
    cyl.paint_uniform_color(color)
    return cyl


def _clean_coords_with_depth(coords_raw, vejledende_dybde_mm,
                              cfg=PIPE_DEPTH_CONFIG, parent_avg_z=None):
    """
    Translate UTM -> local. For vertices with Z = -99 (sentinel), resolve the
    depth using the ordered DepthSource hierarchy defined in *cfg* — exactly
    like base_viewer, but the ground level here is the best-fit plane sampled
    via `_ground_z_at`.

    Returns (coords, sources) where `sources` is a DepthSource int8 array
    (one entry per vertex).
    """
    coords = coords_raw.copy().astype(float)
    if coords.shape[1] == 2:
        coords = np.hstack([coords, np.zeros((len(coords), 1))])

    coords[:, 0] -= TX
    coords[:, 1] -= TY

    n = len(coords)
    sources = np.full(n, DepthSource.NONE, dtype=np.int8)

    # Catch -99 and any near-sentinel values (float imprecision)
    bad = coords[:, 2] <= -98
    sources[~bad] = DepthSource.REGISTERED

    if bad.any():
        ind_depth_m = None
        if vejledende_dybde_mm is not None:
            try:
                d = float(vejledende_dybde_mm)
                if d > 0:
                    ind_depth_m = d / 1000.0
            except (ValueError, TypeError):
                pass

        good_z = coords[~bad, 2]
        feature_mean_z = float(good_z.mean()) if len(good_z) > 0 else None

        # Resolver table: level -> callable(idx) -> float | None (absolute UTM Z)
        def _resolve_vejledende(idx):
            if ind_depth_m is None:
                return None
            g = _ground_z_at(coords[idx, 0], coords[idx, 1])
            return (g + TZ) - ind_depth_m

        def _resolve_feature_mean(idx):
            return feature_mean_z

        def _resolve_layer_mean(idx):
            # parent_avg_z is local; convert to absolute UTM so the final
            # coords[:, 2] -= TZ brings it back to local.
            if parent_avg_z is None:
                return None
            return parent_avg_z + TZ

        def _resolve_ground_plane(idx):
            return _ground_z_at(coords[idx, 0], coords[idx, 1]) + TZ

        _RESOLVERS = {
            DepthSource.VEJLEDENDE:   _resolve_vejledende,
            DepthSource.FEATURE_MEAN: _resolve_feature_mean,
            DepthSource.LAYER_MEAN:   _resolve_layer_mean,
            DepthSource.GROUND_PLANE: _resolve_ground_plane,
        }

        ordered_levels = sorted(
            lv for lv in cfg.enabled_levels if lv != DepthSource.REGISTERED
        )

        for idx in np.where(bad)[0]:
            for level in ordered_levels:
                resolver = _RESOLVERS.get(level)
                if resolver is None:
                    continue
                z = resolver(idx)
                if z is not None:
                    coords[idx, 2] = z
                    sources[idx] = level
                    break

    coords[:, 2] -= TZ

    # Clamp Z to the range of the actual point cloud.
    # This catches any unresolved sentinels, bogus Z=0 values, and
    # wildly wrong depth estimates.  For an overview viewer the utilities
    # should sit within the point cloud's vertical extent.
    coords[:, 2] = np.clip(coords[:, 2], PC_Z_MIN - 2.0, PC_Z_MAX + 2.0)

    return coords, sources


def _segments_in_bbox(coords_utm):
    xs = coords_utm[:, 0]
    ys = coords_utm[:, 1]
    return (xs.max() >= buf_min_x and xs.min() <= buf_max_x and
            ys.max() >= buf_min_y and ys.min() <= buf_max_y)


def _point_in_bbox(x, y):
    return buf_min_x <= x <= buf_max_x and buf_min_y <= y <= buf_max_y


def _pt_in_local_bbox(x, y):
    return (gx_min <= x <= gx_max and gy_min <= y <= gy_max)


def _clip_segment_to_bbox(p1, p2):
    x0, y0 = p1[0], p1[1]
    dx = p2[0] - x0
    dy = p2[1] - y0

    t0, t1 = 0.0, 1.0
    for p_val, q_val in [
        (-dx, (x0 - gx_min)),
        (dx, -(x0 - gx_max)),
        (-dy, (y0 - gy_min)),
        (dy, -(y0 - gy_max)),
    ]:
        if abs(p_val) < 1e-12:
            if q_val < 0:
                return None
        else:
            r = q_val / p_val
            if p_val < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return None

    c1 = p1 + t0 * (p2 - p1)
    c2 = p1 + t1 * (p2 - p1)
    return c1, c2


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Build Graveforesp surface mesh (triangulated polygon)
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding Graveforesp surface mesh ...")

grave_verts_2d = grave_xy_local[:-1]
n_grave = len(grave_verts_2d)

grave_verts_3d = np.zeros((n_grave, 3))
grave_verts_3d[:, 0] = grave_verts_2d[:, 0]
grave_verts_3d[:, 1] = grave_verts_2d[:, 1]
# Follow the local ground surface so the polygon sits on the road height
grave_verts_3d[:, 2] = [_ground_z_at(x, y) for x, y in grave_verts_2d]

grave_triangles = []
for i in range(1, n_grave - 1):
    grave_triangles.append([0, i, i + 1])

grave_mesh = o3d.geometry.TriangleMesh()
grave_mesh.vertices = o3d.utility.Vector3dVector(grave_verts_3d)
grave_mesh.triangles = o3d.utility.Vector3iVector(np.array(grave_triangles, dtype=np.int32))
grave_mesh.paint_uniform_color([0.9, 0.9, 0.2])
grave_mesh.compute_vertex_normals()

print(f"  {n_grave} vertices, {len(grave_triangles)} triangles")

grave_wire_z = np.array([[_ground_z_at(x, y)] for x, y in grave_xy_local])
grave_wire_pts = np.hstack([grave_xy_local, grave_wire_z])
grave_lines = [[i, i + 1] for i in range(len(grave_wire_pts) - 1)]
grave_ls = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(grave_wire_pts),
    lines=o3d.utility.Vector2iVector(grave_lines),
)
grave_ls.paint_uniform_color([1.0, 1.0, 0.0])

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Load ALL GML layers ONCE (was 14 separate reads of the same file)
# ─────────────────────────────────────────────────────────────────────────────
_t_step = time.perf_counter()
print("\n--- Loading all GML layers (single pass) ---")

_cached_gdfs = {}  # layer_name -> GeoDataFrame

for layer_name in list(LINE_LAYERS.keys()) + list(COMPONENT_LAYERS.keys()):
    if layer_name in _cached_gdfs:
        continue
    _tl = time.perf_counter()
    try:
        _cached_gdfs[layer_name] = gpd.read_file(GML_PATH, layer=layer_name)
        print(f"  {layer_name:<35} {len(_cached_gdfs[layer_name]):>5} features  "
              f"[{time.perf_counter() - _tl:.2f}s]")
    except Exception as e:
        print(f"  {layer_name:<35} skip ({e})")

print(f"  [timer] GML loading (all layers): {time.perf_counter() - _t_step:.2f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Build utility line meshes from cached GML data
# ─────────────────────────────────────────────────────────────────────────────
_t_step = time.perf_counter()
print("\n--- Building utility line meshes ---")

layer_meshes = {}
layer_meshes_depth = {}          # layer_name -> depth-source-coloured mesh
_pipe_seg_dsrc = {}              # layer_name -> [DepthSource, ...] per segment
layer_stats = {}
all_pipe_coords = []

pick_seg_midpoints = []
pick_seg_attrs = []
pick_seg_layer = []

_layer_avg_depth_local = {}

def _batch_merge_meshes(mesh_list):
    """Merge meshes in O(n) by collecting all vertices/triangles, then building once."""
    if not mesh_list:
        return None
    if len(mesh_list) == 1:
        mesh_list[0].compute_vertex_normals()
        return mesh_list[0]

    all_verts = []
    all_tris = []
    all_colors = []
    offset = 0
    for m in mesh_list:
        v = np.asarray(m.vertices)
        t = np.asarray(m.triangles)
        c = np.asarray(m.vertex_colors) if m.has_vertex_colors() else None
        all_verts.append(v)
        all_tris.append(t + offset)
        if c is not None and len(c) == len(v):
            all_colors.append(c)
        else:
            # Use the uniform paint color (first vertex of the mesh)
            vc = np.asarray(m.vertex_colors)
            if len(vc) > 0:
                all_colors.append(vc)
            else:
                all_colors.append(np.tile([0.5, 0.5, 0.5], (len(v), 1)))
        offset += len(v)

    combined = o3d.geometry.TriangleMesh()
    combined.vertices = o3d.utility.Vector3dVector(np.vstack(all_verts))
    combined.triangles = o3d.utility.Vector3iVector(np.vstack(all_tris))
    combined.vertex_colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    combined.compute_vertex_normals()
    return combined


def _build_depth_mesh(mesh_list, src_list):
    """
    Merge `mesh_list` like `_batch_merge_meshes`, but paint every mesh with its
    depth-source colour (parallel `src_list`). Produces the alternate
    'Depth Hierarchy' colouring without deep-copying individual meshes.
    """
    if not mesh_list:
        return None
    all_verts, all_tris, all_colors = [], [], []
    offset = 0
    for m, src in zip(mesh_list, src_list):
        v = np.asarray(m.vertices)
        t = np.asarray(m.triangles)
        all_verts.append(v)
        all_tris.append(t + offset)
        all_colors.append(np.tile(_dsrc_linear(src), (len(v), 1)))
        offset += len(v)

    combined = o3d.geometry.TriangleMesh()
    combined.vertices = o3d.utility.Vector3dVector(np.vstack(all_verts))
    combined.triangles = o3d.utility.Vector3iVector(np.vstack(all_tris))
    combined.vertex_colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    combined.compute_vertex_normals()
    return combined

# Track Ledningstrace forsyningsart variants
_ledningstrace_variants = {}

for layer_name, cfg in LINE_LAYERS.items():
    gdf = _cached_gdfs.get(layer_name)
    if gdf is None:
        continue

    default_color = cfg["color"]
    fallback_radius = cfg["fallback_radius"]
    n_features = 0
    n_segments = 0
    _layer_z_vals = []
    layer_mesh_list = []
    layer_src_list  = []   # DepthSource per segment (parallel to layer_mesh_list)

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        if geom.geom_type == "MultiLineString":
            sub_geoms = list(geom.geoms)
        else:
            sub_geoms = [geom]

        diam_mm = 0.0
        if "udvendigDiameter" in row.index:
            try:
                diam_mm = float(row["udvendigDiameter"] or 0)
            except (ValueError, TypeError):
                diam_mm = 0.0

        radius = diam_mm / 2000.0 if diam_mm > 0 else fallback_radius

        # Get Ledningstrace display info and width
        is_trace, display_fa, color = get_ledningstrace_display_info(layer_name, row, default_color)
        if is_trace and display_fa and display_fa not in _ledningstrace_variants:
            _ledningstrace_variants[display_fa] = color

        bredde_m = get_bredde_width(row)
        if is_trace and bredde_m is None:
            bredde_m = 0.25  # fallback: 25 cm

        vejl_dybde = None
        if "vejledendeDybde" in row.index:
            vejl_dybde = row.get("vejledendeDybde", None)

        row_attrs = []
        for col in row.index:
            if col == "geometry":
                continue
            val = row[col]
            val_str = str(val) if (val is not None and str(val) != "nan") else "—"
            row_attrs.append((col, val_str))

        feature_hit = False
        for sub_geom in sub_geoms:
            coords_raw = np.array(sub_geom.coords, dtype=float)
            if not _segments_in_bbox(coords_raw):
                continue

            coords, seg_sources = _clean_coords_with_depth(coords_raw, vejl_dybde)
            all_pipe_coords.append(coords)
            _layer_z_vals.extend(coords[:, 2].tolist())
            feature_hit = True

            for i in range(len(coords) - 1):
                clipped = _clip_segment_to_bbox(coords[i], coords[i + 1])
                if clipped is None:
                    continue
                if bredde_m is not None:
                    cyl = segment_to_plane(clipped[0], clipped[1], bredde_m, color)
                else:
                    cyl = segment_to_cylinder(clipped[0], clipped[1], radius, color)
                if cyl is not None:
                    # Store with compound key for Ledningstrace variants
                    storage_key = get_storage_key(layer_name, display_fa)
                    # Dominant (worst) depth source of the segment's endpoints
                    _seg_src = DepthSource(max(int(seg_sources[i]),
                                               int(seg_sources[i + 1])))
                    layer_mesh_list.append(cyl)
                    layer_src_list.append(_seg_src)
                    midpt = (clipped[0] + clipped[1]) / 2.0
                    pick_seg_midpoints.append(midpt)
                    pick_seg_attrs.append(row_attrs)
                    pick_seg_layer.append(layer_name)
                    n_segments += 1

        if feature_hit:
            n_features += 1

    layer_stats[layer_name] = (n_features, n_segments)
    if _layer_z_vals:
        _layer_avg_depth_local[layer_name] = float(np.mean(_layer_z_vals))

    # Batch merge (O(n) instead of O(n^2))
    combined = _batch_merge_meshes(layer_mesh_list)
    if combined is not None:
        layer_meshes[layer_name] = combined
        _pipe_seg_dsrc[layer_name] = layer_src_list
        _depth_combined = _build_depth_mesh(layer_mesh_list, layer_src_list)
        if _depth_combined is not None:
            layer_meshes_depth[layer_name] = _depth_combined
        print(f"  {layer_name:<35} {n_features:>4} features  {n_segments:>5} segments")

pick_seg_midpoints = np.array(pick_seg_midpoints) if pick_seg_midpoints else np.empty((0, 3))

# Depth hierarchy stats — counted from rendered segments only (matches base_viewer)
_depth_stats = {src: 0 for src in DepthSource if src != DepthSource.NONE}
for _ln, _src_list in _pipe_seg_dsrc.items():
    for _src in _src_list:
        if _src != DepthSource.NONE:
            _depth_stats[_src] = _depth_stats.get(_src, 0) + 1

print(f"\n  Total line segments: {sum(s for _, s in layer_stats.values()):,}")
print(f"  Depth hierarchy stats (rendered pipe segments):")
print(f"    1. Registered Z:     {_depth_stats.get(DepthSource.REGISTERED, 0)}")
print(f"    2. vejledendeDybde:  {_depth_stats.get(DepthSource.VEJLEDENDE, 0)}")
print(f"    3. Feature mean Z:   {_depth_stats.get(DepthSource.FEATURE_MEAN, 0)}")
print(f"    4. Layer mean Z:     {_depth_stats.get(DepthSource.LAYER_MEAN, 0)}")
print(f"    5. Ground plane:     {_depth_stats.get(DepthSource.GROUND_PLANE, 0)}")
print(f"  [timer] Line mesh building: {time.perf_counter() - _t_step:.2f}s")

# Pipe centroid
pipe_centroid = np.array([0.0, 0.0, 0.0])
if all_pipe_coords:
    pipe_centroid = np.vstack(all_pipe_coords).mean(axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Load utility components from cached GML data
# ─────────────────────────────────────────────────────────────────────────────
_t_step = time.perf_counter()
print("\n--- Building utility component meshes ---")

_COMP_TO_LINE = COMP_TO_LINE

comp_meshes = {}
comp_meshes_depth = {}           # layer_name -> depth-source-coloured mesh
_comp_seg_dsrc = {}              # layer_name -> [DepthSource, ...] per component
comp_stats = {}

pick_comp_centres = []
pick_comp_attrs = []
pick_comp_layer = []

for layer_name, cfg in COMPONENT_LAYERS.items():
    gdf_c = _cached_gdfs.get(layer_name)
    if gdf_c is None:
        continue

    color = cfg["color"]
    n_comp = 0
    comp_mesh_list = []
    comp_src_list  = []   # DepthSource per component (parallel to comp_mesh_list)

    parent_line = _COMP_TO_LINE.get(layer_name)
    parent_avg_z = _layer_avg_depth_local.get(parent_line) if parent_line else None

    for _, row in gdf_c.iterrows():
        g = row.geometry
        if g is None:
            continue

        # Extract representative point: centroid for polygons/lines, direct coords for points
        if g.geom_type in ("Point", "PointZ"):
            gx, gy, gz = g.x, g.y, g.z
        else:
            # Polygon, MultiPolygon, LineString, etc. — use centroid
            c = g.centroid
            gx, gy = c.x, c.y
            # Try to get Z from the geometry's representative coord
            if g.has_z:
                # Take the Z of the first coordinate
                try:
                    first_coord = next(iter(g.exterior.coords)) if hasattr(g, 'exterior') else next(iter(g.coords))
                    gz = first_coord[2] if len(first_coord) > 2 else 0.0
                except Exception:
                    gz = 0.0
            else:
                gz = 0.0

        if not _point_in_bbox(gx, gy):
            continue

        pt = np.array([gx - TX, gy - TY, gz - TZ], dtype=float)
        if not _pt_in_local_bbox(pt[0], pt[1]):
            continue

        if gz <= -98 or pt[2] <= -98:
            # Component depth hierarchy: LAYER_MEAN (parent pipe) -> GROUND_PLANE
            if parent_avg_z is not None:
                pt[2] = parent_avg_z
                _comp_src = DepthSource.LAYER_MEAN
            else:
                pt[2] = _ground_z_at(pt[0], pt[1])
                _comp_src = DepthSource.GROUND_PLANE
        else:
            _comp_src = DepthSource.REGISTERED

        # Clamp to point cloud Z range
        pt[2] = np.clip(pt[2], PC_Z_MIN - 2.0, PC_Z_MAX + 2.0)

        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=COMPONENT_SPHERE_RADIUS, resolution=8
        )
        sphere.translate(pt)
        sphere.paint_uniform_color(color)
        comp_mesh_list.append(sphere)
        comp_src_list.append(_comp_src)

        pick_comp_centres.append(pt.copy())
        comp_row_attrs = []
        for col in row.index:
            if col == "geometry":
                continue
            val = row[col]
            val_str = str(val) if (val is not None and str(val) != "nan") else "—"
            comp_row_attrs.append((col, val_str))
        pick_comp_attrs.append(comp_row_attrs)
        pick_comp_layer.append(layer_name)
        n_comp += 1

    comp_stats[layer_name] = n_comp
    combined = _batch_merge_meshes(comp_mesh_list)
    if combined is not None:
        comp_meshes[layer_name] = combined
        _comp_seg_dsrc[layer_name] = comp_src_list
        _depth_combined = _build_depth_mesh(comp_mesh_list, comp_src_list)
        if _depth_combined is not None:
            comp_meshes_depth[layer_name] = _depth_combined
        print(f"  {layer_name:<35} {n_comp:>4} components")

pick_comp_centres = np.array(pick_comp_centres) if pick_comp_centres else np.empty((0, 3))

# Fold component depth sources into the overall depth-hierarchy stats so the
# GUI legend counts reflect everything that gets recoloured.
for _ln, _src_list in _comp_seg_dsrc.items():
    for _src in _src_list:
        if _src != DepthSource.NONE:
            _depth_stats[_src] = _depth_stats.get(_src, 0) + 1
print(f"  [timer] Component mesh building: {time.perf_counter() - _t_step:.2f}s")

_t_load = time.perf_counter()
print(f"\nAll data loaded in {_t_load - _t0:.2f}s")
# ─────────────────────────────────────────────────────────────────────────────
# 9.  Coordinate frame
# ─────────────────────────────────────────────────────────────────────────────
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5, origin=cloud_centroid
)

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Material helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_mesh_material(alpha: float) -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency"
    mat.base_color = [1.0, 1.0, 1.0, float(alpha)]
    return mat


def make_point_material() -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 2.0
    return mat


def make_frame_material() -> rendering.MaterialRecord:
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    return mat


def linear_to_srgb(c: float) -> float:
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Build GUI
# ─────────────────────────────────────────────────────────────────────────────
POINT_CLOUD_GEOM = "point_cloud"
GRAVE_MESH_GEOM  = "grave_surface"
GRAVE_WIRE_GEOM  = "grave_wire"
FRAME_GEOM       = "frame"
HIGHLIGHT_GEOM   = "highlight"

# Geometry names for utility layers
def _line_geom_name(layer):  return f"line_{layer}"
def _comp_geom_name(layer):  return f"comp_{layer}"

# ── Depth Hierarchy state + mesh accessors ───────────────────────────────────
# When active, layers render with their depth-source colouring instead of the
# normal layer colour. All scene re-adds go through these accessors so every
# toggle/opacity path stays consistent.
_depth_hierarchy_active = [False]

def _active_line_mesh(ln):
    if _depth_hierarchy_active[0] and ln in layer_meshes_depth:
        return layer_meshes_depth[ln]
    return layer_meshes[ln]

def _active_comp_mesh(ln):
    if _depth_hierarchy_active[0] and ln in comp_meshes_depth:
        return comp_meshes_depth[ln]
    return comp_meshes[ln]

class_labels_active = [False]

app = gui.Application.instance
app.initialize()

window = app.create_window(
    f"Graveforesp Viewer  |  {len(all_pcd_filtered)} sites  |  "
    f"{total_pts_filt:,} pts  |  press H for help",
    1600, 900,
)
em = window.theme.font_size

scene_widget = gui.SceneWidget()
scene_widget.scene = rendering.Open3DScene(window.renderer)
scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])

# Gentle top-down sun light for subtle depth shading on utility meshes
scene_widget.scene.scene.set_sun_light(
    [0.0, 0.0, -1.0],        # direction: straight down
    [1.0, 1.0, 1.0],         # white colour
    75000,                    # intensity
)
scene_widget.scene.scene.enable_sun_light(True)

# Add point cloud
scene_widget.scene.add_geometry(POINT_CLOUD_GEOM, merged_pcd, make_point_material())

# Add Graveforesp surface
grave_opacity = [0.35]
scene_widget.scene.add_geometry(
    GRAVE_MESH_GEOM, grave_mesh, make_mesh_material(grave_opacity[0])
)

# Add Graveforesp wireframe
line_mat = rendering.MaterialRecord()
line_mat.shader = "unlitLine"
line_mat.line_width = 3.0
scene_widget.scene.add_geometry(GRAVE_WIRE_GEOM, grave_ls, line_mat)

# Add utility line layers
layer_opacity = {}  # layer_name -> [opacity_float]
for layer_name, mesh in layer_meshes.items():
    layer_opacity[layer_name] = [1.0]
    scene_widget.scene.add_geometry(
        _line_geom_name(layer_name), mesh, make_mesh_material(1.0)
    )

# Add utility component layers
for layer_name, mesh in comp_meshes.items():
    if layer_name not in layer_opacity:
        layer_opacity[layer_name] = [1.0]
    scene_widget.scene.add_geometry(
        _comp_geom_name(layer_name), mesh, make_mesh_material(1.0)
    )

# Add frame
scene_widget.scene.add_geometry(FRAME_GEOM, frame, make_frame_material())

bounds = scene_widget.scene.bounding_box
scene_widget.setup_camera(60, bounds, cloud_centroid.tolist())

# ─────────────────────────────────────────────────────────────────────────────
# 11b. Class label toggle
# ─────────────────────────────────────────────────────────────────────────────
def _toggle_class_labels(show_labels: bool):
    if class_colors is None:
        print("[class toggle] No class labels available.")
        return
    class_labels_active[0] = show_labels
    if show_labels:
        merged_pcd.colors = o3d.utility.Vector3dVector(class_colors)
    else:
        merged_pcd.colors = o3d.utility.Vector3dVector(original_colors)
    scene_widget.scene.remove_geometry(POINT_CLOUD_GEOM)
    scene_widget.scene.add_geometry(POINT_CLOUD_GEOM, merged_pcd, make_point_material())
    window.post_redraw()


# ─────────────────────────────────────────────────────────────────────────────
# 12.  Right-side control panel  (matches base_viewer layout)
# ─────────────────────────────────────────────────────────────────────────────
PANEL_WIDTH = int(20 * em)
panel = gui.Vert(int(0.5 * em), gui.Margins(int(em), int(em), int(em), int(em)))

panel.add_child(gui.Label(f"Area: {AREA_NAME}  |  Sites: {len(all_pcd_filtered)}"))
panel.add_child(gui.Label(f"Points: {total_pts_filt:,}  |  Buffer: {BUFFER} m"))
panel.add_child(gui.Label(f"Ground Z: {GROUND_Z:.3f} m ({_pick_method_tag})"))
panel.add_fixed(int(0.3 * em))

# ── Depth Hierarchy toggle — recolours utilities by depth source ────────────
depth_toggle_cb = gui.Checkbox("Depth Hierarchy")
depth_toggle_cb.checked = False

def _dsrc_gui_color(src):
    """sRGB depth-source colour as gui.Color (matches viewer appearance)."""
    r, g, b = _DSRC_COLOR_SRGB[src]
    return gui.Color(r, g, b, 1.0)

_hierarchy_display = [
    ("1. Registered Z",    _depth_stats.get(DepthSource.REGISTERED, 0),   _dsrc_gui_color(DepthSource.REGISTERED)),
    ("2. vejledendeDybde", _depth_stats.get(DepthSource.VEJLEDENDE, 0),   _dsrc_gui_color(DepthSource.VEJLEDENDE)),
    ("3. Feature mean Z",  _depth_stats.get(DepthSource.FEATURE_MEAN, 0), _dsrc_gui_color(DepthSource.FEATURE_MEAN)),
    ("4. Layer mean Z",    _depth_stats.get(DepthSource.LAYER_MEAN, 0),   _dsrc_gui_color(DepthSource.LAYER_MEAN)),
    ("5. Ground plane",    _depth_stats.get(DepthSource.GROUND_PLANE, 0), _dsrc_gui_color(DepthSource.GROUND_PLANE)),
]

_depth_legend_container = gui.Vert(0)
for _label, _count, _color in _hierarchy_display:
    _lbl = gui.Label(f"  {_label}: {_count}")
    _lbl.text_color = _color
    _depth_legend_container.add_child(_lbl)
_depth_legend_container.visible = False


def _on_depth_toggle(checked):
    _depth_hierarchy_active[0] = checked
    _depth_legend_container.visible = checked
    for ln in layer_meshes:
        alpha = pipe_opacity_val[0] if (_ler_active[0] and _layer_visible.get(ln, True)) else 0.0
        scene_widget.scene.remove_geometry(_line_geom_name(ln))
        scene_widget.scene.add_geometry(_line_geom_name(ln), _active_line_mesh(ln),
                                        make_mesh_material(alpha))
    for ln in comp_meshes:
        alpha = pipe_opacity_val[0] if (_ler_active[0] and _layer_visible.get(ln, True)) else 0.0
        scene_widget.scene.remove_geometry(_comp_geom_name(ln))
        scene_widget.scene.add_geometry(_comp_geom_name(ln), _active_comp_mesh(ln),
                                        make_mesh_material(alpha))
    window.set_needs_layout()
    window.post_redraw()


depth_toggle_cb.set_on_checked(_on_depth_toggle)
panel.add_child(depth_toggle_cb)
panel.add_child(_depth_legend_container)
panel.add_fixed(int(0.3 * em))

# ── Show origin axis ──────────────────────────────────────────────────────
origin_frame_visible = [False]
scene_widget.scene.show_geometry(FRAME_GEOM, False)

origin_toggle_cb = gui.Checkbox("Show origin axis")
origin_toggle_cb.checked = False

def _on_origin_toggle(checked):
    origin_frame_visible[0] = checked
    scene_widget.scene.show_geometry(FRAME_GEOM, checked)
    window.post_redraw()

origin_toggle_cb.set_on_checked(_on_origin_toggle)
panel.add_child(origin_toggle_cb)

panel.add_fixed(int(0.8 * em))

# ── Class Label Toggle ────────────────────────────────────────────────────
class_toggle_cb = gui.Checkbox("OpenTrench3D ID Class")
class_toggle_cb.checked = False
if class_colors is None:
    class_toggle_cb.enabled = False

_class_legend_container = gui.Vert(0)
if merged_class_labels is not None:
    for cls_id in sorted(CLASS_LABELS.keys()):
        cfg = CLASS_LABELS[cls_id]
        if cls_id not in np.unique(merged_class_labels):
            continue
        n_pts = int((merged_class_labels == cls_id).sum())
        row = make_legend_row(
            cfg["color"], gui.Label(f"{cls_id}: {cfg['name']} ({n_pts:,})"), em
        )
        _class_legend_container.add_child(row)
_class_legend_container.visible = False


def _on_class_toggle(checked):
    _toggle_class_labels(checked)
    _class_legend_container.visible = checked
    window.set_needs_layout()

class_toggle_cb.set_on_checked(_on_class_toggle)
panel.add_child(class_toggle_cb)
panel.add_child(_class_legend_container)

panel.add_fixed(int(0.8 * em))

# ── Graveforesp Surface ──────────────────────────────────────────────────
grave_visible_cb = gui.Checkbox("Graveforesp surface")
grave_visible_cb.checked = True

def _on_grave_toggle(checked):
    scene_widget.scene.show_geometry(GRAVE_MESH_GEOM, checked)
    window.post_redraw()

grave_visible_cb.set_on_checked(_on_grave_toggle)
panel.add_child(grave_visible_cb)
panel.add_fixed(int(0.2 * em))

grave_opacity_label = gui.Label(f"{grave_opacity[0]:.2f}")
grave_opacity_row = gui.Horiz(int(0.25 * em))
grave_opacity_row.add_child(gui.Label("Opacity"))
grave_opacity_row.add_child(gui.Slider(gui.Slider.DOUBLE))
grave_opacity_row = gui.Horiz(int(0.25 * em))
grave_opacity_row.add_child(gui.Label("Opacity"))

grave_slider = gui.Slider(gui.Slider.DOUBLE)
grave_slider.set_limits(0.0, 1.0)
grave_slider.double_value = grave_opacity[0]


def _apply_grave_opacity(val):
    val = max(0.0, min(1.0, val))
    grave_opacity[0] = val
    grave_slider.double_value = val
    grave_opacity_label.text = f"{val:.2f}"
    scene_widget.scene.remove_geometry(GRAVE_MESH_GEOM)
    scene_widget.scene.add_geometry(GRAVE_MESH_GEOM, grave_mesh, make_mesh_material(val))
    if not grave_visible_cb.checked:
        scene_widget.scene.show_geometry(GRAVE_MESH_GEOM, False)
    window.post_redraw()


grave_slider.set_on_value_changed(lambda v: _apply_grave_opacity(v))

grave_slider_row = gui.Horiz(int(0.25 * em))
grave_slider_row.add_child(gui.Label("Opacity"))
grave_slider_row.add_child(grave_slider)
panel.add_child(grave_slider_row)

panel.add_fixed(int(0.8 * em))

# ── Utility Legend (with per-layer visibility toggles) ────────────────────
_gml_folder = Path(GML_PATH).parent.name
_ler_match = re.match(r"(Ledningspakke)[_\s]*(\d+)", _gml_folder, re.IGNORECASE)
_ler_label = f"{_ler_match.group(1)} {_ler_match.group(2)}" if _ler_match else _gml_folder

_ler_active = [True]
pipe_opacity_val = [1.0]
_layer_visible = {}

ler_toggle_cb = gui.Checkbox(_ler_label)
ler_toggle_cb.checked = True

_ler_legend_container = gui.Vert(int(0.3 * em))

# Global opacity slider
opacity_value_label = gui.Label("1.00")
opacity_slider = gui.Slider(gui.Slider.DOUBLE)
opacity_slider.set_limits(0.0, 1.0)
opacity_slider.double_value = 1.0

slider_row = gui.Horiz(int(0.25 * em))
slider_row.add_child(gui.Label("Opacity"))
slider_row.add_child(opacity_slider)


def _apply_opacity(val: float):
    val = max(0.0, min(1.0, val))
    pipe_opacity_val[0] = val
    opacity_slider.double_value = val
    opacity_value_label.text = f"{val:.2f}"
    for ln in layer_opacity:
        layer_opacity[ln][0] = val
    for ln in layer_meshes:
        alpha = val if _layer_visible.get(ln, True) else 0.0
        mat = make_mesh_material(alpha)
        scene_widget.scene.remove_geometry(_line_geom_name(ln))
        scene_widget.scene.add_geometry(_line_geom_name(ln), _active_line_mesh(ln), mat)
    for ln in comp_meshes:
        alpha = val if _layer_visible.get(ln, True) else 0.0
        mat = make_mesh_material(alpha)
        scene_widget.scene.remove_geometry(_comp_geom_name(ln))
        scene_widget.scene.add_geometry(_comp_geom_name(ln), _active_comp_mesh(ln), mat)
    window.post_redraw()


opacity_slider.set_on_value_changed(lambda val: _apply_opacity(val))
_ler_legend_container.add_child(slider_row)

_pipe_checkboxes = []
_comp_checkboxes = []


def _make_pipe_toggle(ln):
    def _cb(checked):
        _layer_visible[ln] = checked
        alpha = pipe_opacity_val[0] if checked else 0.0
        mat = make_mesh_material(alpha)
        scene_widget.scene.remove_geometry(_line_geom_name(ln))
        scene_widget.scene.add_geometry(_line_geom_name(ln), _active_line_mesh(ln), mat)
        window.post_redraw()
    return _cb


def _make_comp_toggle(ln):
    def _cb(checked):
        _layer_visible[ln] = checked
        alpha = pipe_opacity_val[0] if checked else 0.0
        mat = make_mesh_material(alpha)
        scene_widget.scene.remove_geometry(_comp_geom_name(ln))
        scene_widget.scene.add_geometry(_comp_geom_name(ln), _active_comp_mesh(ln), mat)
        window.post_redraw()
    return _cb


# "Toggle all segments" master checkbox
_all_pipes_cb = gui.Checkbox("All segments")
_all_pipes_cb.checked = True

def _on_toggle_all_pipes(checked):
    for ln, cb in _pipe_checkboxes:
        cb.checked = checked
        _layer_visible[ln] = checked
        alpha = pipe_opacity_val[0] if checked else 0.0
        mat = make_mesh_material(alpha)
        scene_widget.scene.remove_geometry(_line_geom_name(ln))
        scene_widget.scene.add_geometry(_line_geom_name(ln), _active_line_mesh(ln), mat)
    window.post_redraw()

_all_pipes_cb.set_on_checked(_on_toggle_all_pipes)
_ler_legend_container.add_child(_all_pipes_cb)

# Line layers
for layer_name in LINE_LAYERS:
    if layer_name not in layer_meshes:
        continue
    cfg = LINE_LAYERS[layer_name]
    n_feat, n_seg = layer_stats.get(layer_name, (0, 0))

    cb = gui.Checkbox(f"{layer_name} ({n_feat})")
    cb.checked = True
    _layer_visible[layer_name] = True
    cb.set_on_checked(_make_pipe_toggle(layer_name))
    _pipe_checkboxes.append((layer_name, cb))

    _ler_legend_container.add_child(make_legend_row(cfg["color"], cb, em))

# "Toggle all components" master checkbox
_all_comps_cb = gui.Checkbox("All components")
_all_comps_cb.checked = True

def _on_toggle_all_comps(checked):
    for ln, cb in _comp_checkboxes:
        cb.checked = checked
        _layer_visible[ln] = checked
        alpha = pipe_opacity_val[0] if checked else 0.0
        mat = make_mesh_material(alpha)
        scene_widget.scene.remove_geometry(_comp_geom_name(ln))
        scene_widget.scene.add_geometry(_comp_geom_name(ln), _active_comp_mesh(ln), mat)
    window.post_redraw()

_all_comps_cb.set_on_checked(_on_toggle_all_comps)
_ler_legend_container.add_child(_all_comps_cb)

# Component layers
for layer_name, cfg in COMPONENT_LAYERS.items():
    if layer_name not in comp_meshes:
        continue
    n_comp = comp_stats.get(layer_name, 0)

    cb = gui.Checkbox(f"{layer_name} ({n_comp})")
    cb.checked = True
    _layer_visible[layer_name] = True
    cb.set_on_checked(_make_comp_toggle(layer_name))
    _comp_checkboxes.append((layer_name, cb))

    _ler_legend_container.add_child(make_legend_row(cfg["color"], cb, em))


def _on_ler_toggle(checked):
    _ler_active[0] = checked
    _ler_legend_container.visible = checked
    for ln in layer_meshes:
        alpha = pipe_opacity_val[0] if (checked and _layer_visible.get(ln, True)) else 0.0
        mat = make_mesh_material(alpha)
        scene_widget.scene.remove_geometry(_line_geom_name(ln))
        scene_widget.scene.add_geometry(_line_geom_name(ln), _active_line_mesh(ln), mat)
    for ln in comp_meshes:
        alpha = pipe_opacity_val[0] if (checked and _layer_visible.get(ln, True)) else 0.0
        mat = make_mesh_material(alpha)
        scene_widget.scene.remove_geometry(_comp_geom_name(ln))
        scene_widget.scene.add_geometry(_comp_geom_name(ln), _active_comp_mesh(ln), mat)
    window.set_needs_layout()
    window.post_redraw()


ler_toggle_cb.set_on_checked(_on_ler_toggle)
panel.add_child(ler_toggle_cb)
panel.add_child(_ler_legend_container)

panel.add_stretch()

# ── Left-side info panel (shown only when a feature is selected) ──────────
LEFT_PANEL_WIDTH = int(22 * em)
left_panel = gui.Vert(int(0.5 * em), gui.Margins(int(em), int(em), int(em), int(em)))
left_panel.background_color = gui.Color(0.15, 0.15, 0.15, 1.0)

_info_type_lbl = gui.Label("")
_info_type_lbl.text_color = gui.Color(0.85, 0.85, 0.20, 1.0)
left_panel.add_child(_info_type_lbl)
left_panel.add_fixed(int(0.4 * em))

info_scroll = gui.ScrollableVert(int(0.3 * em),
                                  gui.Margins(int(0.5 * em), 0, int(0.5 * em), 0))
left_panel.add_child(info_scroll)

_MAX_ATTRS = 30
_attr_rows = []
for _ in range(_MAX_ATTRS):
    row_h = gui.Horiz(int(0.3 * em))
    k_lbl = gui.Label("")
    v_lbl = gui.Label("")
    k_lbl.text_color = gui.Color(0.65, 0.75, 1.00, 1.0)
    v_lbl.text_color = gui.Color(0.90, 0.90, 0.90, 1.0)
    row_h.add_child(k_lbl)
    row_h.add_fixed(int(0.5 * em))
    row_h.add_child(v_lbl)
    info_scroll.add_child(row_h)
    _attr_rows.append((k_lbl, v_lbl))

left_panel.add_stretch()
_left_panel_visible = [False]


def _show_feature_attrs(feature_type: str, attrs: list):
    """Populate the left-side Selected Feature panel with attribute key-value pairs."""
    _info_type_lbl.text = feature_type
    for i, (k_lbl, v_lbl) in enumerate(_attr_rows):
        if i < len(attrs):
            label, value = attrs[i]
            k_lbl.text = f"{label}:"
            v_lbl.text = value
            k_lbl.visible = True
            v_lbl.visible = True
        else:
            k_lbl.visible = False
            v_lbl.visible = False
    _left_panel_visible[0] = True
    window.set_needs_layout()
    window.post_redraw()


def _hide_left_panel():
    """Hide the left-side info panel when no feature is selected."""
    _left_panel_visible[0] = False
    _info_type_lbl.text = ""
    for k_lbl, v_lbl in _attr_rows:
        k_lbl.visible = False
        v_lbl.visible = False
    window.set_needs_layout()
    window.post_redraw()


def _clear_highlight():
    if scene_widget.scene.has_geometry(HIGHLIGHT_GEOM):
        scene_widget.scene.remove_geometry(HIGHLIGHT_GEOM)


def _place_highlight(centre: np.ndarray):
    _clear_highlight()
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.15, resolution=10)
    marker.translate(centre)
    marker.paint_uniform_color([1.0, 1.0, 0.0])
    marker.compute_vertex_normals()
    marker_mat = rendering.MaterialRecord()
    marker_mat.shader = "defaultUnlit"
    scene_widget.scene.add_geometry(HIGHLIGHT_GEOM, marker, marker_mat)
    window.post_redraw()


# ─────────────────────────────────────────────────────────────────────────────
# 13.  Mouse picking  (Ctrl + Left-Click)
# ─────────────────────────────────────────────────────────────────────────────
PICK_RADIUS_SEG = 2.0
PICK_RADIUS_COMP = 1.0
PICK_RADIUS_PC_PX = 10.0    # screen-space search radius (pixels) for point picking
_last_click = [None]


def _site_attrs_for_index(idx):
    """Build (centre, attrs, label) for the left panel from a merged-point index."""
    pt = all_pts[idx].copy()
    si = int(_site_index_per_point[idx]) if idx < len(_site_index_per_point) else -1
    site = site_names[si] if 0 <= si < len(site_names) else f"site {si}"

    attrs = [
        ("Local X", f"{pt[0]:.3f} m"),
        ("Local Y", f"{pt[1]:.3f} m"),
        ("Local Z", f"{pt[2]:.3f} m"),
        ("UTM X",   f"{pt[0] + TX:.3f}"),
        ("UTM Y",   f"{pt[1] + TY:.3f}"),
        ("UTM Z",   f"{pt[2] + TZ:.3f}"),
    ]
    if merged_class_labels is not None:
        cid = int(merged_class_labels[idx])
        if cid >= 0:
            cname = CLASS_LABELS.get(cid, {}).get("name", "Unknown")
            attrs.append(("Class", f"{cid}: {cname}"))
        else:
            attrs.append(("Class", "—"))
    attrs.append(("Ground Z here", f"{_ground_z_at(pt[0], pt[1]):.3f} m"))
    if 0 <= si < len(_site_point_counts):
        attrs.append(("Site points", f"{_site_point_counts[si]:,}"))

    return pt, attrs, f"{site} (point cloud)"


def _pick_point_cloud_screen(sx, sy):
    """
    Screen-space point pick — independent of the depth buffer. Projects every
    merged point to pixels, then returns the index of the front-most point
    within PICK_RADIUS_PC_PX of the click (sx, sy in scene-frame pixels), or
    None. This works even where the translucent ground surface would otherwise
    intercept a depth-based pick.
    """
    if len(all_pts) == 0:
        return None
    cam = scene_widget.scene.camera
    V = np.asarray(cam.get_view_matrix(), dtype=float)        # world -> eye
    P = np.asarray(cam.get_projection_matrix(), dtype=float)  # eye  -> clip
    W = float(scene_widget.frame.width)
    H = float(scene_widget.frame.height)
    if W <= 0 or H <= 0:
        return None

    homog = np.empty((len(all_pts), 4), dtype=float)
    homog[:, :3] = all_pts
    homog[:, 3] = 1.0
    eye  = homog @ V.T          # (N, 4) camera space
    clip = eye @ P.T            # (N, 4) clip space
    w = clip[:, 3]
    valid = np.abs(w) > 1e-9
    ndc_x = np.where(valid, clip[:, 0] / w, 2.0)
    ndc_y = np.where(valid, clip[:, 1] / w, 2.0)
    ndc_z = np.where(valid, clip[:, 2] / w, 2.0)

    px = (ndc_x * 0.5 + 0.5) * W
    py = (1.0 - (ndc_y * 0.5 + 0.5)) * H

    in_front = valid & (ndc_z >= -1.0) & (ndc_z <= 1.0)
    d2 = (px - sx) ** 2 + (py - sy) ** 2
    d2 = np.where(in_front, d2, np.inf)

    near = d2 <= PICK_RADIUS_PC_PX * PICK_RADIUS_PC_PX
    if not near.any():
        return None
    cand = np.where(near)[0]
    # Front-most candidate: in eye space the camera looks down -Z, so the point
    # nearest the camera has the greatest (least-negative) z.
    return int(cand[int(np.argmax(eye[cand, 2]))])


def _do_pick(depth_image):
    if _last_click[0] is None:
        return
    ex, ey = _last_click[0]
    _last_click[0] = None

    sx = int(ex - scene_widget.frame.x)
    sy = int(ey - scene_widget.frame.y)
    depth_arr = np.asarray(depth_image)
    h, w = depth_arr.shape[:2]
    px = int(np.clip(sx, 0, w - 1))
    py = int(np.clip(sy, 0, h - 1))
    depth = float(depth_arr[py, px])

    centre = attrs = label = None

    # ── Utilities (segments / components) take priority — depth-based ────────
    if depth < 1.0:
        world = scene_widget.scene.camera.unproject(
            ex, ey, depth,
            scene_widget.frame.width,
            scene_widget.frame.height,
        )
        hit = np.array(world[:3], dtype=float)

        # Skip segments whose layer is hidden
        best_seg_d = np.inf
        best_seg_i = -1
        if len(pick_seg_midpoints) > 0:
            dists = np.linalg.norm(pick_seg_midpoints - hit, axis=1)
            for _si, _sl in enumerate(pick_seg_layer):
                if not _layer_visible.get(_sl, True) or not _ler_active[0]:
                    dists[_si] = np.inf
            best_seg_i = int(np.argmin(dists))
            best_seg_d = float(dists[best_seg_i])

        # Skip components whose layer is hidden
        best_comp_d = np.inf
        best_comp_i = -1
        if len(pick_comp_centres) > 0:
            dists = np.linalg.norm(pick_comp_centres - hit, axis=1)
            for _ci, _cl in enumerate(pick_comp_layer):
                if not _layer_visible.get(_cl, True) or not _ler_active[0]:
                    dists[_ci] = np.inf
            best_comp_i = int(np.argmin(dists))
            best_comp_d = float(dists[best_comp_i])

        if best_comp_d < best_seg_d and best_comp_d < PICK_RADIUS_COMP:
            centre = pick_comp_centres[best_comp_i].copy()
            attrs = pick_comp_attrs[best_comp_i]
            label = f"{pick_comp_layer[best_comp_i]} (component)"
        elif best_seg_d < PICK_RADIUS_SEG:
            centre = pick_seg_midpoints[best_seg_i].copy()
            attrs = pick_seg_attrs[best_seg_i]
            label = f"{pick_seg_layer[best_seg_i]} (pipe segment)"

    # ── Fall back to a point-cloud (site) pick in screen space ───────────────
    if centre is None:
        idx = _pick_point_cloud_screen(sx, sy)
        if idx is not None:
            centre, attrs, label = _site_attrs_for_index(idx)

    if centre is None:
        def _clear():
            _clear_highlight()
            _hide_left_panel()
        gui.Application.instance.post_to_main_thread(window, _clear)
        return

    def _update():
        _place_highlight(centre)
        _show_feature_attrs(label, attrs)
        window.set_needs_layout()
        window.post_redraw()
    gui.Application.instance.post_to_main_thread(window, _update)


# Distinguish a click from a drag-to-orbit.
DRAG_THRESHOLD = 8   # pixels — below this the release is treated as a click
_mouse_down_pos = [None]
_mouse_moved    = [False]
_left_was_down  = [False]


def on_mouse(event):
    if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
        if int(event.buttons) & int(gui.MouseButton.LEFT):
            _mouse_down_pos[0] = (event.x, event.y)
            _mouse_moved[0]    = False
            _left_was_down[0]  = True
        return gui.Widget.EventCallbackResult.IGNORED

    if event.type == gui.MouseEvent.Type.MOVE:
        if _left_was_down[0] and _mouse_down_pos[0] is not None:
            dx = event.x - _mouse_down_pos[0][0]
            dy = event.y - _mouse_down_pos[0][1]
            if (dx * dx + dy * dy) > DRAG_THRESHOLD * DRAG_THRESHOLD:
                _mouse_moved[0] = True
        return gui.Widget.EventCallbackResult.IGNORED

    if event.type == gui.MouseEvent.Type.BUTTON_UP:
        if not _left_was_down[0]:
            return gui.Widget.EventCallbackResult.IGNORED
        _left_was_down[0] = False

        if _mouse_moved[0] or _mouse_down_pos[0] is None:
            _mouse_down_pos[0] = None
            return gui.Widget.EventCallbackResult.IGNORED

        # Genuine left-click — fire the pick
        click_pos = _mouse_down_pos[0]
        _mouse_down_pos[0] = None
        _last_click[0] = click_pos
        scene_widget.scene.scene.render_to_depth_image(_do_pick)
        return gui.Widget.EventCallbackResult.HANDLED

    return gui.Widget.EventCallbackResult.IGNORED


scene_widget.set_on_mouse(on_mouse)

# ─────────────────────────────────────────────────────────────────────────────
# 14.  Camera helpers
# ─────────────────────────────────────────────────────────────────────────────
origin_pt = np.array([0.0, 0.0, 0.0])
pc_min = all_pts.min(axis=0)
pc_max = all_pts.max(axis=0)


def _pivot_to(point: np.ndarray):
    d = max(1.0, np.linalg.norm(pc_max - pc_min) * 0.6)
    eye = point + np.array([d, -d, d * 0.6])
    scene_widget.look_at(point.tolist(), eye.tolist(), [0.0, 0.0, 1.0])
    print(f"  Pivot -> [{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}]")


# ─────────────────────────────────────────────────────────────────────────────
# 15.  Key callbacks
# ─────────────────────────────────────────────────────────────────────────────
HANDLED = gui.Widget.EventCallbackResult.HANDLED
IGNORED = gui.Widget.EventCallbackResult.IGNORED


def on_key(event):
    if event.type != gui.KeyEvent.DOWN:
        return IGNORED
    k = event.key

    if k == ord(']'):
        _apply_opacity(min(1.0, pipe_opacity_val[0] + 0.05))
        return HANDLED

    if k == ord('['):
        _apply_opacity(max(0.0, pipe_opacity_val[0] - 0.05))
        return HANDLED

    if k in (ord('L'), ord('l')):
        new_state = not class_labels_active[0]
        class_toggle_cb.checked = new_state
        _toggle_class_labels(new_state)
        return HANDLED

    if k in (ord('D'), ord('d')):
        new_state = not _depth_hierarchy_active[0]
        depth_toggle_cb.checked = new_state
        _on_depth_toggle(new_state)
        return HANDLED

    if k in (ord('C'), ord('c')):
        _pivot_to(cloud_centroid)
        return HANDLED
    if k in (ord('P'), ord('p')):
        _pivot_to(pipe_centroid)
        return HANDLED
    if k == ord('0'):
        _pivot_to(origin_pt)
        return HANDLED

    if k in (ord('H'), ord('h')):
        print("\n-- Shortcuts ---------------------------------------------------")
        print("  Left-click     pick pipe / component / point-cloud site (show info)")
        print("  C              pivot to point cloud centroid")
        print("  P              pivot to pipe centroid (all utilities)")
        print("  0              pivot to world origin (0, 0, 0)")
        print("  ]              increase all utility opacities +0.05")
        print("  [              decrease all utility opacities -0.05")
        print("  L              toggle class label colours on/off")
        print("  D              toggle depth-hierarchy colouring on/off")
        print("  H              show this help")
        print("----------------------------------------------------------------\n")
        return HANDLED

    return IGNORED


scene_widget.set_on_key(on_key)

# ─────────────────────────────────────────────────────────────────────────────
# 16.  Layout + run
# ─────────────────────────────────────────────────────────────────────────────
def on_layout(layout_ctx):
    r = window.content_rect
    if _left_panel_visible[0]:
        left_w = LEFT_PANEL_WIDTH
        left_panel.frame = gui.Rect(r.x, r.y, left_w, r.height)
    else:
        left_w = 0
        left_panel.frame = gui.Rect(-LEFT_PANEL_WIDTH, r.y, 0, r.height)
    scene_widget.frame = gui.Rect(r.x + left_w, r.y, r.width - PANEL_WIDTH - left_w, r.height)
    panel.frame = gui.Rect(r.x + r.width - PANEL_WIDTH, r.y, PANEL_WIDTH, r.height)


window.set_on_layout(on_layout)
window.add_child(left_panel)
window.add_child(scene_widget)
window.add_child(panel)

n_total_segs = sum(s for _, s in layer_stats.values())
n_total_comps = sum(comp_stats.values())
print(f"\nRendering {total_pts_filt:,} points  +  {n_total_segs:,} pipe segments  "
      f"+  {n_total_comps} components")
print("Launching viewer ...\n")

app.run()
print("Viewer closed.")
