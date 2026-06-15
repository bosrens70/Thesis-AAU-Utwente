# -*- coding: utf-8 -*-
"""
Geometric Deviation Viewer — Instances vs LER Utility Registry
===============================================================
Refactored to use core/ for shared configuration and data loading.

Usage: python viewers/deviation_viewer.py
  Change the site by editing PLY_FILE in core/config.py.
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
import geopandas as gpd
import numpy as np
import re
import time
import json

from core.config import (
    PLY_FILE, GML_PATH, AREA_REF_GEOJSON, CROP_RADIUS, CROP_MODE, UTILITY_RECT_BUFFER,
    LINE_LAYERS, COMPONENT_LAYERS, COMP_TO_LINE,
    COMPONENT_SPHERE_RADIUS,
    UTILITY_TYPE_LABELS, UTILITY_TYPE_COLORS, UTILITY_TO_LER_MATCH,
    DEVIATION_THRESHOLDS, DEVIATION_COLORS, DEVIATION_CLASS_LABELS,
    FORSYNINGSART_COLOR_HINTS, FORSYNINGSART_TO_LINE,
    forsyningsart_color as _forsyningsart_color,
)
from core.data_loader import (
    init_site, read_ply_with_utility_type, utility_type_from_filename,
    pick_ground_level, instance_base_name,
)
from core.geometry import (
    batch_point_to_segments, batch_point_to_plane_segments,
    discretize_segment,
    deviation_to_color, deviation_to_color_continuous, linear_to_srgb,
    segment_to_cylinder, segment_to_plane,
    segments_in_rect, point_in_rect, clip_segment_to_rect,
)
from core.ledningstrace import get_bredde_width

# ─────────────────────────────────────────────────────────────────────────────
# INITIALISE — load area offset, point cloud, and GML via core/
# ─────────────────────────────────────────────────────────────────────────────
site = init_site(load_instances=True)

# Unpack area info
TX, TY, TZ = site.area.TX, site.area.TY, site.area.TZ
AREA_NUMBER = site.area.area_number
AREA_NAME   = site.area.area_name

# Unpack point cloud data (DEV1 uses pcd_orig / pts_orig naming)
pcd_orig        = site.pc.pcd
pts_orig        = site.pc.pts
original_colors = site.pc.original_colors
cloud_centroid  = site.pc.cloud_centroid
cloud_centroid_full = site.pc.cloud_centroid_full
pc_min          = site.pc.pc_min
pc_max          = site.pc.pc_max

_cx = site.pc.crop_center_local[0]
_cy = site.pc.crop_center_local[1]
_cx_utm = site.pc.crop_center_utm[0]
_cy_utm = site.pc.crop_center_utm[1]
_crop_r2 = CROP_RADIUS ** 2

# Rectangle region (CROP_MODE == "rect"): full-cloud XY AABB grown by the utility
# buffer.  Selection and clipping are XY-only so every utility passing through the
# footprint is rendered regardless of its depth.  pc_min/pc_max are local;
# UTM = local + (TX, TY).
_rect_min_x = pc_min[0] - UTILITY_RECT_BUFFER
_rect_max_x = pc_max[0] + UTILITY_RECT_BUFFER
_rect_min_y = pc_min[1] - UTILITY_RECT_BUFFER
_rect_max_y = pc_max[1] + UTILITY_RECT_BUFFER
_rect_min_x_utm = _rect_min_x + TX
_rect_max_x_utm = _rect_max_x + TX
_rect_min_y_utm = _rect_min_y + TY
_rect_max_y_utm = _rect_max_y + TY

_ply_path = Path(PLY_FILE)

# Ground level — interactive picking (shared from core/)
GROUND_Z = pick_ground_level(site.pc)

# ─────────────────────────────────────────────────────────────────────────────
# LER LOADING + DEVIATION COMPUTATION + GUI
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 3.  Load LER utility line segments from GML
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Loading LER utility segments ---")
_t_ler0 = time.perf_counter()

all_seg_p1 = []
all_seg_p2 = []
all_seg_layer = []
all_seg_active = []       # True = "i drift", False = "permanent ude af drift"
all_seg_half_width = []   # half-width for plane segments (ledningstrace), 0 for cylinders
all_seg_radius = []       # cylinder radius per segment (used to sample the tube surface)
ler_meshes = {}           # layer -> merged TriangleMesh (for visualisation)
_layer_avg_depth_local = {}  # layer_name -> float (average local Z for component depth fallback)
ler_stats = {}            # layer -> (n_feat_active, n_seg_active, n_feat_inactive, n_seg_inactive)


def _in_crop_utm(coords):
    """Conservative check: any part of the polyline within the crop region (UTM).
    First checks whether any vertex is inside the circle.
    Falls back to an AABB overlap test to catch segments that cross the disc
    but have no vertex inside it — the segment clipper makes the final call.
    """
    if CROP_MODE == "rect":
        return segments_in_rect(coords, _rect_min_x_utm, _rect_min_y_utm,
                                _rect_max_x_utm, _rect_max_y_utm)
    dx = coords[:, 0] - _cx_utm
    dy = coords[:, 1] - _cy_utm
    if (dx * dx + dy * dy <= _crop_r2).any():
        return True
    # AABB fallback
    xs, ys = coords[:, 0], coords[:, 1]
    if xs.max() < _cx_utm - CROP_RADIUS:
        return False
    if xs.min() > _cx_utm + CROP_RADIUS:
        return False
    if ys.max() < _cy_utm - CROP_RADIUS:
        return False
    if ys.min() > _cy_utm + CROP_RADIUS:
        return False
    return True


def _to_local(coords_utm, vejl_dybde_mm=None):
    c = coords_utm.copy().astype(float)
    if c.shape[1] == 2:
        c = np.hstack([c, np.zeros((len(c), 1))])
    c[:, 0] -= TX
    c[:, 1] -= TY
    bad = c[:, 2] == -99
    if bad.any():
        ind_m = None
        if vejl_dybde_mm is not None:
            try:
                d = float(vejl_dybde_mm)
                if d > 0:
                    ind_m = d / 1000.0
            except (ValueError, TypeError):
                pass
        good_z = c[~bad, 2]
        feat_mean = float(good_z.mean()) if len(good_z) > 0 else None
        for idx in np.where(bad)[0]:
            if ind_m is not None:
                c[idx, 2] = (GROUND_Z + TZ) - ind_m
            elif feat_mean is not None:
                c[idx, 2] = feat_mean
            else:
                c[idx, 2] = GROUND_Z + TZ
    c[:, 2] -= TZ
    return c


def _clip_segment_to_crop(p1, p2):
    """
    Clip a 3D segment to the crop region in XY.
    Circle: centre (_cx, _cy), radius CROP_RADIUS, or the rectangle in rect mode.
    Returns (clipped_p1, clipped_p2) or None if entirely outside.
    """
    if CROP_MODE == "rect":
        return clip_segment_to_rect(p1, p2, _rect_min_x, _rect_min_y,
                                    _rect_max_x, _rect_max_y)
    x1 = p1[0] - _cx
    y1 = p1[1] - _cy
    x2 = p2[0] - _cx
    y2 = p2[1] - _cy

    dx = x2 - x1
    dy = y2 - y1
    a = dx * dx + dy * dy

    if a < 1e-12:
        if x1 * x1 + y1 * y1 <= _crop_r2:
            return p1, p2
        return None

    b = 2.0 * (x1 * dx + y1 * dy)
    c = x1 * x1 + y1 * y1 - _crop_r2
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None

    sq = np.sqrt(disc)
    t_enter = (-b - sq) / (2.0 * a)
    t_exit = (-b + sq) / (2.0 * a)

    t0 = max(0.0, t_enter)
    t1 = min(1.0, t_exit)
    if t0 > t1:
        return None

    return p1 + t0 * (p2 - p1), p1 + t1 * (p2 - p1)


for layer_name, cfg in list(LINE_LAYERS.items()):
    try:
        gdf = gpd.read_file(GML_PATH, layer=layer_name)
    except Exception as e:
        print(f"  {layer_name}: skip ({e})")
        continue

    default_color = cfg["color"]
    fallback_r = cfg["fallback_radius"]
    is_trace = (layer_name == "Ledningstrace")
    has_driftsstatus = "driftsstatus" in gdf.columns

    # For Ledningstrace: accumulate per-forsyningsart sub-layers
    _trace_sub_cyls = {}   # display_name -> [meshes]
    _trace_sub_stats = {}  # display_name -> [n_feat_act, n_seg_act, n_feat_inact, n_seg_inact]
    n_feat_act, n_seg_act = 0, 0
    n_feat_inact, n_seg_inact = 0, 0
    layer_cyls = []
    _layer_z_vals = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        subs = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]

        # Determine active / inactive
        is_active = True
        if has_driftsstatus:
            ds = str(row.get("driftsstatus", "") or "").strip().lower()
            if "ude af drift" in ds:
                is_active = False

        diam_mm = 0.0
        if "udvendigDiameter" in row.index:
            try:
                diam_mm = float(row["udvendigDiameter"] or 0)
            except (ValueError, TypeError):
                pass
        radius = diam_mm / 2000.0 if diam_mm > 0 else fallback_r

        bredde_m = get_bredde_width(row)
        if is_trace and bredde_m is None:
            bredde_m = 0.25

        # Resolve display name and colour for Ledningstrace via forsyningsart
        if is_trace and "forsyningsart" in row.index:
            fa = str(row.get("forsyningsart", "") or "").strip()
            if fa:
                display_name = f"Ledningstrace ({fa})"
                color = _forsyningsart_color(fa, default_color)
            else:
                display_name = "Ledningstrace"
                color = default_color
        elif is_trace:
            display_name = "Ledningstrace"
            color = default_color
        else:
            display_name = layer_name
            color = default_color

        vejl = row.get("vejledendeDybde", None) if "vejledendeDybde" in row.index else None

        hit = False
        for sub in subs:
            coords_raw = np.array(sub.coords, dtype=float)
            if not _in_crop_utm(coords_raw):
                continue
            coords = _to_local(coords_raw, vejl)
            _layer_z_vals.extend(coords[:, 2].tolist())
            hit = True
            for i in range(len(coords) - 1):
                clipped = _clip_segment_to_crop(coords[i], coords[i + 1])
                if clipped is None:
                    continue
                cp1, cp2 = clipped
                all_seg_p1.append(cp1)
                all_seg_p2.append(cp2)
                all_seg_layer.append(display_name)
                all_seg_active.append(is_active)
                all_seg_half_width.append(bredde_m / 2.0 if bredde_m is not None else 0.0)
                all_seg_radius.append(radius)
                if bredde_m is not None:
                    mesh = segment_to_plane(cp1, cp2, bredde_m, color)
                else:
                    mesh = segment_to_cylinder(cp1, cp2, radius, color)
                if mesh is not None:
                    if is_trace:
                        _trace_sub_cyls.setdefault(display_name, []).append(mesh)
                    else:
                        layer_cyls.append(mesh)
                if is_active:
                    n_seg_act += 1
                else:
                    n_seg_inact += 1
        if hit:
            if is_active:
                n_feat_act += 1
            else:
                n_feat_inact += 1
            if is_trace:
                _trace_sub_stats.setdefault(display_name, [0, 0, 0, 0])
                if is_active:
                    _trace_sub_stats[display_name][0] += 1
                else:
                    _trace_sub_stats[display_name][2] += 1

    if is_trace:
        for dname, cyls in _trace_sub_cyls.items():
            sub_stats = _trace_sub_stats.get(dname, [0, 0, 0, 0])
            sub_stats[1] = len([c for c in cyls])  # total segments
            ler_stats[dname] = tuple(sub_stats)

            m = cyls[0]
            for c in cyls[1:]:
                m += c
            m.compute_vertex_normals()
            ler_meshes[dname] = m

            fa_val = dname.split("(")[-1].rstrip(")").strip() if "(" in dname else ""
            LINE_LAYERS[dname] = {"color": _forsyningsart_color(fa_val, default_color),
                                  "fallback_radius": fallback_r}
            parts = []
            if sub_stats[0] > 0:
                parts.append(f"{sub_stats[0]} active")
            if sub_stats[2] > 0:
                parts.append(f"{sub_stats[2]} inactive")
            print(f"  {dname:<35} {', '.join(parts):>20}  "
                  f"{len(cyls):>5} segments")
    else:
        ler_stats[layer_name] = (n_feat_act, n_seg_act, n_feat_inact, n_seg_inact)
        if layer_cyls:
            m = layer_cyls[0]
            for c in layer_cyls[1:]:
                m += c
            m.compute_vertex_normals()
            ler_meshes[layer_name] = m
        if n_feat_act + n_feat_inact > 0:
            parts = []
            if n_feat_act > 0:
                parts.append(f"{n_feat_act} active")
            if n_feat_inact > 0:
                parts.append(f"{n_feat_inact} inactive")
            print(f"  {layer_name:<35} {', '.join(parts):>20}  "
                  f"{n_seg_act + n_seg_inact:>5} segments")
    if _layer_z_vals:
        _layer_avg_depth_local[layer_name] = float(np.mean(_layer_z_vals))

seg_p1 = np.array(all_seg_p1) if all_seg_p1 else np.empty((0, 3))
seg_p2 = np.array(all_seg_p2) if all_seg_p2 else np.empty((0, 3))
seg_active = np.array(all_seg_active, dtype=bool) if all_seg_active else np.empty(0, dtype=bool)
seg_half_width = np.array(all_seg_half_width, dtype=float) if all_seg_half_width else np.empty(0, dtype=float)
seg_radius = np.array(all_seg_radius, dtype=float) if all_seg_radius else np.empty(0, dtype=float)
n_total_segs = len(seg_p1)
n_active_segs = int(seg_active.sum()) if len(seg_active) else 0
n_inactive_segs = n_total_segs - n_active_segs

_t_ler1 = time.perf_counter()
print(f"\n  Total: {n_total_segs:,} LER segments loaded in {_t_ler1 - _t_ler0:.1f}s"
      f"  ({n_active_segs} active, {n_inactive_segs} inactive)")

if n_total_segs == 0:
    print("[WARNING] No LER segments found -deviations will be infinite.")

# ─────────────────────────────────────────────────────────────────────────────
# 3b.  Load LER utility components (points) within bbox
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Loading LER utility components within bbox ---")
comp_meshes = {}        # layer_name -> merged TriangleMesh
comp_stats = {}         # layer_name -> int count

for comp_layer, comp_cfg in COMPONENT_LAYERS.items():
    try:
        gdf_c = gpd.read_file(GML_PATH, layer=comp_layer)
    except Exception:
        continue

    color = comp_cfg["color"]
    n_comp = 0
    spheres = []

    parent_line = COMP_TO_LINE.get(comp_layer)
    parent_avg_z = _layer_avg_depth_local.get(parent_line) if parent_line else None

    for _, row in gdf_c.iterrows():
        g = row.geometry
        if g is None or g.geom_type not in ("Point", "PointZ"):
            continue
        if CROP_MODE == "rect":
            if not point_in_rect(g.x, g.y, _rect_min_x_utm, _rect_min_y_utm,
                                 _rect_max_x_utm, _rect_max_y_utm):
                continue
        else:
            dx = g.x - _cx_utm
            dy = g.y - _cy_utm
            if dx * dx + dy * dy > _crop_r2:
                continue

        pt = np.array([g.x - TX, g.y - TY, g.z - TZ], dtype=float)

        if CROP_MODE == "rect":
            if not point_in_rect(pt[0], pt[1], _rect_min_x, _rect_min_y,
                                 _rect_max_x, _rect_max_y):
                continue
        elif (pt[0] - _cx) ** 2 + (pt[1] - _cy) ** 2 > _crop_r2:
            continue

        if g.z == -99 or pt[2] <= -98:
            if parent_avg_z is not None:
                pt[2] = parent_avg_z
            else:
                pt[2] = GROUND_Z

        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=COMPONENT_SPHERE_RADIUS, resolution=12)
        sphere.translate(pt)
        sphere.paint_uniform_color(color)
        spheres.append(sphere)
        n_comp += 1

    comp_stats[comp_layer] = n_comp
    if spheres:
        m = spheres[0]
        for s in spheres[1:]:
            m += s
        m.compute_vertex_normals()
        comp_meshes[comp_layer] = m
    if n_comp > 0:
        print(f"  {comp_layer:<35} {n_comp:>4} components")

print(f"\n  Total: {sum(comp_stats.values())} component spheres")


def _get_matching_segment_mask(utility_type, active_only=None):
    """Return a boolean mask over seg_p1/seg_p2 for segments matching this utility type.

    active_only: None = both, True = only active, False = only inactive.
    """
    match = UTILITY_TO_LER_MATCH.get(utility_type)
    if match is None:
        mask = np.ones(len(seg_p1), dtype=bool)
    else:
        layers = match["layers"]
        mask = np.zeros(len(seg_p1), dtype=bool)
        for i, layer_name in enumerate(all_seg_layer):
            if layer_name in layers:
                mask[i] = True
            elif layer_name.startswith("Ledningstrace"):
                fa = layer_name.split("(")[-1].rstrip(")").strip() if "(" in layer_name else ""
                mapped_line = FORSYNINGSART_TO_LINE.get(fa.lower())
                if mapped_line and mapped_line in layers:
                    mask[i] = True

    if active_only is True:
        mask &= seg_active
    elif active_only is False:
        mask &= ~seg_active
    return mask


def _get_matching_ler_names(utility_type):
    """Return set of LER layer display names that match the given utility type."""
    match = UTILITY_TO_LER_MATCH.get(utility_type)
    if match is None:
        return set()
    layers = match["layers"]
    names = set()
    for layer_name in ler_meshes:
        if layer_name in layers:
            names.add(layer_name)
        elif layer_name.startswith("Ledningstrace"):
            fa = layer_name.split("(")[-1].rstrip(")").strip() if "(" in layer_name else ""
            mapped_line = FORSYNINGSART_TO_LINE.get(fa.lower())
            if mapped_line and mapped_line in layers:
                names.add(layer_name)
    return names


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Load instances + compute deviations against LER segments
# ─────────────────────────────────────────────────────────────────────────────
_ply_stem = _ply_path.stem
_inst_base = instance_base_name(_ply_path)

# New convention: permanent <base>_Instances/ directory
_perm_dir = _ply_path.parent / f"{_inst_base}_Instances"
_inst_dir = None
_inst_files = []
_src_label = "none"

if _perm_dir.is_dir():
    _inst_dir = _perm_dir
    # Find the most recent labeled_* subfolder
    _labeled_dirs = sorted(
        [d for d in _perm_dir.iterdir()
         if d.is_dir() and d.name.startswith("labeled_")],
        key=lambda p: p.name, reverse=True,
    )
    for _ld in _labeled_dirs:
        _files = sorted(_ld.glob("*.ply"))
        if _files:
            _inst_files = _files
            _src_label = _ld.name
            break
    # Legacy fallback: labeled/ (no timestamp)
    if not _inst_files:
        _legacy_labeled = _perm_dir / "labeled"
        if _legacy_labeled.is_dir():
            _inst_files = sorted(_legacy_labeled.glob("*.ply"))
            _src_label = "labeled/"
    # Always include top-level PLY files (e.g. water instance 0_instance_0_type_7.ply)
    _top_level_plys = sorted(_perm_dir.glob("*.ply"))
    if _top_level_plys:
        _inst_files = _top_level_plys + _inst_files
    # Fallback: top-level only when no labeled instances exist
    if not _inst_files:
        _src_label = "root"

# Legacy fallback: old-style timestamped directories
if not _inst_files:
    _inst_candidates = sorted(
        set(_ply_path.parent.glob(f"{_inst_base}_instances_*"))
        | set(_ply_path.parent.glob(f"{_ply_stem}_instances_*")),
        key=lambda p: p.name, reverse=True,
    )
    if _inst_candidates:
        _inst_dir = _inst_candidates[0]
        _legacy_labeled = _inst_dir / "labeled"
        if _legacy_labeled.is_dir():
            _inst_files = sorted(_legacy_labeled.glob("*.ply"))
            _src_label = "labeled/"
        else:
            _inst_files = sorted(_inst_dir.glob("*.ply"))
            _src_label = "root"

if _inst_dir is None:
    raise SystemExit(f"[ERROR] No instance directories for {_inst_base}")
print(f"\nInstance directory: {_inst_dir.name}/")
print(f"  {len(_inst_files)} PLY files ({_src_label})")

if not _inst_files:
    raise SystemExit("[ERROR] No instance PLY files found.")

print("\n--- Computing deviations: instances vs LER ---")
class_instances = {}

for inst_path in _inst_files:
    _ti0 = time.perf_counter()
    pts_inst, colors_inst, ut_arr = read_ply_with_utility_type(inst_path)
    if len(pts_inst) == 0:
        continue

    # Determine utility type (majority vote; fallback to filename)
    ut_unique, ut_counts = np.unique(ut_arr, return_counts=True)
    utility_type = int(ut_unique[np.argmax(ut_counts)])
    if utility_type == 0:
        utility_type = utility_type_from_filename(inst_path.name)
    ut_label = UTILITY_TYPE_LABELS.get(utility_type, f"Unknown({utility_type})")

    # Compute distances: all matching segments (active + inactive combined for heatmap)
    seg_mask_all = _get_matching_segment_mask(utility_type)
    seg_mask_act = _get_matching_segment_mask(utility_type, active_only=True)
    seg_mask_inact = _get_matching_segment_mask(utility_type, active_only=False)
    n_act = int(seg_mask_act.sum())
    n_inact = int(seg_mask_inact.sum())
    n_matched = n_act + n_inact
    has_ler = n_matched > 0

    def _make_stats(d):
        return {
            "mean": float(np.mean(d)), "median": float(np.median(d)),
            "std": float(np.std(d)), "p95": float(np.percentile(d, 95)),
            "max": float(np.max(d)), "min": float(np.min(d)),
            "n_pts": len(d),
        }

    _nan_stats = {"mean": np.nan, "median": np.nan, "std": np.nan,
                  "p95": np.nan, "max": np.nan, "min": np.nan, "n_pts": len(pts_inst)}

    # Combined (active + inactive) for heatmap colouring
    if has_ler:
        dists = batch_point_to_plane_segments(
            pts_inst, seg_p1[seg_mask_all], seg_p2[seg_mask_all],
            seg_half_width[seg_mask_all])
        stats = _make_stats(dists)
    else:
        dists = np.full(len(pts_inst), np.nan)
        stats = dict(_nan_stats)

    # Separate stats for active / inactive
    if n_act > 0:
        dists_act = batch_point_to_plane_segments(
            pts_inst, seg_p1[seg_mask_act], seg_p2[seg_mask_act],
            seg_half_width[seg_mask_act])
        stats_act = _make_stats(dists_act)
    else:
        dists_act = None
        stats_act = dict(_nan_stats)

    if n_inact > 0:
        dists_inact = batch_point_to_plane_segments(
            pts_inst, seg_p1[seg_mask_inact], seg_p2[seg_mask_inact],
            seg_half_width[seg_mask_inact])
        stats_inact = _make_stats(dists_inact)
    else:
        dists_inact = None
        stats_inact = dict(_nan_stats)

    # Deviation point clouds (grey if no matching LER): discrete class bins and
    # a continuous gradient over the same distances.
    _grey = np.tile([0.5, 0.5, 0.5], (len(pts_inst), 1))
    pcd_dev = o3d.geometry.PointCloud()
    pcd_dev.points = o3d.utility.Vector3dVector(pts_inst)
    pcd_dev.colors = o3d.utility.Vector3dVector(
        deviation_to_color(dists) if has_ler else _grey)

    pcd_dev_cont = o3d.geometry.PointCloud()
    pcd_dev_cont.points = o3d.utility.Vector3dVector(pts_inst)
    pcd_dev_cont.colors = o3d.utility.Vector3dVector(
        deviation_to_color_continuous(dists) if has_ler else _grey)

    # Original RGB point cloud
    pcd_rgb = o3d.geometry.PointCloud()
    pcd_rgb.points = o3d.utility.Vector3dVector(pts_inst)
    if colors_inst is not None:
        pcd_rgb.colors = o3d.utility.Vector3dVector(colors_inst.astype(float) / 255.0)
    else:
        pcd_rgb.colors = pcd_dev.colors

    # Utility-class colour
    ut_col = UTILITY_TYPE_COLORS.get(utility_type, [0.5, 0.5, 0.5])
    pcd_class = o3d.geometry.PointCloud()
    pcd_class.points = o3d.utility.Vector3dVector(pts_inst)
    pcd_class.colors = o3d.utility.Vector3dVector(np.tile(ut_col, (len(pts_inst), 1)))

    inst_data = {
        "name": inst_path.stem,
        "utility_type": utility_type,
        "label": ut_label,
        "has_ler": has_ler,
        "n_active_segs": n_act,
        "n_inactive_segs": n_inact,
        "pcd_dev": pcd_dev,
        "pcd_dev_cont": pcd_dev_cont,
        "pcd_rgb": pcd_rgb,
        "pcd_class": pcd_class,
        "distances": dists,
        "stats": stats,
        "stats_active": stats_act,
        "stats_inactive": stats_inact,
    }
    class_instances.setdefault(utility_type, []).append(inst_data)

    _ti1 = time.perf_counter()
    if has_ler:
        tag = f"active={n_act} inactive={n_inact}"
        print(f"  {inst_path.stem}: {len(pts_inst):,} pts  "
              f"type={ut_label}  "
              f"LER({tag})  "
              f"mean={stats['mean']*1000:.1f}mm  "
              f"P95={stats['p95']*1000:.1f}mm  "
              f"max={stats['max']*1000:.1f}mm  "
              f"[{_ti1 - _ti0:.2f}s]")
    else:
        print(f"  {inst_path.stem}: {len(pts_inst):,} pts  "
              f"type={ut_label}  "
              f"** No matching LER utility **  [{_ti1 - _ti0:.2f}s]")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Per-class summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  DEVIATION SUMMARY -Instances vs LER (by utility class)")
print("=" * 72)

class_summaries = {}
for ut, instances in sorted(class_instances.items()):
    label = UTILITY_TYPE_LABELS.get(ut, f"Unknown({ut})")
    has_ler = any(inst["has_ler"] for inst in instances)
    total_pts = sum(inst["stats"]["n_pts"] for inst in instances)
    total_act = sum(inst["n_active_segs"] for inst in instances)
    total_inact = sum(inst["n_inactive_segs"] for inst in instances)

    def _agg_stats(key):
        """Aggregate per-instance stats arrays for a given stats key ('stats_active' or 'stats_inactive')."""
        # We need the raw distances, but we only stored summary stats per split.
        # Use the combined distances and filter isn't possible here,
        # so we report from the per-instance stats (weighted is close enough for display).
        vals = [inst[key] for inst in instances
                if not np.isnan(inst[key].get("mean", np.nan))]
        if not vals:
            return None
        # Re-aggregate from per-instance stats (approximate but representative)
        all_n = sum(v["n_pts"] for v in vals)
        w_mean = sum(v["mean"] * v["n_pts"] for v in vals) / all_n if all_n else np.nan
        return {
            "mean": w_mean,
            "p95": max(v["p95"] for v in vals),
            "max": max(v["max"] for v in vals),
        }

    if has_ler:
        matched_dists = np.concatenate([
            inst["distances"] for inst in instances if inst["has_ler"]])
        summary = {
            "label": label, "n_instances": len(instances), "n_points": total_pts,
            "has_ler": True,
            "n_active_segs": total_act, "n_inactive_segs": total_inact,
            "mean": float(np.mean(matched_dists)),
            "median": float(np.median(matched_dists)),
            "std": float(np.std(matched_dists)),
            "p95": float(np.percentile(matched_dists, 95)),
            "max": float(np.max(matched_dists)),
            "active_agg": _agg_stats("stats_active"),
            "inactive_agg": _agg_stats("stats_inactive"),
        }
    else:
        summary = {
            "label": label, "n_instances": len(instances), "n_points": total_pts,
            "has_ler": False,
            "n_active_segs": 0, "n_inactive_segs": 0,
            "mean": np.nan, "median": np.nan, "std": np.nan,
            "p95": np.nan, "max": np.nan,
            "active_agg": None, "inactive_agg": None,
        }
    class_summaries[ut] = summary

    print(f"\n  {label} (type {ut})")
    print(f"    Instances:  {len(instances)}")
    print(f"    Points:     {total_pts:,}")
    print(f"    LER segs:   {total_act} active, {total_inact} inactive")
    if has_ler:
        print(f"    ── Combined (all matching LER) ──")
        print(f"    Mean:       {summary['mean']*1000:>8.2f} mm")
        print(f"    Median:     {summary['median']*1000:>8.2f} mm")
        print(f"    Std dev:    {summary['std']*1000:>8.2f} mm")
        print(f"    P95:        {summary['p95']*1000:>8.2f} mm")
        print(f"    Max:        {summary['max']*1000:>8.2f} mm")
        if summary["active_agg"]:
            a = summary["active_agg"]
            print(f"    ── Active LER only ──")
            print(f"    Mean:       {a['mean']*1000:>8.2f} mm   "
                  f"P95: {a['p95']*1000:.2f} mm   Max: {a['max']*1000:.2f} mm")
        if summary["inactive_agg"]:
            ia = summary["inactive_agg"]
            print(f"    ── Inactive LER only ──")
            print(f"    Mean:       {ia['mean']*1000:>8.2f} mm   "
                  f"P95: {ia['p95']*1000:.2f} mm   Max: {ia['max']*1000:.2f} mm")
    else:
        print(f"    ** No matching LER utility — deviation not computed **")

print("\n" + "=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
# 5b.  Discretized LER deviation point clouds
# ─────────────────────────────────────────────────────────────────────────────
# Each LER segment is sampled into a dense cloud of points approximating the
# utility surface: a tube of the registered radius for pipes, the flat ribbon
# for traces.  Every sample is coloured by its deviation = distance to the
# nearest measured instance point of a matching utility type, giving the
# accuracy-class heatmap resolved over the registered utility surface.
print("\n--- Discretizing LER surfaces + per-sample deviation ---")
from scipy.spatial import cKDTree

_seg_layer_arr = np.array(all_seg_layer)
_NO_DATA_COLOR = [0.5, 0.5, 0.5]
LER_LENGTH_STEP = 0.02    # m — sample spacing along each segment
LER_SURFACE_STEP = 0.02   # m — surface sample spacing (ribbon width / tube ring)

# Instance points each layer is compared against: the union over all utility
# types whose LER match covers a segment in that layer.
_layer_ref_pts = {}
for ut, instances in class_instances.items():
    mask = _get_matching_segment_mask(ut)   # combined active + inactive
    if not mask.any() or not instances:
        continue
    pts_ut = np.concatenate(
        [np.asarray(inst["pcd_dev"].points) for inst in instances])
    if len(pts_ut) == 0:
        continue
    for ln in set(_seg_layer_arr[mask]):
        _layer_ref_pts.setdefault(ln, []).append(pts_ut)

ler_pcd_dev = {}          # layer -> PointCloud, 3D deviation, discrete colours
ler_pcd_dev_cont = {}     # layer -> PointCloud, 3D deviation, continuous colours
ler_pcd_zdev = {}         # layer -> PointCloud, |Z| deviation, discrete colours
ler_pcd_zdev_cont = {}    # layer -> PointCloud, |Z| deviation, continuous colours
ler_pcd_xydev = {}        # layer -> PointCloud, horizontal deviation, discrete colours
ler_pcd_xydev_cont = {}   # layer -> PointCloud, horizontal deviation, continuous colours
_n_samples_total = 0
for ln in ler_meshes:
    seg_ids = np.where(_seg_layer_arr == ln)[0]
    if len(seg_ids) == 0:
        continue
    ref_list = _layer_ref_pts.get(ln)
    ref_pts = np.concatenate(ref_list) if ref_list else None
    tree = cKDTree(ref_pts) if ref_pts is not None else None

    samp_chunks = []
    col_chunks, col_cont_chunks = [], []
    zcol_chunks, zcol_cont_chunks = [], []
    xycol_chunks, xycol_cont_chunks = [], []
    for idx in seg_ids:
        samp = discretize_segment(
            seg_p1[idx], seg_p2[idx], seg_radius[idx], seg_half_width[idx],
            LER_LENGTH_STEP, LER_SURFACE_STEP)
        if tree is not None:
            # 3D-nearest measured point; the Z and XY deviations are the
            # vertical and horizontal components of the displacement to that
            # same neighbour.
            dev, nn = tree.query(samp, workers=-1)
            zdev = np.abs(samp[:, 2] - ref_pts[nn, 2])
            xydev = np.linalg.norm(samp[:, :2] - ref_pts[nn, :2], axis=1)
            cols = deviation_to_color(dev)
            cols_cont = deviation_to_color_continuous(dev)
            zcols = deviation_to_color(zdev)
            zcols_cont = deviation_to_color_continuous(zdev)
            xycols = deviation_to_color(xydev)
            xycols_cont = deviation_to_color_continuous(xydev)
        else:
            cols = np.tile(_NO_DATA_COLOR, (len(samp), 1))
            cols_cont = zcols = zcols_cont = xycols = xycols_cont = cols
        samp_chunks.append(samp)
        col_chunks.append(cols)
        col_cont_chunks.append(cols_cont)
        zcol_chunks.append(zcols)
        zcol_cont_chunks.append(zcols_cont)
        xycol_chunks.append(xycols)
        xycol_cont_chunks.append(xycols_cont)

    samp_pts = np.concatenate(samp_chunks)
    _n_samples_total += len(samp_pts)

    def _make_pc(color_chunks):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(samp_pts)
        pc.colors = o3d.utility.Vector3dVector(np.concatenate(color_chunks))
        return pc

    ler_pcd_dev[ln] = _make_pc(col_chunks)
    ler_pcd_dev_cont[ln] = _make_pc(col_cont_chunks)
    ler_pcd_zdev[ln] = _make_pc(zcol_chunks)
    ler_pcd_zdev_cont[ln] = _make_pc(zcol_cont_chunks)
    ler_pcd_xydev[ln] = _make_pc(xycol_chunks)
    ler_pcd_xydev_cont[ln] = _make_pc(xycol_cont_chunks)

print(f"  {_n_samples_total:,} LER samples across {len(ler_pcd_dev)} layers")

# Normals for original cloud
try:
    pcd_orig.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=12))
    pcd_orig.orient_normals_towards_camera_location(cloud_centroid + np.array([0, 0, 5]))
except Exception:
    pass

_t_load = time.perf_counter()

# ─────────────────────────────────────────────────────────────────────────────
# 6.  GUI
# ─────────────────────────────────────────────────────────────────────────────
ORIG_GEOM = "original_cloud"
CROP_GEOM = "crop_region"
_color_mode = [0]
_MODE_NAMES = [
    "Point cloud XYZ deviation (discrete)",
    "Point cloud XYZ deviation (continuous)",
    "Original RGB",
    "OpenTrench3D utility class",
    "LER XYZ deviation (discrete)",
    "LER XYZ deviation (continuous)",
    "LER Z deviation (discrete)",
    "LER Z deviation (continuous)",
    "LER XY deviation (discrete)",
    "LER XY deviation (continuous)",
]
# Instance point cloud shown per mode. In the LER deviation modes the heatmap
# lives on the LER segments, so the instance points fall back to original RGB.
_MODE_INST_PCD = ["pcd_dev", "pcd_dev_cont", "pcd_rgb", "pcd_class",
                  "pcd_rgb", "pcd_rgb", "pcd_rgb", "pcd_rgb", "pcd_rgb", "pcd_rgb"]
# LER deviation modes: the LER layers become deviation-coloured point clouds.
# Each maps to the precomputed cloud carrying the right metric + colouring.
_LER_MODE_PCD = {
    4: ler_pcd_dev,          # XYZ deviation, discrete accuracy-class colours
    5: ler_pcd_dev_cont,     # XYZ deviation, continuous gradient
    6: ler_pcd_zdev,         # Z deviation, discrete accuracy-class colours
    7: ler_pcd_zdev_cont,    # Z deviation, continuous gradient
    8: ler_pcd_xydev,        # XY deviation, discrete accuracy-class colours
    9: ler_pcd_xydev_cont,   # XY deviation, continuous gradient
}
_LER_DEV_MODES = tuple(_LER_MODE_PCD)
# Modes that show the discrete accuracy-class heatmap legend
_HEATMAP_MODES = (0, 4, 6, 8)
# Modes that show the continuous deviation-gradient legend
_GRADIENT_MODES = (1, 5, 7, 9)

app = gui.Application.instance
app.initialize()

window = app.create_window(
    f"{_ply_path.stem}  |  Deviation: Instances vs LER  |  H for help",
    1460, 840,
)
em = window.theme.font_size

scene_widget = gui.SceneWidget()
scene_widget.scene = rendering.Open3DScene(window.renderer)
scene_widget.scene.set_background([0.10, 0.10, 0.10, 1.0])
try:
    scene_widget.scene.view.set_post_processing(True)
except Exception:
    pass
scene_widget.scene.scene.set_sun_light([0, 0, -1], [1, 1, 1], 75000)
scene_widget.scene.scene.enable_sun_light(True)


def make_pt_mat(size=3.0):
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.point_size = size
    return mat


def make_mesh_mat(alpha=1.0):
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency"
    mat.base_color = [1, 1, 1, float(alpha)]
    return mat


# Add dimmed original cloud
_dim = original_colors * 0.35
pcd_dim = o3d.geometry.PointCloud()
pcd_dim.points = o3d.utility.Vector3dVector(pts_orig)
pcd_dim.colors = o3d.utility.Vector3dVector(_dim)
try:
    pcd_dim.normals = pcd_orig.normals
except Exception:
    pass
scene_widget.scene.add_geometry(ORIG_GEOM, pcd_dim, make_pt_mat(2.0))

# Crop-region wireframe at ground level (same style as base_viewer): the
# AABB + buffer rectangle that bounds utility selection in rect mode, or the
# disc in circle mode.
if CROP_MODE == "rect":
    _crop_corners = [
        (_rect_min_x, _rect_min_y), (_rect_max_x, _rect_min_y),
        (_rect_max_x, _rect_max_y), (_rect_min_x, _rect_max_y),
    ]
    _crop_pts = np.array([[x, y, GROUND_Z] for x, y in _crop_corners])
    _crop_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
else:
    _N_CIRCLE = 72
    _theta = np.linspace(0.0, 2.0 * np.pi, _N_CIRCLE + 1)
    _crop_pts = np.stack([
        _cx + CROP_RADIUS * np.cos(_theta),
        _cy + CROP_RADIUS * np.sin(_theta),
        np.full(_N_CIRCLE + 1, GROUND_Z),
    ], axis=1)
    _crop_lines = [[i, i + 1] for i in range(_N_CIRCLE)]
_crop_ls = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(_crop_pts),
    lines=o3d.utility.Vector2iVector(_crop_lines))
_crop_ls.paint_uniform_color([1.0, 1.0, 0.0])
_crop_mat = rendering.MaterialRecord()
_crop_mat.shader = "unlitLine"
_crop_mat.line_width = 2.0
scene_widget.scene.add_geometry(CROP_GEOM, _crop_ls, _crop_mat)

# Add LER pipe meshes
_ler_visible = {}
for ln, mesh in ler_meshes.items():
    gn = f"ler_{ln}"
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    scene_widget.scene.add_geometry(gn, mesh, make_mesh_mat(0.6))
    _ler_visible[ln] = True

# Add LER component meshes
_comp_visible = {}
for ln, mesh in comp_meshes.items():
    gn = f"comp_{ln}"
    scene_widget.scene.add_geometry(gn, mesh, make_mesh_mat(0.6))
    _comp_visible[ln] = False
    scene_widget.scene.show_geometry(gn, False)

# Add instance geometries
_inst_gnames = []
_class_visible = {}
for ut, instances in class_instances.items():
    _class_visible[ut] = True
    for i, inst in enumerate(instances):
        gn = f"inst_{ut}_{i}"
        _inst_gnames.append((ut, i, gn))
        scene_widget.scene.add_geometry(gn, inst["pcd_dev"], make_pt_mat(4.0))

# Camera
bounds = scene_widget.scene.bounding_box
scene_widget.setup_camera(60, bounds, cloud_centroid.tolist())

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Colour-mode switch
# ─────────────────────────────────────────────────────────────────────────────
def _apply_ler_color_mode(mode):
    """Swap each LER layer between its solid mesh and a discretized deviation
    point cloud. The cloud carries the metric (XYZ or Z) and colouring
    (discrete accuracy classes or continuous gradient) for the active mode."""
    dev_pcds = _LER_MODE_PCD.get(mode)
    for ln in ler_meshes:
        gn = f"ler_{ln}"
        scene_widget.scene.remove_geometry(gn)
        if dev_pcds is not None and ln in dev_pcds:
            scene_widget.scene.add_geometry(gn, dev_pcds[ln], make_pt_mat(6.0))
        else:
            scene_widget.scene.add_geometry(gn, ler_meshes[ln], make_mesh_mat(_ler_opacity[0]))
        scene_widget.scene.show_geometry(gn, _ler_visible.get(ln, True))


def _apply_color_mode(mode):
    _color_mode[0] = mode
    pcd_key = _MODE_INST_PCD[mode]
    for ut, instances in class_instances.items():
        for i, inst in enumerate(instances):
            gn = f"inst_{ut}_{i}"
            pcd = inst[pcd_key]
            scene_widget.scene.remove_geometry(gn)
            scene_widget.scene.add_geometry(gn, pcd, make_pt_mat(4.0))
            scene_widget.scene.show_geometry(gn, _class_visible.get(ut, True))
    _apply_ler_color_mode(mode)
    window.post_redraw()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Right panel
# ─────────────────────────────────────────────────────────────────────────────
PANEL_W = int(22 * em)
panel = gui.Vert(int(0.5 * em), gui.Margins(int(em), int(em), int(em), int(em)))

panel.add_child(gui.Label(f"Original: {len(pts_orig):,} pts"))
total_inst = sum(inst["stats"]["n_pts"] for v in class_instances.values() for inst in v)
n_inst = sum(len(v) for v in class_instances.values())
panel.add_child(gui.Label(f"Instances: {n_inst} ({total_inst:,} pts)"))
panel.add_child(gui.Label(f"LER segments: {n_total_segs:,} ({n_active_segs}a, {n_inactive_segs}i)"))
panel.add_fixed(int(0.5 * em))

# Colour mode
panel.add_child(gui.Label("Colour mode:"))
combo = gui.Combobox()
for n in _MODE_NAMES:
    combo.add_item(n)
combo.selected_index = 0

_heatmap_legend = gui.Vert(0)
_heatmap_legend.add_child(gui.Label("LER Accuracy Class:"))
for i, (col, lbl) in enumerate(zip(DEVIATION_COLORS, DEVIATION_CLASS_LABELS)):
    sr, sg, sb = (linear_to_srgb(c) for c in col)
    row = gui.Horiz(int(0.3 * em))
    sw = gui.Button(" ")
    sw.background_color = gui.Color(sr, sg, sb, 1.0)
    sw.toggleable = False
    sw.vertical_padding_em = 0.0
    sw.horizontal_padding_em = 0.3
    row.add_child(sw)
    row.add_fixed(int(0.4 * em))
    row.add_child(gui.Label(lbl))
    _heatmap_legend.add_child(row)

# Continuous gradient legend: same anchor colours as the accuracy classes, but
# sampled at intermediate ticks to show the smooth interpolation between them.
_gradient_legend = gui.Vert(0)
_gradient_legend.add_child(gui.Label("LER deviation (gradient):"))
_grad_ticks_mm = [0, 250, 500, 750, 1000, 1500, 2000]
_grad_tick_cols = deviation_to_color_continuous(
    np.asarray(_grad_ticks_mm, dtype=float) / 1000.0)
for _mm, _col in zip(_grad_ticks_mm, _grad_tick_cols):
    sr, sg, sb = (linear_to_srgb(c) for c in _col)
    row = gui.Horiz(int(0.3 * em))
    sw = gui.Button(" ")
    sw.background_color = gui.Color(sr, sg, sb, 1.0)
    sw.toggleable = False
    sw.vertical_padding_em = 0.0
    sw.horizontal_padding_em = 0.3
    row.add_child(sw)
    row.add_fixed(int(0.4 * em))
    _lbl = f">= {_mm} mm" if _mm == _grad_ticks_mm[-1] else f"{_mm} mm"
    row.add_child(gui.Label(_lbl))
    _gradient_legend.add_child(row)
_gradient_legend.visible = False


def _on_mode(val, idx):
    _apply_color_mode(idx)
    _heatmap_legend.visible = (idx in _HEATMAP_MODES)
    _gradient_legend.visible = (idx in _GRADIENT_MODES)
    window.set_needs_layout()


combo.set_on_selection_changed(_on_mode)
panel.add_child(combo)
panel.add_child(_heatmap_legend)
panel.add_child(_gradient_legend)
panel.add_fixed(int(0.5 * em))

# Original cloud toggle
orig_cb = gui.Checkbox("Original cloud")
orig_cb.checked = True
orig_cb.set_on_checked(lambda c: (scene_widget.scene.show_geometry(ORIG_GEOM, c), window.post_redraw()))
panel.add_child(orig_cb)

# Crop-region toggle (XY AABB + buffer rectangle in rect mode)
crop_cb = gui.Checkbox("Crop region (XY AABB + buffer)")
crop_cb.checked = True
crop_cb.set_on_checked(lambda c: (scene_widget.scene.show_geometry(CROP_GEOM, c), window.post_redraw()))
panel.add_child(crop_cb)

# ── Utility filter (per-class view) ──
panel.add_fixed(int(0.5 * em))
panel.add_child(gui.Label("Utility filter:"))

# Build filter entries: (label, utility_type or None for "all")
_filter_entries = [("All utilities", None)]
for _fut in sorted(class_summaries.keys()):
    _fs = class_summaries[_fut]
    _ler_names = _get_matching_ler_names(_fut)
    if _ler_names:
        _ler_short = ", ".join(sorted(_ler_names))
        _filter_entries.append((f"{_fs['label']} <-> {_ler_short}", _fut))
    else:
        _filter_entries.append((f"{_fs['label']}  (no LER)", _fut))

_active_filter = [None]   # None = show all

filter_combo = gui.Combobox()
for _flbl, _ in _filter_entries:
    filter_combo.add_item(_flbl)
filter_combo.selected_index = 0


def _apply_utility_filter(filter_ut):
    """Show/hide instances and LER layers to isolate one utility pair."""
    _active_filter[0] = filter_ut
    matching_ler = _get_matching_ler_names(filter_ut) if filter_ut is not None else None

    # Instances: show only the selected utility type (or all)
    for ut, instances in class_instances.items():
        vis = (filter_ut is None or ut == filter_ut)
        _class_visible[ut] = vis
        for i, inst in enumerate(instances):
            gn = f"inst_{ut}_{i}"
            scene_widget.scene.show_geometry(gn, vis)

    # LER layers: show only those matching the selected utility (or all)
    for ln in ler_meshes:
        if filter_ut is None:
            vis = True
        else:
            vis = ln in matching_ler if matching_ler else False
        _ler_visible[ln] = vis
        scene_widget.scene.show_geometry(f"ler_{ln}", vis)

    window.post_redraw()


def _on_filter(val, idx):
    _, filter_ut = _filter_entries[idx]
    _apply_utility_filter(filter_ut)


filter_combo.set_on_selection_changed(_on_filter)
panel.add_child(filter_combo)

# LER opacity slider
panel.add_fixed(int(0.3 * em))
_ler_opacity = [0.6]
ler_row = gui.Horiz(int(0.25 * em))
ler_row.add_child(gui.Label("LER opacity"))
ler_slider = gui.Slider(gui.Slider.DOUBLE)
ler_slider.set_limits(0.0, 1.0)
ler_slider.double_value = 0.6


def _on_ler_opacity(val):
    _ler_opacity[0] = val
    # In "LER deviation" mode the LER layers are point clouds, not meshes, so
    # the mesh-opacity material does not apply to them.
    if _color_mode[0] not in _LER_DEV_MODES:
        for ln in ler_meshes:
            if _ler_visible.get(ln, True):
                scene_widget.scene.modify_geometry_material(f"ler_{ln}", make_mesh_mat(val))
    for ln in comp_meshes:
        if _comp_visible.get(ln, False):
            scene_widget.scene.modify_geometry_material(f"comp_{ln}", make_mesh_mat(val))
    window.post_redraw()


ler_slider.set_on_value_changed(_on_ler_opacity)
ler_row.add_child(ler_slider)
panel.add_child(ler_row)

# LER layer toggles
panel.add_fixed(int(0.3 * em))
panel.add_child(gui.Label("LER layers:"))

for ln in LINE_LAYERS:
    if ln not in ler_meshes:
        continue
    col = LINE_LAYERS[ln]["color"]
    sr, sg, sb = (linear_to_srgb(c) for c in col)
    st = ler_stats.get(ln, (0, 0, 0, 0))
    nf_act = st[0]
    nf_inact = st[2] if len(st) > 2 else 0

    # Build label: e.g. "Gasledning (2a, 1i)"
    parts = []
    if nf_act > 0:
        parts.append(f"{nf_act}a")
    if nf_inact > 0:
        parts.append(f"{nf_inact}i")
    count_str = ", ".join(parts) if parts else "0"

    row = gui.Horiz(int(0.3 * em))
    sw = gui.Button(" ")
    sw.background_color = gui.Color(sr, sg, sb, 1.0)
    sw.toggleable = False
    sw.vertical_padding_em = 0.0
    sw.horizontal_padding_em = 0.3

    def _make_ler_cb(layer):
        def _cb(checked):
            _ler_visible[layer] = checked
            scene_widget.scene.show_geometry(f"ler_{layer}", checked)
            window.post_redraw()
        return _cb

    cb = gui.Checkbox(f"{ln} ({count_str})")
    cb.checked = True
    cb.set_on_checked(_make_ler_cb(ln))
    row.add_child(sw)
    row.add_fixed(int(0.4 * em))
    row.add_child(cb)
    panel.add_child(row)

# LER component toggles
if comp_meshes:
    panel.add_fixed(int(0.3 * em))
    panel.add_child(gui.Label("LER components:"))
    for comp_ln in COMPONENT_LAYERS:
        if comp_ln not in comp_meshes:
            continue
        comp_col = COMPONENT_LAYERS[comp_ln]["color"]
        csr, csg, csb = (linear_to_srgb(c) for c in comp_col)
        n_c = comp_stats.get(comp_ln, 0)

        crow = gui.Horiz(int(0.3 * em))
        csw = gui.Button(" ")
        csw.background_color = gui.Color(csr, csg, csb, 1.0)
        csw.toggleable = False
        csw.vertical_padding_em = 0.0
        csw.horizontal_padding_em = 0.3

        def _make_comp_cb(layer):
            def _cb(checked):
                _comp_visible[layer] = checked
                scene_widget.scene.show_geometry(f"comp_{layer}", checked)
                window.post_redraw()
            return _cb

        ccb = gui.Checkbox(f"{comp_ln} ({n_c})")
        ccb.checked = False
        ccb.set_on_checked(_make_comp_cb(comp_ln))
        crow.add_child(csw)
        crow.add_fixed(int(0.4 * em))
        crow.add_child(ccb)
        panel.add_child(crow)

# Instance class toggles + stats
panel.add_fixed(int(0.8 * em))
panel.add_child(gui.Label("Instance classes:"))
panel.add_fixed(int(0.3 * em))

for ut in sorted(class_summaries.keys()):
    s = class_summaries[ut]
    col = UTILITY_TYPE_COLORS.get(ut, [0.5, 0.5, 0.5])
    sr, sg, sb = (linear_to_srgb(c) for c in col)

    row = gui.Horiz(int(0.3 * em))
    sw = gui.Button(" ")
    sw.background_color = gui.Color(sr, sg, sb, 1.0)
    sw.toggleable = False
    sw.vertical_padding_em = 0.0
    sw.horizontal_padding_em = 0.3

    def _make_cls_cb(u):
        def _cb(checked):
            _class_visible[u] = checked
            for _u, _i, gn in _inst_gnames:
                if _u == u:
                    scene_widget.scene.show_geometry(gn, checked)
            window.post_redraw()
        return _cb

    _match = UTILITY_TO_LER_MATCH.get(ut)
    if _match:
        _ler_layers = _match["layers"]
        if _ler_layers:
            _ler_name = sorted(_ler_layers)[0]
        else:
            _ler_name = "no LER"
    else:
        _ler_name = "no LER"
    _cls_label = (f"{s['label']} -> {_ler_name} ({s['n_instances']})"
                  if _ler_name != "no LER"
                  else f"{s['label']} (no LER) ({s['n_instances']})")
    cb = gui.Checkbox(_cls_label)
    cb.checked = True
    cb.set_on_checked(_make_cls_cb(ut))
    row.add_child(sw)
    row.add_fixed(int(0.4 * em))
    row.add_child(cb)
    panel.add_child(row)

    stat_box = gui.Vert(0, gui.Margins(int(2.5 * em), 0, 0, 0))
    n_a = s.get("n_active_segs", 0)
    n_i = s.get("n_inactive_segs", 0)
    if s.get("has_ler", True):
        stat_box.add_child(gui.Label(
            f"{s['n_points']:,} pts  |  LER: {n_a}a {n_i}i"))
        stat_box.add_child(gui.Label(
            f"mean {s['mean']*1000:.1f}  |  P95 {s['p95']*1000:.1f}  |  max {s['max']*1000:.1f} mm"))
        if s.get("active_agg") and s.get("inactive_agg"):
            a = s["active_agg"]
            ia = s["inactive_agg"]
            stat_box.add_child(gui.Label(
                f"active mean {a['mean']*1000:.0f}  |  inactive mean {ia['mean']*1000:.0f} mm"))
    else:
        stat_box.add_child(gui.Label(f"{s['n_points']:,} pts"))
        stat_box.add_child(gui.Label("No matching LER utility"))
    panel.add_child(stat_box)
    panel.add_fixed(int(0.3 * em))

panel.add_stretch()

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Keys + layout
# ─────────────────────────────────────────────────────────────────────────────
HANDLED = gui.Widget.EventCallbackResult.HANDLED
IGNORED = gui.Widget.EventCallbackResult.IGNORED


def _pivot_to(pt):
    d = max(1.0, np.linalg.norm(pc_max - pc_min) * 0.6)
    eye = pt + np.array([d, -d, d * 0.6])
    scene_widget.look_at(pt.tolist(), eye.tolist(), [0, 0, 1])


def on_key(event):
    if event.type != gui.KeyEvent.DOWN:
        return IGNORED
    k = event.key
    if k in (ord('C'), ord('c')):
        _pivot_to(cloud_centroid)
        return HANDLED
    if k in (ord('H'), ord('h')):
        print("\n  C   pivot to centroid    H   help\n")
        return HANDLED
    return IGNORED


scene_widget.set_on_key(on_key)


def on_layout(ctx):
    r = window.content_rect
    scene_widget.frame = gui.Rect(r.x, r.y, r.width - PANEL_W, r.height)
    panel.frame = gui.Rect(r.x + r.width - PANEL_W, r.y, PANEL_W, r.height)


window.set_on_layout(on_layout)
window.add_child(scene_widget)
window.add_child(panel)

print(f"\nLaunching viewer ...\n")
app.run()
print("Viewer closed.")
