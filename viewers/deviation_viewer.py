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
from shapely.geometry import LineString as ShapelyLine, Point as ShapelyPoint, box as shapely_box
from shapely.ops import unary_union

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
    load_or_pick_ground_level, load_or_pick_trench, trench_path_from_vertices,
    instance_base_name,
    feature_accuracy_tolerance, accuracy_class_coverage,
)
from core.geometry import (
    batch_point_to_segments, batch_point_to_plane_segments,
    batch_point_to_plane_segment_components,
    discretize_segment,
    deviation_to_color, deviation_to_color_continuous, linear_to_srgb,
    segment_to_cylinder, segment_to_plane,
    segments_in_rect, point_in_rect, clip_segment_to_rect,
    accuracy_buffer_polygon, polygon_to_o3d_mesh, polygon_to_o3d_lineset,
    merge_linesets, drape_z_from_polylines,
)
from core.ledningstrace import get_bredde_width
from core.rendering import (
    point_material_shaded, point_material_flat, mesh_material, line_material,
    setup_scene_lighting,
)

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

# Ground level: cached per site (delete <site>_ground.json to re-pick).
GROUND_Z = load_or_pick_ground_level(site.pc, _ply_path)

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

# ─────────────────────────────────────────────────────────────────────────────
# 3c.  Registered accuracy buffers (noejagtighedsklasse, 2D)
# ─────────────────────────────────────────────────────────────────────────────
# For every LER line/component feature that registers an accuracy class, draw a
# flat 2D buffer around its centerline (a circle around components) whose
# half-width equals the registered horizontal tolerance, coloured by class. The
# attribute is checked per feature, so a buffer is built only where this dataset
# actually records the class. Each feature's buffer sits at its resolved depth.
print("\n--- Building registered accuracy buffers (noejagtighedsklasse, 2D) ---")

# Crop region as a shapely polygon in local coords (clips every buffer to view).
if CROP_MODE == "rect":
    _acc_clip = shapely_box(_rect_min_x, _rect_min_y, _rect_max_x, _rect_max_y)
else:
    _acc_clip = ShapelyPoint(_cx, _cy).buffer(CROP_RADIUS)

accbuf_fill = {}     # layer -> merged TriangleMesh (translucent fill)
accbuf_outline = {}  # layer -> merged LineSet (outline)
accbuf_stats = {}    # layer -> (n_registered_in_view, n_in_view)
_acc_cov_rows = []   # (layer, has_column, n_registered_total, n_total)


def _store_accbuf(layer_name, fills, outlines, n_reg_view, n_in_view):
    if fills:
        m = fills[0]
        for f in fills[1:]:
            m += f
        m.compute_vertex_normals()
        accbuf_fill[layer_name] = m
    if outlines:
        ml = merge_linesets(outlines)
        if ml is not None:
            accbuf_outline[layer_name] = ml
    accbuf_stats[layer_name] = (n_reg_view, n_in_view)


# ── Line layers ──────────────────────────────────────────────────────────────
for layer_name, cfg in list(LINE_LAYERS.items()):
    # Skip the synthetic per-forsyningsart Ledningstrace sub-layers added above;
    # the real "Ledningstrace" layer (no parenthesis) is still processed.
    if layer_name.startswith("Ledningstrace ("):
        continue
    try:
        gdf = gpd.read_file(GML_PATH, layer=layer_name)
    except Exception:
        continue

    has_col, n_reg_total, n_total = accuracy_class_coverage(gdf)
    _acc_cov_rows.append((layer_name, has_col, n_reg_total, n_total))
    if not has_col:
        continue

    is_trace = (layer_name == "Ledningstrace")
    # Group buffers by display name so the utility filter can isolate them. For
    # Ledningstrace this splits per forsyningsart, keyed identically to the LER
    # meshes (e.g. "Ledningstrace (Vand)"); other layers form a single group.
    grp_fills, grp_outlines = {}, {}     # display_name -> [meshes] / [linesets]
    grp_in_view, grp_reg_view = {}, {}
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        subs = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        vejl = row.get("vejledendeDybde", None) if "vejledendeDybde" in row.index else None

        if is_trace and "forsyningsart" in row.index:
            fa = str(row.get("forsyningsart", "") or "").strip()
            display_name = f"Ledningstrace ({fa})" if fa else "Ledningstrace"
        elif is_trace:
            display_name = "Ledningstrace"
        else:
            display_name = layer_name

        local_lines = []      # XY arrays for buffering
        local_lines_xyz = []  # XYZ arrays for draping the buffer onto the depth
        for sub in subs:
            cr = np.array(sub.coords, dtype=float)
            if not _in_crop_utm(cr):
                continue
            cl = _to_local(cr, vejl)
            local_lines.append(cl[:, :2])
            local_lines_xyz.append(cl)
        if not local_lines:
            continue
        grp_in_view[display_name] = grp_in_view.get(display_name, 0) + 1

        tol = feature_accuracy_tolerance(row)
        if tol is None:
            continue                      # in view but accuracy class not registered
        half_width, cls_idx = tol
        color = DEVIATION_COLORS[cls_idx - 1]
        # Drape the flat buffer onto the utility's depth profile: each buffer
        # vertex takes the Z of the nearest point on the registered centerline.
        _lines_xyz = local_lines_xyz
        z = lambda xy, _l=_lines_xyz: drape_z_from_polylines(xy, _l)

        polys = []
        for ln_xy in local_lines:
            g = ShapelyPoint(ln_xy[0]) if len(ln_xy) < 2 else ShapelyLine(ln_xy)
            poly = accuracy_buffer_polygon(g, half_width, _acc_clip)
            if poly is not None and not poly.is_empty:
                polys.append(poly)
        if not polys:
            continue
        merged = unary_union(polys)
        fm = polygon_to_o3d_mesh(merged, z, color)
        om = polygon_to_o3d_lineset(merged, z, color)
        if fm is not None:
            grp_fills.setdefault(display_name, []).append(fm)
        if om is not None:
            grp_outlines.setdefault(display_name, []).append(om)
        grp_reg_view[display_name] = grp_reg_view.get(display_name, 0) + 1

    for dname in set(grp_fills) | set(grp_outlines) | set(grp_in_view):
        _store_accbuf(dname, grp_fills.get(dname, []), grp_outlines.get(dname, []),
                      grp_reg_view.get(dname, 0), grp_in_view.get(dname, 0))

# Accuracy buffers are built for line layers only; components are excluded.

print("\n  Registered accuracy class (noejagtighedsklasse) coverage:")
for _ln, _has, _nreg, _ntot in _acc_cov_rows:
    status = f"{_nreg}/{_ntot} registered" if _has else "no column"
    print(f"    {_ln:<32} {status}")
_n_acc_view = sum(v[0] for v in accbuf_stats.values())
print(f"  Buffers built for {_n_acc_view} features within the view")


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


def _get_matching_accbuf_keys(utility_type):
    """Return the accuracy-buffer layer keys that match the given utility type.

    Covers line layers, their components (via COMP_TO_LINE) and Ledningstrace
    sub-layers whose forsyningsart maps to a matching line, so the utility filter
    shows only the selected utility's registered-accuracy buffers."""
    match = UTILITY_TO_LER_MATCH.get(utility_type)
    if match is None:
        return set()
    line_layers = match["layers"]
    comp_layers = {c for c, pl in COMP_TO_LINE.items() if pl in line_layers}
    keys = set()
    for ln in set(accbuf_fill) | set(accbuf_outline):
        if ln in line_layers or ln in comp_layers:
            keys.add(ln)
        elif ln.startswith("Ledningstrace"):
            fa = ln.split("(")[-1].rstrip(")").strip() if "(" in ln else ""
            mapped_line = FORSYNINGSART_TO_LINE.get(fa.lower())
            if mapped_line and mapped_line in line_layers:
                keys.add(ln)
    return keys


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

    # Combined (active + inactive) for heatmap colouring. The XY and Z
    # components are taken at the same nearest segment, so XYZ^2 = XY^2 + Z^2.
    if has_ler:
        dists, xy_dists, z_dists = batch_point_to_plane_segment_components(
            pts_inst, seg_p1[seg_mask_all], seg_p2[seg_mask_all],
            seg_half_width[seg_mask_all])
        stats = _make_stats(dists)
    else:
        dists = np.full(len(pts_inst), np.nan)
        xy_dists = np.full(len(pts_inst), np.nan)
        z_dists = np.full(len(pts_inst), np.nan)
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

    def _dev_pcd(values, continuous):
        """Instance cloud coloured by a deviation metric (grey if no LER)."""
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts_inst)
        if has_ler:
            fn = deviation_to_color_continuous if continuous else deviation_to_color
            pc.colors = o3d.utility.Vector3dVector(fn(values))
        else:
            pc.colors = o3d.utility.Vector3dVector(_grey)
        return pc

    pcd_dev_xy = _dev_pcd(xy_dists, False)
    pcd_dev_xy_cont = _dev_pcd(xy_dists, True)
    pcd_dev_z = _dev_pcd(z_dists, False)
    pcd_dev_z_cont = _dev_pcd(z_dists, True)

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
        "pcd_dev_xy": pcd_dev_xy,
        "pcd_dev_xy_cont": pcd_dev_xy_cont,
        "pcd_dev_z": pcd_dev_z,
        "pcd_dev_z_cont": pcd_dev_z_cont,
        "pcd_rgb": pcd_rgb,
        "pcd_class": pcd_class,
        "distances": dists,
        "dists_active": dists_act,
        "dists_inactive": dists_inact,
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

# ───── Trench footprint: restricts deviation colouring + statistics ─────
# The user marks the trench outline by picking points (Shift+Click) on the
# cloud at startup; the footprint is the XY polygon through those points
# (convex hull by default, pick order optionally). Only points whose XY falls
# inside the polygon are coloured by the deviation modes and counted in the
# per-class statistics; everything outside is greyed. The polygon is cached in
# a JSON next to the site PLY so it survives restarts. With no trench defined
# the whole cloud is coloured / measured as before.
TRENCH_POLYGON_MODE = "hull"          # "hull" (convex) or "order" (pick order)
# Resolve the trench via the shared cache (load <site>_trench.json, else pick).
_trench_verts, _trench_mode = load_or_pick_trench(
    site.pc, _ply_path, mode=TRENCH_POLYGON_MODE)
_trench_path_obj = trench_path_from_vertices(_trench_verts, _trench_mode)


def _inside_mask(points_xyz):
    """Boolean mask of points whose XY lies inside the trench, or None when no
    trench is defined (meaning no restriction)."""
    if _trench_path_obj is None:
        return None
    xy = np.asarray(points_xyz)[:, :2]
    return _trench_path_obj.contains_points(xy)


# Inside-trench mask per instance (keyed (ut, i)); absent key => unrestricted.
_inst_inside = {}
for _ut, _insts in class_instances.items():
    for _i, _inst in enumerate(_insts):
        _m = _inside_mask(np.asarray(_inst["pcd_dev"].points))
        if _m is not None:
            _inst_inside[(_ut, _i)] = _m


def _build_class_summaries():
    """Per-class deviation statistics over inside-trench points (or all points
    when no trench is defined). Reads the current _inst_inside masks so it can
    be recomputed if the trench changes."""
    summaries = {}
    for ut, instances in sorted(class_instances.items()):
        label = UTILITY_TYPE_LABELS.get(ut, f"Unknown({ut})")
        has_ler = any(inst["has_ler"] for inst in instances)
        total_act = sum(inst["n_active_segs"] for inst in instances)
        total_inact = sum(inst["n_inactive_segs"] for inst in instances)

        def _masked(idx, arr):
            if arr is None:
                return None
            m = _inst_inside.get((ut, idx))
            return arr if m is None else arr[m]

        total_pts = 0
        for idx, inst in enumerate(instances):
            m = _inst_inside.get((ut, idx))
            total_pts += int(inst["stats"]["n_pts"] if m is None else int(m.sum()))

        def _pool(arr_key, only_ler=False):
            parts = []
            for idx, inst in enumerate(instances):
                if only_ler and not inst["has_ler"]:
                    continue
                a = _masked(idx, inst.get(arr_key))
                if a is not None and len(a):
                    parts.append(a)
            return np.concatenate(parts) if parts else np.array([])

        def _agg(arr_key):
            alld = _pool(arr_key)
            if alld.size == 0:
                return None
            return {"mean": float(np.mean(alld)),
                    "p95": float(np.percentile(alld, 95)),
                    "max": float(np.max(alld))}

        base = {
            "label": label, "n_instances": len(instances), "n_points": total_pts,
            "has_ler": has_ler,
            "n_active_segs": total_act if has_ler else 0,
            "n_inactive_segs": total_inact if has_ler else 0,
        }
        matched = _pool("distances", only_ler=True) if has_ler else np.array([])
        if matched.size:
            base.update({
                "mean": float(np.mean(matched)),
                "median": float(np.median(matched)),
                "std": float(np.std(matched)),
                "p95": float(np.percentile(matched, 95)),
                "max": float(np.max(matched)),
                "active_agg": _agg("dists_active"),
                "inactive_agg": _agg("dists_inactive"),
            })
        else:
            base.update({
                "mean": np.nan, "median": np.nan, "std": np.nan,
                "p95": np.nan, "max": np.nan,
                "active_agg": None, "inactive_agg": None,
            })
        summaries[ut] = base
    return summaries


class_summaries = _build_class_summaries()

for ut in sorted(class_summaries.keys()):
    s = class_summaries[ut]
    print(f"\n  {s['label']} (type {ut})")
    print(f"    Instances:  {s['n_instances']}")
    print(f"    Points (in trench): {s['n_points']:,}")
    print(f"    LER segs:   {s['n_active_segs']} active, {s['n_inactive_segs']} inactive")
    if s["has_ler"] and not np.isnan(s["mean"]):
        print(f"    -- Combined (all matching LER) --")
        print(f"    Mean:       {s['mean']*1000:>8.2f} mm")
        print(f"    Median:     {s['median']*1000:>8.2f} mm")
        print(f"    Std dev:    {s['std']*1000:>8.2f} mm")
        print(f"    P95:        {s['p95']*1000:>8.2f} mm")
        print(f"    Max:        {s['max']*1000:>8.2f} mm")
        if s["active_agg"]:
            a = s["active_agg"]
            print(f"    -- Active LER only --")
            print(f"    Mean:       {a['mean']*1000:>8.2f} mm   "
                  f"P95: {a['p95']*1000:.2f} mm   Max: {a['max']*1000:.2f} mm")
        if s["inactive_agg"]:
            ia = s["inactive_agg"]
            print(f"    -- Inactive LER only --")
            print(f"    Mean:       {ia['mean']*1000:>8.2f} mm   "
                  f"P95: {ia['p95']*1000:.2f} mm   Max: {ia['max']*1000:.2f} mm")
    elif s["has_ler"]:
        print(f"    ** No measured points inside the trench **")
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
# types whose LER match covers a segment in that layer. Only utility types with
# an explicit LER match contribute; unlabelled / unmatched instances (whose
# match mask would otherwise cover every segment) are skipped so that
# unsegmented LER layers, e.g. Gasledning or Foeringsroer, get no reference
# points and therefore no deviation.
_layer_ref_pts = {}
for ut, instances in class_instances.items():
    if UTILITY_TO_LER_MATCH.get(ut) is None:
        continue
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
# Raw deviation values per layer, retained for the QGIS LAS export (the
# point clouds above only keep baked colours). None where no measured
# neighbour exists (no LER match), matching the no-data colouring.
ler_raw_xyz = {}          # layer -> float array, 3D deviation (m)
ler_raw_xy = {}           # layer -> float array, horizontal deviation (m)
ler_raw_z = {}            # layer -> float array, |Z| deviation (m)
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
    dev_chunks, zdev_chunks, xydev_chunks = [], [], []
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
            dev = zdev = xydev = np.full(len(samp), np.nan)
        samp_chunks.append(samp)
        col_chunks.append(cols)
        col_cont_chunks.append(cols_cont)
        zcol_chunks.append(zcols)
        zcol_cont_chunks.append(zcols_cont)
        xycol_chunks.append(xycols)
        xycol_cont_chunks.append(xycols_cont)
        dev_chunks.append(dev)
        zdev_chunks.append(zdev)
        xydev_chunks.append(xydev)

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
    ler_raw_xyz[ln] = np.concatenate(dev_chunks)
    ler_raw_xy[ln] = np.concatenate(xydev_chunks)
    ler_raw_z[ln] = np.concatenate(zdev_chunks)

print(f"  {_n_samples_total:,} LER samples across {len(ler_pcd_dev)} layers")

# Inside-trench mask per LER layer (the six dev clouds of a layer share points,
# so one mask suffices); absent key => unrestricted.
_ler_inside = {}
for _ln, _pc in ler_pcd_dev.items():
    _m = _inside_mask(np.asarray(_pc.points))
    if _m is not None:
        _ler_inside[_ln] = _m

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
    "Point cloud XYZ deviation (discrete)",      # 0
    "Point cloud XYZ deviation (continuous)",    # 1
    "Point cloud XY deviation (discrete)",       # 2
    "Point cloud XY deviation (continuous)",     # 3
    "Point cloud Z deviation (discrete)",        # 4
    "Point cloud Z deviation (continuous)",      # 5
    "Original RGB",                              # 6
    "LER utility class",                         # 7
    "LER XYZ deviation (discrete)",              # 8
    "LER XYZ deviation (continuous)",            # 9
    "LER XY deviation (discrete)",               # 10
    "LER XY deviation (continuous)",             # 11
    "LER Z deviation (discrete)",                # 12
    "LER Z deviation (continuous)",              # 13
]
# Instance point cloud shown per mode. In the LER deviation modes the heatmap
# lives on the LER segments, so the instance points fall back to original RGB.
_MODE_INST_PCD = ["pcd_dev", "pcd_dev_cont",
                  "pcd_dev_xy", "pcd_dev_xy_cont",
                  "pcd_dev_z", "pcd_dev_z_cont",
                  "pcd_rgb", "pcd_class",
                  "pcd_rgb", "pcd_rgb", "pcd_rgb", "pcd_rgb", "pcd_rgb", "pcd_rgb"]
# LER deviation modes: the LER layers become deviation-coloured point clouds.
# Each maps to the precomputed cloud carrying the right metric + colouring.
_LER_MODE_PCD = {
    8: ler_pcd_dev,           # XYZ deviation, discrete accuracy-class colours
    9: ler_pcd_dev_cont,      # XYZ deviation, continuous gradient
    10: ler_pcd_xydev,        # XY deviation, discrete accuracy-class colours
    11: ler_pcd_xydev_cont,   # XY deviation, continuous gradient
    12: ler_pcd_zdev,         # Z deviation, discrete accuracy-class colours
    13: ler_pcd_zdev_cont,    # Z deviation, continuous gradient
}
_LER_DEV_MODES = tuple(_LER_MODE_PCD)
# Instance-deviation modes (the measured points themselves are deviation
# coloured); these are the modes whose instance clouds the trench restricts.
_PC_DEV_MODES = (0, 1, 2, 3, 4, 5)
# Modes that show the discrete accuracy-class heatmap legend
_HEATMAP_MODES = (0, 2, 4, 8, 10, 12)
# Modes that show the continuous deviation-gradient legend
_GRADIENT_MODES = (1, 3, 5, 9, 11, 13)

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
setup_scene_lighting(scene_widget.scene, post_processing=True)


# Instance clouds are coloured by deviation / class, so they use the shaded
# (lit) material for a depth cue; the background RGB cloud uses the flat one.
def make_pt_mat(size=3.0):
    return point_material_shaded(size)


def make_pt_mat_unlit(size=3.0):
    return point_material_flat(size)


def _srgb_to_linear_arr(c):
    """Vectorised sRGB -> linear. Open3D's Filament renderer treats vertex
    colours as linear and re-encodes to sRGB for display, so PLY colours (already
    sRGB) must be linearised first or they render too bright."""
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def make_mesh_mat(alpha=1.0):
    return mesh_material(alpha)


def make_ler_pt_mat(size=6.0, alpha=1.0):
    """Point material for the LER deviation clouds. A transparency shader with a
    white base colour preserves the per-point deviation colours while letting
    the LER-opacity slider fade the cloud, mirroring the mesh material used in
    the non-deviation modes."""
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency"
    mat.base_color = [1.0, 1.0, 1.0, float(alpha)]
    mat.point_size = size
    return mat


def _trench_colored_pcd(base_pcd, inside_mask, outside_colors):
    """Inside the trench keep the cloud's deviation colours; outside, restore
    its original colours (``outside_colors``). Returns the cloud unchanged when
    no trench is defined (inside_mask is None) or the arrays do not match."""
    if inside_mask is None:
        return base_pcd
    cols = np.asarray(base_pcd.colors)
    outside = np.asarray(outside_colors)
    if cols.shape[0] != inside_mask.shape[0] or outside.shape != cols.shape:
        return base_pcd
    new_cols = cols.copy()
    new_cols[~inside_mask] = outside[~inside_mask]
    out = o3d.geometry.PointCloud()
    out.points = base_pcd.points
    out.colors = o3d.utility.Vector3dVector(new_cols)
    return out


# Background original cloud. Dimmed in every mode except "Original RGB", where
# it is shown at full brightness so the whole scene (trench included) reads in
# true RGB rather than the darkened backdrop. Colours are linearised so Filament
# re-encodes them to the original sRGB; the cloud is drawn unlit so it reads flat
# like a 2D viewer instead of being lit and tonemapped (which looked too bright).
_orig_lin = _srgb_to_linear_arr(original_colors)
pcd_dim = o3d.geometry.PointCloud()
pcd_dim.points = o3d.utility.Vector3dVector(pts_orig)
pcd_dim.colors = o3d.utility.Vector3dVector(_orig_lin * 0.35)

pcd_full = o3d.geometry.PointCloud()
pcd_full.points = o3d.utility.Vector3dVector(pts_orig)
pcd_full.colors = o3d.utility.Vector3dVector(_orig_lin)
try:
    pcd_dim.normals = pcd_orig.normals
    pcd_full.normals = pcd_orig.normals
except Exception:
    pass

ORIG_RGB_MODE = 6              # the "Original RGB" colour mode index
_orig_visible = [True]         # tracks the "Original cloud" checkbox state
scene_widget.scene.add_geometry(ORIG_GEOM, pcd_dim, make_pt_mat_unlit(2.0))


def _apply_orig_cloud_mode(mode):
    """Show the background cloud at full brightness in Original RGB mode, dimmed
    otherwise. Preserves the current visibility set by the 'Original cloud' box."""
    scene_widget.scene.remove_geometry(ORIG_GEOM)
    base = pcd_full if mode == ORIG_RGB_MODE else pcd_dim
    scene_widget.scene.add_geometry(ORIG_GEOM, base, make_pt_mat_unlit(2.0))
    scene_widget.scene.show_geometry(ORIG_GEOM, _orig_visible[0])

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
_crop_mat = line_material(2.0)
scene_widget.scene.add_geometry(CROP_GEOM, _crop_ls, _crop_mat)

# Trench outline overlay (only when a trench is defined). Drawn as a closed
# cyan polygon at ground level, in the same style as the crop region.
TRENCH_GEOM = "trench_outline"
if _trench_path_obj is not None:
    _tv = np.asarray(_trench_path_obj.vertices, dtype=float)
    if len(_tv) > 1 and np.allclose(_tv[0], _tv[-1]):
        _tv = _tv[:-1]
    _tpts = np.column_stack([_tv[:, 0], _tv[:, 1], np.full(len(_tv), GROUND_Z)])
    _tlines = [[i, (i + 1) % len(_tv)] for i in range(len(_tv))]
    _trench_ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(_tpts),
        lines=o3d.utility.Vector2iVector(_tlines))
    _trench_ls.paint_uniform_color([0.0, 1.0, 1.0])
    _trench_mat = line_material(3.0)
    scene_widget.scene.add_geometry(TRENCH_GEOM, _trench_ls, _trench_mat)

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

# Registered accuracy buffers (noejagtighedsklasse, 2D), hidden by default.
ACC_FILL_PREFIX = "accfill_"
ACC_OUT_PREFIX = "accout_"
_acc_show = [False]        # master toggle for the buffers
_acc_fill_show = [True]    # fill on/off (outline always shown when buffers are on)
# Per-layer buffer visibility, driven by the utility filter (all visible = show
# everything the master toggle allows).
_acc_layer_vis = {ln: True for ln in set(accbuf_fill) | set(accbuf_outline)}
_acc_outline_mat = line_material(2.0)
# Dash pattern (metres) for the accuracy-buffer outlines, so they read as
# dashed and are easy to tell apart from the solid utility lines. Open3D's line
# shader has no dash support, so each outline edge is broken into short dash
# segments with gaps. Each accuracy class gets its own (dash, gap) pattern so
# the five classes are tellable apart by line style as well as by colour; the
# pattern lengthens with class (class 1 finest, class 5 coarsest).
ACC_DASH_BY_CLASS = {
    1: (0.10, 0.10),
    2: (0.20, 0.15),
    3: (0.35, 0.20),
    4: (0.55, 0.30),
    5: (0.80, 0.40),
}
ACC_DASH_DEFAULT = (0.20, 0.15)
_CLASS_COLORS = [np.asarray(c, dtype=float) for c in DEVIATION_COLORS]


def _dash_params_for_color(col):
    """Dash and gap length (metres) for an outline line, chosen by its accuracy
    class. The class is identified by matching the line colour to the class
    palette; falls back to a default pattern if no class matches."""
    if col is not None:
        for idx, cc in enumerate(_CLASS_COLORS):
            if np.allclose(col, cc, atol=1e-3):
                return ACC_DASH_BY_CLASS[idx + 1]
    return ACC_DASH_DEFAULT


def _dash_lineset(ls):
    """Return a dashed copy of a LineSet by splitting each edge into on/off
    segments. The pattern depends on the line's accuracy class (class 1 differs
    from the rest), and the per-line colour is preserved."""
    pts = np.asarray(ls.points)
    lines = np.asarray(ls.lines)
    if len(pts) == 0 or len(lines) == 0:
        return ls
    cols = np.asarray(ls.colors)
    has_cols = len(cols) == len(lines)
    new_pts, new_lines, new_cols = [], [], []
    for li in range(len(lines)):
        a, b = lines[li]
        col = cols[li] if has_cols else None
        dash, gap = _dash_params_for_color(col)
        period = dash + gap
        p0, p1 = pts[a], pts[b]
        seg = p1 - p0
        length = float(np.linalg.norm(seg))
        if length < 1e-9:
            continue
        direction = seg / length
        t = 0.0
        while t < length:
            t_end = min(t + dash, length)
            i = len(new_pts)
            new_pts.append(p0 + direction * t)
            new_pts.append(p0 + direction * t_end)
            new_lines.append([i, i + 1])
            if col is not None:
                new_cols.append(col)
            t += period
    out = o3d.geometry.LineSet()
    out.points = o3d.utility.Vector3dVector(np.asarray(new_pts, dtype=float))
    out.lines = o3d.utility.Vector2iVector(np.asarray(new_lines, dtype=np.int32))
    if new_cols and len(new_cols) == len(new_lines):
        out.colors = o3d.utility.Vector3dVector(np.asarray(new_cols, dtype=float))
    return out


for ln, mesh in accbuf_fill.items():
    scene_widget.scene.add_geometry(ACC_FILL_PREFIX + ln, mesh, make_mesh_mat(0.30))
    scene_widget.scene.show_geometry(ACC_FILL_PREFIX + ln, False)
for ln, ls in accbuf_outline.items():
    scene_widget.scene.add_geometry(ACC_OUT_PREFIX + ln, _dash_lineset(ls), _acc_outline_mat)
    scene_widget.scene.show_geometry(ACC_OUT_PREFIX + ln, False)


def _update_acc_buffers():
    """Apply the master toggle, fill toggle and per-layer (filter) visibility."""
    show = _acc_show[0]
    for ln in accbuf_fill:
        vis = show and _acc_fill_show[0] and _acc_layer_vis.get(ln, True)
        scene_widget.scene.show_geometry(ACC_FILL_PREFIX + ln, vis)
    for ln in accbuf_outline:
        vis = show and _acc_layer_vis.get(ln, True)
        scene_widget.scene.show_geometry(ACC_OUT_PREFIX + ln, vis)
    window.post_redraw()

# Add instance geometries. Visibility is tracked per instance (ut, index) so the
# utility filter can isolate a single instance; the class checkboxes and the
# colour-mode switch read the same dict.
_inst_gnames = []
_inst_visible = {}
for ut, instances in class_instances.items():
    for i, inst in enumerate(instances):
        gn = f"inst_{ut}_{i}"
        _inst_gnames.append((ut, i, gn))
        _inst_visible[(ut, i)] = True
        # Startup is mode 0 (a deviation mode): inside-trench points get the
        # deviation colour, outside points keep their original RGB.
        _init_pcd = _trench_colored_pcd(inst["pcd_dev"], _inst_inside.get((ut, i)),
                                        np.asarray(inst["pcd_rgb"].colors))
        scene_widget.scene.add_geometry(gn, _init_pcd, make_pt_mat(4.0))

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
            base = dev_pcds[ln]
            _lcol = LINE_LAYERS.get(ln, {}).get("color", [0.5, 0.5, 0.5])
            _fb = np.tile(_lcol, (len(np.asarray(base.points)), 1))
            disp = _trench_colored_pcd(base, _ler_inside.get(ln), _fb)
            scene_widget.scene.add_geometry(gn, disp,
                                            make_ler_pt_mat(6.0, _ler_opacity[0]))
        else:
            scene_widget.scene.add_geometry(gn, ler_meshes[ln], make_mesh_mat(_ler_opacity[0]))
        scene_widget.scene.show_geometry(gn, _ler_visible.get(ln, True))


def _apply_color_mode(mode):
    _color_mode[0] = mode
    pcd_key = _MODE_INST_PCD[mode]
    restrict = mode in _PC_DEV_MODES
    for ut, instances in class_instances.items():
        for i, inst in enumerate(instances):
            gn = f"inst_{ut}_{i}"
            pcd = inst[pcd_key]
            if restrict:
                pcd = _trench_colored_pcd(pcd, _inst_inside.get((ut, i)),
                                          np.asarray(inst["pcd_rgb"].colors))
            scene_widget.scene.remove_geometry(gn)
            scene_widget.scene.add_geometry(gn, pcd, make_pt_mat(4.0))
            scene_widget.scene.show_geometry(gn, _inst_visible.get((ut, i), True))
    _apply_ler_color_mode(mode)
    _apply_orig_cloud_mode(mode)
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
_heatmap_legend.add_child(gui.Label("Accuracy class:"))
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
_gradient_legend.add_child(gui.Label("Deviation (gradient):"))
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


def _on_orig(c):
    _orig_visible[0] = c
    scene_widget.scene.show_geometry(ORIG_GEOM, c)
    window.post_redraw()


orig_cb.set_on_checked(_on_orig)
panel.add_child(orig_cb)

# Crop-region toggle (XY AABB + buffer rectangle in rect mode)
crop_cb = gui.Checkbox("Crop region (XY AABB + buffer)")
crop_cb.checked = True
crop_cb.set_on_checked(lambda c: (scene_widget.scene.show_geometry(CROP_GEOM, c), window.post_redraw()))
panel.add_child(crop_cb)

# Trench outline toggle + status (only meaningful when a trench is defined).
if _trench_path_obj is not None:
    trench_cb = gui.Checkbox(
        f"Trench outline ({len(_trench_verts)} pts, {_trench_mode})")
    trench_cb.checked = True
    trench_cb.set_on_checked(
        lambda c: (scene_widget.scene.show_geometry(TRENCH_GEOM, c), window.post_redraw()))
    panel.add_child(trench_cb)
else:
    panel.add_child(gui.Label("Trench: none (whole cloud)"))

# Registered accuracy buffer toggle (noejagtighedsklasse, 2D).
if accbuf_fill or accbuf_outline:
    acc_cb = gui.Checkbox(f"Accuracy buffer 2D ({_n_acc_view} feats)")
    acc_cb.checked = False

    def _on_acc(c):
        _acc_show[0] = c
        _update_acc_buffers()

    acc_cb.set_on_checked(_on_acc)
    panel.add_child(acc_cb)

    accfill_cb = gui.Checkbox("   buffer fill")
    accfill_cb.checked = True

    def _on_acc_fill(c):
        _acc_fill_show[0] = c
        _update_acc_buffers()

    accfill_cb.set_on_checked(_on_acc_fill)
    panel.add_child(accfill_cb)
else:
    panel.add_child(gui.Label("Accuracy buffer: none registered"))

# ── Utility filter (per-class view) ──
panel.add_fixed(int(0.5 * em))
panel.add_child(gui.Label("Utility filter:"))

# Build filter entries: (label, selector). The selector is None for "show all"
# or an (utility_type, instance_index) pair isolating a single instance. Each
# instance gets its own entry; a per-class "#k" suffix disambiguates classes
# that hold more than one instance (e.g. the two TelecomunicationLine clouds).
_filter_entries = [("All utilities", None)]
for _fut in sorted(class_instances.keys()):
    _fs = class_summaries[_fut]
    _instances = class_instances[_fut]
    _ler_names = _get_matching_ler_names(_fut)
    _ler_suffix = (f" <-> {', '.join(sorted(_ler_names))}"
                   if _ler_names else "  (no LER)")
    _multi = len(_instances) > 1
    for _i in range(len(_instances)):
        _num = f" #{_i + 1}" if _multi else ""
        _filter_entries.append((f"{_fs['label']}{_num}{_ler_suffix}", (_fut, _i)))

_active_filter = [None]   # None = show all
# Class-checkbox widgets, populated when the "Instance classes" panel is built;
# the filter uses these to keep checkbox states truthful.
_class_checkboxes = {}

filter_combo = gui.Combobox()
for _flbl, _ in _filter_entries:
    filter_combo.add_item(_flbl)
filter_combo.selected_index = 0


def _apply_utility_filter(sel):
    """Show/hide geometry to isolate a single instance, or show everything.

    ``sel`` is ``None`` (show all) or an ``(utility_type, index)`` pair. The LER
    layers shown are those matching the selected instance's utility type."""
    _active_filter[0] = sel
    sel_ut = sel[0] if sel is not None else None
    matching_ler = _get_matching_ler_names(sel_ut) if sel_ut is not None else None

    # Instances: show only the selected one (or all)
    for ut, instances in class_instances.items():
        for i in range(len(instances)):
            vis = (sel is None or (ut, i) == sel)
            _inst_visible[(ut, i)] = vis
            scene_widget.scene.show_geometry(f"inst_{ut}_{i}", vis)

    # LER layers: show only those matching the selected instance's type (or all)
    for ln in ler_meshes:
        if sel is None:
            vis = True
        else:
            vis = ln in matching_ler if matching_ler else False
        _ler_visible[ln] = vis
        scene_widget.scene.show_geometry(f"ler_{ln}", vis)

    # Accuracy buffers follow the same per-utility matching as the LER layers.
    matching_acc = _get_matching_accbuf_keys(sel_ut) if sel_ut is not None else None
    for ln in _acc_layer_vis:
        if sel is None:
            _acc_layer_vis[ln] = True
        else:
            _acc_layer_vis[ln] = ln in matching_acc if matching_acc else False
    _update_acc_buffers()

    # Keep the class checkboxes truthful: checked if any of the class's
    # instances is currently visible.
    for ut, cb in _class_checkboxes.items():
        cb.checked = any(_inst_visible[(ut, j)]
                         for j in range(len(class_instances[ut])))

    window.post_redraw()


def _on_filter(val, idx):
    _, sel = _filter_entries[idx]
    _apply_utility_filter(sel)


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
    # In the LER deviation modes the LER layers are deviation-coloured point
    # clouds, so the slider fades them via the point material; in the other
    # modes they are solid meshes faded via the mesh material.
    in_ler_dev = _color_mode[0] in _LER_DEV_MODES
    for ln in ler_meshes:
        if _ler_visible.get(ln, True):
            mat = make_ler_pt_mat(6.0, val) if in_ler_dev else make_mesh_mat(val)
            scene_widget.scene.modify_geometry_material(f"ler_{ln}", mat)
    for ln in comp_meshes:
        if _comp_visible.get(ln, False):
            scene_widget.scene.modify_geometry_material(f"comp_{ln}", make_mesh_mat(val))
    window.post_redraw()


ler_slider.set_on_value_changed(_on_ler_opacity)
ler_row.add_child(ler_slider)
panel.add_child(ler_row)

# Export the trench-restricted discrete LER deviation modes (XYZ, XY, Z) to LAS
# for QGIS. Only the samples inside the picked trench are written; with no
# trench the whole LER cloud is exported.
panel.add_fixed(int(0.3 * em))
_export_status = gui.Label("")


def _on_export_ler_las():
    from core.ler_las_export import export_ler_deviation_las
    out_dir = _ply_path.parent / f"{_ply_path.stem}_LER_deviation_LAS"
    # Export only the LER layers currently shown in the deviation colour mode:
    # visible in the LER layers panel and passing the utility filter. Layers
    # with no computed deviation (unsegmented) are further dropped in the
    # exporter via the NaN filter, so the LAS matches the on-screen discrete
    # deviation mode exactly.
    export_layers = [ln for ln in ler_pcd_dev if _ler_visible.get(ln, True)]
    samples_by_layer = {ln: np.asarray(ler_pcd_dev[ln].points)
                        for ln in export_layers}
    raw_by_metric = {"xyz": ler_raw_xyz, "xy": ler_raw_xy, "z": ler_raw_z}
    print(f"\nExporting LER deviation LAS to {out_dir} ...")
    print(f"  Visible layers to export ({len(export_layers)}): "
          f"{', '.join(export_layers) if export_layers else '(none)'}")
    try:
        written = export_ler_deviation_las(
            _ply_path.stem, out_dir, (TX, TY, TZ),
            samples_by_layer, raw_by_metric, _ler_inside)
    except Exception as exc:
        print(f"  [ERROR] export failed: {exc}")
        _export_status.text = f"Export failed: {exc}"
        window.post_redraw()
        return
    if written:
        las_n = sum(1 for p in written if p.suffix == ".las")
        _export_status.text = f"Exported {las_n} LAS to {out_dir.name}"
    else:
        _export_status.text = "Nothing to export (no in-trench LER samples)"
    window.post_redraw()


_export_btn = gui.Button("Export LER deviation to LAS (QGIS)")
_export_btn.set_on_clicked(_on_export_ler_las)
panel.add_child(_export_btn)
panel.add_child(_export_status)

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
            for _u, _i, gn in _inst_gnames:
                if _u == u:
                    _inst_visible[(_u, _i)] = checked
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
    _class_checkboxes[ut] = cb
    row.add_child(sw)
    row.add_fixed(int(0.4 * em))
    row.add_child(cb)
    panel.add_child(row)
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


def _top_view():
    """Bird's-eye view looking straight down, framed on the trench footprint
    when one is defined, otherwise on the whole scene."""
    if _trench_path_obj is not None:
        v = np.asarray(_trench_path_obj.vertices, dtype=float)
        cx, cy = float(v[:, 0].mean()), float(v[:, 1].mean())
        span = max(float(v[:, 0].ptp()), float(v[:, 1].ptp()))
        cz = float(GROUND_Z)
    else:
        cx, cy, cz = (float(cloud_centroid[0]), float(cloud_centroid[1]),
                      float(cloud_centroid[2]))
        span = max(float(pc_max[0] - pc_min[0]), float(pc_max[1] - pc_min[1]))
    h = max(1.0, span) * 1.2
    scene_widget.look_at([cx, cy, cz], [cx, cy, cz + h], [0.0, 1.0, 0.0])


def on_key(event):
    if event.type != gui.KeyEvent.DOWN:
        return IGNORED
    k = event.key
    if k in (ord('C'), ord('c')):
        _pivot_to(cloud_centroid)
        return HANDLED
    if k in (ord('T'), ord('t')):
        _top_view()
        return HANDLED
    if k in (ord('H'), ord('h')):
        print("\n  C   pivot to centroid    T   top view of trench    H   help\n")
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
