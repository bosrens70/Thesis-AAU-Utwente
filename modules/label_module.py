# -*- coding: utf-8 -*-
"""
Single Point Cloud Viewer with Instance Labels + Surrounding Utilities
======================================================================
Refactored to use core/ for shared configuration and data loading.

Besides assigning a utility-type label to an instance, an instance can also
be linked to one specific LER line feature: left-click a utility line while
that instance is active to record the match (layer + gml_id), or use
"Suggest LER match" to have the closest/best-aligned nearby feature proposed
automatically (ranked by proximity, direction, diameter and colour
similarity — see core/ler_matching.py) and accept or cycle through it.
"Mark as NOT in LER" records that an instance has no registry counterpart at
all. Matches are saved to ler_matches.json next to the labelled PLYs;
deviation_module.py reads it and, when a match exists, measures that
instance against only its linked LER feature instead of every nearby feature
of the same utility type.

Usage: python modules/label_module.py
  Change the site in core/site_local.py.
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
import glob as _globmod
from datetime import datetime

from core.config import (
    PLY_FILE, GML_PATH, AREA_REF_GEOJSON, CROP_RADIUS, CROP_MODE, UTILITY_RECT_BUFFER,
    PANEL_WIDTH_EM,
    CLASS_LABELS, DEFAULT_CLASS_COLOR,
    LEDNINGSPAKKE_LABEL, layer_display_name,
    LINE_LAYERS, COMPONENT_LAYERS, COMP_TO_LINE,
    COMPONENT_SPHERE_RADIUS, PIPE_LEGEND_UI_ORDER,
    INSTANCE_COLORS, INSTANCE_LABEL_OPTIONS,
    TARGET_CLASS, UTILITY_TO_LER_MATCH, UTILITY_TYPE_COLORS,
    DepthSource, DEPTH_STATS_KEY as _STATS_KEY,
    forsyningsart_color,
)
from core.data_loader import (
    init_site, discover_instances, load_or_pick_ground_level, load_trench,
)
from core.site_status import (
    LABELED_PREFIX, LABELED_FNAME_RE, resolve_labeled_dir, format_label_summary,
)
from core.gui_helpers import (
    make_legend_row, make_master_pipe_toggle, make_master_comp_toggle,
    LerLegendSection,
    pivot_top_down, top_view, trench_or_scene_frame,
)
from core.ler_matching import build_feature_index, score_candidates
from core.geometry import (
    segment_to_cylinder, segment_to_plane, point_to_segment_dists,
    linear_to_srgb,
)
from core.crop import CropRegion
from core.depth import (clean_coords_with_depth as _core_clean_coords,
                        MAX_DEPTH_BELOW_GROUND)
from core.ledningstrace import (
    get_ledningstrace_display_info, get_storage_key, get_bredde_width,
    is_trace_key, ribbon_alpha,
)
from core.trace_render import (
    build_trace_centerlines, add_trace_centerlines, set_layer_material,
)
from core.rendering import (
    point_material_shaded, point_material_flat, mesh_material, line_material,
    flat_material, setup_scene_lighting,
)

# ─────────────────────────────────────────────────────────────────────────────
# INITIALISE — load area offset, point cloud, GML, and instances via core/
# ─────────────────────────────────────────────────────────────────────────────
# GML is read layer-by-layer below (the loop needs per-feature control), so
# init_site must not pre-load it a second time.
site = init_site(load_gml=False, load_instances=True)

# Unpack area info
TX, TY, TZ = site.area.TX, site.area.TY, site.area.TZ
AREA_NUMBER = site.area.area_number
AREA_NAME   = site.area.area_name

# Unpack point cloud data
pcd             = site.pc.pcd
pts             = site.pc.pts
original_colors = site.pc.original_colors
class_labels    = site.pc.class_labels
cloud_centroid  = site.pc.cloud_centroid
cloud_centroid_full = site.pc.cloud_centroid_full
pc_min          = site.pc.pc_min
pc_max          = site.pc.pc_max

_crop_cx_local = site.pc.crop_center_local[0]
_crop_cy_local = site.pc.crop_center_local[1]
_crop_cx_utm   = site.pc.crop_center_utm[0]
_crop_cy_utm   = site.pc.crop_center_utm[1]
_crop_r2       = CROP_RADIUS * CROP_RADIUS

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

# Instance directory from core discovery
INSTANCE_DIR = str(site.instance_dir) if site.instance_dir else ""

# ─────────────────────────────────────────────────────────────────────────────
# VIEWER-SPECIFIC CODE BELOW (instances, ground picking, mesh creation, GUI)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 2b. Load instance PLY files and compute bounding boxes
# ─────────────────────────────────────────────────────────────────────────────
INSTANCE_COLORS = [
    [1.00, 0.20, 0.20],  # red
    [0.20, 0.60, 1.00],  # blue
    [0.20, 0.90, 0.20],  # green
    [1.00, 0.60, 0.00],  # orange
    [0.80, 0.20, 1.00],  # purple
    [0.00, 0.90, 0.90],  # cyan
    [1.00, 1.00, 0.20],  # yellow
    [1.00, 0.40, 0.70],  # pink
    [0.60, 0.40, 0.20],  # brown
    [0.50, 1.00, 0.50],  # lime
]

_instance_dir = Path(INSTANCE_DIR)
_instance_files = site.instance_files if site.instance_files else []

instance_data = []
for _i, _inst_path in enumerate(_instance_files):
    _inst_pcd = o3d.io.read_point_cloud(str(_inst_path))
    _inst_pts = np.asarray(_inst_pcd.points)
    if len(_inst_pts) == 0:
        continue
    _obb = _inst_pcd.get_oriented_bounding_box()
    _col = INSTANCE_COLORS[_i % len(INSTANCE_COLORS)]
    _obb.color = _col
    instance_data.append({
        "name": _inst_path.stem,
        "path": _inst_path,
        "pcd": _inst_pcd,
        "obb": _obb,
        "color": _col,
        "n_pts": len(_inst_pts),
    })

instance_data.sort(key=lambda d: d["n_pts"], reverse=True)
for _i, _d in enumerate(instance_data):
    _d["color"] = INSTANCE_COLORS[_i % len(INSTANCE_COLORS)]
    _d["obb"].color = _d["color"]

if instance_data:
    print(f"\n  Loaded {len(instance_data)} instances from {_instance_dir.name}/ (sorted largest first)")
    for _i, _d in enumerate(instance_data):
        print(f"    [{_i}] {_d['name']}: {_d['n_pts']:,} points")
else:
    print(f"\n  [warn] No instance PLY files found in {INSTANCE_DIR}")

INSTANCE_LABEL_OPTIONS = [
    "PowerLine",
    "DrainageLine",
    "OilPipeLine",
    "GasLine",
    "ThermalLine",
    "Conduit",
    "WaterLine",
    "TelecomunicationLine",
    "OtherLine",
    "LineUnknownServiceType",
]

_instance_labels = {}
_instance_ler_match = {}  # idx -> {"layer": str, "gml_id": str} — exclusive LER link
_current_inst_idx = [0]

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Ground level: cached per site in <stem>_ground.json next to the PLY, so it
#     is picked once and then shared with the other modules. Delete that file to
#     re-pick.
# ─────────────────────────────────────────────────────────────────────────────
GROUND_Z = load_or_pick_ground_level(site.pc, _ply_path)
print(f"  Ground level (UTM)   = {GROUND_Z + TZ:.3f} m")

_ground_normal = np.array([0.0, 0.0, 1.0])
_ground_center = np.array([_crop_cx_local, _crop_cy_local, GROUND_Z])


def _ground_z_at(x_local, y_local):
    """Return ground Z at a local XY position (flat plane)."""
    return GROUND_Z

# Depth estimation counters
_depth_stats = {"estimated": 0, "fallback_feature_mean": 0, "fallback_global": 0}

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Geometry helpers (shared implementations in core/)
# ─────────────────────────────────────────────────────────────────────────────
# Crop-region selection/clipping: one shared implementation in core.crop.
_crop_region = CropRegion.from_pointcloud(site.pc, TX, TY)
_point_in_bbox        = _crop_region.contains_utm
_pt_in_local_bbox     = _crop_region.contains_local
_segments_in_bbox     = _crop_region.polyline_in_region_utm
_clip_segment_to_bbox = _crop_region.clip_local


def _clean_coords_with_depth(coords_raw, vejledende_dybde_mm):
    """UTM -> local translation + DepthSource fallback (core.depth), bound to
    this viewer's flat ground level. Returns (coords, sources); the caller
    counts the fallback statistics from ``sources``."""
    return _core_clean_coords(coords_raw, vejledende_dybde_mm,
                              TX=TX, TY=TY, TZ=TZ, ground_z_at=_ground_z_at)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Load utility lines (pipes / cables) within bbox
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Loading utility lines within bbox ---")
all_pipe_meshes   = []          # flat list — kept for wireframe merge only
_pipe_layer_cyls  = {}          # layer_name -> [TriangleMesh, ...]  per-layer
_pipe_layer_seg_pts = {}        # layer_name -> ([p1, ...], [p2, ...]) for XRay centerlines
layer_stats = {}
all_pipe_coords = []

# Picking data — segment endpoints, midpoints, and their GML attributes
pick_seg_p1        = []   # list of np.array([x,y,z])  — segment start
pick_seg_p2        = []   # list of np.array([x,y,z])  — segment end
pick_seg_midpoints = []   # list of np.array([x,y,z])  — for highlight placement
pick_seg_attrs     = []   # list of [(label, value), ...]
pick_seg_layer     = []   # layer name per segment
pick_seg_gml_id    = []   # GML gml_id per segment (identifies the whole feature)

# Store per-utility-type average depth for component fallback
_layer_avg_depth_local = {}

# Track Ledningstrace forsyningsart variants for GUI legend
_ledningstrace_variants = {}  # forsyningsart -> color mapping

# Track colors for all storage keys (including compound keys for Ledningstrace variants)
_storage_key_colors = {}  # storage_key -> color

for layer_name, cfg in LINE_LAYERS.items():
    try:
        gdf = gpd.read_file(GML_PATH, layer=layer_name)
    except Exception as e:
        print(f"  {layer_name}: skip ({e})")
        continue

    default_color   = cfg["color"]
    fallback_radius = cfg["fallback_radius"]
    n_features = 0
    n_segments = 0
    _layer_z_vals = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        # Handle MultiLineString (e.g. Ledningstrace) by extracting sub-lines
        if geom.geom_type == "MultiLineString":
            sub_geoms = list(geom.geoms)
        else:
            sub_geoms = [geom]

        # Determine radius and colour (same for all sub-geometries of one feature)
        diam_mm = 0.0
        if "udvendigDiameter" in row.index:
            try:
                diam_mm = float(row["udvendigDiameter"] or 0)
            except (ValueError, TypeError):
                diam_mm = 0.0

        radius = diam_mm / 2000.0 if diam_mm > 0 else fallback_radius

        # Get Ledningstrace display info (color, forsyningsart) and width
        is_trace, display_fa, color = get_ledningstrace_display_info(layer_name, row, default_color)
        if is_trace and display_fa and display_fa not in _ledningstrace_variants:
            _ledningstrace_variants[display_fa] = color

        bredde_m = get_bredde_width(row)
        if is_trace and bredde_m is None:
            bredde_m = 0.25  # fallback: 25 cm

        # Get indicative depth for this feature
        vejl_dybde = None
        if "vejledendeDybde" in row.index:
            vejl_dybde = row.get("vejledendeDybde", None)

        # Extract all GML attributes for picking display
        row_attrs = []
        for col in row.index:
            if col == "geometry":
                continue
            val     = row[col]
            val_str = str(val) if (val is not None and str(val) != "nan") else "—"
            row_attrs.append((col, val_str))

        gml_id_val = str(row.get("gml_id", "") or "")

        feature_hit = False
        for sub_geom in sub_geoms:
            coords_raw = np.array(sub_geom.coords, dtype=float)
            if not _segments_in_bbox(coords_raw):
                continue

            coords, _seg_srcs = _clean_coords_with_depth(coords_raw, vejl_dybde)
            for _src in _seg_srcs:
                _key = _STATS_KEY.get(DepthSource(_src))
                if _key in _depth_stats:
                    _depth_stats[_key] += 1
            all_pipe_coords.append(coords)
            _layer_z_vals.extend(coords[:, 2].tolist())
            feature_hit = True

            for i in range(len(coords) - 1):
                clipped = _clip_segment_to_bbox(coords[i], coords[i + 1])
                if clipped is None:
                    continue
                # Use planes for Ledningstrace (with width from bredde_m), cylinders for other utility lines
                if bredde_m is not None:
                    mesh = segment_to_plane(clipped[0], clipped[1], bredde_m, color)
                else:
                    # Registered Z is the pipe crown (top), not its axis; lower the
                    # drawn cylinder by its radius so its crown sits on the line.
                    # The pick line (below) stays on the registered crown.
                    _dz = np.array([0.0, 0.0, radius])
                    mesh = segment_to_cylinder(clipped[0] - _dz, clipped[1] - _dz, radius, color)
                if mesh is not None:
                    all_pipe_meshes.append(mesh)
                    storage_key = get_storage_key(layer_name, display_fa)
                    _pipe_layer_cyls.setdefault(storage_key, []).append(mesh)
                    # Track color for this storage key
                    if storage_key not in _storage_key_colors:
                        _storage_key_colors[storage_key] = color
                    if storage_key not in _pipe_layer_seg_pts:
                        _pipe_layer_seg_pts[storage_key] = ([], [])
                    _pipe_layer_seg_pts[storage_key][0].append(clipped[0].copy())
                    _pipe_layer_seg_pts[storage_key][1].append(clipped[1].copy())
                    midpt = (clipped[0] + clipped[1]) / 2.0
                    pick_seg_p1.append(clipped[0].copy())
                    pick_seg_p2.append(clipped[1].copy())
                    pick_seg_midpoints.append(midpt)
                    pick_seg_attrs.append(row_attrs)
                    pick_seg_layer.append(storage_key)
                    pick_seg_gml_id.append(gml_id_val)
                    n_segments += 1

        if feature_hit:
            n_features += 1

    layer_stats[layer_name] = (n_features, n_segments)
    if _layer_z_vals:
        _layer_avg_depth_local[layer_name] = float(np.mean(_layer_z_vals))
    if n_features > 0:
        print(f"  {layer_name:<35} {n_features:>4} features  {n_segments:>5} segments")

pick_seg_p1        = np.array(pick_seg_p1)        if pick_seg_p1        else np.empty((0, 3))
pick_seg_p2        = np.array(pick_seg_p2)        if pick_seg_p2        else np.empty((0, 3))
pick_seg_midpoints = np.array(pick_seg_midpoints) if pick_seg_midpoints else np.empty((0, 3))

# One record per LER feature (grouped by gml_id), used to suggest the most
# likely instance <-> LER feature match (see "Suggest LER match" below).
_ler_feature_index = build_feature_index(
    pick_seg_p1, pick_seg_p2, pick_seg_layer, pick_seg_gml_id, pick_seg_attrs)

print(f"\n  Total: {len(all_pipe_meshes):,} cylinder segments")
print(f"\n  Depth estimation stats:")
print(f"    Estimated from vejledendeDybde + ground model: {_depth_stats['estimated']}")
print(f"    Fallback to feature mean Z:                    {_depth_stats['fallback_feature_mean']}")
print(f"    Fallback to global ground level:               {_depth_stats['fallback_global']}")

# Per-layer merged pipe meshes (used for individual visibility toggles)
_pipe_layer_meshes = {}
for _ln, _cyls in _pipe_layer_cyls.items():
    _m = _cyls[0]
    for _c in _cyls[1:]:
        _m += _c
    _m.compute_vertex_normals()
    _pipe_layer_meshes[_ln] = _m

# Per-layer XRay centerline LineSets — one line per clipped segment, rendered
# with depth_func="always" so thin pipes are visible through thick ones.
# Traces are excluded: they always carry their own centreline tube (below), so
# an XRay line for them would just double it up.
_pipe_layer_centerlines = {}
for _ln, (p1s, p2s) in _pipe_layer_seg_pts.items():
    if is_trace_key(_ln):
        continue
    _cl_pts   = []
    _cl_lines = []
    for _ci, (_cp1, _cp2) in enumerate(zip(p1s, p2s)):
        _cl_pts.extend([_cp1, _cp2])
        _cl_lines.append([2 * _ci, 2 * _ci + 1])
    _cl_ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(_cl_pts)),
        lines=o3d.utility.Vector2iVector(_cl_lines),
    )
    # Use tracked color for this storage key (works for both regular and Ledningstrace variants)
    _color = _storage_key_colors.get(_ln, [1.0, 1.0, 1.0])
    _cl_ls.paint_uniform_color(_color)
    _pipe_layer_centerlines[_ln] = _cl_ls

# Trace centrelines: the corridor ribbon is drawn transparent (see
# core/trace_render.py), so the registered centreline is drawn as a thin tube
# through the same lit material as the pipes.
_trace_centerlines = build_trace_centerlines(
    pick_seg_p1, pick_seg_p2, pick_seg_layer,
    lambda k: _storage_key_colors.get(k, [1.0, 1.0, 1.0]))

# Combined wireframe (all layers) for the wireframe overlay toggle.
# Build from per-layer meshes using the non-mutating `+` operator so that
# _pipe_layer_meshes entries are not corrupted (using `+=` on all_pipe_meshes[0]
# would mutate the first layer's merged mesh to contain all layers).
combined_pipe_wire = None
if _pipe_layer_meshes:
    _wf_meshes = list(_pipe_layer_meshes.values())
    _wire_src = _wf_meshes[0]
    for _m in _wf_meshes[1:]:
        _wire_src = _wire_src + _m  # non-mutating: creates a new merged mesh each time
    combined_pipe_wire = o3d.geometry.LineSet.create_from_triangle_mesh(_wire_src)
    combined_pipe_wire.paint_uniform_color([1.0, 1.0, 1.0])

# Pipe centroid
pipe_centroid = np.array([0.0, 0.0, 0.0])
if all_pipe_coords:
    pipe_centroid = np.vstack(all_pipe_coords).mean(axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Load utility components (points) within bbox
# ─────────────────────────────────────────────────────────────────────────────
# Map component layer -> corresponding line layer for depth estimation
_COMP_TO_LINE = {
    "Vandkomponent":               "Vandledning",
    "Afloebskomponent":            "Afloebsledning",
    "Gaskomponent":                "Gasledning",
    "Elkomponent":                 "Elledning",
    "Telekommunikationskomponent": "Telekommunikationsledning",
    "TermiskKomponent":            "TermiskLedning",
}

print("\n--- Loading utility components within bbox ---")
all_comp_meshes    = []     # flat list (kept for count reporting)
_comp_layer_spheres = {}    # layer_name -> [TriangleMesh, ...]  per-layer
comp_stats = {}
_comp_depth_stats = {"from_pipe_avg": 0, "from_ground": 0}

# Picking data for components
pick_comp_centres = []
pick_comp_attrs   = []
pick_comp_layer   = []

for layer_name, cfg in COMPONENT_LAYERS.items():
    try:
        gdf_c = gpd.read_file(GML_PATH, layer=layer_name)
    except Exception:
        continue

    color = cfg["color"]
    n_comp = 0

    # Get the average depth of the corresponding line layer for fallback
    parent_line = _COMP_TO_LINE.get(layer_name)
    parent_avg_z = _layer_avg_depth_local.get(parent_line) if parent_line else None

    for _, row in gdf_c.iterrows():
        g = row.geometry
        if g is None:
            continue
        # Components are usually Points; skip non-point geometries (e.g. Polygon)
        if g.geom_type not in ("Point", "PointZ"):
            continue
        if not _point_in_bbox(g.x, g.y):
            continue

        pt = np.array([g.x - TX, g.y - TY, g.z - TZ], dtype=float)

        # Crop to local buffered bbox
        if not _pt_in_local_bbox(pt[0], pt[1]):
            continue

        _gz_local = _ground_z_at(pt[0], pt[1])
        if g.z == -99 or pt[2] <= -98 or pt[2] < _gz_local - MAX_DEPTH_BELOW_GROUND:
            # Component has no reliable Z — estimate from parent pipe depth
            # or from ground model
            if parent_avg_z is not None:
                # Use average depth of the corresponding utility type
                pt[2] = parent_avg_z
                _comp_depth_stats["from_pipe_avg"] += 1
            else:
                pt[2] = _gz_local
                _comp_depth_stats["from_ground"] += 1

        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=COMPONENT_SPHERE_RADIUS, resolution=12
        )
        sphere.translate(pt)
        sphere.paint_uniform_color(color)
        all_comp_meshes.append(sphere)
        _comp_layer_spheres.setdefault(layer_name, []).append(sphere)

        # Store picking data
        pick_comp_centres.append(pt.copy())
        comp_row_attrs = []
        for col in row.index:
            if col == "geometry":
                continue
            val     = row[col]
            val_str = str(val) if (val is not None and str(val) != "nan") else "—"
            comp_row_attrs.append((col, val_str))
        pick_comp_attrs.append(comp_row_attrs)
        pick_comp_layer.append(layer_name)

        n_comp += 1

    comp_stats[layer_name] = n_comp
    if n_comp > 0:
        print(f"  {layer_name:<35} {n_comp:>4} components")

pick_comp_centres = np.array(pick_comp_centres) if pick_comp_centres else np.empty((0, 3))

print(f"\n  Total: {len(all_comp_meshes)} component spheres")
print(f"  Component depth estimation:")
print(f"    From parent pipe average Z: {_comp_depth_stats['from_pipe_avg']}")
print(f"    From ground model:          {_comp_depth_stats['from_ground']}")

# Per-layer merged component meshes
_comp_layer_meshes = {}
for _ln, _spheres in _comp_layer_spheres.items():
    _m = _spheres[0]
    for _s in _spheres[1:]:
        _m += _s
    _m.compute_vertex_normals()
    _comp_layer_meshes[_ln] = _m

_t_load = time.perf_counter()

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Coordinate frame + circular crop wireframe + point cloud normals
# ─────────────────────────────────────────────────────────────────────────────
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5, origin=cloud_centroid
)

# Estimate normals on the cropped point cloud so we can shade it with the
# `defaultLit` shader.  
try:
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(
        cloud_centroid + np.array([0.0, 0.0, 5.0])
    )
except Exception as _e:
    print(f"  [warn] point cloud normal estimation failed: {_e}")

# Wireframe showing the crop boundary on the ground plane
if CROP_MODE == "rect":
    _n = _ground_normal

    def _ground_z_at(x, y):
        """Z on the fitted ground plane at local (x, y)."""
        if abs(_n[2]) < 1e-9:
            return _ground_center[2]
        return _ground_center[2] - (
            _n[0] * (x - _ground_center[0]) + _n[1] * (y - _ground_center[1])
        ) / _n[2]

    _rect_corners = [
        (_rect_min_x, _rect_min_y), (_rect_max_x, _rect_min_y),
        (_rect_max_x, _rect_max_y), (_rect_min_x, _rect_max_y),
    ]
    bbox_wire_pts = np.array([[x, y, _ground_z_at(x, y)] for x, y in _rect_corners])
    bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
else:
    _N_CIRCLE = 72
    _theta = np.linspace(0.0, 2.0 * np.pi, _N_CIRCLE + 1)
    # Build two tangent vectors in the ground plane
    _n = _ground_normal
    if abs(_n[0]) < 0.9:
        _t1 = np.cross(_n, np.array([1.0, 0.0, 0.0]))
    else:
        _t1 = np.cross(_n, np.array([0.0, 1.0, 0.0]))
    _t1 /= np.linalg.norm(_t1)
    _t2 = np.cross(_n, _t1)
    bbox_wire_pts = np.array([
        _ground_center + CROP_RADIUS * (np.cos(t) * _t1 + np.sin(t) * _t2)
        for t in _theta
    ])
    bbox_lines = [[i, i + 1] for i in range(_N_CIRCLE)]
bbox_ls = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(bbox_wire_pts),
    lines=o3d.utility.Vector2iVector(bbox_lines),
)
bbox_ls.paint_uniform_color([1.0, 1.0, 0.0])

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Material helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_mesh_material(alpha: float) -> rendering.MaterialRecord:
    # Lit + transparent so the opacity slider still works AND the pipes
    # get shaded by normals (gives depth cues that flat-colour rendering lacks).
    return mesh_material(alpha)


def make_dotted_bbox_lineset(
    obb: o3d.geometry.OrientedBoundingBox,
    dash_len: float = 0.08,
    gap_len: float = 0.05,
) -> o3d.geometry.LineSet:
    """Create a dotted-style bbox by splitting each edge into short segments."""
    solid_ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    pts = np.asarray(solid_ls.points)
    lines = np.asarray(solid_ls.lines)

    out_pts = []
    out_lines = []

    for i0, i1 in lines:
        p0 = pts[int(i0)]
        p1 = pts[int(i1)]
        edge_vec = p1 - p0
        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len <= 1e-9:
            continue

        t = 0.0
        while t < edge_len:
            t_dash_end = min(t + dash_len, edge_len)
            a = p0 + edge_vec * (t / edge_len)
            b = p0 + edge_vec * (t_dash_end / edge_len)
            base = len(out_pts)
            out_pts.extend([a, b])
            out_lines.append([base, base + 1])
            t += dash_len + gap_len

    dotted_ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(out_pts)),
        lines=o3d.utility.Vector2iVector(np.asarray(out_lines, dtype=np.int32)),
    )
    return dotted_ls


def make_point_material() -> rendering.MaterialRecord:
    # Shaded (defaultLit) + estimated normals + SSAO post-processing is the
    # closest Open3D equivalent to an EDL shader. Points near geometric ridges
    # end up darker, giving a depth cue for the class-coloured cloud.
    return point_material_shaded(3.0)


def make_pipe_wire_material() -> rendering.MaterialRecord:
    return line_material(1.5)


def make_centerline_material() -> rendering.MaterialRecord:
    mat = line_material(2.5)
    try:
        # Render centerlines through all occluding geometry so thin pipes
        # remain visible even when embedded inside thick pipe cylinders.
        mat.depth_func = "always"
    except AttributeError:
        pass  # older Open3D — centerlines depth-test normally
    return mat


def make_frame_material() -> rendering.MaterialRecord:
    return flat_material()


def _add_mesh(scene, name, mesh, mat):
    """Add a TriangleMesh to the scene, ensuring vertex normals exist first."""
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    scene.add_geometry(name, mesh, mat)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Build GUI
# ─────────────────────────────────────────────────────────────────────────────
POINT_CLOUD_GEOM = "point_cloud"
PIPE_WIRE_GEOM   = "pipes_wire"
FRAME_GEOM       = "frame"
BBOX_GEOM        = "bbox_wire"
HIGHLIGHT_GEOM   = "highlight"

def _inst_bbox_gn(idx): return f"inst_bbox_{idx}"
def _inst_pts_gn(idx):  return f"inst_pts_{idx}"

_instance_visible = {i: True for i in range(len(instance_data))}

# Per-layer geometry names
def _pipe_gn(ln):       return f"pipe_{ln}"
def _comp_gn(ln):       return f"comp_{ln}"
def _centerline_gn(ln): return f"centerline_{ln}"

# Per-layer visibility state (True = shown)
_layer_visible = {ln: True for ln in LINE_LAYERS}
_layer_visible.update({ln: False for ln in COMPONENT_LAYERS})  # start with all components hidden
if "Ledningstrace" in _layer_visible:
    _layer_visible["Ledningstrace"] = False  # start with Ledningstrace hidden

pipe_opacity = [1.0]
origin_pt    = np.array([0.0, 0.0, 0.0])
pick_active  = [False]
origin_frame_visible  = [False] # toggled by the "Show origin axis" checkbox
pipe_wireframe_active = [False] # toggled by the "Wireframe pipes" checkbox
centerline_xray_active = [False] # toggled by the "XRay centerlines" checkbox
ler_utilities_visible = [True]   # toggled by the "Show LER utilities" checkbox

app = gui.Application.instance
app.initialize()

window = app.create_window(
    f"{_ply_path.stem}  |  Utilities + depth + class labels  |  press H for help",
    1460, 840,
)
em = window.theme.font_size

scene_widget = gui.SceneWidget()
scene_widget.scene = rendering.Open3DScene(window.renderer)
scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])

# Post-processing (SSAO + tone-mapping) and a top-down sun light for shading.
setup_scene_lighting(scene_widget.scene, post_processing=True)

# Add point cloud
scene_widget.scene.add_geometry(POINT_CLOUD_GEOM, pcd, make_point_material())

# Add per-layer pipe meshes (filled); wireframe is a separate combined overlay.
# A trace's corridor ribbon goes on more transparent than the rest.
for _ln, _mesh in _pipe_layer_meshes.items():
    _alpha0 = 1.0 if _layer_visible.get(_ln, True) else 0.0
    _add_mesh(scene_widget.scene, _pipe_gn(_ln), _mesh,
              make_mesh_material(ribbon_alpha(_ln, _alpha0)))

# Add trace centrelines at the unscaled opacity, so they read like the pipes
add_trace_centerlines(
    scene_widget.scene, _trace_centerlines, pipe_opacity[0], make_mesh_material,
    visible_of=lambda k: _layer_visible.get(k, True))

# Add combined wireframe overlay (hidden by default)
if combined_pipe_wire is not None:
    scene_widget.scene.add_geometry(
        PIPE_WIRE_GEOM, combined_pipe_wire, make_pipe_wire_material()
    )
    scene_widget.scene.show_geometry(PIPE_WIRE_GEOM, False)

# Add per-layer XRay centerlines (hidden by default)
for _ln, _cls in _pipe_layer_centerlines.items():
    scene_widget.scene.add_geometry(_centerline_gn(_ln), _cls, make_centerline_material())
    scene_widget.scene.show_geometry(_centerline_gn(_ln), False)

# Add per-layer component meshes
for _ln, _mesh in _comp_layer_meshes.items():
    _alpha0 = 1.0 if _layer_visible.get(_ln, True) else 0.0
    _add_mesh(scene_widget.scene, _comp_gn(_ln), _mesh, make_mesh_material(_alpha0))

# Add frame and bbox wireframe
scene_widget.scene.add_geometry(FRAME_GEOM, frame, make_frame_material())
scene_widget.scene.show_geometry(FRAME_GEOM, origin_frame_visible[0])

line_mat = line_material(3.0)
scene_widget.scene.add_geometry(BBOX_GEOM, bbox_ls, line_mat)

# Add instance bounding boxes and point clouds (only first visible initially)
for _idx, _inst in enumerate(instance_data):
    _bb_mat = line_material(4.0)
    _bb_ls = make_dotted_bbox_lineset(_inst["obb"])
    _bb_ls.paint_uniform_color([1.0, 1.0, 1.0])  # force per-instance bbox to white
    scene_widget.scene.add_geometry(_inst_bbox_gn(_idx), _bb_ls, _bb_mat)

    # Labelled instance clouds are RGB, so they render flat (unlit).
    _inst_pt_mat = point_material_flat(3.0)
    scene_widget.scene.add_geometry(_inst_pts_gn(_idx), _inst["pcd"], _inst_pt_mat)

    _show = (_idx == 0)
    scene_widget.scene.show_geometry(_inst_bbox_gn(_idx), _show)
    scene_widget.scene.show_geometry(_inst_pts_gn(_idx), _show)

bounds = scene_widget.scene.bounding_box
_init_d = max(1.0, np.linalg.norm(pc_max - pc_min) * 0.6)
_init_eye = cloud_centroid + np.array([0.0, 0.0, _init_d])
scene_widget.look_at(cloud_centroid.tolist(), _init_eye.tolist(), [0.0, 1.0, 0.0])


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Right-side control panel
# ─────────────────────────────────────────────────────────────────────────────
PANEL_WIDTH = int(PANEL_WIDTH_EM * em)
panel = gui.Vert(int(0.5 * em), gui.Margins(int(em), int(em), int(em), int(em)))

# Title
panel.add_child(gui.Label(f"Points: {len(pts):,}"))
if CROP_MODE == "rect":
    panel.add_child(gui.Label(
        f"Crop: cloud AABB + {UTILITY_RECT_BUFFER:.0f} m (rect)"))
else:
    panel.add_child(gui.Label(f"Crop radius: {CROP_RADIUS} m (circular)"))
panel.add_fixed(int(0.5 * em))

origin_toggle_cb = gui.Checkbox("Show origin axis")
origin_toggle_cb.checked = False


def _on_origin_toggle(checked):
    origin_frame_visible[0] = checked
    scene_widget.scene.show_geometry(FRAME_GEOM, checked)
    window.post_redraw()


origin_toggle_cb.set_on_checked(_on_origin_toggle)
panel.add_child(origin_toggle_cb)
panel.add_fixed(int(0.5 * em))

# ── Utility Legend (uniform LerLegendSection, see core/gui_helpers.py) ───────
_ler_section = LerLegendSection(em, LEDNINGSPAKKE_LABEL)
ler_toggle_cb = _ler_section.master_cb
opacity_slider = _ler_section.opacity_slider


def _on_ler_toggle(checked):
    ler_utilities_visible[0] = checked
    for ln in _pipe_layer_meshes:
        if not _layer_visible.get(ln, True):
            continue
        alpha = pipe_opacity[0] if checked else 0.0
        set_layer_material(scene_widget.scene, _pipe_gn(ln), ln, alpha,
                           make_mesh_material)
    for ln in _comp_layer_meshes:
        if not _layer_visible.get(ln, True):
            continue
        alpha = pipe_opacity[0] if checked else 0.0
        scene_widget.scene.modify_geometry_material(_comp_gn(ln), make_mesh_material(alpha))


_ler_section.set_on_master(window, _on_ler_toggle)
_ler_section.add_to(panel)
panel.add_fixed(int(0.3 * em))


def _make_pipe_toggle(ln):
    def _cb(checked):
        _layer_visible[ln] = checked
        _ler = ler_utilities_visible[0]
        if ln in _pipe_layer_meshes:
            alpha = pipe_opacity[0] if (_ler and checked and not pipe_wireframe_active[0]) else 0.0
            set_layer_material(scene_widget.scene, _pipe_gn(ln), ln, alpha,
                               make_mesh_material)
        if ln in _pipe_layer_centerlines:
            scene_widget.scene.show_geometry(
                _centerline_gn(ln), _ler and checked and centerline_xray_active[0]
            )
        window.post_redraw()
    return _cb


def _make_comp_toggle(ln):
    def _cb(checked):
        _layer_visible[ln] = checked
        _ler = ler_utilities_visible[0]
        if ln in _comp_layer_meshes:
            alpha = pipe_opacity[0] if (_ler and checked) else 0.0
            scene_widget.scene.modify_geometry_material(_comp_gn(ln), make_mesh_material(alpha))
        window.post_redraw()
    return _cb


# Track checkboxes for master toggles
_pipe_checkboxes = []
_comp_checkboxes = []

# "Toggle all segments" master checkbox
_all_pipes_cb = _ler_section.add_all_segments(
    True, make_master_pipe_toggle(_pipe_checkboxes, _layer_visible,
                                  _pipe_layer_meshes, scene_widget,
                                  _pipe_gn, make_mesh_material,
                                  pipe_opacity, window))

# Line layers — only show legend entry if the layer produced actual geometry
for layer_name, cfg in LINE_LAYERS.items():
    # Skip Ledningstrace here; we'll handle variants below
    if layer_name == "Ledningstrace":
        continue
    if layer_name not in _pipe_layer_meshes:
        continue
    n_feat, _ = layer_stats.get(layer_name, (0, 0))

    cb = _ler_section.add_layer_row(cfg["color"],
                                    f"{layer_display_name(layer_name)} ({n_feat})",
                                    _layer_visible.get(layer_name, True),
                                    _make_pipe_toggle(layer_name))
    _pipe_checkboxes.append((layer_name, cb))

# Ledningstrace variants — create separate entry for each forsyningsart
if _ledningstrace_variants:
    for fa, fa_color in sorted(_ledningstrace_variants.items()):
        variant_key = f"Ledningstrace ({fa})"
        if variant_key not in _pipe_layer_meshes:
            continue
        cb = _ler_section.add_layer_row(fa_color, layer_display_name(variant_key),
                                        _layer_visible.get(variant_key, True),
                                        _make_pipe_toggle(variant_key))
        _pipe_checkboxes.append((variant_key, cb))

# "Toggle all components" master checkbox
_all_comps_cb = _ler_section.add_all_components(
    False, make_master_comp_toggle(_comp_checkboxes, _layer_visible,
                                   _comp_layer_meshes, scene_widget,
                                   _comp_gn, make_mesh_material,
                                   pipe_opacity, window))

# Component layers — only show legend entry if the layer produced actual geometry
for layer_name, cfg in COMPONENT_LAYERS.items():
    if layer_name not in _comp_layer_meshes:
        continue
    n_comp = comp_stats.get(layer_name, 0)

    _layer_visible[layer_name] = False
    cb = _ler_section.add_layer_row(cfg["color"],
                                    f"{layer_display_name(layer_name)} ({n_comp})",
                                    False, _make_comp_toggle(layer_name))
    _comp_checkboxes.append((layer_name, cb))


# ── Utility Opacity (slider lives in the legend section) ─────────────────────
def _apply_opacity(val: float):
    val = max(0.0, min(1.0, val))
    pipe_opacity[0] = val
    opacity_slider.double_value = val
    _ler = ler_utilities_visible[0]

    for ln in _pipe_layer_meshes:
        alpha = val if (_ler and _layer_visible.get(ln, True) and not pipe_wireframe_active[0]) else 0.0
        set_layer_material(scene_widget.scene, _pipe_gn(ln), ln, alpha,
                           make_mesh_material)

    for ln in _comp_layer_meshes:
        alpha = val if (_ler and _layer_visible.get(ln, True)) else 0.0
        scene_widget.scene.modify_geometry_material(_comp_gn(ln), make_mesh_material(alpha))

    window.post_redraw()


_ler_section.set_on_opacity(_apply_opacity)
panel.add_fixed(int(0.4 * em))


panel.add_stretch()

# ─────────────────────────────────────────────────────────────────────────────
# 10b.  Left-side Instance Labeling panel
# ─────────────────────────────────────────────────────────────────────────────
LEFT_PANEL_WIDTH = int(18 * em)
left_panel = gui.Vert(int(0.15 * em), gui.Margins(int(em), int(em), int(em), int(em)))

if instance_data:
    _inst_progress_lbl = gui.Label(
        f"Instance 1 / {len(instance_data)}:  {instance_data[0]['name']}"
    )
    _inst_progress_lbl.text_color = gui.Color(1.0, 1.0, 0.3, 1.0)
    left_panel.add_child(_inst_progress_lbl)

    _inst_pts_lbl = gui.Label(f"  {instance_data[0]['n_pts']:,} points")
    _inst_pts_lbl.text_color = gui.Color(0.7, 0.7, 0.7, 1.0)
    left_panel.add_child(_inst_pts_lbl)

    _inst_assigned_lbl = gui.Label("")
    _inst_assigned_lbl.text_color = gui.Color(0.3, 1.0, 0.3, 1.0)
    _inst_assigned_lbl.visible = False
    left_panel.add_child(_inst_assigned_lbl)
    left_panel.add_fixed(int(0.25 * em))

    _suggest_btn = gui.Button("Suggest LER match")
    _suggest_btn.set_on_clicked(lambda: _suggest_ler_match())
    left_panel.add_child(_suggest_btn)

    _suggest_lbl = gui.Label("")
    _suggest_lbl.text_color = gui.Color(0.6, 0.6, 0.6, 1.0)
    left_panel.add_child(_suggest_lbl)

    _suggest_nav_row = gui.Horiz(int(0.3 * em))
    _accept_suggest_btn = gui.Button("Accept")
    _accept_suggest_btn.set_on_clicked(lambda: _accept_suggestion())
    _suggest_nav_row.add_child(_accept_suggest_btn)
    _next_suggest_btn = gui.Button("Next candidate")
    _next_suggest_btn.set_on_clicked(lambda: _next_suggestion())
    _suggest_nav_row.add_child(_next_suggest_btn)
    left_panel.add_child(_suggest_nav_row)
    left_panel.add_fixed(int(0.25 * em))

    _ler_match_lbl = gui.Label("LER match: none (click a line)")
    _ler_match_lbl.text_color = gui.Color(0.6, 0.6, 0.6, 1.0)
    left_panel.add_child(_ler_match_lbl)

    _no_ler_btn = gui.Button("Mark as NOT in LER")
    _no_ler_btn.set_on_clicked(lambda: _mark_no_ler())
    left_panel.add_child(_no_ler_btn)

    _clear_match_btn = gui.Button("Clear LER match")
    _clear_match_btn.set_on_clicked(lambda: _clear_ler_match())
    left_panel.add_child(_clear_match_btn)
    left_panel.add_fixed(int(0.25 * em))

    left_panel.add_child(gui.Label("Assign label (or press 1-0):"))

# Labelled output: resume the most recent labelled session when one exists, so
# labels and LER matches carry across sessions and a matching-only pass can
# attach to labels saved earlier. Which session that is, is decided by
# core/site_status.py so the status tool agrees. A fresh timestamped folder is
# named only when the site has no labelled session yet, and is not created on
# disk until the first save (see _ensure_output_dir).
_labeled_output_dir = None
if instance_data:
    _prev_labeled, _empty_labeled, _superseded_labeled = resolve_labeled_dir(_instance_dir)
    if _prev_labeled:
        _labeled_output_dir = _prev_labeled.path
    else:
        _label_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _labeled_output_dir = _instance_dir / f"{LABELED_PREFIX}{_label_stamp}"
    for _sd in _superseded_labeled:
        print(f"  [note] superseded label session kept: {_sd.path.name}/ "
              f"({_sd.n_ply} PLYs)")
    if _empty_labeled:
        print(f"  [note] {len(_empty_labeled)} empty labeled_* folder(s) ignored; "
              f"remove them with: python tools/pipeline_status.py --prune-empty")


_LABEL_TO_ID = {name: i + 1 for i, name in enumerate(INSTANCE_LABEL_OPTIONS)}

# Reverse of UTILITY_TO_LER_MATCH: matched LER layer -> implied type label, so
# an accepted match can label an instance that has no label yet.
_LER_LAYER_TO_LABEL = {
    layer: INSTANCE_LABEL_OPTIONS[lid - 1]
    for lid, _cfg in UTILITY_TO_LER_MATCH.items()
    if 1 <= lid <= len(INSTANCE_LABEL_OPTIONS)
    for layer in _cfg["layers"]
}

# Prefill labels and LER matches from the resumed session, and start at the
# first instance that still has no label.
if _labeled_output_dir and _labeled_output_dir.exists():
    for _f in sorted(_labeled_output_dir.glob("*.ply")):
        _fm = LABELED_FNAME_RE.match(_f.name)
        if not _fm:
            continue
        _pidx, _plid = int(_fm.group(1)), int(_fm.group(2))
        if _pidx < len(instance_data) and 1 <= _plid <= len(INSTANCE_LABEL_OPTIONS):
            _instance_labels[_pidx] = INSTANCE_LABEL_OPTIONS[_plid - 1]
    _matches_path = _labeled_output_dir / "ler_matches.json"
    if _matches_path.exists():
        try:
            with open(str(_matches_path), encoding="utf-8") as _fh:
                for _fname, _pmatch in json.load(_fh).items():
                    _fm = LABELED_FNAME_RE.match(_fname)
                    if _fm and int(_fm.group(1)) < len(instance_data):
                        _instance_ler_match[int(_fm.group(1))] = _pmatch
        except (json.JSONDecodeError, OSError) as _e:
            print(f"  [warn] Could not read {_matches_path.name}: {_e}")
    if _instance_labels or _instance_ler_match:
        print(f"  [resume] {_labeled_output_dir.name}/: {len(_instance_labels)} labels, "
              f"{len(_instance_ler_match)} LER matches loaded")
        _first_unlabeled = next(
            (i for i in range(len(instance_data)) if i not in _instance_labels), None)
        if _first_unlabeled is not None:
            _current_inst_idx[0] = _first_unlabeled


def _ensure_output_dir():
    """Create the labelled output folder on the first write.

    Deferred on purpose: the folder used to be created at startup, so opening
    the viewer (or just closing the ground-level picker) and quitting without
    labelling anything left an empty labeled_* folder behind. Several sites
    accumulated a stack of those.
    """
    if _labeled_output_dir and not _labeled_output_dir.is_dir():
        _labeled_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [new] label session {_labeled_output_dir.name}/")
    return _labeled_output_dir


def _label_status_text():
    """Live '4/4 labelled, 3 matched' counter, formatted by core/site_status.py
    so it reads the same as the pipeline_status table."""
    n_total = len(instance_data)
    n_labeled = len([i for i in _instance_labels if i < n_total])
    n_matched = sum(1 for i, m in _instance_ler_match.items()
                    if i < n_total and m.get("gml_id"))
    n_no_ler = sum(1 for i, m in _instance_ler_match.items()
                   if i < n_total and m.get("no_ler"))
    return format_label_summary(n_labeled, n_total, n_matched, n_no_ler)


def _refresh_window_title():
    """Keep the progress for this site visible in the title bar, so the state
    is readable without scrolling the panel or the console."""
    window.title = (f"{_ply_path.stem}  |  {_label_status_text()}  |  "
                    f"press H for help")


def _write_ler_matches_json():
    """Persist idx -> {layer, gml_id} for every labelled instance that has an
    exclusive LER link, keyed by the saved PLY filename. Rewritten in full each
    time so relabelling (which changes the filename) never leaves a stale key."""
    if not _labeled_output_dir:
        return
    _refresh_window_title()
    out = {}
    for idx, match in _instance_ler_match.items():
        if idx not in _instance_labels:
            continue
        label_id = _LABEL_TO_ID.get(_instance_labels[idx], 0)
        fname = f"{TARGET_CLASS}_instance_{idx}_type_{label_id}.ply"
        out[fname] = match
    if not out:
        return
    out_path = _ensure_output_dir() / "ler_matches.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def _save_instance_ply(idx, label_name):
    if not _labeled_output_dir or idx >= len(instance_data):
        return
    inst = instance_data[idx]
    pcd = inst["pcd"]
    pts = np.asarray(pcd.points)
    has_colors = pcd.has_colors()
    colors = np.asarray(pcd.colors) if has_colors else None
    has_normals = pcd.has_normals()
    normals = np.asarray(pcd.normals) if has_normals else None
    n = len(pts)
    label_id = _LABEL_TO_ID.get(label_name, 0)

    fname = f"{TARGET_CLASS}_instance_{idx}_type_{label_id}.ply"
    out_path = _ensure_output_dir() / fname

    with open(str(out_path), "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_colors:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        if has_normals:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        f.write("property int utility_type\n")
        f.write("end_header\n")
        for i in range(n):
            parts = [f"{pts[i, 0]:.6f}", f"{pts[i, 1]:.6f}", f"{pts[i, 2]:.6f}"]
            if has_colors:
                r, g, b = int(colors[i, 0] * 255), int(colors[i, 1] * 255), int(colors[i, 2] * 255)
                parts.extend([str(r), str(g), str(b)])
            if has_normals:
                parts.extend([f"{normals[i, 0]:.6f}", f"{normals[i, 1]:.6f}", f"{normals[i, 2]:.6f}"])
            parts.append(str(label_id))
            f.write(" ".join(parts) + "\n")

    print(f"  [saved] {out_path}  (utility_type={label_id}: {label_name})")
    _write_ler_matches_json()
    _refresh_window_title()


def _check_all_labeled():
    """Show the completion message once every instance carries a label. Shared by
    the label, autolabel, and startup paths so the message is consistent."""
    if instance_data and len(_instance_labels) == len(instance_data):
        _inst_progress_lbl.text = "All instances labeled!"
        _inst_pts_lbl.text = ""
        print("  [done] All instances have been labeled.")
        return True
    return False


def _utility_color_for_label(label_name):
    """DLF utility colour (RGB 0-1) for a label, via its utility-type id."""
    return UTILITY_TYPE_COLORS.get(_LABEL_TO_ID.get(label_name, 0),
                                   UTILITY_TYPE_COLORS[0])


def _apply_instance_color(idx):
    """Render a labelled instance in its utility-type colour, so an instance that
    has been linked to a utility is visually distinct from the raw RGB of the
    unlabelled ones. Restores the original RGB when the instance has no label.
    Re-adds the point geometry (Open3D cannot repaint vertices in place) and
    keeps its current visibility."""
    gn = _inst_pts_gn(idx)
    if not scene_widget.scene.has_geometry(gn):
        return
    label_name = _instance_labels.get(idx)
    if label_name is not None:
        pcd = o3d.geometry.PointCloud(instance_data[idx]["pcd"])  # copy, keep original
        pcd.paint_uniform_color(_utility_color_for_label(label_name))
    else:
        pcd = instance_data[idx]["pcd"]  # original RGB
    was_visible = (idx == _current_inst_idx[0])
    scene_widget.scene.remove_geometry(gn)
    scene_widget.scene.add_geometry(gn, pcd, point_material_flat(3.0))
    scene_widget.scene.show_geometry(gn, was_visible)


def _maybe_autolabel_from_layer(idx, layer_name):
    """When an instance has no type label yet, infer one from the matched LER
    layer (reverse of UTILITY_TO_LER_MATCH) and save it, so a match recorded on
    its own still persists a labelled PLY. No-op if the instance is already
    labelled or the layer has no type mapping (e.g. a Ledningstrace variant)."""
    if idx in _instance_labels:
        return
    label_name = _LER_LAYER_TO_LABEL.get(layer_name)
    if label_name is None:
        return
    _instance_labels[idx] = label_name
    print(f"  [label] Instance {idx} ({instance_data[idx]['name']}) -> {label_name}"
          f"  (inferred from LER layer {layer_name})")
    _inst_assigned_lbl.text = f"  Label: {label_name}"
    _inst_assigned_lbl.visible = True
    _save_instance_ply(idx, label_name)
    _apply_instance_color(idx)
    _check_all_labeled()


def _show_instance(idx):
    if not instance_data:
        return
    for i in range(len(instance_data)):
        vis = (i == idx)
        scene_widget.scene.show_geometry(_inst_bbox_gn(i), vis)
        scene_widget.scene.show_geometry(_inst_pts_gn(i), vis)
    inst = instance_data[idx]
    _inst_progress_lbl.text = (
        f"Instance {idx + 1} / {len(instance_data)}:  {inst['name']}"
    )
    _inst_pts_lbl.text = f"  {inst['n_pts']:,} points"
    if idx in _instance_labels:
        _inst_assigned_lbl.text = f"  Label: {_instance_labels[idx]}"
        _inst_assigned_lbl.visible = True
    else:
        _inst_assigned_lbl.visible = False
    _refresh_ler_match_label(idx)
    _m = _instance_ler_match.get(idx)
    _gid = _m.get("gml_id") if _m else None
    if _gid and _gid in pick_seg_gml_id:
        _gi = pick_seg_gml_id.index(_gid)
        _place_ler_match_highlight(pick_seg_p1[_gi], pick_seg_p2[_gi])
    else:
        _clear_ler_match_highlight()
    _suggestion_state["candidates"] = []
    _suggestion_state["idx"] = 0
    _suggest_lbl.text = ""
    _clear_suggestion_highlight()
    obb_center = np.asarray(inst["obb"].center)
    _pivot_to(obb_center)
    window.set_needs_layout()
    window.post_redraw()


def _assign_label(label_name):
    if not instance_data:
        return
    idx = _current_inst_idx[0]
    _instance_labels[idx] = label_name
    print(f"  [label] Instance {idx} ({instance_data[idx]['name']}) -> {label_name}")
    _inst_assigned_lbl.text = f"  Label: {label_name}"
    _inst_assigned_lbl.visible = True
    _save_instance_ply(idx, label_name)
    _apply_instance_color(idx)
    # Advance to next unlabeled instance, or next instance if all labeled
    next_idx = None
    for i in range(idx + 1, len(instance_data)):
        if i not in _instance_labels:
            next_idx = i
            break
    if next_idx is None:
        for i in range(0, idx):
            if i not in _instance_labels:
                next_idx = i
                break
    if next_idx is not None:
        _current_inst_idx[0] = next_idx
        _show_instance(next_idx)
    else:
        _check_all_labeled()
    window.set_needs_layout()
    window.post_redraw()


if instance_data:
    def _make_label_cb(label_name):
        def _cb():
            _assign_label(label_name)
        return _cb

    for _li, _label_name in enumerate(INSTANCE_LABEL_OPTIONS):
        _lbl_btn = gui.Button(f"{_li + 1}. {_label_name}")
        _lbl_btn.set_on_clicked(_make_label_cb(_label_name))
        left_panel.add_child(_lbl_btn)

    left_panel.add_fixed(int(0.25 * em))

    _nav_row = gui.Horiz(int(0.3 * em))

    _prev_btn = gui.Button("Prev")
    def _on_prev():
        if _current_inst_idx[0] > 0:
            _current_inst_idx[0] -= 1
            _show_instance(_current_inst_idx[0])
    _prev_btn.set_on_clicked(_on_prev)
    _nav_row.add_child(_prev_btn)

    _skip_btn = gui.Button("Skip")
    def _on_skip():
        if _current_inst_idx[0] + 1 < len(instance_data):
            _current_inst_idx[0] += 1
            _show_instance(_current_inst_idx[0])
    _skip_btn.set_on_clicked(_on_skip)
    _nav_row.add_child(_skip_btn)

    _next_btn = gui.Button("Next")
    def _on_next():
        if _current_inst_idx[0] + 1 < len(instance_data):
            _current_inst_idx[0] += 1
            _show_instance(_current_inst_idx[0])
    _next_btn.set_on_clicked(_on_next)
    _nav_row.add_child(_next_btn)

    left_panel.add_child(_nav_row)
    left_panel.add_fixed(int(0.2 * em))

    _save_info = gui.Label(f"Saves to: {_labeled_output_dir.name}/")
    _save_info.text_color = gui.Color(0.5, 0.5, 0.5, 1.0)
    left_panel.add_child(_save_info)

left_panel.add_stretch()



# ─────────────────────────────────────────────────────────────────────────────
# 11b.  Exclusive LER matching — left-click a utility line to link it to the
# current instance, so deviation_module.py measures that instance against only
# this one registered feature instead of every nearby feature of the same type.
# ─────────────────────────────────────────────────────────────────────────────
LER_MATCH_PICK_RADIUS = 0.30  # m — same tolerance as base_module's segment picking
LER_MATCH_HIGHLIGHT_GEOM = "ler_match_highlight"


def _clear_ler_match_highlight():
    try:
        scene_widget.scene.remove_geometry(LER_MATCH_HIGHLIGHT_GEOM)
    except Exception:
        pass


def _place_ler_match_highlight(p1, p2):
    _clear_ler_match_highlight()
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array([p1, p2])),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    ls.paint_uniform_color([1.0, 0.0, 1.0])  # magenta — distinct from any LER layer colour
    mat = line_material(6.0)
    try:
        mat.depth_func = "always"  # stay visible through occluding pipe meshes
    except AttributeError:
        pass
    scene_widget.scene.add_geometry(LER_MATCH_HIGHLIGHT_GEOM, ls, mat)


def _refresh_ler_match_label(idx):
    m = _instance_ler_match.get(idx)
    if m and m.get("no_ler"):
        _ler_match_lbl.text = "LER match: confirmed NOT in LER"
        _ler_match_lbl.text_color = gui.Color(1.0, 0.55, 0.15, 1.0)
    elif m:
        gid = m.get("gml_id", "")
        gid_short = gid[-28:] if len(gid) > 28 else gid
        _ler_match_lbl.text = f"LER match: {layer_display_name(m['layer'])}\n({gid_short})"
        _ler_match_lbl.text_color = gui.Color(0.3, 1.0, 1.0, 1.0)
    else:
        _ler_match_lbl.text = "LER match: none (click a line)"
        _ler_match_lbl.text_color = gui.Color(0.6, 0.6, 0.6, 1.0)


def _mark_no_ler():
    """Confirm this instance has no counterpart anywhere in LER, so
    deviation_module.py skips the nearest-of-type fallback for it entirely
    instead of risking a match against an unrelated nearby same-type feature."""
    idx = _current_inst_idx[0]
    _instance_ler_match[idx] = {"no_ler": True}
    _refresh_ler_match_label(idx)
    _clear_ler_match_highlight()
    _clear_suggestion_highlight()
    _write_ler_matches_json()
    window.post_redraw()


def _clear_ler_match():
    idx = _current_inst_idx[0]
    if idx in _instance_ler_match:
        del _instance_ler_match[idx]
    _refresh_ler_match_label(idx)
    _clear_ler_match_highlight()
    _clear_suggestion_highlight()
    _write_ler_matches_json()
    window.post_redraw()


# ── "Suggest LER match": rank nearby LER features by proximity, direction,
# diameter and colour similarity to the current instance, so the user can
# accept a proposed link instead of hunting for the right line to click.
# The suggestion is only a starting point — Accept just records the same
# match a manual click would, and the user can pick "Next candidate" or fall
# back to clicking a different line themselves if the top guess is wrong.
LER_SUGGEST_HIGHLIGHT_GEOM = "ler_suggest_highlight"
_suggestion_state = {"candidates": [], "idx": 0}


def _clear_suggestion_highlight():
    try:
        scene_widget.scene.remove_geometry(LER_SUGGEST_HIGHLIGHT_GEOM)
    except Exception:
        pass


def _place_suggestion_highlight(p1, p2):
    _clear_suggestion_highlight()
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array([p1, p2])),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    ls.paint_uniform_color([1.0, 0.85, 0.0])  # yellow — tentative, vs. magenta for a confirmed match
    mat = line_material(6.0)
    try:
        mat.depth_func = "always"
    except AttributeError:
        pass
    scene_widget.scene.add_geometry(LER_SUGGEST_HIGHLIGHT_GEOM, ls, mat)


def _show_current_suggestion():
    cands = _suggestion_state["candidates"]
    i = _suggestion_state["idx"]
    if not cands:
        _suggest_lbl.text = "No nearby LER candidates found"
        _suggest_lbl.text_color = gui.Color(0.9, 0.5, 0.4, 1.0)
        _clear_suggestion_highlight()
        return
    c = cands[i]
    parts_str = ", ".join(f"{k}={v:.2f}" for k, v in c["breakdown"].items())
    _suggest_lbl.text = (f"Suggestion {i + 1}/{len(cands)}: {layer_display_name(c['layer'])}\n"
                        f"score={c['score']:.2f}  ({parts_str})")
    _suggest_lbl.text_color = gui.Color(1.0, 0.85, 0.2, 1.0)
    _place_suggestion_highlight(*c["rep_segment"])
    window.post_redraw()


def _suggest_ler_match():
    if not instance_data:
        return
    idx = _current_inst_idx[0]
    inst = instance_data[idx]
    pts = np.asarray(inst["pcd"].points)
    colors = np.asarray(inst["pcd"].colors) if inst["pcd"].has_colors() else None

    allowed_layers = None
    label_name = _instance_labels.get(idx)
    if label_name:
        match_cfg = UTILITY_TO_LER_MATCH.get(_LABEL_TO_ID.get(label_name))
        if match_cfg:
            allowed_layers = match_cfg["layers"]

    candidates = score_candidates(pts, colors, _ler_feature_index, allowed_layers=allowed_layers)
    _suggestion_state["candidates"] = candidates
    _suggestion_state["idx"] = 0
    _show_current_suggestion()


def _next_suggestion():
    cands = _suggestion_state["candidates"]
    if not cands:
        return
    _suggestion_state["idx"] = (_suggestion_state["idx"] + 1) % len(cands)
    _show_current_suggestion()


def _accept_suggestion():
    cands = _suggestion_state["candidates"]
    if not cands:
        return
    c = cands[_suggestion_state["idx"]]
    idx = _current_inst_idx[0]
    _instance_ler_match[idx] = {"layer": c["layer"], "gml_id": c["gml_id"]}
    print(f"  [ler-match] Instance {idx} ({instance_data[idx]['name']}) "
          f"-> {c['layer']}  gml_id={c['gml_id']}  (accepted suggestion, score={c['score']:.2f})")
    _maybe_autolabel_from_layer(idx, c["layer"])
    _refresh_ler_match_label(idx)
    _place_ler_match_highlight(*c["rep_segment"])
    _clear_suggestion_highlight()
    _write_ler_matches_json()
    window.post_redraw()


_ler_last_click = [None]


def _do_pick_ler(depth_image):
    if _ler_last_click[0] is None or not instance_data:
        return
    ex, ey = _ler_last_click[0]
    _ler_last_click[0] = None

    sx = int(ex - scene_widget.frame.x)
    sy = int(ey - scene_widget.frame.y)
    depth_arr = np.asarray(depth_image)
    h, w = depth_arr.shape[:2]
    px = int(np.clip(sx, 0, w - 1))
    py = int(np.clip(sy, 0, h - 1))
    depth = float(depth_arr[py, px])
    if depth >= 1.0 or len(pick_seg_p1) == 0:
        return

    world = scene_widget.scene.camera.unproject(
        sx, sy, depth, scene_widget.frame.width, scene_widget.frame.height,
    )
    hit = np.array(world[:3], dtype=float)

    seg_dists = point_to_segment_dists(hit, pick_seg_p1, pick_seg_p2)
    for _si, _sl in enumerate(pick_seg_layer):
        if not _layer_visible.get(_sl, True) or not ler_utilities_visible[0]:
            seg_dists[_si] = np.inf
    best_i = int(np.argmin(seg_dists))
    best_d = float(seg_dists[best_i])
    if best_d > LER_MATCH_PICK_RADIUS:
        return

    idx = _current_inst_idx[0]
    layer_name = pick_seg_layer[best_i]
    gml_id = pick_seg_gml_id[best_i]
    _instance_ler_match[idx] = {"layer": layer_name, "gml_id": gml_id}
    print(f"  [ler-match] Instance {idx} ({instance_data[idx]['name']}) "
          f"-> {layer_name}  gml_id={gml_id}")

    def _update():
        _maybe_autolabel_from_layer(idx, layer_name)
        _place_ler_match_highlight(pick_seg_p1[best_i], pick_seg_p2[best_i])
        _clear_suggestion_highlight()
        _refresh_ler_match_label(idx)
        _write_ler_matches_json()
        window.post_redraw()
    gui.Application.instance.post_to_main_thread(window, _update)


# Distinguish a genuine click (pick) from a drag-to-orbit, same approach as
# base_module.py's segment picking.
_LER_DRAG_THRESHOLD = 8  # pixels
_ler_mouse_down_pos = [None]
_ler_mouse_moved = [False]
_ler_left_was_down = [False]


def _on_mouse_ler(event):
    if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
        if int(event.buttons) & int(gui.MouseButton.LEFT):
            _ler_mouse_down_pos[0] = (event.x, event.y)
            _ler_mouse_moved[0] = False
            _ler_left_was_down[0] = True
        return gui.Widget.EventCallbackResult.IGNORED

    if event.type == gui.MouseEvent.Type.MOVE:
        if _ler_left_was_down[0] and _ler_mouse_down_pos[0] is not None:
            dx = event.x - _ler_mouse_down_pos[0][0]
            dy = event.y - _ler_mouse_down_pos[0][1]
            if (dx * dx + dy * dy) > _LER_DRAG_THRESHOLD * _LER_DRAG_THRESHOLD:
                _ler_mouse_moved[0] = True
        return gui.Widget.EventCallbackResult.IGNORED

    if event.type == gui.MouseEvent.Type.BUTTON_UP:
        if not _ler_left_was_down[0]:
            return gui.Widget.EventCallbackResult.IGNORED
        _ler_left_was_down[0] = False
        if _ler_mouse_moved[0] or _ler_mouse_down_pos[0] is None:
            _ler_mouse_down_pos[0] = None
            return gui.Widget.EventCallbackResult.IGNORED

        click_pos = _ler_mouse_down_pos[0]
        _ler_mouse_down_pos[0] = None
        _ler_last_click[0] = click_pos
        scene_widget.scene.scene.render_to_depth_image(_do_pick_ler)
        # HANDLED so Open3D does not also pan/translate the view on this click
        return gui.Widget.EventCallbackResult.HANDLED

    return gui.Widget.EventCallbackResult.IGNORED


if instance_data:
    scene_widget.set_on_mouse(_on_mouse_ler)


# ─────────────────────────────────────────────────────────────────────────────
# 12.  Camera helpers
# ─────────────────────────────────────────────────────────────────────────────
def _pivot_to(point: np.ndarray):
    d = max(1.0, np.linalg.norm(pc_max - pc_min) * 0.6)
    pivot_top_down(scene_widget, point, d)
    print(f"  Pivot -> [{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}]")


_trench_path = load_trench(_ply_path)


def _top_view():
    """Bird's-eye view looking straight down, framed on the trench footprint
    when one is defined, otherwise on the whole scene."""
    top_view(scene_widget, *trench_or_scene_frame(_trench_path, cloud_centroid,
                                                  pc_min, pc_max))

# ─────────────────────────────────────────────────────────────────────────────
# 13.  Key callbacks
# ─────────────────────────────────────────────────────────────────────────────
HANDLED = gui.Widget.EventCallbackResult.HANDLED
IGNORED = gui.Widget.EventCallbackResult.IGNORED


def on_key(event):
    if event.type != gui.KeyEvent.DOWN:
        return IGNORED
    k = event.key

    # Number keys 1-9 and 0 (=10) for quick instance labeling
    if instance_data:
        _num_keys = {ord(str(i)): i - 1 for i in range(1, 10)}
        _num_keys[ord('0')] = 9
        if k in _num_keys:
            li = _num_keys[k]
            if li < len(INSTANCE_LABEL_OPTIONS):
                _assign_label(INSTANCE_LABEL_OPTIONS[li])
                return HANDLED

    if k == ord(']'):
        _apply_opacity(pipe_opacity[0] + 0.05); return HANDLED
    if k == ord('['):
        _apply_opacity(pipe_opacity[0] - 0.05); return HANDLED

    if k in (ord('C'), ord('c')):
        print("Pivot -> cloud centroid")
        _pivot_to(cloud_centroid)
        return HANDLED
    if k in (ord('P'), ord('p')):
        print("Pivot -> pipe centroid")
        _pivot_to(pipe_centroid)
        return HANDLED
    if k in (ord('T'), ord('t')):
        print("Top view of trench")
        _top_view()
        return HANDLED

    if k in (ord('H'), ord('h')):
        print("\n-- Shortcuts ---------------------------------------------------")
        print("  1-0            assign label to current instance (1-10)")
        print("  Left-click     link a utility line to the current instance")
        print("                 (exclusive LER match, see left panel)")
        print("                 or use 'Mark as NOT in LER' if it isn't registered")
        print("                 or try 'Suggest LER match' for a proposed link")
        print("  C              pivot to point cloud centroid")
        print("  P              pivot to pipe centroid (all utilities)")
        print("  T              top view of trench (or scene if none)")
        print("  ]              increase opacity +0.05")
        print("  [              decrease opacity -0.05")
        print("  H              show this help")
        print("----------------------------------------------------------------\n")
        return HANDLED

    return IGNORED


scene_widget.set_on_key(on_key)

# ─────────────────────────────────────────────────────────────────────────────
# 14.  Layout + run
# ─────────────────────────────────────────────────────────────────────────────
def on_layout(layout_ctx):
    r = window.content_rect
    _lw = LEFT_PANEL_WIDTH if instance_data else 0
    left_panel.frame   = gui.Rect(r.x, r.y, _lw, r.height)
    scene_widget.frame = gui.Rect(r.x + _lw, r.y, r.width - _lw - PANEL_WIDTH, r.height)
    panel.frame        = gui.Rect(r.x + r.width - PANEL_WIDTH, r.y, PANEL_WIDTH, r.height)


window.set_on_layout(on_layout)
if instance_data:
    window.add_child(left_panel)
window.add_child(scene_widget)
window.add_child(panel)

# Summary
n_total_segs  = sum(s for _, s in layer_stats.values())
n_total_comps = sum(comp_stats.values())
print(f"\nRendering {len(pts):,} points  +  {n_total_segs:,} pipe segments  "
      f"+  {n_total_comps} component spheres")
print("Launching viewer ...\n")

# Reflect any resumed session in the opening view: colour instances that were
# already labelled, display the instance the resume logic selected (first
# unlabelled), and show the completion message when everything is labelled.
if instance_data:
    for _lidx in _instance_labels:
        _apply_instance_color(_lidx)
    _show_instance(_current_inst_idx[0])
    _check_all_labeled()
_refresh_window_title()

app.run()
print("Viewer closed.")
