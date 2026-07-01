# -*- coding: utf-8 -*-
"""
Single Point Cloud Viewer with Instance Labels + Surrounding Utilities
======================================================================
Refactored to use core/ for shared configuration and data loading.

Usage: python viewers/label_viewer.py
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
import glob as _globmod
from datetime import datetime

from core.config import (
    PLY_FILE, GML_PATH, AREA_REF_GEOJSON, CROP_RADIUS, CROP_MODE, UTILITY_RECT_BUFFER,
    CLASS_LABELS, DEFAULT_CLASS_COLOR,
    LINE_LAYERS, COMPONENT_LAYERS, COMP_TO_LINE,
    COMPONENT_SPHERE_RADIUS, PIPE_LEGEND_UI_ORDER,
    INSTANCE_COLORS, INSTANCE_LABEL_OPTIONS,
    TARGET_CLASS,
    forsyningsart_color,
)
from core.data_loader import init_site, discover_instances, pick_ground_level, load_trench
from core.gui_helpers import make_legend_row, make_master_pipe_toggle, make_master_comp_toggle
from core.geometry import (
    segment_to_plane,
    segments_in_rect, point_in_rect, clip_segment_to_rect,
)
from core.ledningstrace import get_ledningstrace_display_info, get_storage_key, get_bredde_width
from core.rendering import (
    point_material_shaded, point_material_flat, mesh_material, line_material,
    flat_material, setup_scene_lighting,
)

# ─────────────────────────────────────────────────────────────────────────────
# INITIALISE — load area offset, point cloud, GML, and instances via core/
# ─────────────────────────────────────────────────────────────────────────────
site = init_site(load_instances=True)

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
_current_inst_idx = [0]

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Pick ground-level points (shared function from core/)
# ─────────────────────────────────────────────────────────────────────────────
GROUND_Z = pick_ground_level(site.pc)
print(f"  Ground level (UTM)   = {GROUND_Z + TZ:.3f} m")

_ground_normal = np.array([0.0, 0.0, 1.0])
_ground_center = np.array([_crop_cx_local, _crop_cy_local, GROUND_Z])


def _ground_z_at(x_local, y_local):
    """Return ground Z at a local XY position (flat plane)."""
    return GROUND_Z

# Depth estimation counters
_depth_stats = {"estimated": 0, "fallback_feature_mean": 0, "fallback_global": 0}

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def segment_to_cylinder(p1, p2, radius, color, resolution=12): # define a function to convert a segment to a cylinder
    vec    = p2 - p1 # calculate the vector between the two points
    length = np.linalg.norm(vec)
    if length < 1e-6: # if the length is less than 1e-6, return None
        return None

    cyl = o3d.geometry.TriangleMesh.create_cylinder( 
        radius=radius, height=length, resolution=resolution, split=1) # create a cylinder   
    z_axis    = np.array([0.0, 0.0, 1.0]) # define the z-axis
    direction = vec / length # calculate the direction of the vector
    cross     = np.cross(z_axis, direction) # calculate the cross product of the z-axis and the direction
    cross_norm = np.linalg.norm(cross) # calculate the norm of the cross product
    dot        = np.dot(z_axis, direction) # calculate the dot product of the z-axis and the direction

    if cross_norm > 1e-6:
        axis  = cross / cross_norm # calculate the axis of the cylinder
        angle = np.arctan2(cross_norm, dot) # calculate the angle of the cylinder
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle) # calculate the rotation matrix
        cyl.rotate(R, center=[0.0, 0.0, 0.0]) # rotate the cylinder
    elif dot < 0:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([1.0, 0.0, 0.0]) * np.pi # define the rotation matrix for a 180 degree rotation around the x-axis
        )
        cyl.rotate(R, center=[0.0, 0.0, 0.0]) # rotate the cylinder

    cyl.translate((p1 + p2) / 2.0) # translate the cylinder to the midpoint of the segment
    cyl.paint_uniform_color(color) # paint the cylinder with the color
    return cyl # return the cylinder


def _batch_point_to_segment_dists(p, p1s, p2s): # define a function to calculate the minimum distance from a point to a segment
    """ 
    Vectorised minimum distances from point p to each segment p1s[i]->p2s[i].
    p   : (3,)
    p1s : (N, 3)
    p2s : (N, 3)
    Returns dists : (N,) # return the distances
    """
    d     = p2s - p1s                        # (N, 3) # calculate the vector between the two points
    denom = np.einsum('ij,ij->i', d, d)      # (N,)  squared lengths # calculate the squared lengths of the vector
    v     = p - p1s                          # (N, 3) # calculate the vector from the point to the segment
    t     = np.einsum('ij,ij->i', v, d)     # (N,)  dot products # calculate the dot products of the vector and the segment
    # avoid divide-by-zero for degenerate segments
    safe  = denom > 1e-12 # avoid divide-by-zero for degenerate segments
    t_clamped        = np.where(safe, np.clip(t / np.where(safe, denom, 1.0), 0.0, 1.0), 0.0) # calculate the clamped values
    closest = p1s + t_clamped[:, None] * d       # (N, 3) nearest points on segments
    diff    = p - closest                         # (N, 3)
    dists   = np.sqrt(np.einsum('ij,ij->i', diff, diff))  # (N,)
    return dists


def _clean_coords_with_depth(coords_raw, vejledende_dybde_mm): # define a function to clean the coordinates with depth
    """
    Translate UTM -> local.  For vertices with Z = -99 (no reliable
    measurement), estimate depth using:
        Z = ground_level(XY) - vejledendeDybde / 1000
    If vejledendeDybde is not available, fall back to mean of valid Z
    on the same feature, or the global ground level.
    """
    coords = coords_raw.copy().astype(float) # convert the coordinates to float if the shape of the coordinates is 2, add a zero column to the coordinates
    if coords.shape[1] == 2:
        coords = np.hstack([coords, np.zeros((len(coords), 1))]) # add a zero column to the coordinates

    # Translate XY to local first (Z stays in absolute UTM for now)
    coords[:, 0] -= TX # subtract the translation from the x-coordinates
    coords[:, 1] -= TY # subtract the translation from the y-coordinates

    bad = coords[:, 2] == -99 # check if the z-coordinates are -99
    if bad.any():
        # Parse indicative depth (mm -> m)
        ind_depth_m = None # initialize the indicative depth
        if vejledende_dybde_mm is not None:
            try:
                d = float(vejledende_dybde_mm) # convert the indicative depth to float
                if d > 0:
                    ind_depth_m = d / 1000.0 # convert the indicative depth to meters
            except (ValueError, TypeError):
                pass

        # Mean of valid Z on this feature (absolute UTM Z)
        good_z = coords[~bad, 2] # get the valid z-coordinates
        feature_mean_z = float(good_z.mean()) if len(good_z) > 0 else None

        for idx in np.where(bad)[0]:
            local_x = coords[idx, 0]
            local_y = coords[idx, 1]
            ground_z_here = _ground_z_at(local_x, local_y)
            if ind_depth_m is not None:
                coords[idx, 2] = (ground_z_here + TZ) - ind_depth_m
                _depth_stats["estimated"] += 1
            elif feature_mean_z is not None:
                coords[idx, 2] = feature_mean_z
                _depth_stats["fallback_feature_mean"] += 1
            else:
                coords[idx, 2] = ground_z_here + TZ
                _depth_stats["fallback_global"] += 1

    # Now translate Z to local
    coords[:, 2] -= TZ
    return coords


def _segments_in_bbox(coords_utm):
    """Conservative check: any part of the polyline within the crop region (UTM)."""
    if CROP_MODE == "rect":
        return segments_in_rect(coords_utm, _rect_min_x_utm, _rect_min_y_utm,
                                _rect_max_x_utm, _rect_max_y_utm)
    dx = coords_utm[:, 0] - _crop_cx_utm
    dy = coords_utm[:, 1] - _crop_cy_utm
    d2 = dx * dx + dy * dy
    if (d2 <= _crop_r2).any():
        return True
    # Fallback AABB overlap — catches segments that cross the disc but
    # have no vertex inside it; the segment clipper makes the final call.
    xs, ys = coords_utm[:, 0], coords_utm[:, 1]
    if xs.max() < _crop_cx_utm - CROP_RADIUS: return False
    if xs.min() > _crop_cx_utm + CROP_RADIUS: return False
    if ys.max() < _crop_cy_utm - CROP_RADIUS: return False
    if ys.min() > _crop_cy_utm + CROP_RADIUS: return False
    return True


def _point_in_bbox(x, y):
    if CROP_MODE == "rect":
        return point_in_rect(x, y, _rect_min_x_utm, _rect_min_y_utm,
                             _rect_max_x_utm, _rect_max_y_utm)
    dx = x - _crop_cx_utm
    dy = y - _crop_cy_utm
    return (dx * dx + dy * dy) <= _crop_r2


def _pt_in_local_bbox(x, y):
    if CROP_MODE == "rect":
        return point_in_rect(x, y, _rect_min_x, _rect_min_y,
                             _rect_max_x, _rect_max_y)
    dx = x - _crop_cx_local
    dy = y - _crop_cy_local
    return (dx * dx + dy * dy) <= _crop_r2


def _clip_segment_to_bbox(p1, p2):
    """
    Clip a 3D line segment (p1 -> p2) to the local circular crop in XY.
    Circle: center = (_crop_cx_local, _crop_cy_local), radius = CROP_RADIUS.
    Returns (clipped_p1, clipped_p2) or None if entirely outside.

    Z is linearly interpolated along the segment parameter, matching how
    the previous Liang-Barsky rectangular clipper handled it.
    """
    if CROP_MODE == "rect":
        return clip_segment_to_rect(p1, p2, _rect_min_x, _rect_min_y,
                                    _rect_max_x, _rect_max_y)
    x1 = p1[0] - _crop_cx_local
    y1 = p1[1] - _crop_cy_local
    x2 = p2[0] - _crop_cx_local
    y2 = p2[1] - _crop_cy_local

    dx = x2 - x1
    dy = y2 - y1
    a  = dx * dx + dy * dy

    if a < 1e-12:
        # Degenerate — segment is a single point
        if x1 * x1 + y1 * y1 <= _crop_r2:
            return p1, p2
        return None

    b = 2.0 * (x1 * dx + y1 * dy)
    c = x1 * x1 + y1 * y1 - _crop_r2
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None

    sq      = np.sqrt(disc)
    t_enter = (-b - sq) / (2.0 * a)
    t_exit  = (-b + sq) / (2.0 * a)

    t0 = max(0.0, t_enter)
    t1 = min(1.0, t_exit)
    if t0 > t1:
        return None

    c1 = p1 + t0 * (p2 - p1)
    c2 = p1 + t1 * (p2 - p1)
    return c1, c2

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

        feature_hit = False
        for sub_geom in sub_geoms:
            coords_raw = np.array(sub_geom.coords, dtype=float)
            if not _segments_in_bbox(coords_raw):
                continue

            coords = _clean_coords_with_depth(coords_raw, vejl_dybde)
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
                    mesh = segment_to_cylinder(clipped[0], clipped[1], radius, color)
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
_pipe_layer_centerlines = {}
for _ln, (p1s, p2s) in _pipe_layer_seg_pts.items():
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

        if g.z == -99 or pt[2] <= -98:
            # Component has no reliable Z — estimate from parent pipe depth
            # or from ground model
            if parent_avg_z is not None:
                # Use average depth of the corresponding utility type
                pt[2] = parent_avg_z
                _comp_depth_stats["from_pipe_avg"] += 1
            else:
                pt[2] = _ground_z_at(pt[0], pt[1])
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


def linear_to_srgb(c: float) -> float:
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


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
scene_widget.scene.set_background([0.10, 0.10, 0.10, 1.0])

# Post-processing (SSAO + tone-mapping) and a top-down sun light for shading.
setup_scene_lighting(scene_widget.scene, post_processing=True)

# Add point cloud
scene_widget.scene.add_geometry(POINT_CLOUD_GEOM, pcd, make_point_material())

# Add per-layer pipe meshes (filled); wireframe is a separate combined overlay
for _ln, _mesh in _pipe_layer_meshes.items():
    _alpha0 = 1.0 if _layer_visible.get(_ln, True) else 0.0
    _add_mesh(scene_widget.scene, _pipe_gn(_ln), _mesh, make_mesh_material(_alpha0))

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
PANEL_WIDTH = int(20 * em)
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

# ── Utility Legend (Ledningspakke_2803288) ───────────────────────────────────
ler_toggle_cb = gui.Checkbox("Ledningspakke_2803288")
ler_toggle_cb.checked = True


def _on_ler_toggle(checked):
    ler_utilities_visible[0] = checked
    for ln in _pipe_layer_meshes:
        if not _layer_visible.get(ln, True):
            continue
        alpha = pipe_opacity[0] if checked else 0.0
        scene_widget.scene.modify_geometry_material(_pipe_gn(ln), make_mesh_material(alpha))
    for ln in _comp_layer_meshes:
        if not _layer_visible.get(ln, True):
            continue
        alpha = pipe_opacity[0] if checked else 0.0
        scene_widget.scene.modify_geometry_material(_comp_gn(ln), make_mesh_material(alpha))
    window.post_redraw()


ler_toggle_cb.set_on_checked(_on_ler_toggle)
panel.add_child(ler_toggle_cb)
panel.add_fixed(int(0.3 * em))


def _make_pipe_toggle(ln):
    def _cb(checked):
        _layer_visible[ln] = checked
        _ler = ler_utilities_visible[0]
        if ln in _pipe_layer_meshes:
            alpha = pipe_opacity[0] if (_ler and checked and not pipe_wireframe_active[0]) else 0.0
            scene_widget.scene.modify_geometry_material(_pipe_gn(ln), make_mesh_material(alpha))
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
_all_pipes_cb = gui.Checkbox("All segments")
_all_pipes_cb.checked = True
_all_pipes_cb.set_on_checked(make_master_pipe_toggle(_pipe_checkboxes, _layer_visible,
                                                      _pipe_layer_meshes, scene_widget,
                                                      _pipe_gn, make_mesh_material,
                                                      pipe_opacity, window))
panel.add_child(_all_pipes_cb)

# Line layers — only show legend entry if the layer produced actual geometry
for layer_name, cfg in LINE_LAYERS.items():
    # Skip Ledningstrace here; we'll handle variants below
    if layer_name == "Ledningstrace":
        continue
    if layer_name not in _pipe_layer_meshes:
        continue
    n_feat, _ = layer_stats.get(layer_name, (0, 0))

    cb = gui.Checkbox(f"{layer_name} ({n_feat})")
    cb.checked = _layer_visible.get(layer_name, True)
    cb.set_on_checked(_make_pipe_toggle(layer_name))
    _pipe_checkboxes.append((layer_name, cb))

    panel.add_child(make_legend_row(cfg["color"], cb, em))

# Ledningstrace variants — create separate entry for each forsyningsart
if _ledningstrace_variants:
    for fa, fa_color in sorted(_ledningstrace_variants.items()):
        variant_key = f"Ledningstrace ({fa})"
        if variant_key not in _pipe_layer_meshes:
            continue
        cb = gui.Checkbox(f"Ledningstrace ({fa})")
        cb.checked = _layer_visible.get(variant_key, True)
        cb.set_on_checked(_make_pipe_toggle(variant_key))
        _pipe_checkboxes.append((variant_key, cb))
        panel.add_child(make_legend_row(fa_color, cb, em))

# "Toggle all components" master checkbox
_all_comps_cb = gui.Checkbox("All components")
_all_comps_cb.checked = False
_all_comps_cb.set_on_checked(make_master_comp_toggle(_comp_checkboxes, _layer_visible,
                                                      _comp_layer_meshes, scene_widget,
                                                      _comp_gn, make_mesh_material,
                                                      pipe_opacity, window))
panel.add_child(_all_comps_cb)

# Component layers — only show legend entry if the layer produced actual geometry
for layer_name, cfg in COMPONENT_LAYERS.items():
    if layer_name not in _comp_layer_meshes:
        continue
    n_comp = comp_stats.get(layer_name, 0)

    cb = gui.Checkbox(f"{layer_name} ({n_comp})")
    cb.checked = False
    _layer_visible[layer_name] = False
    cb.set_on_checked(_make_comp_toggle(layer_name))
    _comp_checkboxes.append((layer_name, cb))

    panel.add_child(make_legend_row(cfg["color"], cb, em))

# ── Utility Opacity ──────────────────────────────────────────────────────────
panel.add_fixed(int(0.8 * em))

opacity_value_label = gui.Label("1.00")
lbl_row = gui.Horiz(int(0.25 * em))
lbl_row.add_child(gui.Label("Utility Opacity"))
lbl_row.add_stretch()
lbl_row.add_child(opacity_value_label)
panel.add_child(lbl_row)

opacity_slider = gui.Slider(gui.Slider.DOUBLE)
opacity_slider.set_limits(0.0, 1.0)
opacity_slider.double_value = 1.0


def _apply_opacity(val: float):
    val = max(0.0, min(1.0, val))
    pipe_opacity[0] = val
    opacity_slider.double_value = val
    opacity_value_label.text    = f"{val:.2f}"
    _ler = ler_utilities_visible[0]

    for ln in _pipe_layer_meshes:
        alpha = val if (_ler and _layer_visible.get(ln, True) and not pipe_wireframe_active[0]) else 0.0
        scene_widget.scene.modify_geometry_material(_pipe_gn(ln), make_mesh_material(alpha))

    for ln in _comp_layer_meshes:
        alpha = val if (_ler and _layer_visible.get(ln, True)) else 0.0
        scene_widget.scene.modify_geometry_material(_comp_gn(ln), make_mesh_material(alpha))

    window.post_redraw()


opacity_slider.set_on_value_changed(lambda val: _apply_opacity(val))
panel.add_child(opacity_slider)
panel.add_fixed(int(0.4 * em))

panel.add_child(gui.Label("Quick set"))
btn_row = gui.Horiz(int(0.2 * em))
for pct in (0, 25, 50, 75, 100):
    btn = gui.Button(f"{pct}%")
    def _make_cb(a):
        def _cb(): _apply_opacity(a)
        return _cb
    btn.set_on_clicked(_make_cb(pct / 100.0))
    btn_row.add_child(btn)
panel.add_child(btn_row)

panel.add_stretch()

# ─────────────────────────────────────────────────────────────────────────────
# 10b.  Left-side Instance Labeling panel
# ─────────────────────────────────────────────────────────────────────────────
LEFT_PANEL_WIDTH = int(18 * em)
left_panel = gui.Vert(int(0.5 * em), gui.Margins(int(em), int(em), int(em), int(em)))

if instance_data:
    _inst_progress_lbl = gui.Label(
        f"Instance 1 / {len(instance_data)}:  {instance_data[0]['name']}"
    )
    _inst_progress_lbl.text_color = gui.Color(1.0, 1.0, 0.3, 1.0)
    left_panel.add_child(_inst_progress_lbl)
    left_panel.add_fixed(int(0.1 * em))

    _inst_pts_lbl = gui.Label(f"  {instance_data[0]['n_pts']:,} points")
    _inst_pts_lbl.text_color = gui.Color(0.7, 0.7, 0.7, 1.0)
    left_panel.add_child(_inst_pts_lbl)
    left_panel.add_fixed(int(0.1 * em))

    _inst_assigned_lbl = gui.Label("")
    _inst_assigned_lbl.text_color = gui.Color(0.3, 1.0, 0.3, 1.0)
    _inst_assigned_lbl.visible = False
    left_panel.add_child(_inst_assigned_lbl)
    left_panel.add_fixed(int(0.4 * em))

    left_panel.add_child(gui.Label("Assign label (or press 1-0):"))
    left_panel.add_fixed(int(0.2 * em))

_label_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_labeled_output_dir = _instance_dir / f"labeled_{_label_stamp}" if instance_data else None
if _labeled_output_dir and not _labeled_output_dir.exists():
    _labeled_output_dir.mkdir(parents=True)


_LABEL_TO_ID = {name: i + 1 for i, name in enumerate(INSTANCE_LABEL_OPTIONS)}


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
    out_path = _labeled_output_dir / fname

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
    elif len(_instance_labels) == len(instance_data):
        _inst_progress_lbl.text = "All instances labeled!"
        _inst_pts_lbl.text = ""
        print("  [done] All instances have been labeled.")
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
        left_panel.add_fixed(int(0.1 * em))

    left_panel.add_fixed(int(0.4 * em))

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
    left_panel.add_fixed(int(0.3 * em))

    _save_info = gui.Label(f"Saves to: {_labeled_output_dir.name}/")
    _save_info.text_color = gui.Color(0.5, 0.5, 0.5, 1.0)
    left_panel.add_child(_save_info)

left_panel.add_stretch()



# ─────────────────────────────────────────────────────────────────────────────
# 12.  Camera helpers
# ─────────────────────────────────────────────────────────────────────────────
def _pivot_to(point: np.ndarray):
    d   = max(1.0, np.linalg.norm(pc_max - pc_min) * 0.6)
    eye = point + np.array([0.0, 0.0, d])
    scene_widget.look_at(point.tolist(), eye.tolist(), [0.0, 1.0, 0.0])
    print(f"  Pivot -> [{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}]")


_trench_path = load_trench(_ply_path)


def _top_view():
    """Bird's-eye view looking straight down, framed on the trench footprint
    when one is defined, otherwise on the whole scene."""
    if _trench_path is not None:
        v = np.asarray(_trench_path.vertices, dtype=float)
        cx, cy = float(v[:, 0].mean()), float(v[:, 1].mean())
        span = max(float(v[:, 0].ptp()), float(v[:, 1].ptp()))
    else:
        cx, cy = float(cloud_centroid[0]), float(cloud_centroid[1])
        span = max(float(pc_max[0] - pc_min[0]), float(pc_max[1] - pc_min[1]))
    cz = float(cloud_centroid[2])
    h = max(1.0, span) * 1.2
    scene_widget.look_at([cx, cy, cz], [cx, cy, cz + h], [0.0, 1.0, 0.0])

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

app.run()
print("Viewer closed.")
