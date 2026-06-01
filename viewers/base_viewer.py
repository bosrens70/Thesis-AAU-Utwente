# -*- coding: utf-8 -*-
"""
Single Point Cloud Viewer with Surrounding Utilities — Indicative Depth
+ Class Label Colour Toggle  +  Left-Click Segment Picking
========================================================================
Refactored to use core/ for shared configuration and data loading.

Usage: python viewers/base_viewer.py
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
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
import geopandas as gpd
import numpy as np
import re
import time
import copy
from core.config import (
    PLY_FILE, GML_PATH, AREA_REF_GEOJSON, CROP_RADIUS,
    CLASS_LABELS, DEFAULT_CLASS_COLOR,
    LINE_LAYERS, COMPONENT_LAYERS, COMP_TO_LINE,
    COMPONENT_SPHERE_RADIUS, PIPE_LEGEND_UI_ORDER,
    DepthSource, DepthConfig,
    PIPE_DEPTH_CONFIG, COMPONENT_DEPTH_CONFIG, DEPTH_STATS_KEY as _STATS_KEY,
)
from core.data_loader import init_site, pick_ground_level

# ─────────────────────────────────────────────────────────────────────────────
# INITIALISE — load area offset, point cloud, and GML via core/
# ─────────────────────────────────────────────────────────────────────────────
site = init_site(load_instances=False)

_t_script_start = time.perf_counter()

# Unpack area info
TX, TY, TZ = site.area.TX, site.area.TY, site.area.TZ
AREA_NUMBER = site.area.area_number
AREA_NAME   = site.area.area_name

# Unpack point cloud data
pcd             = site.pc.pcd
pts             = site.pc.pts
original_colors = site.pc.original_colors
class_labels    = site.pc.class_labels
class_colors    = site.pc.class_colors
cloud_centroid  = site.pc.cloud_centroid
cloud_centroid_full = site.pc.cloud_centroid_full
pc_min          = site.pc.pc_min
pc_max          = site.pc.pc_max

_crop_cx_local = site.pc.crop_center_local[0]
_crop_cy_local = site.pc.crop_center_local[1]
_crop_cx_utm   = site.pc.crop_center_utm[0]
_crop_cy_utm   = site.pc.crop_center_utm[1]
_crop_r2       = CROP_RADIUS * CROP_RADIUS

_ply_path = Path(PLY_FILE)

# Alias for backward compat
_DEFAULT_CLASS_COLOR = DEFAULT_CLASS_COLOR

# ─────────────────────────────────────────────────────────────────────────────
# VIEWER-SPECIFIC CODE BELOW (ground picking, mesh creation, GUI)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 3.  Pick ground-level points (shared function from core/)
# ─────────────────────────────────────────────────────────────────────────────
GROUND_Z = pick_ground_level(site.pc)
_pick_method = site.pc.ground_z_method
print(f"  Ground level (UTM)   = {GROUND_Z + TZ:.3f} m")

# Flat ground plane (a*x + b*y + c) — within a 2 m crop radius the tilt is negligible.
_ground_a, _ground_b, _ground_c = 0.0, 0.0, GROUND_Z


def _ground_level_local(x: float, y: float) -> float:
    """Evaluate fitted (or flat fallback) ground plane in local coordinates."""
    return (_ground_a * float(x)) + (_ground_b * float(y)) + _ground_c

# Depth estimation counters (incremented only inside crop disc XY)
_depth_stats = {"registered": 0, "estimated": 0, "fallback_feature_mean": 0, "fallback_layer_mean": 0, "fallback_global": 0}

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


def segment_to_plane(p1, p2, width, color):
    """Create a flat horizontal quad between p1 and p2 with the given width (metres)."""
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return None
    fwd = vec / length
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(fwd, up)
    side_len = np.linalg.norm(side)
    if side_len < 1e-6:
        side = np.array([1.0, 0.0, 0.0])
    else:
        side = side / side_len
    half_w = width / 2.0
    offset = side * half_w
    v0 = p1 - offset
    v1 = p1 + offset
    v2 = p2 + offset
    v3 = p2 - offset
    verts = np.array([v0, v1, v2, v3], dtype=float)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(tris),
    )
    mesh.paint_uniform_color(color)
    return mesh


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


def _point_in_bbox(x, y):
    dx = x - _crop_cx_utm
    dy = y - _crop_cy_utm
    return (dx * dx + dy * dy) <= _crop_r2


def _pt_in_local_bbox(x, y):
    dx = x - _crop_cx_local
    dy = y - _crop_cy_local
    return (dx * dx + dy * dy) <= _crop_r2


def _clean_coords_with_depth(coords_raw, vejledende_dybde_mm,
                              cfg=PIPE_DEPTH_CONFIG, parent_avg_z=None):
    """
    Translate UTM -> local.  For vertices with Z = -99, resolve depth using
    the ordered DepthSource hierarchy defined in *cfg*.

    Returns (coords, sources) where sources is a DepthSource int8 array
    (one entry per vertex) when cfg.track_per_vertex is True, else just coords.
    """
    coords = coords_raw.copy().astype(float)
    if coords.shape[1] == 2:
        coords = np.hstack([coords, np.zeros((len(coords), 1))])

    coords[:, 0] -= TX
    coords[:, 1] -= TY

    n = len(coords)
    sources = np.full(n, DepthSource.NONE, dtype=np.int8)

    bad = coords[:, 2] == -99
    sources[~bad] = DepthSource.REGISTERED

    # Count registered vertices
    _depth_stats["registered"] += int((~bad).sum())

    if bad.any():
        # Pre-compute resolver inputs once per feature
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

        # Resolver table: level -> callable(idx) -> float | None
        def _resolve_vejledende(idx):
            if ind_depth_m is None:
                return None
            g = _ground_level_local(coords[idx, 0], coords[idx, 1])
            return (g + TZ) - ind_depth_m

        def _resolve_feature_mean(idx):
            return feature_mean_z

        def _resolve_layer_mean(idx):
            # parent_avg_z is in local coords; convert to absolute UTM
            # so the final coords[:, 2] -= TZ brings it back to local
            if parent_avg_z is None:
                return None
            return parent_avg_z + TZ

        def _resolve_ground_plane(idx):
            g = _ground_level_local(coords[idx, 0], coords[idx, 1])
            return g + TZ

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
                    _depth_stats[_STATS_KEY[level]] += 1
                    break

    # Translate Z to local
    coords[:, 2] -= TZ

    if cfg.track_per_vertex:
        return coords, sources
    return coords


def _segments_in_bbox(coords_utm):
    """Conservative check: any part of the polyline within the circular crop (UTM)."""
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


def _clip_segment_to_bbox(p1, p2):
    """
    Clip a 3D line segment (p1 -> p2) to the local circular crop in XY.
    Circle: center = (_crop_cx_local, _crop_cy_local), radius = CROP_RADIUS.
    Returns (clipped_p1, clipped_p2) or None if entirely outside.

    Z is linearly interpolated along the segment parameter, matching how
    the previous Liang-Barsky rectangular clipper handled it.
    """
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
_t_pipes0 = time.perf_counter()
all_pipe_meshes   = []          # flat list — kept for count reporting
_pipe_layer_cyls  = {}          # layer_name -> [TriangleMesh, ...]  per-layer
_pipe_seg_dsrc    = {}          # layer_name -> [DepthSource, ...]   per-segment
layer_stats = {}
all_pipe_coords  = []
all_pipe_sources = []   # per-vertex DepthSource arrays, parallel to all_pipe_coords

# Picking data — segment endpoints, midpoints, and their GML attributes
pick_seg_p1        = []   # list of np.array([x,y,z])  — segment start
pick_seg_p2        = []   # list of np.array([x,y,z])  — segment end
pick_seg_midpoints = []   # list of np.array([x,y,z])  — for highlight placement
pick_seg_attrs     = []   # list of [(label, value), ...]
pick_seg_layer     = []   # layer name per segment

# Store per-utility-type average depth for component fallback
_layer_avg_depth_local = {}

for layer_name, cfg in LINE_LAYERS.items():
    _t_layer0 = time.perf_counter()
    try:
        gdf = gpd.read_file(GML_PATH, layer=layer_name)
    except Exception as e:
        print(f"  {layer_name}: skip ({e})")
        continue
    _t_layer_read = time.perf_counter()

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

        # Ledningstrace: read 'bredde' (width in mm) for flat plane rendering
        bredde_m = None
        if layer_name == "Ledningstrace":
            bredde_m = 0.25  # fallback: 25 cm
            if "bredde" in row.index:
                try:
                    b = float(row["bredde"] or 0)
                    if b > 0:
                        bredde_m = b / 1000.0
                except (ValueError, TypeError):
                    pass

        color = default_color

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

            coords, seg_sources = _clean_coords_with_depth(coords_raw, vejl_dybde)
            all_pipe_coords.append(coords)
            all_pipe_sources.append(seg_sources)
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
                    all_pipe_meshes.append(cyl)
                    _pipe_layer_cyls.setdefault(layer_name, []).append(cyl)
                    # Store dominant (worst) depth source for this segment
                    _seg_src = DepthSource(max(int(seg_sources[i]), int(seg_sources[i + 1])))
                    _pipe_seg_dsrc.setdefault(layer_name, []).append(_seg_src)
                    midpt = (clipped[0] + clipped[1]) / 2.0
                    pick_seg_p1.append(clipped[0].copy())
                    pick_seg_p2.append(clipped[1].copy())
                    pick_seg_midpoints.append(midpt)
                    pick_seg_attrs.append(row_attrs)
                    pick_seg_layer.append(layer_name)
                    n_segments += 1

        if feature_hit:
            n_features += 1

    _t_layer1 = time.perf_counter()
    layer_stats[layer_name] = (n_features, n_segments)
    if _layer_z_vals:
        _layer_avg_depth_local[layer_name] = float(np.mean(_layer_z_vals))
    if n_features > 0:
        print(f"  {layer_name:<35} {n_features:>4} features  {n_segments:>5} segments"
              f"  [read {_t_layer_read - _t_layer0:.2f}s | process {_t_layer1 - _t_layer_read:.2f}s]")

pick_seg_p1        = np.array(pick_seg_p1)        if pick_seg_p1        else np.empty((0, 3))
pick_seg_p2        = np.array(pick_seg_p2)        if pick_seg_p2        else np.empty((0, 3))
pick_seg_midpoints = np.array(pick_seg_midpoints) if pick_seg_midpoints else np.empty((0, 3))

_t_pipes1 = time.perf_counter()
print(f"\n  Total: {len(all_pipe_meshes):,} cylinder segments  [{_t_pipes1 - _t_pipes0:.2f}s total]")
print(f"\n  Depth hierarchy stats (all pipe vertices):")
print(f"    1. Registered Z:        {_depth_stats['registered']}")
print(f"    2. vejledendeDybde:      {_depth_stats['estimated']}")
print(f"    3. Feature mean Z:       {_depth_stats['fallback_feature_mean']}")
print(f"    4. Layer mean Z:         {_depth_stats['fallback_layer_mean']}")
print(f"    5. Ground plane:         {_depth_stats['fallback_global']}")

# Consolidated per-vertex depth source arrays
if all_pipe_sources:
    _all_pipe_sources_flat = np.concatenate(all_pipe_sources)
    for src in DepthSource:
        if src == DepthSource.NONE:
            continue
        _cnt = int(np.sum(_all_pipe_sources_flat == src))
        if _cnt > 0:
            print(f"    [{src.name:<14}] {_cnt:>6} vertices (all, incl. outside crop)")
    del _all_pipe_sources_flat

# Per-layer merged pipe meshes (used for individual visibility toggles)
_pipe_layer_meshes = {}
for _ln, _cyls in _pipe_layer_cyls.items():
    _m = _cyls[0]
    for _c in _cyls[1:]:
        _m += _c
    _m.compute_vertex_normals()
    _pipe_layer_meshes[_ln] = _m

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
}

print("\n--- Loading utility components within bbox ---")
_t_comp0 = time.perf_counter()
all_comp_meshes    = []     # flat list (kept for count reporting)
_comp_layer_spheres = {}    # layer_name -> [TriangleMesh, ...]  per-layer
_comp_seg_dsrc     = {}     # layer_name -> [DepthSource, ...]   per-component
comp_stats = {}
_comp_depth_stats = {"from_pipe_avg": 0, "from_ground": 0}
all_comp_sources = []       # per-component DepthSource values

# Picking data for components
pick_comp_centres = []
pick_comp_attrs   = []
pick_comp_layer   = []

for layer_name, comp_cfg in COMPONENT_LAYERS.items():
    _t_clayer0 = time.perf_counter()
    try:
        gdf_c = gpd.read_file(GML_PATH, layer=layer_name)
    except Exception:
        continue
    _t_clayer_read = time.perf_counter()

    color = comp_cfg["color"]
    n_comp = 0

    # Get the average depth of the corresponding line layer for fallback
    parent_line = _COMP_TO_LINE.get(layer_name)
    parent_avg_z = _layer_avg_depth_local.get(parent_line) if parent_line else None

    for _, row in gdf_c.iterrows():
        g = row.geometry
        if g is None:
            continue
        if g.geom_type not in ("Point", "PointZ"):
            continue
        if not _point_in_bbox(g.x, g.y):
            continue

        # Use the unified resolver via _clean_coords_with_depth
        coords_utm = np.array([[g.x, g.y, g.z]], dtype=float)
        pt_arr, src_arr = _clean_coords_with_depth(
            coords_utm, None,
            cfg=COMPONENT_DEPTH_CONFIG, parent_avg_z=parent_avg_z,
        )
        pt = pt_arr[0]
        comp_source = DepthSource(int(src_arr[0]))

        if not _pt_in_local_bbox(pt[0], pt[1]):
            continue

        all_comp_sources.append(comp_source)

        # Legacy counters for backward-compatible print output
        if comp_source == DepthSource.LAYER_MEAN:
            _comp_depth_stats["from_pipe_avg"] += 1
        elif comp_source == DepthSource.GROUND_PLANE:
            _comp_depth_stats["from_ground"] += 1

        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=COMPONENT_SPHERE_RADIUS, resolution=12
        )
        sphere.translate(pt)
        sphere.paint_uniform_color(color)
        all_comp_meshes.append(sphere)
        _comp_layer_spheres.setdefault(layer_name, []).append(sphere)
        _comp_seg_dsrc.setdefault(layer_name, []).append(comp_source)

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

    _t_clayer1 = time.perf_counter()
    comp_stats[layer_name] = n_comp
    if n_comp > 0:
        print(f"  {layer_name:<35} {n_comp:>4} components"
              f"  [read {_t_clayer_read - _t_clayer0:.2f}s | process {_t_clayer1 - _t_clayer_read:.2f}s]")

pick_comp_centres = np.array(pick_comp_centres) if pick_comp_centres else np.empty((0, 3))

_t_comp1 = time.perf_counter()
print(f"\n  Total: {len(all_comp_meshes)} component spheres  [{_t_comp1 - _t_comp0:.2f}s total]")
print(f"  Component depth estimation:")
print(f"    From parent pipe average Z: {_comp_depth_stats['from_pipe_avg']}")
print(f"    From ground model:          {_comp_depth_stats['from_ground']}")
if all_comp_sources:
    for src in DepthSource:
        if src == DepthSource.NONE:
            continue
        _cnt = sum(1 for s in all_comp_sources if s == src)
        if _cnt > 0:
            print(f"    [{src.name:<14}] {_cnt:>6} components")

# Per-layer merged component meshes
_comp_layer_meshes = {}
for _ln, _spheres in _comp_layer_spheres.items():
    _m = _spheres[0]
    for _s in _spheres[1:]:
        _m += _s
    _m.compute_vertex_normals()
    _comp_layer_meshes[_ln] = _m

def srgb_to_linear(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4

# ── Depth-source colour map (sRGB — used directly for GUI labels) ──────────
_DSRC_COLOR_SRGB = {
    DepthSource.REGISTERED:   [0.4, 1.0, 0.4],   # green
    DepthSource.VEJLEDENDE:   [0.4, 0.8, 1.0],   # light blue
    DepthSource.FEATURE_MEAN: [1.0, 0.7, 0.3],   # orange
    DepthSource.LAYER_MEAN:   [1.0, 0.7, 0.3],   # orange
    DepthSource.GROUND_PLANE: [1.0, 0.4, 0.4],   # red
    DepthSource.NONE:         [0.5, 0.5, 0.5],   # grey
}

def _dsrc_linear(src):
    """Convert sRGB depth-source colour to linear for Open3D meshes."""
    s = _DSRC_COLOR_SRGB.get(src, [0.5, 0.5, 0.5])
    return [srgb_to_linear(c) for c in s]

# Build depth-coloured per-layer pipe meshes
_pipe_layer_meshes_depth = {}
for _ln, _cyls in _pipe_layer_cyls.items():
    _dsrcs = _pipe_seg_dsrc.get(_ln, [])
    _coloured = []
    for _i, _c in enumerate(_cyls):
        _dc = copy.deepcopy(_c)
        _src = _dsrcs[_i] if _i < len(_dsrcs) else DepthSource.NONE
        _dc.paint_uniform_color(_dsrc_linear(_src))
        _coloured.append(_dc)
    if _coloured:
        _m = _coloured[0]
        for _c2 in _coloured[1:]:
            _m += _c2
        _m.compute_vertex_normals()
        _pipe_layer_meshes_depth[_ln] = _m

# Build depth-coloured per-layer component meshes
_comp_layer_meshes_depth = {}
for _ln, _spheres in _comp_layer_spheres.items():
    _dsrcs = _comp_seg_dsrc.get(_ln, [])
    _coloured = []
    for _i, _s in enumerate(_spheres):
        _ds = copy.deepcopy(_s)
        _src = _dsrcs[_i] if _i < len(_dsrcs) else DepthSource.NONE
        _ds.paint_uniform_color(_dsrc_linear(_src))
        _coloured.append(_ds)
    if _coloured:
        _m = _coloured[0]
        for _s2 in _coloured[1:]:
            _m += _s2
        _m.compute_vertex_normals()
        _comp_layer_meshes_depth[_ln] = _m

_t_load = time.perf_counter()
print(f"\nAll data loaded in {_t_load - _t_script_start:.2f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Coordinate frame + circular crop wireframe + point cloud normals
# ─────────────────────────────────────────────────────────────────────────────
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5, origin=cloud_centroid
)

# Estimate normals — reduced parameters (only used for defaultLit shading).
_t_norm0 = time.perf_counter()
try:
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=12)
    )
    pcd.orient_normals_towards_camera_location(
        cloud_centroid + np.array([0.0, 0.0, 5.0])
    )
except Exception as _e:
    print(f"  [warn] point cloud normal estimation failed: {_e}")
_t_norm1 = time.perf_counter()
print(f"  [timer] Normal estimation: {_t_norm1 - _t_norm0:.3f}s")

# Circle wireframe showing the crop boundary at ground level
_N_CIRCLE = 72
_theta = np.linspace(0.0, 2.0 * np.pi, _N_CIRCLE + 1)
_circle_x = _crop_cx_local + CROP_RADIUS * np.cos(_theta)
_circle_y = _crop_cy_local + CROP_RADIUS * np.sin(_theta)
bbox_wire_pts = np.stack([
    _circle_x,
    _circle_y,
    np.array([_ground_level_local(x, y) for x, y in zip(_circle_x, _circle_y)]),
], axis=1)
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
    mat            = rendering.MaterialRecord()
    mat.shader     = "defaultLitTransparency"
    mat.base_color = [1.0, 1.0, 1.0, float(alpha)]
    return mat


def make_point_material() -> rendering.MaterialRecord:
    # `defaultLit` + estimated normals + SSAO post-processing is the closest
    # Open3D equivalent to an EDL shader. Points near geometric ridges end
    # up darker, giving a strong depth cue for the class-coloured cloud.
    mat            = rendering.MaterialRecord()
    mat.shader     = "defaultLit"
    mat.point_size = 3.0
    return mat


def make_frame_material() -> rendering.MaterialRecord:
    mat        = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    return mat


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
FRAME_GEOM       = "frame"
BBOX_GEOM        = "bbox_wire"
HIGHLIGHT_GEOM      = "highlight"
HIGHLIGHT_AXES_GEOM = "highlight_axes"

# Per-layer geometry names
def _pipe_gn(ln):       return f"pipe_{ln}"
def _comp_gn(ln):       return f"comp_{ln}"

# Per-layer visibility state (True = shown)
_layer_visible = {ln: True for ln in list(LINE_LAYERS) + list(COMPONENT_LAYERS)}
_layer_visible["Ledningstrace"] = True

pipe_opacity = [1.0]
origin_pt    = np.array([0.0, 0.0, 0.0])
pick_active  = [False]
class_labels_active = [False]   # toggled by L key or checkbox
origin_frame_visible  = [False]  # toggled by the "Show origin axis" checkbox

_t_gui0 = time.perf_counter()
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

# Enable post-processing (SSAO + tone-mapping)
try:
    scene_widget.scene.view.set_post_processing(True)
except Exception as _e:
    print(f"  [warn] could not enable post-processing: {_e}")

# Top-down directional sun light for utility mesh shading
scene_widget.scene.scene.set_sun_light(
    [0.0, 0.0, -1.0],        # direction: straight down
    [1.0, 1.0, 1.0],         # white colour
    75000,                    # intensity
)
scene_widget.scene.scene.enable_sun_light(True)

# Add point cloud
scene_widget.scene.add_geometry(POINT_CLOUD_GEOM, pcd, make_point_material())

# Add per-layer pipe meshes (filled)
for _ln, _mesh in _pipe_layer_meshes.items():
    alpha = pipe_opacity[0] if _layer_visible.get(_ln, True) else 0.0
    _add_mesh(scene_widget.scene, _pipe_gn(_ln), _mesh, make_mesh_material(alpha))

# Add per-layer component meshes (hidden by default)
for _ln, _mesh in _comp_layer_meshes.items():
    _add_mesh(scene_widget.scene, _comp_gn(_ln), _mesh, make_mesh_material(0.0))

# Add frame and bbox wireframe
scene_widget.scene.add_geometry(FRAME_GEOM, frame, make_frame_material())
scene_widget.scene.show_geometry(FRAME_GEOM, False)

line_mat            = rendering.MaterialRecord()
line_mat.shader     = "unlitLine"
line_mat.line_width = 3.0
scene_widget.scene.add_geometry(BBOX_GEOM, bbox_ls, line_mat)

bounds = scene_widget.scene.bounding_box
scene_widget.setup_camera(60, bounds, cloud_centroid.tolist())


# ─────────────────────────────────────────────────────────────────────────────
# 9b. Class label toggle function
# ─────────────────────────────────────────────────────────────────────────────
def _toggle_class_labels(show_labels: bool):
    """Switch point cloud colours between original RGB and class labels."""
    if class_colors is None:
        print("[class toggle] No class labels available in this PLY.")
        return

    class_labels_active[0] = show_labels

    if show_labels:
        pcd.colors = o3d.utility.Vector3dVector(class_colors)
        print("[class toggle] ON  — showing class label colours")
    else:
        pcd.colors = o3d.utility.Vector3dVector(original_colors)
        print("[class toggle] OFF — showing original RGB colours")

    # Update the point cloud in the scene
    scene_widget.scene.remove_geometry(POINT_CLOUD_GEOM)
    scene_widget.scene.add_geometry(POINT_CLOUD_GEOM, pcd, make_point_material())
    window.post_redraw()


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Right-side control panel
# ─────────────────────────────────────────────────────────────────────────────
PANEL_WIDTH = int(20 * em)
panel = gui.Vert(int(0.5 * em), gui.Margins(int(em), int(em), int(em), int(em)))

panel.add_child(gui.Label(f"Points: {len(pts):,}"))
panel.add_child(gui.Label(f"Crop radius: {CROP_RADIUS} m (circular)"))
panel.add_child(gui.Label(f"Ground Z: {GROUND_Z:.3f} m ({_pick_method})"))
panel.add_fixed(int(0.3 * em))

# ── Depth Hierarchy toggle — recolours utilities by depth source ────────────
_depth_hierarchy_active = [False]

depth_toggle_cb = gui.Checkbox("Depth Hierarchy")
depth_toggle_cb.checked = False

def _dsrc_gui_color(src):
    """sRGB depth-source colour as gui.Color (matches viewer appearance)."""
    r, g, b = _DSRC_COLOR_SRGB[src]
    return gui.Color(r, g, b, 1.0)

_hierarchy_display = [
    ("1. Registered Z",       _depth_stats["registered"],            _dsrc_gui_color(DepthSource.REGISTERED)),
    ("2. vejledendeDybde",    _depth_stats["estimated"],             _dsrc_gui_color(DepthSource.VEJLEDENDE)),
    ("3. Feature mean Z",     _depth_stats["fallback_feature_mean"], _dsrc_gui_color(DepthSource.FEATURE_MEAN)),
    ("4. Layer mean Z",       _depth_stats["fallback_layer_mean"],   _dsrc_gui_color(DepthSource.LAYER_MEAN)),
    ("5. Ground plane",       _depth_stats["fallback_global"],       _dsrc_gui_color(DepthSource.GROUND_PLANE)),
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
    for ln in _pipe_layer_meshes:
        mesh = _pipe_layer_meshes_depth[ln] if checked and ln in _pipe_layer_meshes_depth else _pipe_layer_meshes[ln]
        alpha = pipe_opacity[0] if _layer_visible.get(ln, True) else 0.0
        scene_widget.scene.remove_geometry(_pipe_gn(ln))
        _add_mesh(scene_widget.scene, _pipe_gn(ln), mesh, make_mesh_material(alpha))
    for ln in _comp_layer_meshes:
        mesh = _comp_layer_meshes_depth[ln] if checked and ln in _comp_layer_meshes_depth else _comp_layer_meshes[ln]
        alpha = pipe_opacity[0] if _layer_visible.get(ln, True) else 0.0
        scene_widget.scene.remove_geometry(_comp_gn(ln))
        _add_mesh(scene_widget.scene, _comp_gn(ln), mesh, make_mesh_material(alpha))
    window.set_needs_layout()
    window.post_redraw()


depth_toggle_cb.set_on_checked(_on_depth_toggle)
panel.add_child(depth_toggle_cb)
panel.add_child(_depth_legend_container)

panel.add_fixed(int(0.3 * em))

origin_toggle_cb = gui.Checkbox("Show origin axis")
origin_toggle_cb.checked = False

def _on_origin_toggle(checked):
    origin_frame_visible[0] = checked
    scene_widget.scene.show_geometry(FRAME_GEOM, checked)
    window.post_redraw()

origin_toggle_cb.set_on_checked(_on_origin_toggle)
panel.add_child(origin_toggle_cb)

panel.add_fixed(int(0.8 * em))

# ── Class Label Toggle ──────────────────────────────────────────────────────
class_toggle_cb = gui.Checkbox("OpenTrench3D ID Class")
class_toggle_cb.checked = False
if class_colors is None:
    class_toggle_cb.enabled = False

_class_legend_container = gui.Vert(0)
if class_labels is not None:
    for cls_id in sorted(CLASS_LABELS.keys()):
        cfg = CLASS_LABELS[cls_id]
        if cls_id not in np.unique(class_labels):
            continue
        n_pts = int((class_labels == cls_id).sum())
        col = cfg["color"]
        sr, sg, sb = (linear_to_srgb(c) for c in col)

        row     = gui.Horiz(int(0.3 * em))
        swatch  = gui.Button(" ")
        swatch.background_color = gui.Color(sr, sg, sb, 1.0)
        swatch.toggleable = False
        swatch.vertical_padding_em = 0.0
        swatch.horizontal_padding_em = 0.3
        row.add_child(swatch)
        row.add_fixed(int(0.4 * em))
        row.add_child(gui.Label(f"{cls_id}: {cfg['name']} ({n_pts:,})"))
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

# ── Utility Legend (with per-layer visibility toggles) ───────────────────────
_gml_folder = Path(GML_PATH).parent.name
_ler_match = re.match(r"(Ledningspakke)[_\s]*(\d+)", _gml_folder, re.IGNORECASE)
_ler_label = f"{_ler_match.group(1)} {_ler_match.group(2)}" if _ler_match else _gml_folder

_ler_active = [True]
ler_toggle_cb = gui.Checkbox(_ler_label)
ler_toggle_cb.checked = True

_ler_legend_container = gui.Vert(int(0.3 * em))

# Opacity slider (label + slider + value on one row)
opacity_value_label = gui.Label("1.00")
opacity_slider = gui.Slider(gui.Slider.DOUBLE)
opacity_slider.set_limits(0.0, 1.0)
opacity_slider.double_value = 1.0

slider_row = gui.Horiz(int(0.25 * em))
slider_row.add_child(gui.Label("Opacity"))
slider_row.add_child(opacity_slider)


def _apply_opacity(val: float):
    val = max(0.0, min(1.0, val))
    pipe_opacity[0] = val
    opacity_slider.double_value = val
    opacity_value_label.text    = f"{val:.2f}"

    for ln in _pipe_layer_meshes:
        alpha = val if _layer_visible.get(ln, True) else 0.0
        scene_widget.scene.modify_geometry_material(_pipe_gn(ln), make_mesh_material(alpha))

    for ln in _comp_layer_meshes:
        alpha = val if _layer_visible.get(ln, True) else 0.0
        scene_widget.scene.modify_geometry_material(_comp_gn(ln), make_mesh_material(alpha))

    window.post_redraw()


opacity_slider.set_on_value_changed(lambda val: _apply_opacity(val))
_ler_legend_container.add_child(slider_row)

_pipe_checkboxes = []   # (layer_name, checkbox) for "toggle all" control
_comp_checkboxes = []


def _make_pipe_toggle(ln):
    def _cb(checked):
        _layer_visible[ln] = checked
        if ln in _pipe_layer_meshes:
            alpha = pipe_opacity[0] if checked else 0.0
            scene_widget.scene.modify_geometry_material(_pipe_gn(ln), make_mesh_material(alpha))
        window.post_redraw()
    return _cb


def _make_comp_toggle(ln):
    def _cb(checked):
        _layer_visible[ln] = checked
        if ln in _comp_layer_meshes:
            alpha = pipe_opacity[0] if checked else 0.0
            scene_widget.scene.modify_geometry_material(_comp_gn(ln), make_mesh_material(alpha))
        window.post_redraw()
    return _cb


# "Toggle all segments" master checkbox
_all_pipes_cb = gui.Checkbox("All segments")
_all_pipes_cb.checked = True

def _on_toggle_all_pipes(checked):
    for ln, cb in _pipe_checkboxes:
        cb.checked = checked
        _layer_visible[ln] = checked
        if ln in _pipe_layer_meshes:
            alpha = pipe_opacity[0] if checked else 0.0
            scene_widget.scene.modify_geometry_material(_pipe_gn(ln), make_mesh_material(alpha))
    window.post_redraw()

_all_pipes_cb.set_on_checked(_on_toggle_all_pipes)
_ler_legend_container.add_child(_all_pipes_cb)

# Line layers — only show legend entry if the layer produced actual geometry
for layer_name in PIPE_LEGEND_UI_ORDER:
    if layer_name not in _pipe_layer_meshes:
        continue
    cfg = LINE_LAYERS[layer_name]
    n_feat, _ = layer_stats.get(layer_name, (0, 0))
    col = cfg["color"]
    sr, sg, sb = (linear_to_srgb(c) for c in col)
    row    = gui.Horiz(int(0.3 * em))
    swatch = gui.Button(" ")
    swatch.background_color = gui.Color(sr, sg, sb, 1.0)
    swatch.toggleable = False
    swatch.vertical_padding_em = 0.0
    swatch.horizontal_padding_em = 0.3

    cb = gui.Checkbox(f"{layer_name} ({n_feat})")
    cb.checked = _layer_visible.get(layer_name, True)
    cb.set_on_checked(_make_pipe_toggle(layer_name))
    _pipe_checkboxes.append((layer_name, cb))

    row.add_child(swatch)
    row.add_fixed(int(0.4 * em))
    row.add_child(cb)
    _ler_legend_container.add_child(row)

# "Toggle all components" master checkbox
_all_comps_cb = gui.Checkbox("All components")
_all_comps_cb.checked = False

def _on_toggle_all_comps(checked):
    for ln, cb in _comp_checkboxes:
        cb.checked = checked
        _layer_visible[ln] = checked
        if ln in _comp_layer_meshes:
            alpha = pipe_opacity[0] if checked else 0.0
            scene_widget.scene.modify_geometry_material(_comp_gn(ln), make_mesh_material(alpha))
    window.post_redraw()

_all_comps_cb.set_on_checked(_on_toggle_all_comps)
_ler_legend_container.add_child(_all_comps_cb)

# Component layers — only show legend entry if the layer produced actual geometry
for layer_name, cfg in COMPONENT_LAYERS.items():
    if layer_name not in _comp_layer_meshes:
        continue
    n_comp = comp_stats.get(layer_name, 0)
    col = cfg["color"]
    sr, sg, sb = (linear_to_srgb(c) for c in col)

    row    = gui.Horiz(int(0.3 * em))
    swatch = gui.Button(" ")
    swatch.background_color = gui.Color(sr, sg, sb, 1.0)
    swatch.toggleable = False
    swatch.vertical_padding_em = 0.0
    swatch.horizontal_padding_em = 0.3

    cb = gui.Checkbox(f"{layer_name} ({n_comp})")
    cb.checked = False
    _layer_visible[layer_name] = False
    cb.set_on_checked(_make_comp_toggle(layer_name))
    _comp_checkboxes.append((layer_name, cb))

    row.add_child(swatch)
    row.add_fixed(int(0.4 * em))
    row.add_child(cb)
    _ler_legend_container.add_child(row)


def _on_ler_toggle(checked):
    _ler_active[0] = checked
    _ler_legend_container.visible = checked
    # Show/hide all utility geometry
    for ln in _pipe_layer_meshes:
        if checked:
            # Restore per-layer visibility state
            alpha = pipe_opacity[0] if _layer_visible.get(ln, True) else 0.0
        else:
            alpha = 0.0
        scene_widget.scene.modify_geometry_material(_pipe_gn(ln), make_mesh_material(alpha))
    for ln in _comp_layer_meshes:
        if checked:
            alpha = pipe_opacity[0] if _layer_visible.get(ln, True) else 0.0
        else:
            alpha = 0.0
        scene_widget.scene.modify_geometry_material(_comp_gn(ln), make_mesh_material(alpha))
    window.set_needs_layout()
    window.post_redraw()


ler_toggle_cb.set_on_checked(_on_ler_toggle)
panel.add_child(ler_toggle_cb)
panel.add_child(_ler_legend_container)

panel.add_stretch()

# ── Left-side info panel (shown only when a feature is selected) ─────────────
LEFT_PANEL_WIDTH = int(22 * em)
left_panel = gui.Vert(int(0.5 * em), gui.Margins(int(em), int(em), int(em), int(em)))
left_panel.background_color = gui.Color(0.15, 0.15, 0.15, 1.0)

_info_type_lbl            = gui.Label("")
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
            label, value  = attrs[i]
            k_lbl.text    = f"{label}:"
            v_lbl.text    = value
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
    if scene_widget.scene.has_geometry(HIGHLIGHT_AXES_GEOM):
        scene_widget.scene.remove_geometry(HIGHLIGHT_AXES_GEOM)
    pick_active[0] = False


def _place_highlight(centre: np.ndarray):
    _clear_highlight()
    r = 0.15
    # Use a wireframe highlight so the selected component's original colour
    # remains visible instead of being covered by a filled yellow sphere.
    marker_wire_src = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=12)
    marker_wire_src.translate(centre)
    marker_wire = o3d.geometry.LineSet.create_from_triangle_mesh(marker_wire_src)
    marker_wire.paint_uniform_color([1.0, 1.0, 0.0])
    marker_mat            = rendering.MaterialRecord()
    marker_mat.shader     = "unlitLine"
    marker_mat.line_width = 0.5
    scene_widget.scene.add_geometry(HIGHLIGHT_GEOM, marker_wire, marker_mat)

    axes_pts = [
        centre + np.array([-r, 0, 0]), centre + np.array([r, 0, 0]),
        centre + np.array([0, -r, 0]), centre + np.array([0, r, 0]),
        centre + np.array([0, 0, -r]), centre + np.array([0, 0, r]),
        centre,
    ]
    axes_lines = [
        [0, 1], [2, 3], [4, 5],
        [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5],
    ]
    axes_ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(axes_pts)),
        lines=o3d.utility.Vector2iVector(axes_lines),
    )
    axes_ls.paint_uniform_color([1.0, 1.0, 1.0])
    axes_mat            = rendering.MaterialRecord()
    axes_mat.shader     = "unlitLine"
    axes_mat.line_width = 2.0
    scene_widget.scene.add_geometry(HIGHLIGHT_AXES_GEOM, axes_ls, axes_mat)

    pick_active[0] = True
    window.post_redraw()

# ─────────────────────────────────────────────────────────────────────────────
# 11.  Mouse picking  (Left-Click)
# ─────────────────────────────────────────────────────────────────────────────
# Max distance from the unprojected click point to a segment/component (metres).
# Segments use true point-to-segment distance; keep this tight (~cylinder radius
# + a small margin) so misclicks on empty space are rejected cleanly.
PICK_RADIUS_SEG  = 0.30   # m — adjusted for typical pipe cylinder radius
PICK_RADIUS_COMP = 0.20   # m — adjusted for component sphere radius
_last_click = [None]


def _do_pick(depth_image):
    """Callback invoked after the depth buffer has been rendered."""
    if _last_click[0] is None:
        return
    ex, ey = _last_click[0]
    _last_click[0] = None

    sx = int(ex - scene_widget.frame.x)
    sy = int(ey - scene_widget.frame.y)
    depth_arr = np.asarray(depth_image)
    h, w      = depth_arr.shape[:2]
    px        = int(np.clip(sx, 0, w - 1))
    py        = int(np.clip(sy, 0, h - 1))
    depth     = float(depth_arr[py, px])

    if depth >= 1.0:
        # Clicked on background — clear selection and hide left panel
        def _clear():
            _clear_highlight()
            _hide_left_panel()
        gui.Application.instance.post_to_main_thread(window, _clear)
        return

    world = scene_widget.scene.camera.unproject(
        sx, sy, depth,
        scene_widget.frame.width,
        scene_widget.frame.height,
    )
    hit = np.array(world[:3], dtype=float)

    # ── Find nearest pipe segment using true point-to-segment distance ────────
    # Skip segments whose layer is hidden
    best_seg_d = np.inf
    best_seg_i = -1
    if len(pick_seg_p1) > 0:
        seg_dists = _batch_point_to_segment_dists(hit, pick_seg_p1, pick_seg_p2)
        for _si, _sl in enumerate(pick_seg_layer):
            if not _layer_visible.get(_sl, True) or not _ler_active[0]:
                seg_dists[_si] = np.inf
        best_seg_i = int(np.argmin(seg_dists))
        best_seg_d = float(seg_dists[best_seg_i])

    # ── Find nearest component sphere centre ──────────────────────────────────
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

    # ── Pick whichever is closer, within radius thresholds ───────────────────
    if best_comp_d < best_seg_d and best_comp_d < PICK_RADIUS_COMP:
        centre = pick_comp_centres[best_comp_i].copy()
        attrs  = pick_comp_attrs[best_comp_i]
        label  = f"{pick_comp_layer[best_comp_i]} (component)"
        print(f"\n[pick] -> {label}")
        for k, v in attrs:
            print(f"    {k:<30} = {v}")
        print()
    elif best_seg_d < PICK_RADIUS_SEG:
        centre = pick_seg_midpoints[best_seg_i].copy()
        attrs  = pick_seg_attrs[best_seg_i]
        label  = f"{pick_seg_layer[best_seg_i]} (pipe segment)"
        print(f"\n[pick] -> {label}")
        for k, v in attrs:
            print(f"    {k:<30} = {v}")
        print()
    else:
        # Nothing close enough — clear selection and hide left panel
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
# We track state with a flag (_left_was_down) rather than reading event.buttons
# at BUTTON_UP, because Open3D sets event.buttons to the *released* button at
# that point (non-zero), making an "== 0" check always fail.
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
        # Return IGNORED so the scene widget still receives the event for orbit
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
        print(f"[pick] Left-click at {click_pos}")
        _last_click[0] = click_pos
        scene_widget.scene.scene.render_to_depth_image(_do_pick)
        # Return HANDLED so Open3D does not also pan/translate the view
        return gui.Widget.EventCallbackResult.HANDLED

    return gui.Widget.EventCallbackResult.IGNORED


scene_widget.set_on_mouse(on_mouse)

# ─────────────────────────────────────────────────────────────────────────────
# 12.  Camera helpers
# ─────────────────────────────────────────────────────────────────────────────
def _pivot_to(point: np.ndarray):
    d   = max(1.0, np.linalg.norm(pc_max - pc_min) * 0.6)
    eye = point + np.array([d, -d, d * 0.6])
    scene_widget.look_at(point.tolist(), eye.tolist(), [0.0, 0.0, 1.0])
    print(f"  Pivot -> [{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 13.  Key callbacks
# ─────────────────────────────────────────────────────────────────────────────
HANDLED = gui.Widget.EventCallbackResult.HANDLED
IGNORED = gui.Widget.EventCallbackResult.IGNORED


def on_key(event):
    if event.type != gui.KeyEvent.DOWN:
        return IGNORED
    k = event.key

    if k in (ord('L'), ord('l')):
        new_state = not class_labels_active[0]
        class_toggle_cb.checked = new_state
        _on_class_toggle(new_state)
        return HANDLED

    if k in (ord('C'), ord('c')):
        print("Pivot -> cloud centroid")
        _pivot_to(cloud_centroid)
        return HANDLED
    if k in (ord('P'), ord('p')):
        print("Pivot -> pipe centroid")
        _pivot_to(pipe_centroid)
        return HANDLED
    if k == ord('0'):
        print("Pivot -> world origin")
        _pivot_to(origin_pt)
        return HANDLED

    if k in (ord('H'), ord('h')):
        print("\n-- Shortcuts ---------------------------------------------------")
        print("  Left-click     pick pipe segment or component (show attributes)")
        print("  C              pivot to point cloud centroid")
        print("  P              pivot to pipe centroid (all utilities)")
        print("  0              pivot to world origin (0, 0, 0)")
        print("  L              toggle class label colours on/off")
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
    if _left_panel_visible[0]:
        left_w = LEFT_PANEL_WIDTH
        left_panel.frame = gui.Rect(r.x, r.y, left_w, r.height)
    else:
        left_w = 0
        # Move off-screen so nothing is rendered
        left_panel.frame = gui.Rect(-LEFT_PANEL_WIDTH, r.y, 0, r.height)
    scene_widget.frame = gui.Rect(r.x + left_w, r.y, r.width - PANEL_WIDTH - left_w, r.height)
    panel.frame        = gui.Rect(r.x + r.width - PANEL_WIDTH, r.y, PANEL_WIDTH, r.height)


window.set_on_layout(on_layout)
window.add_child(left_panel)
window.add_child(scene_widget)
window.add_child(panel)

# Summary
_t_gui1 = time.perf_counter()
n_total_segs  = sum(s for _, s in layer_stats.values())
n_total_comps = sum(comp_stats.values())
print(f"\nRendering {len(pts):,} points  +  {n_total_segs:,} pipe segments  "
      f"+  {n_total_comps} component spheres")
print(f"  [timer] GUI setup: {_t_gui1 - _t_gui0:.3f}s")
_t_total = _t_gui1 - _t_script_start
print(f"  [timer] Total startup (incl. ground picking): {_t_total:.2f}s")
print("Launching viewer ...\n")

app.run()
print("Viewer closed.")
