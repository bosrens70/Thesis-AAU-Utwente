# -*- coding: utf-8 -*-
"""
Single Point Cloud Viewer with Surrounding Utilities — Indicative Depth
+ Class Label Colour Toggle  +  Left-Click Segment Picking
========================================================================
Refactored to use core/ for shared configuration and data loading.

Usage: python modules/base_module.py
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
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
import geopandas as gpd
import numpy as np
import time
import copy
from core.config import (
    PLY_FILE, GML_PATH, AREA_REF_GEOJSON, CROP_RADIUS, CROP_MODE, UTILITY_RECT_BUFFER,
    PANEL_WIDTH_EM,
    CLASS_LABELS, DEFAULT_CLASS_COLOR,
    LINE_LAYERS, COMPONENT_LAYERS, COMP_TO_LINE,
    COMPONENT_SPHERE_RADIUS, PIPE_LEGEND_UI_ORDER, LEDNINGSPAKKE_LABEL,
    layer_display_name,
    DepthSource, DepthConfig,
    PIPE_DEPTH_CONFIG, COMPONENT_DEPTH_CONFIG,
    forsyningsart_color,
)
from core.data_loader import init_site, load_or_pick_ground_level, load_trench
from core.geometry import (
    point_to_segment_dists, srgb_to_linear, linear_to_srgb,
)
from core.signature_legend import SignatureLegendSection
from core.crop import CropRegion
from core.depth import clean_coords_with_depth as _core_clean_coords
from core.rendering import (
    point_material_shaded, point_material_flat, mesh_material, line_material,
    flat_material,
    setup_scene_lighting,
)
from core.gui_helpers import (
    make_legend_row, LerLegendSection,
    pivot_oblique, top_view, trench_or_scene_frame,
)
from core.ledningstrace import (
    get_ledningstrace_display_info, get_storage_key, get_bredde_width,
    ribbon_alpha,
)
from core.trace_render import (
    build_trace_centerlines, add_trace_centerlines, set_layer_material,
)
from core import symbology as sym
from core.signature_render import (
    PolylineDash, line_segment_mesh,
    feature_signature_meshes_3d, stitch_clipped_segments, merge_meshes,
    add_signature_meshes, set_signature_material,
)

# ─────────────────────────────────────────────────────────────────────────────
# INITIALISE — load area offset, point cloud, and GML via core/
# ─────────────────────────────────────────────────────────────────────────────
# GML is read layer-by-layer below (the loop needs per-feature control), so
# init_site must not pre-load it a second time.
site = init_site(load_gml=False, load_instances=False)

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

# Alias for backward compat
_DEFAULT_CLASS_COLOR = DEFAULT_CLASS_COLOR

# ─────────────────────────────────────────────────────────────────────────────
# VIEWER-SPECIFIC CODE BELOW (ground picking, mesh creation, GUI)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 3.  Pick ground-level points (shared function from core/)
# ─────────────────────────────────────────────────────────────────────────────
GROUND_Z = load_or_pick_ground_level(site.pc, _ply_path)
_pick_method = site.pc.ground_z_method
print(f"  Ground level (UTM)   = {GROUND_Z + TZ:.3f} m")

# Flat ground plane (a*x + b*y + c) — within a 2 m crop radius the tilt is negligible.
_ground_a, _ground_b, _ground_c = 0.0, 0.0, GROUND_Z


def _ground_level_local(x: float, y: float) -> float:
    """Evaluate fitted (or flat fallback) ground plane in local coordinates."""
    return (_ground_a * float(x)) + (_ground_b * float(y)) + _ground_c

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Geometry helpers (shared implementations in core/)
# ─────────────────────────────────────────────────────────────────────────────
# Crop-region selection/clipping: one shared implementation in core.crop.
_crop_region = CropRegion.from_pointcloud(site.pc, TX, TY)
_point_in_bbox        = _crop_region.contains_utm
_pt_in_local_bbox     = _crop_region.contains_local
_segments_in_bbox     = _crop_region.polyline_in_region_utm
_clip_segment_to_bbox = _crop_region.clip_local


def _clean_coords_with_depth(coords_raw, vejledende_dybde_mm,
                              cfg=PIPE_DEPTH_CONFIG, parent_avg_z=None):
    """UTM -> local translation + DepthSource fallback (core.depth), bound to
    this viewer's ground model and offsets."""
    return _core_clean_coords(coords_raw, vejledende_dybde_mm,
                              TX=TX, TY=TY, TZ=TZ,
                              ground_z_at=_ground_level_local,
                              cfg=cfg, parent_avg_z=parent_avg_z)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Load utility lines (pipes / cables) within bbox
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Loading utility lines within bbox ---")
_t_pipes0 = time.perf_counter()
all_pipe_meshes   = []          # flat list — kept for count reporting
_pipe_layer_cyls  = {}          # layer_name -> [TriangleMesh, ...]  per-layer
_pipe_seg_dsrc    = {}          # layer_name -> [DepthSource, ...]   per-segment
_sig_layer_meshes = {}          # layer_name -> [TriangleMesh, ...]  signatures
layer_stats = {}
all_pipe_coords  = []
all_pipe_sources = []   # per-vertex DepthSource arrays, parallel to all_pipe_coords

# Picking data — segment endpoints, midpoints, and their GML attributes
pick_seg_p1        = []   # list of np.array([x,y,z])  — segment start
pick_seg_p2        = []   # list of np.array([x,y,z])  — segment end
pick_seg_midpoints = []   # list of np.array([x,y,z])  — for highlight placement
pick_seg_attrs     = []   # list of [(label, value), ...]
pick_seg_layer     = []   # layer name per segment
# Dash pattern per segment, (PolylineDash, index in its polyline) or None. Only
# how the segment is drawn: the arrays above still hold every segment in full.
pick_seg_dash      = []

# Store per-utility-type average depth for component fallback
_layer_avg_depth_local = {}

# Track Ledningstrace forsyningsart variants for GUI legend
_ledningstrace_variants = {}  # forsyningsart -> color mapping

# Track colors for all storage keys (including compound keys for Ledningstrace variants)
_storage_key_colors = {}  # storage_key -> color

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

        # Get Ledningstrace display info (color, forsyningsart) and width
        is_trace, display_fa, color = get_ledningstrace_display_info(layer_name, row, default_color)
        if is_trace and display_fa and display_fa not in _ledningstrace_variants:
            _ledningstrace_variants[display_fa] = color
        storage_key = get_storage_key(layer_name, display_fa)

        bredde_m = get_bredde_width(row)
        if is_trace and bredde_m is None:
            bredde_m = 0.25  # fallback: 25 cm

        # LER signature choice: dashed for driftsstatus "under etablering", red
        # triangles for fareklasse "meget farlig", El voltage ticks. The dash is
        # cut into the line itself below; the markers are a separate overlay.
        # Neither reaches the picking arrays.
        _sig_style, _sig_hazard, _sig_ticks = sym.signature_choice(row, layer_name)
        _sig_any = _sig_hazard or _sig_ticks > 0

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

            # Registered Z is the pipe crown (top), not its axis; lower the drawn
            # cylinder by its radius so its crown sits on the line. The pick line
            # below stays on the registered crown. A trace's ribbon is already at
            # the registered top level, so it drops by nothing.
            _axis_dz = np.array([0.0, 0.0, 0.0 if bredde_m is not None else radius])
            # Dash phase belongs to the whole polyline, so it is resolved once per
            # polyline and read per segment; taken per segment it would restart at
            # every vertex and lose any dash straddling one.
            _dash = PolylineDash(coords - _axis_dz) if _sig_style == "dashed" else None

            _sig_chords = []
            for i in range(len(coords) - 1):
                clipped = _clip_segment_to_bbox(coords[i], coords[i + 1])
                if clipped is None:
                    continue
                _ax1, _ax2 = clipped[0] - _axis_dz, clipped[1] - _axis_dz
                cyl = line_segment_mesh(_ax1, _ax2, color, radius=radius,
                                        width=bredde_m, dash=_dash, index=i)
                if cyl is not None:
                    all_pipe_meshes.append(cyl)
                    if _sig_any:
                        _sig_chords.append((_ax1, _ax2))
                    _pipe_layer_cyls.setdefault(storage_key, []).append(cyl)
                    # Track color for this storage key
                    if storage_key not in _storage_key_colors:
                        _storage_key_colors[storage_key] = color
                    # Store dominant (worst) depth source for this segment
                    _seg_src = DepthSource(max(int(seg_sources[i]), int(seg_sources[i + 1])))
                    _pipe_seg_dsrc.setdefault(storage_key, []).append(_seg_src)
                    midpt = (clipped[0] + clipped[1]) / 2.0
                    pick_seg_p1.append(clipped[0].copy())
                    pick_seg_p2.append(clipped[1].copy())
                    pick_seg_midpoints.append(midpt)
                    pick_seg_attrs.append(row_attrs)
                    pick_seg_layer.append(storage_key)
                    pick_seg_dash.append((_dash, i) if _dash is not None else None)
                    n_segments += 1

            for _piece in stitch_clipped_segments(_sig_chords):
                _sig_layer_meshes.setdefault(storage_key, []).extend(
                    feature_signature_meshes_3d(
                        _piece, color, hazard=_sig_hazard,
                        tick_count=_sig_ticks, radius=radius, width=bredde_m))

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

# Depth hierarchy stats — count from rendered segments only
_depth_stats = {src: 0 for src in DepthSource if src != DepthSource.NONE}
for _ln, _src_list in _pipe_seg_dsrc.items():
    for _src in _src_list:
        if _src != DepthSource.NONE:
            _depth_stats[_src] = _depth_stats.get(_src, 0) + 1

print(f"\n  Depth hierarchy stats (rendered segments):")
print(f"    1. Registered Z:        {_depth_stats.get(DepthSource.REGISTERED, 0)}")
print(f"    2. vejledendeDybde:      {_depth_stats.get(DepthSource.VEJLEDENDE, 0)}")
print(f"    3. Feature mean Z:       {_depth_stats.get(DepthSource.FEATURE_MEAN, 0)}")
print(f"    4. Layer mean Z:         {_depth_stats.get(DepthSource.LAYER_MEAN, 0)}")
print(f"    5. Ground plane:         {_depth_stats.get(DepthSource.GROUND_PLANE, 0)}")

# Per-layer merged pipe meshes (used for individual visibility toggles)
_pipe_layer_meshes = {}
for _ln, _cyls in _pipe_layer_cyls.items():
    _m = _cyls[0]
    for _c in _cyls[1:]:
        _m += _c
    _m.compute_vertex_normals()
    _pipe_layer_meshes[_ln] = _m

# Per-layer merged signature overlays. Held apart from the pipe mesh because the
# legend colours of a signature are fixed: merged in, they would be repainted by
# the depth-hierarchy recolouring and stop meaning what the legend says.
_sig_meshes = {}
for _ln, _sms in _sig_layer_meshes.items():
    _m = merge_meshes(_sms)
    if _m is not None:
        _sig_meshes[_ln] = _m

# Trace centrelines: a trace's corridor ribbon is drawn transparent (see
# core/trace_render.py), so its registered centreline is drawn separately as a
# thin tube, like any other utility. Built from the picking arrays above.
_trace_centerlines = build_trace_centerlines(
    pick_seg_p1, pick_seg_p2, pick_seg_layer,
    lambda k: _storage_key_colors.get(k, [1.0, 1.0, 1.0]),
    dash_of_index=lambda i: pick_seg_dash[i])

# Pipe centroid
pipe_centroid = np.array([0.0, 0.0, 0.0])
if all_pipe_coords:
    pipe_centroid = np.vstack(all_pipe_coords).mean(axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Load utility components (points) within bbox
# ─────────────────────────────────────────────────────────────────────────────
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
    parent_line = COMP_TO_LINE.get(layer_name)
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

# Wireframe showing the crop boundary at ground level
if CROP_MODE == "rect":
    _rect_corners = [
        (_rect_min_x, _rect_min_y), (_rect_max_x, _rect_min_y),
        (_rect_max_x, _rect_max_y), (_rect_min_x, _rect_max_y),
    ]
    bbox_wire_pts = np.array([
        [x, y, _ground_level_local(x, y)] for x, y in _rect_corners
    ])
    bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
else:
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
    return mesh_material(alpha)


def make_point_material(class_colored: bool = False) -> rendering.MaterialRecord:
    """Material for the scene cloud, chosen by what its colours mean.

    Class colours are categorical, so shaded (defaultLit) + estimated normals +
    SSAO is the closest Open3D equivalent to an EDL shader: points near
    geometric ridges end up darker, and colour fidelity does not matter for a
    label palette. Raw scanner RGB is measured colour, so it is drawn unlit and
    flat the way CloudCompare and other 2D viewers show it; lighting brightened
    it well past its source.
    """
    if class_colored:
        return point_material_shaded(3.0)
    return point_material_flat(3.0)


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
signatures_on         = [True]   # toggled by the "LER signatures" checkbox

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
scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])

# Post-processing (SSAO + tone-mapping) and a top-down sun light for shading.
setup_scene_lighting(scene_widget.scene, post_processing=True)

# Add point cloud
scene_widget.scene.add_geometry(POINT_CLOUD_GEOM, pcd,
                                make_point_material(class_labels_active[0]))

# Add per-layer pipe meshes (filled); a trace's ribbon goes on more transparent
for _ln, _mesh in _pipe_layer_meshes.items():
    alpha = pipe_opacity[0] if _layer_visible.get(_ln, True) else 0.0
    _add_mesh(scene_widget.scene, _pipe_gn(_ln), _mesh,
              make_mesh_material(ribbon_alpha(_ln, alpha)))

# Add trace centrelines at the unscaled opacity, so they read like the pipes
add_trace_centerlines(
    scene_widget.scene, _trace_centerlines, pipe_opacity[0], make_mesh_material,
    visible_of=lambda k: _layer_visible.get(k, True))

# Add the LER signature overlays, at the unscaled opacity for the same reason
add_signature_meshes(
    scene_widget.scene, _sig_meshes, pipe_opacity[0], make_mesh_material,
    visible_of=lambda k: _layer_visible.get(k, True),
    signatures_on=signatures_on[0])

# Add per-layer component meshes (hidden by default)
for _ln, _mesh in _comp_layer_meshes.items():
    _add_mesh(scene_widget.scene, _comp_gn(_ln), _mesh, make_mesh_material(0.0))

# Add frame and bbox wireframe
scene_widget.scene.add_geometry(FRAME_GEOM, frame, make_frame_material())
scene_widget.scene.show_geometry(FRAME_GEOM, False)

line_mat = line_material(3.0)
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

    # Update the point cloud in the scene. The material changes with the mode,
    # not just the colours: class colours are shaded, raw RGB is flat.
    scene_widget.scene.remove_geometry(POINT_CLOUD_GEOM)
    scene_widget.scene.add_geometry(POINT_CLOUD_GEOM, pcd,
                                    make_point_material(show_labels))
    window.post_redraw()


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Right-side control panel
# ─────────────────────────────────────────────────────────────────────────────
PANEL_WIDTH = int(PANEL_WIDTH_EM * em)
panel = gui.Vert(int(0.5 * em), gui.Margins(int(em), int(em), int(em), int(em)))

panel.add_child(gui.Label(f"Points: {len(pts):,}"))
if CROP_MODE == "rect":
    panel.add_child(gui.Label(
        f"Crop: cloud AABB + {UTILITY_RECT_BUFFER:.0f} m (rect)"))
else:
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
    ("1. Registered Z",       _depth_stats.get(DepthSource.REGISTERED, 0),    _dsrc_gui_color(DepthSource.REGISTERED)),
    ("2. vejledendeDybde",    _depth_stats.get(DepthSource.VEJLEDENDE, 0),    _dsrc_gui_color(DepthSource.VEJLEDENDE)),
    ("3. Feature mean Z",     _depth_stats.get(DepthSource.FEATURE_MEAN, 0), _dsrc_gui_color(DepthSource.FEATURE_MEAN)),
    ("4. Layer mean Z",       _depth_stats.get(DepthSource.LAYER_MEAN, 0),   _dsrc_gui_color(DepthSource.LAYER_MEAN)),
    ("5. Ground plane",       _depth_stats.get(DepthSource.GROUND_PLANE, 0), _dsrc_gui_color(DepthSource.GROUND_PLANE)),
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
        _add_mesh(scene_widget.scene, _pipe_gn(ln), mesh,
                  make_mesh_material(ribbon_alpha(ln, alpha)))
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

# ── LER Signature Toggle ────────────────────────────────────────────────────
# The cartographic signatures of the LER "Signaturforklaring", on by default so
# this viewer reads like the ERR plan and like LER itself.
signature_toggle_cb = gui.Checkbox("LER signatures")
signature_toggle_cb.checked = signatures_on[0]
if not _sig_meshes:
    signature_toggle_cb.enabled = False


def _on_signature_toggle(checked):
    signatures_on[0] = checked
    for ln in _sig_meshes:
        alpha = pipe_opacity[0] if (_ler_active[0] and _layer_visible.get(ln, True)) else 0.0
        set_signature_material(scene_widget.scene, ln, alpha,
                               make_mesh_material, checked)
    window.post_redraw()


signature_toggle_cb.set_on_checked(_on_signature_toggle)
panel.add_child(signature_toggle_cb)

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

# ── Utility Legend (uniform LerLegendSection, see core/gui_helpers.py) ───────
_ler_active = [True]
_ler_section = LerLegendSection(em, LEDNINGSPAKKE_LABEL)
ler_toggle_cb = _ler_section.master_cb
_ler_legend_container = _ler_section.container
opacity_slider = _ler_section.opacity_slider


def _apply_opacity(val: float):
    val = max(0.0, min(1.0, val))
    pipe_opacity[0] = val
    opacity_slider.double_value = val

    for ln in _pipe_layer_meshes:
        alpha = val if _layer_visible.get(ln, True) else 0.0
        set_layer_material(scene_widget.scene, _pipe_gn(ln), ln, alpha,
                           make_mesh_material)

    for ln in _sig_meshes:
        alpha = val if _layer_visible.get(ln, True) else 0.0
        set_signature_material(scene_widget.scene, ln, alpha, make_mesh_material,
                               signatures_on[0])

    for ln in _comp_layer_meshes:
        alpha = val if _layer_visible.get(ln, True) else 0.0
        scene_widget.scene.modify_geometry_material(_comp_gn(ln), make_mesh_material(alpha))

    window.post_redraw()


_ler_section.set_on_opacity(_apply_opacity)

_pipe_checkboxes = []   # (layer_name, checkbox) for "toggle all" control
_comp_checkboxes = []


def _make_pipe_toggle(ln):
    def _cb(checked):
        _layer_visible[ln] = checked
        alpha = pipe_opacity[0] if checked else 0.0
        if ln in _pipe_layer_meshes:
            set_layer_material(scene_widget.scene, _pipe_gn(ln), ln, alpha,
                               make_mesh_material)
        set_signature_material(scene_widget.scene, ln, alpha, make_mesh_material,
                               signatures_on[0])
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
def _on_toggle_all_pipes(checked):
    for ln, cb in _pipe_checkboxes:
        cb.checked = checked
        _layer_visible[ln] = checked
        alpha = pipe_opacity[0] if checked else 0.0
        if ln in _pipe_layer_meshes:
            set_layer_material(scene_widget.scene, _pipe_gn(ln), ln, alpha,
                               make_mesh_material)
        set_signature_material(scene_widget.scene, ln, alpha, make_mesh_material,
                               signatures_on[0])
    window.post_redraw()

_all_pipes_cb = _ler_section.add_all_segments(True, _on_toggle_all_pipes)

# Line layers — only show legend entry if the layer produced actual geometry
for layer_name in PIPE_LEGEND_UI_ORDER:
    # Skip Ledningstrace here; we'll handle variants below
    if layer_name == "Ledningstrace":
        continue
    if layer_name not in _pipe_layer_meshes:
        continue
    cfg = LINE_LAYERS[layer_name]
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
def _on_toggle_all_comps(checked):
    for ln, cb in _comp_checkboxes:
        cb.checked = checked
        _layer_visible[ln] = checked
        if ln in _comp_layer_meshes:
            alpha = pipe_opacity[0] if checked else 0.0
            scene_widget.scene.modify_geometry_material(_comp_gn(ln), make_mesh_material(alpha))
    window.post_redraw()

_all_comps_cb = _ler_section.add_all_components(False, _on_toggle_all_comps)

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


def _on_ler_toggle(checked):
    _ler_active[0] = checked
    # Show/hide all utility geometry (legend collapse is handled by the section)
    for ln in _pipe_layer_meshes:
        alpha = pipe_opacity[0] if (checked and _layer_visible.get(ln, True)) else 0.0
        set_layer_material(scene_widget.scene, _pipe_gn(ln), ln, alpha,
                           make_mesh_material)
    for ln in _sig_meshes:
        alpha = pipe_opacity[0] if (checked and _layer_visible.get(ln, True)) else 0.0
        set_signature_material(scene_widget.scene, ln, alpha, make_mesh_material,
                               signatures_on[0])
    for ln in _comp_layer_meshes:
        alpha = pipe_opacity[0] if (checked and _layer_visible.get(ln, True)) else 0.0
        scene_widget.scene.modify_geometry_material(_comp_gn(ln), make_mesh_material(alpha))


_ler_section.set_on_master(window, _on_ler_toggle)
_ler_section.add_to(panel)

# -- LER signature legend ("Signaturforklaring", core/signature_legend.py) ----
# The utility legend above explains colour; this one explains form, which is
# the half a colour swatch cannot show. Collapsed by default, like LER's own.
_sig_legend = SignatureLegendSection(em, components="point")
_sig_legend.add_to(panel)

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
    marker_mat = line_material(0.5)
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
    axes_mat = line_material(2.0)
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
        seg_dists = point_to_segment_dists(hit, pick_seg_p1, pick_seg_p2)
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
        label  = f"{layer_display_name(pick_comp_layer[best_comp_i])} (component)"
        print(f"\n[pick] -> {label}")
        for k, v in attrs:
            print(f"    {k:<30} = {v}")
        print()
    elif best_seg_d < PICK_RADIUS_SEG:
        centre = pick_seg_midpoints[best_seg_i].copy()
        attrs  = pick_seg_attrs[best_seg_i]
        label  = f"{layer_display_name(pick_seg_layer[best_seg_i])} (pipe segment)"
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
    pivot_oblique(scene_widget, point, np.linalg.norm(pc_max - pc_min))
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
    if k in (ord('T'), ord('t')):
        print("Top view of trench")
        _top_view()
        return HANDLED

    if k in (ord('H'), ord('h')):
        print("\n-- Shortcuts ---------------------------------------------------")
        print("  Left-click     pick pipe segment or component (show attributes)")
        print("  C              pivot to point cloud centroid")
        print("  P              pivot to pipe centroid (all utilities)")
        print("  0              pivot to world origin (0, 0, 0)")
        print("  T              top view of trench (or scene if none)")
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
