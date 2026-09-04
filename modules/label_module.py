# -*- coding: utf-8 -*-
"""
Single Point Cloud Viewer with Instance Labels + Surrounding Utilities
======================================================================
Refactored to use core/ for shared configuration and data loading.

Besides assigning a utility-type label to an instance, an instance can also
be linked to one specific LER utility line: left-click a utility line while
that instance is active to record the match, or use "Suggest LER match" to
have the closest/best-aligned nearby line proposed automatically (ranked by
proximity, direction, diameter and colour similarity — see
core/ler_matching.py) and accept or cycle through it.

A link covers the whole line, not the clicked fragment. The registry splits
one physical utility into several features, so core/ler_lines.py groups the
features that continue into each other and the match records every gml_id on
the run. Lines that merely run alongside each other stay separate. Shift-click
adds or removes a single feature when that grouping needs correcting.
"Mark as NOT in LER" records that an instance has no registry counterpart at
all. Matches are saved to ler_matches.json next to the labelled PLYs;
deviation_module.py reads it and, when a match exists, measures that
instance against only its linked LER feature instead of every nearby feature
of the same utility type.

Two kinds of instance are listed. The clustered ones come from segment_module
(class 1, "Other Utility"). After them come the per-class instances written by
tools/convert_main_utility_to_water_instance.py for the classes segment_module
does not cluster: class 0 "Main Utility" and class 3 "Inactive Utility". Those
arrive already labelled, since their utility type is in their own filename, but
without a match they too fall back to being measured against every feature of
their type, so they are listed here to be linked to one specific LER feature.
They keep their file in the root instance directory and record their matches
there, rather than being copied into a label session.

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
import json
from datetime import datetime

from core.config import (
    PLY_FILE, GML_PATH, CROP_RADIUS, CROP_MODE, UTILITY_RECT_BUFFER,
    PANEL_WIDTH_EM,
    LEDNINGSPAKKE_LABEL, layer_display_name,
    LINE_LAYERS, COMPONENT_LAYERS, COMP_TO_LINE,
    COMPONENT_SPHERE_RADIUS, TRACE_CENTERLINE_RADIUS, INSTANCE_COLORS, INSTANCE_LABEL_OPTIONS,
    TARGET_CLASS, UTILITY_TO_LER_MATCH, UTILITY_TYPE_COLORS,
    DepthSource, DEPTH_STATS_KEY as _STATS_KEY,
    PIPE_DEPTH_CONFIG, COMPONENT_DEPTH_CONFIG,
    ler_layers_for_type, trace_forsyningsart, FORSYNINGSART_TO_LINE,
)
from core.data_loader import (
    init_site, load_or_pick_ground_level, load_trench,
    utility_type_from_filename,
)
from core.site_status import (
    LABELED_PREFIX, LABELED_FNAME_RE, ANY_LABELED_FNAME_RE, MATCHES_FILENAME,
    CONFLICTS_FILENAME, resolve_labeled_dir, format_label_summary,
)
from core.gui_helpers import (
    make_master_pipe_toggle, make_master_comp_toggle,
    LerLegendSection,
    pivot_top_down, top_view, trench_or_scene_frame,
)
from core.ler_matching import (build_feature_index, score_candidates,
                               merge_index_by_line)
from core.ler_lines import group_features_into_lines, line_members
from core.geometry import (
    point_to_segment_dists,
)
from core.signature_legend import SignatureLegendSection
from core.crop import CropRegion
from core.depth import clean_coords_with_depth as _core_clean_coords
from core.ledningstrace import (
    get_ledningstrace_display_info, get_storage_key, get_bredde_width,
    is_trace_key, ribbon_alpha,
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
from core.rendering import (
    point_material_flat, mesh_material, line_material,
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

# Rectangle region (CROP_MODE == "rect"): full-cloud XY AABB grown by the utility
# buffer.  Selection and clipping are XY-only so every utility passing through the
# footprint is rendered regardless of its depth.  pc_min/pc_max are local.
_rect_min_x = pc_min[0] - UTILITY_RECT_BUFFER
_rect_max_x = pc_max[0] + UTILITY_RECT_BUFFER
_rect_min_y = pc_min[1] - UTILITY_RECT_BUFFER
_rect_max_y = pc_max[1] + UTILITY_RECT_BUFFER

_ply_path = Path(PLY_FILE)

# Instance directory from core discovery
INSTANCE_DIR = str(site.instance_dir) if site.instance_dir else ""

# ─────────────────────────────────────────────────────────────────────────────
# VIEWER-SPECIFIC CODE BELOW (instances, ground picking, mesh creation, GUI)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 2b. Load instance PLY files and compute bounding boxes
# ─────────────────────────────────────────────────────────────────────────────

_instance_dir = Path(INSTANCE_DIR)
_instance_files = site.instance_files if site.instance_files else []
_class_instance_files = site.class_instance_files if site.class_instance_files else []


def _read_instance(inst_path, src_class, cluster_id):
    """One instance_data entry, or None when the PLY holds no points.

    ``src_class`` and ``cluster_id`` record where the instance came from and so
    what it is saved as (see _instance_fname). A clustered instance passes
    cluster_id=None, because its id is its position in the size sort below and
    that is only known once every instance has been read.
    """
    pcd = o3d.io.read_point_cloud(str(inst_path))
    n_pts = len(np.asarray(pcd.points))
    if n_pts == 0:
        return None
    return {
        "name": inst_path.stem,
        "path": inst_path,
        "pcd": pcd,
        "obb": pcd.get_oriented_bounding_box(),
        "color": INSTANCE_COLORS[0],
        "n_pts": n_pts,
        "src_class": src_class,
        "cluster_id": cluster_id,
    }


instance_data = []
for _inst_path in _instance_files:
    _entry = _read_instance(_inst_path, TARGET_CLASS, None)
    if _entry:
        instance_data.append(_entry)

# Largest first. A clustered instance's position in this order is the index
# baked into its saved filename, so the order is a persistence key, not a
# display preference.
instance_data.sort(key=lambda d: d["n_pts"], reverse=True)
for _i, _d in enumerate(instance_data):
    _d["cluster_id"] = _i

# The loose per-class instances (class 0 "Main Utility", class 3 "Inactive
# Utility", written by tools/convert_main_utility_to_water_instance.py) are
# appended after that sort and never mixed into it: inserting anything ahead of
# a clustered instance would shift its index and silently re-map every label
# already saved for this site. They keep the class and cluster id from their own
# filename, so a class blob split by hand into several files becomes several
# instances, each able to carry its own LER match.
for _inst_path in _class_instance_files:
    _m = ANY_LABELED_FNAME_RE.match(_inst_path.name)
    _entry = _read_instance(_inst_path, int(_m.group(1)), int(_m.group(2)))
    if _entry:
        instance_data.append(_entry)

for _i, _d in enumerate(instance_data):
    _d["color"] = INSTANCE_COLORS[_i % len(INSTANCE_COLORS)]
    _d["obb"].color = _d["color"]

# (src_class, cluster_id) -> index, for resolving a saved filename back to the
# instance it describes. The utility type in the name is deliberately not part
# of the key: relabelling changes it, and a recorded match must survive that.
_IDX_BY_SRC = {(_d["src_class"], _d["cluster_id"]): _i
               for _i, _d in enumerate(instance_data)}

if instance_data:
    print(f"\n  Loaded {len(instance_data)} instances from {_instance_dir.name}/ (sorted largest first)")
    for _i, _d in enumerate(instance_data):
        _tag = "" if _d["src_class"] == TARGET_CLASS else f"  [class {_d['src_class']}]"
        print(f"    [{_i}] {_d['name']}: {_d['n_pts']:,} points{_tag}")
else:
    print(f"\n  [warn] No instance PLY files found in {INSTANCE_DIR}")


_instance_labels = {}
_instance_ler_match = {}  # idx -> {"layer": str, "gml_id": str} — exclusive LER link
# idx -> the most recent pick whose LER layer contradicted the instance's label,
# whether it was refused or deliberately overridden. Persisted next to the
# matches so a refusal leaves a trace instead of vanishing.
_instance_match_conflicts = {}
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


def _clean_coords_with_depth(coords_raw, vejledende_dybde_mm,
                             cfg=PIPE_DEPTH_CONFIG, parent_avg_z=None):
    """UTM -> local translation + DepthSource fallback (core.depth), bound to
    this viewer's flat ground level. Returns (coords, sources); the caller
    counts the fallback statistics from ``sources``."""
    return _core_clean_coords(coords_raw, vejledende_dybde_mm,
                              TX=TX, TY=TY, TZ=TZ, ground_z_at=_ground_z_at,
                              cfg=cfg, parent_avg_z=parent_avg_z)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Load utility lines (pipes / cables) within bbox
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Loading utility lines within bbox ---")
all_pipe_meshes   = []          # flat list, kept for count reporting only
_pipe_layer_cyls  = {}          # layer_name -> [TriangleMesh, ...]  per-layer
_sig_layer_meshes = {}          # layer_name -> [TriangleMesh, ...]  signatures
_pipe_layer_seg_pts = {}        # layer_name -> ([p1, ...], [p2, ...]) for XRay centerlines
layer_stats = {}
all_pipe_coords = []

# Picking data — segment endpoints, midpoints, and their GML attributes
pick_seg_p1        = []   # list of np.array([x,y,z])  — segment start
pick_seg_p2        = []   # list of np.array([x,y,z])  — segment end
pick_seg_midpoints = []   # list of np.array([x,y,z])  — for highlight placement
pick_seg_attrs     = []   # list of [(label, value), ...]
pick_seg_layer     = []   # layer name per segment
# Dash pattern per segment, (PolylineDash, index in its polyline) or None. Only
# how the segment is drawn: the arrays above still hold every segment in full.
pick_seg_dash      = []
pick_seg_gml_id    = []   # GML gml_id per segment (identifies the whole feature)

# One entry per loaded feature part, (storage_key, gml_id, raw coords, attrs),
# fed to core/ler_lines.py once every layer is in. The registry splits one
# physical utility into several features, so this is what lets a match cover the
# whole run. Deliberately the raw GML coordinates, not the bbox-clipped
# segments: a clipping artefact at the crop boundary must not be able to
# fabricate a junction.
_line_features = []

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

        gml_id_val = str(row.get("gml_id", "") or "")

        feature_hit = False
        for sub_geom in sub_geoms:
            coords_raw = np.array(sub_geom.coords, dtype=float)
            if not _segments_in_bbox(coords_raw):
                continue
            _line_features.append((storage_key, gml_id_val, coords_raw, row_attrs))

            coords, _seg_srcs = _clean_coords_with_depth(coords_raw, vejl_dybde)
            for _src in _seg_srcs:
                _key = _STATS_KEY.get(DepthSource(_src))
                if _key in _depth_stats:
                    _depth_stats[_key] += 1
            all_pipe_coords.append(coords)
            _layer_z_vals.extend(coords[:, 2].tolist())
            feature_hit = True

            # Planes for Ledningstrace (width from bredde_m), cylinders for the
            # other utility lines. Registered Z is the pipe crown (top), not its
            # axis; lower the drawn cylinder by its radius so its crown sits on
            # the line. The pick line below stays on the registered crown.
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
                mesh = line_segment_mesh(_ax1, _ax2, color, radius=radius,
                                         width=bredde_m, dash=_dash, index=i)
                if mesh is not None:
                    all_pipe_meshes.append(mesh)
                    if _sig_any:
                        _sig_chords.append((_ax1, _ax2))
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
                    pick_seg_dash.append((_dash, i) if _dash is not None else None)
                    n_segments += 1

            for _piece in stitch_clipped_segments(_sig_chords):
                _sig_layer_meshes.setdefault(storage_key, []).extend(
                    feature_signature_meshes_3d(
                        _piece, color, hazard=_sig_hazard,
                        tick_count=_sig_ticks, radius=radius, width=bredde_m))

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

# Features that form one physical utility, so a match covers the whole run
# rather than the fragment that happened to be clicked.
_line_of, _lines = group_features_into_lines(_line_features)
# The same index, merged per line: "Suggest LER match" proposes whole utility
# lines, and the score sees the run's full extent and direction.
_ler_line_index = merge_index_by_line(_ler_feature_index, _line_of)
_n_multi = sum(1 for ms in _lines.values() if len(ms) > 1)
if _n_multi:
    print(f"  {len(_line_of)} features grouped into {len(_lines)} utility lines "
          f"({_n_multi} spanning more than one feature)")


def _line_gml_ids(gml_id):
    """Every gml_id on the same utility line as this one."""
    return line_members(_line_of, _lines, gml_id)

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

# Per-layer merged signature overlays. Kept out of the pipe mesh so the fixed
# legend colours of a signature survive every recolouring of the utilities, and
# out of the combined wireframe, which is built from the pipe meshes below.
_sig_meshes = {}
for _ln, _sms in _sig_layer_meshes.items():
    _m = merge_meshes(_sms)
    if _m is not None:
        _sig_meshes[_ln] = _m

# Per-layer XRay centerline LineSets, one line per clipped segment. Built and
# added to the scene but never shown: centerline_xray_active is never set true,
# and Open3D 0.19 has no depth_func, so neither half of the effect is live.
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
    lambda k: _storage_key_colors.get(k, [1.0, 1.0, 1.0]),
    dash_of_index=lambda i: pick_seg_dash[i])

# Combined wireframe (all layers). Added to the scene but hidden: there is no
# wireframe toggle, so pipe_wireframe_active never becomes true.
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
    parent_line = COMP_TO_LINE.get(layer_name)
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

        # Same resolver and the same component configuration as every other
        # module: REGISTERED -> LAYER_MEAN -> GROUND_PLANE.
        pt_arr, _src_arr = _clean_coords_with_depth(
            np.array([[g.x, g.y, g.z]], dtype=float), None,
            cfg=COMPONENT_DEPTH_CONFIG, parent_avg_z=parent_avg_z,
        )
        pt = pt_arr[0]

        # Crop to local buffered bbox
        if not _pt_in_local_bbox(pt[0], pt[1]):
            continue

        _comp_src = DepthSource(int(_src_arr[0]))
        if _comp_src == DepthSource.LAYER_MEAN:
            _comp_depth_stats["from_pipe_avg"] += 1
        elif _comp_src == DepthSource.GROUND_PLANE:
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


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Coordinate frame + crop wireframe + point cloud normals
# ─────────────────────────────────────────────────────────────────────────────
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5, origin=cloud_centroid
)

# Estimate normals on the cropped point cloud. The cloud itself is drawn unlit
# and flat (point_material_flat), so this does not affect its own shading.
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
    # This viewer's scene cloud is always raw scanner RGB, which is measured
    # colour, so it is drawn unlit and flat the way CloudCompare and other 2D
    # viewers show it. Lighting it brightened the cloud well past its source.
    return point_material_flat(3.0)


def make_pipe_wire_material() -> rendering.MaterialRecord:
    return line_material(1.5)


def make_centerline_material() -> rendering.MaterialRecord:
    mat = line_material(2.5)
    try:
        # Intended to render centerlines through occluding geometry so thin
        # pipes stay visible inside thick ones. Open3D 0.19 has no
        # MaterialRecord.depth_func, so this is a no-op and they depth-test
        # normally.
        mat.depth_func = "always"
    except AttributeError:
        pass
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

def _inst_bbox_gn(idx): return f"inst_bbox_{idx}"
def _inst_pts_gn(idx):  return f"inst_pts_{idx}"


# Per-layer geometry names
def _pipe_gn(ln):       return f"pipe_{ln}"
def _comp_gn(ln):       return f"comp_{ln}"
def _centerline_gn(ln): return f"centerline_{ln}"

# Per-layer visibility state (True = shown)
_layer_visible = {ln: True for ln in LINE_LAYERS}
_layer_visible.update({ln: False for ln in COMPONENT_LAYERS})  # start with all components hidden
# Traces start shown, like every other line layer. A trace keys on its
# per-forsyningsart storage key ("Ledningstrace (el)"), which is absent from
# LINE_LAYERS, so it relies on the True default of every _layer_visible lookup.

pipe_opacity = [1.0]
origin_frame_visible  = [False] # toggled by the "Show origin axis" checkbox
# Both overlays are built, but neither has a checkbox, so these stay False for
# the whole session and the geometry they gate is never shown.
pipe_wireframe_active = [False]
centerline_xray_active = [False]
ler_utilities_visible = [True]   # toggled by the "Show LER utilities" checkbox
signatures_on         = [True]   # toggled by the "LER signatures" checkbox

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

# Add the LER signature overlays, at the unscaled opacity for the same reason
add_signature_meshes(
    scene_widget.scene, _sig_meshes, pipe_opacity[0], make_mesh_material,
    visible_of=lambda k: _layer_visible.get(k, True),
    signatures_on=signatures_on[0])

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

# The cartographic signatures of the LER "Signaturforklaring", on by default so
# this viewer reads like the ERR plan and like LER itself.
signature_toggle_cb = gui.Checkbox("LER signatures")
signature_toggle_cb.checked = signatures_on[0]
if not _sig_meshes:
    signature_toggle_cb.enabled = False


def _on_signature_toggle(checked):
    signatures_on[0] = checked
    for ln in _sig_meshes:
        alpha = pipe_opacity[0] if (ler_utilities_visible[0]
                                    and _layer_visible.get(ln, True)) else 0.0
        set_signature_material(scene_widget.scene, ln, alpha,
                               make_mesh_material, checked)
    window.post_redraw()


signature_toggle_cb.set_on_checked(_on_signature_toggle)
panel.add_child(signature_toggle_cb)
panel.add_fixed(int(0.5 * em))

# ── Utility Legend (uniform LerLegendSection, see core/gui_helpers.py) ───────
_ler_section = LerLegendSection(em, LEDNINGSPAKKE_LABEL)
opacity_slider = _ler_section.opacity_slider


def _on_ler_toggle(checked):
    ler_utilities_visible[0] = checked
    for ln in _pipe_layer_meshes:
        if not _layer_visible.get(ln, True):
            continue
        alpha = pipe_opacity[0] if checked else 0.0
        set_layer_material(scene_widget.scene, _pipe_gn(ln), ln, alpha,
                           make_mesh_material)
    for ln in _sig_meshes:
        if not _layer_visible.get(ln, True):
            continue
        alpha = pipe_opacity[0] if checked else 0.0
        set_signature_material(scene_widget.scene, ln, alpha, make_mesh_material,
                               signatures_on[0])
    for ln in _comp_layer_meshes:
        if not _layer_visible.get(ln, True):
            continue
        alpha = pipe_opacity[0] if checked else 0.0
        scene_widget.scene.modify_geometry_material(_comp_gn(ln), make_mesh_material(alpha))


_ler_section.set_on_master(window, _on_ler_toggle)
_ler_section.add_to(panel)

# -- LER signature legend ("Signaturforklaring", core/signature_legend.py) ----
# The utility legend above explains colour; this one explains form, which is
# the half a colour swatch cannot show. Collapsed by default, like LER's own.
_sig_legend = SignatureLegendSection(em, components="point")
_sig_legend.add_to(panel)
panel.add_fixed(int(0.3 * em))


def _make_pipe_toggle(ln):
    def _cb(checked):
        _layer_visible[ln] = checked
        _ler = ler_utilities_visible[0]
        alpha = pipe_opacity[0] if (_ler and checked and not pipe_wireframe_active[0]) else 0.0
        if ln in _pipe_layer_meshes:
            set_layer_material(scene_widget.scene, _pipe_gn(ln), ln, alpha,
                               make_mesh_material)
        set_signature_material(scene_widget.scene, ln, alpha, make_mesh_material,
                               signatures_on[0])
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
_ler_section.add_all_segments(
    True, make_master_pipe_toggle(_pipe_checkboxes, _layer_visible,
                                  _pipe_layer_meshes, scene_widget,
                                  _pipe_gn, make_mesh_material,
                                  pipe_opacity, window,
                                  signatures_on=signatures_on))

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
_ler_section.add_all_components(
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

    for ln in _sig_meshes:
        alpha = val if (_ler and _layer_visible.get(ln, True) and not pipe_wireframe_active[0]) else 0.0
        set_signature_material(scene_widget.scene, ln, alpha, make_mesh_material,
                               signatures_on[0])

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


def _index_for_fname(fname):
    """Instance index for a saved PLY filename, or None when it names no
    instance this site still has."""
    m = ANY_LABELED_FNAME_RE.match(str(fname))
    if not m:
        return None
    return _IDX_BY_SRC.get((int(m.group(1)), int(m.group(2))))


def _load_matches_json(match_dir):
    """Read one ler_matches.json into _instance_ler_match. Returns how many
    entries were adopted, so the caller can report the resume."""
    path = Path(match_dir) / MATCHES_FILENAME
    if not path.is_file():
        return 0
    try:
        with open(str(path), encoding="utf-8") as fh:
            loaded = json.load(fh)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [warn] Could not read {path.name}: {e}")
        return 0
    n = 0
    for fname, match in loaded.items():
        idx = _index_for_fname(fname)
        if idx is None:
            continue
        # A record written before utility lines existed names a single feature,
        # which is a fragment of the run it was meant to cover. Expand it to its
        # line in memory; it is only written back in the new form the next time
        # something is saved, and "gml_id" is kept, so the change is reversible.
        if not match.get("no_ler") and not match.get("gml_ids") and match.get("gml_id"):
            ids = _line_gml_ids(match["gml_id"])
            match = dict(match, gml_ids=ids,
                         line_id=_line_of.get(match["gml_id"], match["gml_id"]))
            if len(ids) > 1:
                print(f"  [line] {fname}: match expanded from 1 feature to the "
                      f"{len(ids)}-feature line it belongs to")
        _instance_ler_match[idx] = match
        n += 1
    return n


# Prefill labels and LER matches from the resumed session.
if _labeled_output_dir and _labeled_output_dir.exists():
    for _f in sorted(_labeled_output_dir.glob("*.ply")):
        _fm = LABELED_FNAME_RE.match(_f.name)
        if not _fm:
            continue
        _pidx, _plid = _index_for_fname(_f.name), int(_fm.group(2))
        if _pidx is not None and 1 <= _plid <= len(INSTANCE_LABEL_OPTIONS):
            _instance_labels[_pidx] = INSTANCE_LABEL_OPTIONS[_plid - 1]
            # Remember the file this label already lives in, so relabelling in a
            # later session replaces it instead of leaving a second copy.
            instance_data[_pidx]["saved_as"] = _f
    _n_resumed_matches = _load_matches_json(_labeled_output_dir)
    if _instance_labels or _instance_ler_match:
        print(f"  [resume] {_labeled_output_dir.name}/: {len(_instance_labels)} labels, "
              f"{_n_resumed_matches} LER matches loaded")

# The per-class instances arrive already labelled: their utility type is in
# their own filename. Their matches live in the root instance directory, next to
# the PLYs they describe, rather than in a label session they are not part of.
for _idx, _d in enumerate(instance_data):
    if _d["src_class"] == TARGET_CLASS:
        continue
    _tid = utility_type_from_filename(_d["path"].name)
    if 1 <= _tid <= len(INSTANCE_LABEL_OPTIONS):
        _instance_labels[_idx] = INSTANCE_LABEL_OPTIONS[_tid - 1]
if _class_instance_files:
    _n_class_matches = _load_matches_json(_instance_dir)
    if _n_class_matches:
        print(f"  [resume] {_instance_dir.name}/{MATCHES_FILENAME}: "
              f"{_n_class_matches} LER match(es) for class instances")

# Start at the first instance that still needs a label.
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


def _is_class_instance(idx):
    """True for the loose per-class instances, which are not part of a
    clustered segment run and are saved differently."""
    return instance_data[idx]["src_class"] != TARGET_CLASS


def _instance_fname(idx, label_id):
    """Filename an instance is saved under, and the key its LER match is
    recorded against. For a clustered instance the class is TARGET_CLASS and
    the cluster id is its position in the size sort, which reproduces the
    original ``1_instance_<idx>_type_<id>.ply`` convention exactly."""
    d = instance_data[idx]
    return f"{d['src_class']}_instance_{d['cluster_id']}_type_{label_id}.ply"


def _instance_out_dir(idx):
    """Directory holding an instance's PLY and its ler_matches.json.

    Clustered instances belong to the label session. The per-class ones stay in
    the root instance directory, where they were created and where
    deviation_module already reads them, so labelling one never leaves a second
    copy of the same points for the deviation viewer to count twice.
    """
    return _instance_dir if _is_class_instance(idx) else _labeled_output_dir


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
    exclusive LER link, keyed by the saved PLY filename.

    One file per output directory, since a match belongs next to the PLY it
    describes: the clustered instances write into the label session, the
    per-class ones into the root instance directory. deviation_module reads
    every directory its instances come from, so both are picked up. Each file is
    rewritten in full, so relabelling (which changes the filename) and clearing
    the last match never leave a stale key behind.
    """
    _refresh_window_title()
    by_dir = {}
    for idx, match in _instance_ler_match.items():
        if idx not in _instance_labels:
            continue
        out_dir = _instance_out_dir(idx)
        if not out_dir:
            continue
        label_id = _LABEL_TO_ID.get(_instance_labels[idx], 0)
        by_dir.setdefault(out_dir, {})[_instance_fname(idx, label_id)] = match

    for out_dir in {d for d in map(_instance_out_dir, range(len(instance_data))) if d}:
        out = by_dir.get(out_dir, {})
        path = Path(out_dir) / MATCHES_FILENAME
        if not out and not path.is_file():
            continue        # nothing to record here, and nothing stale to clear
        if out and out_dir == _labeled_output_dir:
            _ensure_output_dir()
        if not path.parent.is_dir():
            continue
        with open(str(path), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


def _write_match_conflicts_json():
    """Persist the refused and overridden picks, keyed by PLY filename.

    Same one-file-per-output-directory rule as the matches, and rewritten in
    full, so resolving a conflict clears it rather than leaving a stale entry.
    """
    by_dir = {}
    for idx, rec in _instance_match_conflicts.items():
        if idx not in _instance_labels:
            continue
        out_dir = _instance_out_dir(idx)
        if not out_dir:
            continue
        label_id = _LABEL_TO_ID.get(_instance_labels[idx], 0)
        by_dir.setdefault(out_dir, {})[_instance_fname(idx, label_id)] = rec

    for out_dir in {d for d in map(_instance_out_dir, range(len(instance_data))) if d}:
        out = by_dir.get(out_dir, {})
        path = Path(out_dir) / CONFLICTS_FILENAME
        if not out:
            if path.is_file():
                path.unlink()       # last conflict resolved: drop the file
            continue
        if out_dir == _labeled_output_dir:
            _ensure_output_dir()
        if not path.parent.is_dir():
            continue
        with open(str(path), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


def _save_instance_ply(idx, label_name):
    if idx >= len(instance_data) or not _instance_out_dir(idx):
        return
    inst = instance_data[idx]
    label_id = _LABEL_TO_ID.get(label_name, 0)
    fname = _instance_fname(idx, label_id)

    if _is_class_instance(idx):
        # The root file already holds these points under this type, so there is
        # nothing to write. Writing a copy into the label session would make
        # deviation_module load the same cloud twice and count it twice.
        out_path = Path(_instance_dir) / fname
        if out_path == inst["path"] and out_path.is_file():
            inst["saved_as"] = out_path
            _write_ler_matches_json()
            _refresh_window_title()
            return
    else:
        out_path = _ensure_output_dir() / fname

    pcd = inst["pcd"]
    pts = np.asarray(pcd.points)
    has_colors = pcd.has_colors()
    colors = np.asarray(pcd.colors) if has_colors else None
    has_normals = pcd.has_normals()
    normals = np.asarray(pcd.normals) if has_normals else None
    n = len(pts)

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

    # Relabelling writes a new filename, so the file written under the previous
    # type has to go with it. Left behind, deviation_module globs both and counts
    # the same points twice, in two different utility classes, measured against
    # two different LER layers.
    #
    # Only this instance's own previous save is removed, never a file matched by
    # name pattern: a hand-split valve can legitimately sit beside its pipe under
    # the same class and cluster id, distinguished only by the type suffix, and a
    # pattern-based sweep would delete it.
    prev = inst.get("saved_as")
    if prev and prev != out_path and Path(prev).is_file():
        Path(prev).unlink()
        print(f"  [replaced] {Path(prev).name}")
    inst["saved_as"] = out_path

    if _is_class_instance(idx):
        old_path = inst["path"]
        if old_path != out_path and old_path.parent == out_path.parent and old_path.is_file():
            old_path.unlink()
            print(f"  [replaced] {old_path.name}")
        inst["path"] = out_path
        inst["name"] = out_path.stem

    print(f"  [saved] {out_path}  (utility_type={label_id}: {label_name})")
    _write_ler_matches_json()
    _refresh_window_title()


def _instance_confirmed(idx):
    """True when both halves of an instance are decided: a type label and an LER
    link. "Not in LER" counts as decided; it is a confirmed answer about the
    register, not a missing one. A refused pick does not: it lives in
    _instance_match_conflicts, and the instance still has no link."""
    return idx in _instance_labels and idx in _instance_ler_match


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


def _label_for_ler_layer(layer_name):
    """The type label a matched LER layer implies, or None.

    A trace resolves through its forsyningsart, so matching
    ``Ledningstrace (telekommunikation)`` implies TelecomunicationLine just as
    ``Telekommunikationsledning`` does. Previously traces mapped to nothing and
    a match on one left the instance unlabelled.
    """
    label_name = _LER_LAYER_TO_LABEL.get(layer_name)
    if label_name is not None:
        return label_name
    fa = trace_forsyningsart(layer_name)
    if fa:
        return _LER_LAYER_TO_LABEL.get(FORSYNINGSART_TO_LINE.get(fa))
    return None


def _maybe_autolabel_from_layer(idx, layer_name):
    """When an instance has no type label yet, infer one from the matched LER
    layer (reverse of UTILITY_TO_LER_MATCH, traces via forsyningsart) and save
    it, so a match recorded on its own still persists a labelled PLY. No-op if
    the instance is already labelled or the layer implies no type."""
    if idx in _instance_labels:
        return
    label_name = _label_for_ler_layer(layer_name)
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
    _place_ler_match_highlight(_match_gml_ids(_instance_ler_match.get(idx)))
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
    # Relabelling can invalidate a match recorded under the previous label. The
    # label is the primary datum, so the match yields: it is dropped and logged
    # rather than left contradicting the label it is stored beside.
    _existing = _instance_ler_match.get(idx)
    if (_existing and not _existing.get("no_ler")
            and _match_conflicts(idx, _existing.get("layer", ""))):
        _record_match_conflict(idx, _existing.get("layer", ""),
                               _match_gml_ids(_existing), overridden=False)
        _instance_ler_match.pop(idx, None)
        print(f"  [ler-match] cleared: it no longer agrees with {label_name}")
        _clear_ler_match_highlight()
        _refresh_ler_match_label(idx)
    _inst_assigned_lbl.text = f"  Label: {label_name}"
    _inst_assigned_lbl.visible = True
    _save_instance_ply(idx, label_name)
    _apply_instance_color(idx)
    # Move on only once this instance is finished. A label on its own leaves the
    # LER link undecided, and advancing there made it easy to label a whole set
    # and match none of it. Confirming the link does not advance either: a pick
    # is often corrected by clicking a different line, and shift+click adjusts
    # the link one feature at a time, so the view has to stay where it is.
    # Prev / Skip / Next remain the way to move by hand.
    if _instance_confirmed(idx):
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
        print(f"  [label] staying on instance {idx}: LER match not confirmed yet"
              f"  (click a line, 'Suggest LER match', or 'Mark as NOT in LER')")
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


# A pipe's cylinder is drawn a radius below the registered line (see the load
# loop), which leaves the line itself clear and a highlight on it visible. A
# trace has no such gap: its corridor ribbon is exactly coplanar with the line
# and its centreline tube is centred on it, so a highlight drawn there is
# buried inside both. Lift it clear of the tube to put a trace highlight where
# every other layer's already sits.
_TRACE_HIGHLIGHT_LIFT = np.array([0.0, 0.0, TRACE_CENTERLINE_RADIUS + 0.005])
_NO_LIFT = np.zeros(3)


def _line_lineset(gml_ids, color):
    """LineSet over every loaded segment of the given features, or None when
    none of them is present in this footprint."""
    wanted = set(gml_ids or [])
    pts, lines = [], []
    for _i, _gid in enumerate(pick_seg_gml_id):
        if _gid not in wanted:
            continue
        _lift = (_TRACE_HIGHLIGHT_LIFT if is_trace_key(pick_seg_layer[_i])
                 else _NO_LIFT)
        pts.extend([pick_seg_p1[_i] + _lift, pick_seg_p2[_i] + _lift])
        lines.append([len(pts) - 2, len(pts) - 1])
    if not lines:
        return None
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(pts)),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.paint_uniform_color(color)
    return ls


def _highlight_material():
    mat = line_material(6.0)
    try:
        # Would draw through occluding pipe meshes, but MaterialRecord has no
        # depth_func in Open3D 0.19, so this is a no-op here and the highlight
        # depth-tests normally. Placing it clear of the geometry (above) is
        # what actually keeps it visible.
        mat.depth_func = "always"
    except AttributeError:
        pass
    return mat


def _place_ler_match_highlight(gml_ids):
    """Outline every segment of the linked utility line.

    A line is usually several GML features, so highlighting only the clicked
    one made a whole-line link look like a fragment.
    """
    _clear_ler_match_highlight()
    # magenta — distinct from any LER layer colour
    ls = _line_lineset(gml_ids, [1.0, 0.0, 1.0])
    if ls is not None:
        scene_widget.scene.add_geometry(LER_MATCH_HIGHLIGHT_GEOM, ls, _highlight_material())


# ── Label / match consistency ────────────────────────────────────────────────
# A match names the registered counterpart of a labelled instance, so the two
# have to describe the same kind of utility. Nothing enforced this before, and
# three matches in the dataset ended up contradicting their label. The rule
# comes from core.config.ler_layers_for_type, the same one the deviation viewer
# uses to decide which LER layers a type may be measured against, so the two
# modules cannot drift apart again.


def _allowed_layers_for_label(label_name):
    """Layer keys a labelled instance may match, or None for no restriction.

    None means the label carries no LER mapping (an unlabelled instance, or a
    class the register has no counterpart layer for), in which case any pick is
    accepted and the label is inferred from it instead.
    """
    if not label_name:
        return None
    return ler_layers_for_type(_LABEL_TO_ID.get(label_name),
                               set(pick_seg_layer))


def _match_conflicts(idx, layer_name):
    """True when ``layer_name`` contradicts the instance's current label."""
    allowed = _allowed_layers_for_label(_instance_labels.get(idx))
    return allowed is not None and layer_name not in allowed


def _record_match_conflict(idx, layer_name, gml_ids, overridden):
    """Log a conflicting pick so a refusal is never silently lost.

    Refused picks land here only; an overridden one is also written into the
    match itself, so the deviation viewer and pipeline_status can count it.
    """
    label = _instance_labels.get(idx, "(unlabelled)")
    rec = {"label": label, "layer": layer_name,
           "gml_ids": list(gml_ids), "overridden": bool(overridden)}
    _instance_match_conflicts[idx] = rec
    verb = "overridden" if overridden else "refused"
    print(f"  [ler-match] {verb}: instance {idx} "
          f"({instance_data[idx]['name']}) is labelled {label}, but "
          f"{layer_display_name(layer_name)} belongs to another utility type."
          + ("" if overridden else "  Ctrl+click to record it anyway."))
    _write_match_conflicts_json()


def _match_gml_ids(match):
    """The features a stored match covers, newest key first so a record written
    before utility lines existed still resolves."""
    if not match:
        return []
    ids = match.get("gml_ids")
    if ids:
        return list(ids)
    gid = match.get("gml_id")
    return [gid] if gid else []


def _refresh_ler_match_label(idx):
    m = _instance_ler_match.get(idx)
    if m and m.get("no_ler"):
        _ler_match_lbl.text = "LER match: confirmed NOT in LER"
        _ler_match_lbl.text_color = gui.Color(1.0, 0.55, 0.15, 1.0)
    elif m:
        ids = _match_gml_ids(m)
        gid = m.get("line_id") or m.get("gml_id", "")
        gid_short = gid[-28:] if len(gid) > 28 else gid
        extent = f" (line, {len(ids)} features)" if len(ids) > 1 else ""
        flag = "  [CONFLICTS WITH LABEL]" if m.get("conflict") else ""
        _ler_match_lbl.text = (f"LER match: {layer_display_name(m['layer'])}{extent}"
                               f"{flag}\n({gid_short})")
        _ler_match_lbl.text_color = (gui.Color(1.0, 0.4, 0.4, 1.0) if m.get("conflict")
                                     else gui.Color(0.3, 1.0, 1.0, 1.0))
    elif idx in _instance_match_conflicts:
        _rec = _instance_match_conflicts[idx]
        _ler_match_lbl.text = (f"LER match: refused "
                               f"{layer_display_name(_rec['layer'])}"
                               f"\ndoes not match {_rec['label']}")
        _ler_match_lbl.text_color = gui.Color(1.0, 0.4, 0.4, 1.0)
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
    # Clearing the match clears the refusal with it, so the panel does not keep
    # reporting a conflict the user has just walked away from.
    _instance_match_conflicts.pop(idx, None)
    _refresh_ler_match_label(idx)
    _clear_ler_match_highlight()
    _clear_suggestion_highlight()
    _write_ler_matches_json()
    _write_match_conflicts_json()
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


def _place_suggestion_highlight(gml_ids):
    """Outline the whole candidate line, so what Accept would record is what is
    shown."""
    _clear_suggestion_highlight()
    # yellow — tentative, vs. magenta for a confirmed match
    ls = _line_lineset(gml_ids, [1.0, 0.85, 0.0])
    if ls is not None:
        scene_widget.scene.add_geometry(LER_SUGGEST_HIGHLIGHT_GEOM, ls, _highlight_material())


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
    extent = f", {len(c['gml_ids'])} features" if len(c["gml_ids"]) > 1 else ""
    _suggest_lbl.text = (f"Suggestion {i + 1}/{len(cands)}: {layer_display_name(c['layer'])}"
                        f"{extent}\nscore={c['score']:.2f}  ({parts_str})")
    _suggest_lbl.text_color = gui.Color(1.0, 0.85, 0.2, 1.0)
    _place_suggestion_highlight(c["gml_ids"])
    window.post_redraw()


def _suggest_ler_match():
    if not instance_data:
        return
    idx = _current_inst_idx[0]
    inst = instance_data[idx]
    pts = np.asarray(inst["pcd"].points)
    colors = np.asarray(inst["pcd"].colors) if inst["pcd"].has_colors() else None

    # Restricted through the shared rule, so a Ledningstrace whose forsyningsart
    # matches the label is offered alongside the utility's own line layer. The
    # bare UTILITY_TO_LER_MATCH layer set used to exclude every trace, which is
    # why no trace could ever be suggested for a labelled instance.
    allowed_layers = _allowed_layers_for_label(_instance_labels.get(idx))

    # Scored per utility line, not per feature, so a suggestion covers the whole
    # run instead of whichever fragment happens to sit nearest the instance.
    candidates = score_candidates(pts, colors, _ler_line_index, allowed_layers=allowed_layers)
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
    # Suggestions are already restricted to the label's allowed layers, so this
    # can only fire for an unlabelled instance whose label arrived in between.
    if _match_conflicts(idx, c["layer"]):
        _record_match_conflict(idx, c["layer"], c["gml_ids"], overridden=False)
        return
    _instance_match_conflicts.pop(idx, None)
    # Candidates come from the line-merged index, so gml_id is the line_id and
    # gml_ids are the features it covers.
    _instance_ler_match[idx] = {"layer": c["layer"], "gml_id": c["gml_ids"][0],
                                "gml_ids": list(c["gml_ids"]), "line_id": c["gml_id"]}
    print(f"  [ler-match] Instance {idx} ({instance_data[idx]['name']}) "
          f"-> {c['layer']}  line={c['gml_id']} ({len(c['gml_ids'])} feature(s))"
          f"  (accepted suggestion, score={c['score']:.2f})")
    _maybe_autolabel_from_layer(idx, c["layer"])
    _refresh_ler_match_label(idx)
    _place_ler_match_highlight(c["gml_ids"])
    _clear_suggestion_highlight()
    _write_ler_matches_json()
    _write_match_conflicts_json()
    window.post_redraw()


_ler_last_click = [None]
_ler_last_click_shift = [False]   # shift held: adjust the link, do not replace it
_ler_last_click_ctrl = [False]    # ctrl held: record a match that contradicts the label


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

    # A pick that contradicts the label is refused, because the deviation viewer
    # trusts the match without rechecking it. Ctrl+click records it anyway and
    # flags it, for the case where the register genuinely disagrees with the
    # trench and the disagreement is the finding.
    conflict = _match_conflicts(idx, layer_name)
    if conflict:
        overridden = _ler_last_click_ctrl[0]
        _record_match_conflict(idx, layer_name, _line_gml_ids(gml_id), overridden)
        if not overridden:
            return
    elif idx in _instance_match_conflicts:
        # A consistent pick supersedes the recorded conflict for this instance.
        _instance_match_conflicts.pop(idx, None)
        _write_match_conflicts_json()

    if _ler_last_click_shift[0]:
        # Shift-click adjusts the current link one feature at a time, for the
        # cases the automatic grouping gets wrong: a run the registry leaves a
        # gap in, or a neighbour it joined that should have stayed separate.
        prev = _instance_ler_match.get(idx)
        ids = _match_gml_ids(prev) if prev and not prev.get("no_ler") else []
        if prev and not prev.get("no_ler") and prev.get("layer") != layer_name:
            print(f"  [ler-match] ignored: {layer_display_name(layer_name)} is not "
                  f"the linked layer ({layer_display_name(prev['layer'])})")
            return
        if gml_id in ids:
            ids = [g for g in ids if g != gml_id]
            action = "removed from"
        else:
            ids = ids + [gml_id]
            action = "added to"
        if ids:
            _instance_ler_match[idx] = {"layer": layer_name, "gml_id": ids[0],
                                        "gml_ids": ids,
                                        "line_id": (prev or {}).get("line_id") or ids[0]}
        else:
            _instance_ler_match.pop(idx, None)
        print(f"  [ler-match] Instance {idx} ({instance_data[idx]['name']}): "
              f"{gml_id} {action} the link ({len(ids)} feature(s))")
    else:
        ids = _line_gml_ids(gml_id)
        _instance_ler_match[idx] = {"layer": layer_name, "gml_id": gml_id,
                                    "gml_ids": ids, "line_id": _line_of.get(gml_id, gml_id)}
        print(f"  [ler-match] Instance {idx} ({instance_data[idx]['name']}) "
              f"-> {layer_name}  line={_line_of.get(gml_id, gml_id)} "
              f"({len(ids)} feature(s), clicked {gml_id})")

    # An overridden match carries the conflict with it, so the deviation viewer
    # and pipeline_status can count register disagreements rather than treating
    # the link as an ordinary one.
    if conflict and idx in _instance_ler_match:
        _instance_ler_match[idx]["conflict"] = True
        _instance_ler_match[idx]["label_at_match"] = _instance_labels.get(idx)

    def _update():
        _maybe_autolabel_from_layer(idx, layer_name)
        _place_ler_match_highlight(_match_gml_ids(_instance_ler_match.get(idx)))
        _clear_suggestion_highlight()
        _refresh_ler_match_label(idx)
        _write_ler_matches_json()
        _write_match_conflicts_json()
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
        try:
            _ler_last_click_shift[0] = event.is_modifier_down(gui.KeyModifier.SHIFT)
            _ler_last_click_ctrl[0] = event.is_modifier_down(gui.KeyModifier.CTRL)
        except AttributeError:
            _ler_last_click_shift[0] = False
            _ler_last_click_ctrl[0] = False
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
        print("                 (the whole line, every connected feature on it)")
        print("  Shift-click    add/remove one feature from the current link,")
        print("                 when the line grouping needs correcting")
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
