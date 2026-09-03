# -*- coding: utf-8 -*-
"""
Geometric Deviation Viewer — Instances vs LER Utility Registry
===============================================================
Refactored to use core/ for shared configuration and data loading.

If label_module.py recorded an exclusive LER match for an instance
(ler_matches.json next to its labelled PLYs, naming a whole utility line by the
gml_ids of every feature on it), that instance
is measured against only its linked LER feature, in both directions: the
instance's own deviation stats/colouring, and that feature's discretized
LER-surface deviation clouds. Instances without a recorded match keep the
original behaviour of measuring against every nearby LER feature whose layer
matches the instance's utility type.

Two deviation metrics are reported side by side:

* **Crown line** (headline) — the measured top centreline recovered from the
  instance cloud by core/crown.py, compared line-to-line against the registered
  line. Both sides are then the same datum LER actually registers (horizontal
  centreline, vertical top), so the comparison carries no radius bias.
* **Point cloud** — every measured point against the registered line, the
  original metric. It is defined for every instance, including the ones whose
  shape yields no crown line, but it compares a surface against a line.

Usage: python modules/deviation_module.py
  Change the site in core/site_local.py.
"""

import copy
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
from shapely.geometry import LineString as ShapelyLine, Point as ShapelyPoint, box as shapely_box
from shapely.ops import unary_union

from core.config import (
    PLY_FILE, GML_PATH, AREA_REF_GEOJSON, CROP_RADIUS, CROP_MODE, UTILITY_RECT_BUFFER,
    PANEL_WIDTH_EM, LEDNINGSPAKKE_LABEL, layer_display_name,
    LINE_LAYERS, COMPONENT_LAYERS, COMP_TO_LINE,
    COMPONENT_SPHERE_RADIUS,
    PIPE_DEPTH_CONFIG, COMPONENT_DEPTH_CONFIG,
    UTILITY_TYPE_LABELS, UTILITY_TYPE_COLORS, UTILITY_TO_LER_MATCH,
    DEVIATION_THRESHOLDS, DEVIATION_COLORS, DEVIATION_CLASS_LABELS,
    DEVIATION_GRADIENT_TICKS,
    ACCURACY_CLASS_COLORS, ACCURACY_UNREGISTERED_COLOR, ACCURACY_UNREGISTERED_LABEL,
    accuracy_class_color,
    KLIC_XY_THRESHOLDS, KLIC_XY_COLORS, KLIC_XY_CLASS_LABELS,
    FORSYNINGSART_COLOR_HINTS, FORSYNINGSART_TO_LINE,
    forsyningsart_color as _forsyningsart_color,
    ler_layers_for_type,
    LEDNINGSTRACE_FALLBACK_WIDTH,
)
from core.data_loader import (
    init_site, read_ply_with_utility_type, utility_type_from_filename,
    load_or_pick_ground_level, load_or_pick_trench, trench_path_from_vertices,
    instance_base_name,
    feature_accuracy_tolerance, accuracy_class_coverage,
)
from core.site_status import (
    instance_dir_for, resolve_labeled_dir, root_class_instances,
)
from core.geometry import (
    batch_point_to_segments, batch_point_to_plane_segments,
    batch_point_to_plane_segment_components,
    discretize_segment,
    deviation_to_color, deviation_to_color_continuous,
    accuracy_buffer_polygon, polygon_to_o3d_mesh, polygon_to_o3d_lineset,
    merge_linesets, drape_z_from_polylines,
)
from core.signature_legend import SignatureLegendSection
from core.crop import CropRegion
from core.crown import crown_line
from core.depth import clean_coords_with_depth as _core_clean_coords
from core.gui_helpers import (
    make_legend_row, LerLegendSection,
    PanelTextFitter,
    pivot_oblique, top_view, trench_or_scene_frame,
)
from core.ledningstrace import get_bredde_width, is_trace_key, ribbon_alpha
from core.trace_render import (
    build_trace_centerlines, add_trace_centerlines, trace_centerline_gn,
)
from core import symbology as sym
from core.signature_render import (
    PolylineDash, line_segment_mesh,
    feature_signature_meshes_3d, stitch_clipped_segments, merge_meshes,
    signature_gn, show_signatures,
)
from core.rendering import (
    point_material_shaded, point_material_flat, mesh_material, line_material,
    setup_scene_lighting,
)

# ─────────────────────────────────────────────────────────────────────────────
# INITIALISE — load area offset, point cloud, and GML via core/
# ─────────────────────────────────────────────────────────────────────────────
# GML is read layer-by-layer below (the loop needs per-feature control), so
# init_site must not pre-load it a second time.
site = init_site(load_gml=False, load_instances=True)

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
all_seg_crown_offset = [] # top->axis lowering per segment: the tube radius for every
                          # pipe (registered or fallback diameter), 0 for traces
all_seg_gml_id = []       # GML gml_id per segment — identifies the whole feature
all_seg_acc_class = []    # registered noejagtighedsklasse per segment: 1..5, 0 = none
all_seg_depth_source = [] # DepthSource per segment: how its Z was arrived at. The
                          # worst of the two endpoints, so a segment counts as
                          # REGISTERED only when both ends genuinely are.
all_seg_owner = []        # ledningsejer per segment (mandatory on every feature)
all_seg_etabl = []        # etableringstidspunkt per segment, "" when absent
# Dash pattern per segment, (PolylineDash, index in its polyline) or None. Only
# how the segment is drawn: every array above still holds every segment in full,
# so nothing the deviation is measured against changes.
all_seg_dash = []
ler_meshes = {}           # layer -> merged TriangleMesh (for visualisation)
ler_meshes_acc = {}       # layer -> same mesh, painted by registered accuracy class
_sig_layer_meshes = {}    # layer -> [TriangleMesh, ...] LER signature overlay parts
_layer_avg_depth_local = {}  # layer_name -> float (average local Z for component depth fallback)
ler_stats = {}            # layer -> (n_feat_active, n_seg_active, n_feat_inactive, n_seg_inactive)


# Crop-region selection/clipping: one shared implementation in core.crop.
_crop_region = CropRegion.from_pointcloud(site.pc, TX, TY)
_in_crop_utm          = _crop_region.polyline_in_region_utm
_clip_segment_to_crop = _crop_region.clip_local


def _to_local(coords_utm, vejl_dybde_mm=None,
              cfg=PIPE_DEPTH_CONFIG, parent_avg_z=None):
    """UTM -> local translation + DepthSource fallback (core.depth), bound to
    this viewer's flat ground level.

    Returns ``(coords, sources)``. The per-vertex sources are kept rather than
    dropped because they decide whether a vertical comparison means anything: a
    vertex resolved by GROUND_PLANE carries the ground level picked from this very
    point cloud, so measuring the cloud against it is circular. Only
    DepthSource.REGISTERED is an independent claim about depth.
    """
    return _core_clean_coords(coords_utm, vejl_dybde_mm,
                              TX=TX, TY=TY, TZ=TZ,
                              ground_z_at=lambda x, y: GROUND_Z,
                              cfg=cfg, parent_avg_z=parent_avg_z)


# Accuracy-class colouring of the utility geometry itself (the "Colour by
# accuracy class" toggle). A layer is one merged mesh, but the class is
# registered per feature, so the parts are recoloured after the merge: each part
# contributes a known number of vertices, in merge order, and gets its feature's
# class colour. Only the colours are rebuilt, never the geometry.
_acc_view_reg = 0     # features in view whose accuracy class is registered
_acc_view_total = 0   # features in view


def _acc_colored_mesh(mesh, parts):
    """Copy of a merged layer mesh painted by registered accuracy class.

    ``parts`` is ``[(n_vertices, class_idx), ...]`` in merge order, with
    ``class_idx`` 0 where the feature registers no class. Returns ``None`` when
    the vertex counts do not add up to the merged mesh, so the caller keeps the
    plain utility colouring rather than painting the wrong features.
    """
    if not parts:
        return None
    counts = np.array([n for n, _ in parts], dtype=int)
    if int(counts.sum()) != len(mesh.vertices):
        return None
    cols = np.array([accuracy_class_color(c) for _, c in parts], dtype=float)
    out = copy.deepcopy(mesh)
    out.vertex_colors = o3d.utility.Vector3dVector(np.repeat(cols, counts, axis=0))
    return out


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
    _trace_sub_acc = {}    # display_name -> [(n_vertices, class_idx), ...]
    _trace_sub_stats = {}  # display_name -> [n_feat_act, n_seg_act, n_feat_inact, n_seg_inact]
    n_feat_act, n_seg_act = 0, 0
    n_feat_inact, n_seg_inact = 0, 0
    layer_cyls = []
    layer_acc = []         # (n_vertices, class_idx) per mesh in layer_cyls
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
            bredde_m = LEDNINGSTRACE_FALLBACK_WIDTH

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
        gml_id_val = str(row.get("gml_id", "") or "")

        # Registered horizontal accuracy class of this feature (0 = not
        # registered), carried per segment so the geometry can be coloured by it.
        _tol = feature_accuracy_tolerance(row)
        acc_class = _tol[1] if _tol is not None else 0

        # Owner and establishment date, carried per segment. They separate an
        # independently registered feature from one surveyed by the same campaign
        # that captured the point cloud, which a deviation figure must not pool.
        owner_val = str(row.get("ledningsejer", "") or "").strip()
        etabl_val = str(row.get("etableringstidspunkt", "") or "").strip()[:10]

        # LER signature choice: dashed for driftsstatus "under etablering", red
        # triangles for fareklasse "meget farlig", El voltage ticks. Purely a
        # display choice: neither the dash cut into the line nor the markers
        # enter the segment arrays the deviation is measured against.
        _sig_style, _sig_hazard, _sig_ticks = sym.signature_choice(row, layer_name)
        _sig_any = _sig_hazard or _sig_ticks > 0
        # The registered vertical coordinate is the crown (top) of the utility,
        # not its axis, for every LER utility (featurekatalog, geometri
        # attribute: "Vertikale koordinater af geometrien angives for overkanten
        # af ledningen"), so the drawn cylinder is lowered by its radius and its
        # crown lands on the registered line. A trace's ribbon is already at that
        # level and drops by nothing. Per feature, not per segment: the dash
        # phase below needs the whole polyline on the axis actually drawn.
        _crown_offset = radius if bredde_m is None else 0.0
        _axis_dz = np.array([0.0, 0.0, _crown_offset])

        hit = False
        for sub in subs:
            coords_raw = np.array(sub.coords, dtype=float)
            if not _in_crop_utm(coords_raw):
                continue
            coords, z_src = _to_local(coords_raw, vejl)
            _layer_z_vals.extend(coords[:, 2].tolist())
            hit = True
            # Dash phase belongs to the whole polyline, so it is resolved once per
            # polyline and read per segment; taken per segment it would restart at
            # every vertex and lose any dash straddling one.
            _dash = PolylineDash(coords - _axis_dz) if _sig_style == "dashed" else None
            _sig_chords = []
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
                all_seg_crown_offset.append(_crown_offset)
                all_seg_gml_id.append(gml_id_val)
                all_seg_acc_class.append(acc_class)
                # DepthSource is ordered worst-last, so max() of the two endpoints
                # is the weaker claim of the pair.
                all_seg_depth_source.append(int(max(z_src[i], z_src[i + 1])))
                all_seg_owner.append(owner_val)
                all_seg_etabl.append(etabl_val)
                all_seg_dash.append((_dash, i) if _dash is not None else None)
                _ax1, _ax2 = cp1 - _axis_dz, cp2 - _axis_dz
                mesh = line_segment_mesh(_ax1, _ax2, color, radius=radius,
                                         width=bredde_m, dash=_dash, index=i)
                if _sig_any:
                    _sig_chords.append((_ax1, _ax2))
                if mesh is not None:
                    _part = (len(mesh.vertices), acc_class)
                    if is_trace:
                        _trace_sub_cyls.setdefault(display_name, []).append(mesh)
                        _trace_sub_acc.setdefault(display_name, []).append(_part)
                    else:
                        layer_cyls.append(mesh)
                        layer_acc.append(_part)
                if is_active:
                    n_seg_act += 1
                else:
                    n_seg_inact += 1
            for _piece in stitch_clipped_segments(_sig_chords):
                _sig_layer_meshes.setdefault(display_name, []).extend(
                    feature_signature_meshes_3d(
                        _piece, color, hazard=_sig_hazard,
                        tick_count=_sig_ticks, radius=radius, width=bredde_m))
        if hit:
            _acc_view_total += 1
            if acc_class:
                _acc_view_reg += 1
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
            _m_acc = _acc_colored_mesh(m, _trace_sub_acc.get(dname, []))
            if _m_acc is not None:
                ler_meshes_acc[dname] = _m_acc

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
            _m_acc = _acc_colored_mesh(m, layer_acc)
            if _m_acc is not None:
                ler_meshes_acc[layer_name] = _m_acc
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
seg_crown_offset = np.array(all_seg_crown_offset, dtype=float) if all_seg_crown_offset else np.empty(0, dtype=float)
seg_gml_id = np.array(all_seg_gml_id, dtype=object) if all_seg_gml_id else np.empty(0, dtype=object)
seg_acc_class = np.array(all_seg_acc_class, dtype=int) if all_seg_acc_class else np.empty(0, dtype=int)
seg_depth_source = (np.array(all_seg_depth_source, dtype=int) if all_seg_depth_source
                    else np.empty(0, dtype=int))
seg_owner = (np.array(all_seg_owner, dtype=object) if all_seg_owner
             else np.empty(0, dtype=object))
seg_etabl = (np.array(all_seg_etabl, dtype=object) if all_seg_etabl
             else np.empty(0, dtype=object))
n_total_segs = len(seg_p1)
n_active_segs = int(seg_active.sum()) if len(seg_active) else 0
n_inactive_segs = n_total_segs - n_active_segs

# Per-layer merged signature overlays. Kept out of ler_meshes because that mesh
# is swapped for a deviation-coloured cloud and repainted by accuracy class,
# neither of which a fixed legend colour may follow.
sig_meshes = {}
for _sln, _sms in _sig_layer_meshes.items():
    _sm = merge_meshes(_sms)
    if _sm is not None:
        sig_meshes[_sln] = _sm

# Trace centrelines: the corridor ribbon is drawn transparent (see
# core/trace_render.py), so the registered centreline is drawn as a thin tube
# through the same mesh material as the pipes. Shown only in the solid colour
# modes; the LER deviation modes already render a trace as a deviation-coloured
# centreline cloud, which this tube would cover.
_trace_centerlines = build_trace_centerlines(
    seg_p1, seg_p2, all_seg_layer,
    lambda k: LINE_LAYERS.get(k, {}).get("color", [0.5, 0.5, 0.5]),
    dash_of_index=lambda i: all_seg_dash[i])
# Second set for the accuracy-class colouring. The centreline is the visible
# part of a trace (the corridor ribbon is nearly transparent), so it has to
# carry the class too, per segment: one trace layer holds several features and
# they need not share a class.
_trace_centerlines_acc = build_trace_centerlines(
    seg_p1, seg_p2, all_seg_layer,
    lambda k: LINE_LAYERS.get(k, {}).get("color", [0.5, 0.5, 0.5]),
    color_of_index=lambda i: accuracy_class_color(seg_acc_class[i]),
    dash_of_index=lambda i: all_seg_dash[i])

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
comp_meshes_acc = {}    # layer_name -> same mesh, painted by registered accuracy class
comp_meshes_flat = {}   # layer_name -> same mesh, centres moved to street level
comp_stats = {}         # layer_name -> int count
_comp_acc_cov_rows = [] # (layer, has_column, n_registered_total, n_total)

for comp_layer, comp_cfg in COMPONENT_LAYERS.items():
    try:
        gdf_c = gpd.read_file(GML_PATH, layer=comp_layer)
    except Exception:
        continue

    _comp_acc_cov_rows.append((comp_layer,) + accuracy_class_coverage(gdf_c))

    color = comp_cfg["color"]
    n_comp = 0
    spheres = []
    spheres_flat = []   # same spheres, centres at street level (_FLAT_LER_MODES)
    comp_acc = []       # (n_vertices, class_idx) per sphere, in merge order

    parent_line = COMP_TO_LINE.get(comp_layer)
    parent_avg_z = _layer_avg_depth_local.get(parent_line) if parent_line else None

    for _, row in gdf_c.iterrows():
        g = row.geometry
        if g is None or g.geom_type not in ("Point", "PointZ"):
            continue
        if not _crop_region.contains_utm(g.x, g.y):
            continue

        # Same resolver and the same component configuration as every other
        # module: REGISTERED -> LAYER_MEAN -> GROUND_PLANE.
        pt_arr, _comp_src_arr = _to_local(
            np.array([[g.x, g.y, g.z]], dtype=float), None,
            cfg=COMPONENT_DEPTH_CONFIG, parent_avg_z=parent_avg_z,
        )
        pt = pt_arr[0]

        if not _crop_region.contains_local(pt[0], pt[1]):
            continue

        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=COMPONENT_SPHERE_RADIUS, resolution=12)
        sphere.translate(pt)
        sphere.paint_uniform_color(color)
        spheres.append(sphere)
        # Twin for the modes that draw the register flat. A translation, not a
        # flattening: the component keeps its shape, only its centre moves.
        _flat = copy.deepcopy(sphere)
        _flat.translate([0.0, 0.0, GROUND_Z - pt[2]])
        spheres_flat.append(_flat)
        _tol_c = feature_accuracy_tolerance(row)
        _cls_c = _tol_c[1] if _tol_c is not None else 0
        comp_acc.append((len(sphere.vertices), _cls_c))
        _acc_view_total += 1
        if _cls_c:
            _acc_view_reg += 1
        n_comp += 1

    comp_stats[comp_layer] = n_comp
    if spheres:
        m = spheres[0]
        for s in spheres[1:]:
            m += s
        m.compute_vertex_normals()
        comp_meshes[comp_layer] = m
        _m_acc = _acc_colored_mesh(m, comp_acc)
        if _m_acc is not None:
            comp_meshes_acc[comp_layer] = _m_acc
        _mf = spheres_flat[0]
        for s in spheres_flat[1:]:
            _mf += s
        _mf.compute_vertex_normals()
        comp_meshes_flat[comp_layer] = _mf
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
            cl, _ = _to_local(cr, vejl)
            local_lines.append(cl[:, :2])
            local_lines_xyz.append(cl)
        if not local_lines:
            continue
        grp_in_view[display_name] = grp_in_view.get(display_name, 0) + 1

        tol = feature_accuracy_tolerance(row)
        if tol is None:
            continue                      # in view but accuracy class not registered
        half_width, cls_idx = tol
        color = ACCURACY_CLASS_COLORS[cls_idx - 1]
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
for _ln, _has, _nreg, _ntot in _acc_cov_rows + _comp_acc_cov_rows:
    status = f"{_nreg}/{_ntot} registered" if _has else "no column"
    print(f"    {_ln:<32} {status}")
_n_acc_view = sum(v[0] for v in accbuf_stats.values())
print(f"  Buffers built for {_n_acc_view} features within the view")
print(f"  Class colouring: {_acc_view_reg}/{_acc_view_total} features in view "
      f"carry a class")


def _get_matching_segment_mask(utility_type, active_only=None):
    """Return a boolean mask over seg_p1/seg_p2 for segments matching this utility type.

    active_only: None = both, True = only active, False = only inactive.
    """
    allowed = ler_layers_for_type(utility_type, set(all_seg_layer))
    if allowed is None:
        # No mapping for this type, so the instance carries no usable label.
        # Match nothing rather than everything: widening to all layers would
        # measure the instance against the nearest geometry of any utility and
        # return a finite, plausible-looking deviation built on no information.
        # This also matches _get_matching_ler_names(), which already returns an
        # empty set here.
        mask = np.zeros(len(seg_p1), dtype=bool)
    else:
        mask = np.array([ln in allowed for ln in all_seg_layer], dtype=bool)
        if not len(mask):
            mask = np.zeros(len(seg_p1), dtype=bool)

    if active_only is True:
        mask &= seg_active
    elif active_only is False:
        mask &= ~seg_active
    return mask


def _get_matching_ler_names(utility_type):
    """Return set of LER layer display names that match the given utility type."""
    return ler_layers_for_type(utility_type, ler_meshes) or set()


def _get_matching_accbuf_keys(utility_type):
    """Return the accuracy-buffer layer keys that match the given utility type.

    Covers line layers, their components (via COMP_TO_LINE) and Ledningstrace
    sub-layers whose forsyningsart maps to a matching line, so the utility filter
    shows only the selected utility's registered-accuracy buffers."""
    return ler_layers_for_type(utility_type,
                               set(accbuf_fill) | set(accbuf_outline),
                               include_components=True) or set()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Load instances + compute deviations against LER segments
# ─────────────────────────────────────────────────────────────────────────────
_ply_stem = _ply_path.stem
_inst_base = instance_base_name(_ply_path)

# New convention: permanent <base>_Instances/ directory
_perm_dir = instance_dir_for(_ply_path)
_inst_dir = None
_inst_files = []
_src_label = "none"

if _perm_dir.is_dir():
    _inst_dir = _perm_dir
    # Which labelled session counts is decided by core/site_status.py, so this
    # reads the same folder the label viewer writes and the status tool reports.
    _labeled, _empty_labeled, _superseded_labeled = resolve_labeled_dir(_perm_dir)
    if _labeled:
        _inst_files = list(_labeled.ply_files)
        _src_label = _labeled.path.name
        for _sd in _superseded_labeled:
            print(f"  [note] ignoring superseded label session {_sd.path.name}/")
    # Always include top-level PLY files (e.g. water instance 0_instance_0_type_7.ply).
    # root_class_instances() is the same predicate tools/pipeline_status.py counts
    # with, so the reported instance count and the measured population agree by
    # construction. It also keeps out the hand-split parts that CloudCompare left
    # beside the pipes ("0_instance_1_type_valve.ply", "..._type_tiewrap.ply"):
    # those carry no utility_type property and no numeric type token, so they used
    # to resolve to type 0 and be measured against every LER layer at once.
    # A root file shadowed by a same-named file in the label session is dropped:
    # label_module writes the per-class instances back to the root rather than
    # copying them, but an older session may still hold a copy, and loading both
    # would count the same points twice.
    _labeled_names = {p.name for p in _inst_files}
    _root_insts = root_class_instances(_perm_dir)
    _kept_root_names = {p.name for p in _root_insts}
    for _sp in sorted(_perm_dir.glob("*.ply")):
        if _sp.name not in _kept_root_names:
            print(f"  [skip] {_sp.name}: not a labelled instance, not measured")
    _top_level_plys = [p for p in _root_insts if p.name not in _labeled_names]
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

# Exclusive instance -> LER feature links recorded by label_module.py, keyed by
# PLY filename: {"layer": <display name>, "gml_id": <GML gml_id>}. When present
# for an instance, that instance is measured against only this one feature
# instead of every nearby feature whose layer matches the instance's type.
# ler_matches.json lives next to the PLYs it describes; _inst_files can mix
# files from more than one directory (e.g. top-level water instances plus a
# labeled_* subfolder), so every distinct parent directory is checked.
_ler_matches = {}
for _matches_dir in dict.fromkeys(p.parent for p in _inst_files):
    _matches_path = _matches_dir / "ler_matches.json"
    if not _matches_path.is_file():
        continue
    try:
        with open(_matches_path, "r", encoding="utf-8") as f:
            _loaded = json.load(f)
        _ler_matches.update(_loaded)
        print(f"\nLoaded {len(_loaded)} exclusive LER match(es) from "
              f"{_matches_dir.name}/{_matches_path.name}")
    except Exception as e:
        print(f"\n[warn] failed to read {_matches_path}: {e}")

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
    if utility_type == 0:
        print(f"  [warn] {inst_path.name}: no utility_type in the PLY and no "
              f"numeric type in the filename. It matches no LER layer, so it is "
              f"reported without a deviation.")

    ler_match = _ler_matches.get(inst_path.name)
    confirmed_no_ler = bool(ler_match and ler_match.get("no_ler"))
    # An exclusive link names a whole utility line: the registry routinely
    # splits one physical run into several features, and label_module records
    # every gml_id on it. "gml_ids" is the current form; a record written before
    # utility lines existed carries only the single "gml_id".
    match_gml_ids = []
    if ler_match and not confirmed_no_ler:
        match_gml_ids = [g for g in (ler_match.get("gml_ids")
                                     or [ler_match.get("gml_id")]) if g]

    # Compute distances: all matching segments (active + inactive combined for
    # heatmap).
    #   - confirmed_no_ler: label_module recorded that this instance has no
    #     counterpart anywhere in LER, so no segment is ever measured against
    #     it (has_ler stays False) rather than falling back to the nearest
    #     same-type feature, which could be an unrelated, already-registered
    #     utility that merely happens to sit nearby.
    #   - match_gml_ids: an exclusive match restricts this to the linked utility
    #     line (every feature on it, covering all of their clipped
    #     sub-segments) instead of every segment whose layer matches the
    #     instance's utility type.
    #   - otherwise: the original nearest-of-type behaviour.
    if confirmed_no_ler:
        seg_mask_all = np.zeros(len(seg_p1), dtype=bool)
        seg_mask_act = seg_mask_all
        seg_mask_inact = seg_mask_all
    elif match_gml_ids:
        seg_mask_all = np.isin(seg_gml_id, match_gml_ids)
        seg_mask_act = seg_mask_all & seg_active
        seg_mask_inact = seg_mask_all & ~seg_active
    else:
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

    def _signed_stats(d):
        """Summary of a signed vertical offset (measured minus registered).

        Deliberately not _make_stats: a magnitude's mean is positive whatever
        the direction, so it cannot answer "does the register sit above or below
        the utility". The median and the share above zero can. p5/p95 rather
        than min/max, because a single crown station at the end of a run is a
        poor witness for a systematic offset.
        """
        f = np.asarray(d, dtype=float)
        f = f[np.isfinite(f)]
        if len(f) == 0:
            return {"mean": np.nan, "median": np.nan, "p5": np.nan,
                    "p95": np.nan, "frac_above": np.nan, "n_pts": 0}
        return {"mean": float(np.mean(f)), "median": float(np.median(f)),
                "p5": float(np.percentile(f, 5)), "p95": float(np.percentile(f, 95)),
                "frac_above": float(np.mean(f > 0.0)), "n_pts": len(f)}

    # Combined (active + inactive) for heatmap colouring. The XY and Z
    # components are taken at the same nearest segment, so XYZ^2 = XY^2 + Z^2.
    # z_signed keeps the direction the magnitude throws away: positive means the
    # measured point sits above the registered line (see core/geometry.py).
    if has_ler:
        dists, xy_dists, z_dists, z_signed = batch_point_to_plane_segment_components(
            pts_inst, seg_p1[seg_mask_all], seg_p2[seg_mask_all],
            seg_half_width[seg_mask_all])
        stats = _make_stats(dists)
    else:
        dists = np.full(len(pts_inst), np.nan)
        xy_dists = np.full(len(pts_inst), np.nan)
        z_dists = np.full(len(pts_inst), np.nan)
        z_signed = np.full(len(pts_inst), np.nan)
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

    def _dev_pcd(values, continuous, thresholds=None, palette=None):
        """Instance cloud coloured by a deviation metric (grey if no LER)."""
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts_inst)
        if has_ler:
            fn = deviation_to_color_continuous if continuous else deviation_to_color
            pc.colors = o3d.utility.Vector3dVector(fn(values, thresholds, palette))
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

    # ── Crown line: the measured counterpart of the registered datum ─────────
    # LER registers the horizontal centreline carrying the Z of the pipe top, so
    # the crown polyline is what should be compared against it. The crown comes
    # from rolling a ball upwards through the cloud (core/crown.py), which stops
    # at the first surface above it and so returns the pipe rather than anything
    # resting on it. Measuring it against the same unlowered seg_p1/seg_p2 the
    # point clouds use makes both sides the top datum, with no radius bias.
    crown = crown_line(pts_inst)
    crown_dists = crown_xy_dists = crown_z_dists = crown_z_signed = np.empty(0)
    crown_stats = dict(_nan_stats)
    crown_stats["n_pts"] = crown.n_stations
    if crown.ok and has_ler:
        crown_dists, crown_xy_dists, crown_z_dists, crown_z_signed = (
            batch_point_to_plane_segment_components(
                crown.points, seg_p1[seg_mask_all], seg_p2[seg_mask_all],
                seg_half_width[seg_mask_all]))
        crown_stats = _make_stats(crown_dists)

    inst_data = {
        "name": inst_path.stem,
        "utility_type": utility_type,
        "label": ut_label,
        "has_ler": has_ler,
        "ler_match": ler_match,
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
        "z_signed": z_signed,
        "z_signed_stats": _signed_stats(z_signed),
        "crown": crown,
        "crown_distances": crown_dists,
        "crown_xy": crown_xy_dists,
        "crown_z": crown_z_dists,
        "crown_z_signed": crown_z_signed,
        "crown_z_signed_stats": _signed_stats(crown_z_signed),
        "crown_stats": crown_stats,
    }
    class_instances.setdefault(utility_type, []).append(inst_data)

    _ti1 = time.perf_counter()
    if confirmed_no_ler:
        match_tag = "  [confirmed: not in LER]"
    elif match_gml_ids:
        _extent = (f", line of {len(match_gml_ids)} features"
                   if len(match_gml_ids) > 1 else "")
        match_tag = f"  [exclusive match: {ler_match['layer']}{_extent}]"
    else:
        match_tag = ""
    if has_ler:
        tag = f"active={n_act} inactive={n_inact}"
        print(f"  {inst_path.stem}: {len(pts_inst):,} pts  "
              f"type={ut_label}  "
              f"LER({tag})  "
              f"mean={stats['mean']*1000:.1f}mm  "
              f"P95={stats['p95']*1000:.1f}mm  "
              f"max={stats['max']*1000:.1f}mm  "
              f"[{_ti1 - _ti0:.2f}s]{match_tag}")
    else:
        print(f"  {inst_path.stem}: {len(pts_inst):,} pts  "
              f"type={ut_label}  "
              f"** No matching LER utility **  [{_ti1 - _ti0:.2f}s]{match_tag}")

    if crown.ok:
        _r = crown.median_radius
        _dtxt = f"D={2 * _r * 1000:.0f}mm" if _r else "D=n/a"
        _crown_dev = (f"mean={crown_stats['mean']*1000:.1f}mm  "
                      f"P95={crown_stats['p95']*1000:.1f}mm"
                      if has_ler else "no LER to compare")
        _arms = (f"{crown.n_parts} of {crown.n_branches} arms"
                 if crown.n_branches > 1 else "1 arm")
        print(f"      crown line: {_arms}, {crown.n_stations} stations over "
              f"{crown.run_length:.2f} m ({crown.coverage*100:.0f}% covered, "
              f"radius measured at {crown.n_radius})  {_dtxt}  "
              f"{_crown_dev}")
        # Both sides are the top of the utility, so the sign is readable as a
        # depth statement rather than just an offset.
        _sz = inst_data["crown_z_signed_stats"]
        if has_ler and np.isfinite(_sz["median"]):
            _above = _sz["median"] > 0
            print(f"        vertical: crown sits {abs(_sz['median'])*1000:.0f}mm "
                  f"{'above' if _above else 'below'} the registered top (median), "
                  f"so measured {'shallower' if _above else 'deeper'} than "
                  f"registered  [{_sz['frac_above']*100:.0f}% of stations above]")
    else:
        print(f"      crown line: none ({crown.reason})")
    # An arm the footprint found but the crown does not cover is missing from the
    # statistics, so it is named rather than left implicit.
    for _note in crown.notes:
        print(f"        not covered: {_note}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Per-class summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  DEVIATION SUMMARY -Instances vs LER (by utility class)")
print("=" * 72)

# ───── Trench footprint: restricts the LER deviation clouds ─────
# The user marks the trench outline by picking points (Shift+Click) on the
# cloud at startup; the footprint is the XY polygon through those points
# (convex hull by default, pick order optionally), cached in a JSON next to the
# site PLY so it survives restarts.
#
# It restricts the LER side only. A registered line spans the whole crop region,
# so cutting it down to the excavated stretch is what the footprint is for. An
# instance is already utility exposed in the trench, and a hull through a handful
# of picked points rarely brackets a pipe running into the trench wall, so
# restricting the instances only ever clipped their ends out of the colouring and
# out of the statistics.
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


def _build_class_summaries():
    """Per-class deviation statistics over every measured point of the class.
    The footprint restricts the LER clouds only, so a pipe end reaching past the
    picked outline still counts here."""
    summaries = {}
    for ut, instances in sorted(class_instances.items()):
        label = UTILITY_TYPE_LABELS.get(ut, f"Unknown({ut})")
        has_ler = any(inst["has_ler"] for inst in instances)
        total_act = sum(inst["n_active_segs"] for inst in instances)
        total_inact = sum(inst["n_inactive_segs"] for inst in instances)

        total_pts = sum(inst["stats"]["n_pts"] for inst in instances)

        def _pool(arr_key, only_ler=False):
            parts = []
            for inst in instances:
                if only_ler and not inst["has_ler"]:
                    continue
                a = inst.get(arr_key)
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

        # ── Crown line (headline metric) ────────────────────────────────────
        # Pooled over the class's crown lines, station by station. Unrestricted
        # like the point-cloud metric, so the two now span the same stretch of
        # pipe and stay comparable.
        def _crown_pool(arr_key):
            parts = []
            for inst in instances:
                if not inst["crown"].ok or not inst["has_ler"]:
                    continue
                a = inst.get(arr_key)
                if a is not None and len(a):
                    parts.append(a)
            return np.concatenate(parts) if parts else np.array([])

        def _crown_agg(arr_key):
            a = _crown_pool(arr_key)
            if a.size == 0:
                return None
            return {"mean": float(np.mean(a)), "median": float(np.median(a)),
                    "std": float(np.std(a)), "p95": float(np.percentile(a, 95)),
                    "max": float(np.max(a)), "n": int(a.size)}

        def _crown_signed_agg(arr_key):
            """Pooled signed vertical offset. Separate from _crown_agg because a
            max over a signed set names the most extreme station in one direction
            only, which says nothing about a systematic bias; p5/p95 bracket both
            directions and frac_above states which way the class leans."""
            a = _crown_pool(arr_key)
            a = a[np.isfinite(a)] if a.size else a
            if a.size == 0:
                return None
            return {"mean": float(np.mean(a)), "median": float(np.median(a)),
                    "p5": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
                    "frac_above": float(np.mean(a > 0.0)), "n": int(a.size)}

        _crowns = [inst["crown"] for inst in instances if inst["crown"].ok]
        _radii = [c.median_radius for c in _crowns if c.median_radius]

        base = {
            "label": label, "n_instances": len(instances), "n_points": total_pts,
            "has_ler": has_ler,
            "n_active_segs": total_act if has_ler else 0,
            "n_inactive_segs": total_inact if has_ler else 0,
            "n_crown_lines": len(_crowns),
            # Independent measurement of udvendigDiameter, from the circle fits.
            "measured_diameter": 2.0 * float(np.median(_radii)) if _radii else None,
            "crown": _crown_agg("crown_distances"),
            "crown_xy": _crown_agg("crown_xy"),
            "crown_z": _crown_agg("crown_z"),
            "crown_z_signed": _crown_signed_agg("crown_z_signed"),
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
    print(f"    Points:     {s['n_points']:,}")
    print(f"    LER segs:   {s['n_active_segs']} active, {s['n_inactive_segs']} inactive")
    _cr = s["crown"]
    _dia = s["measured_diameter"]
    print(f"    Crown lines: {s['n_crown_lines']}/{s['n_instances']} instances"
          + (f"   measured D {_dia*1000:.0f} mm" if _dia else ""))
    if _cr:
        # Headline: both sides on the registered datum (centreline XY, top Z).
        print(f"    -- CROWN LINE vs registered line ({_cr['n']} stations) --")
        print(f"    Mean:       {_cr['mean']*1000:>8.2f} mm")
        print(f"    Median:     {_cr['median']*1000:>8.2f} mm")
        print(f"    Std dev:    {_cr['std']*1000:>8.2f} mm")
        print(f"    P95:        {_cr['p95']*1000:>8.2f} mm")
        print(f"    Max:        {_cr['max']*1000:>8.2f} mm")
        for _lbl, _key in (("XY", "crown_xy"), ("Z", "crown_z")):
            _c = s[_key]
            if _c:
                print(f"      {_lbl:<3} mean {_c['mean']*1000:>7.2f} mm   "
                      f"P95 {_c['p95']*1000:>7.2f} mm   "
                      f"max {_c['max']*1000:>7.2f} mm")
        _cz = s["crown_z_signed"]
        if _cz:
            print(f"      Z signed (+ = crown above the registered top, so "
                  f"shallower than registered)")
            print(f"          median {_cz['median']*1000:>+7.2f} mm   "
                  f"mean {_cz['mean']*1000:>+7.2f} mm   "
                  f"P5..P95 {_cz['p5']*1000:>+7.2f} .. {_cz['p95']*1000:>+7.2f} mm   "
                  f"{_cz['frac_above']*100:.0f}% above")
    elif s["n_crown_lines"] == 0:
        print(f"    -- CROWN LINE: none recovered (see per-instance reasons) --")
    if s["has_ler"] and not np.isnan(s["mean"]):
        print(f"    -- POINT CLOUD vs registered line (all matching LER) --")
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
        print(f"    ** No measured points carry a deviation **")
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

# Instances with an exclusive LER match (see label_module.py) contribute only
# to their linked line's own reference set, keyed by gml_id, so those features'
# discretized deviation is measured against just that one instance. Every
# feature on the linked line gets the same reference points, so a run split
# across several features is measured consistently along its whole length.
_matched_feature_pts = {}
for _instances in class_instances.values():
    for _inst in _instances:
        _mi = _inst.get("ler_match")
        if not _mi or _mi.get("no_ler"):
            continue
        _gids = [g for g in (_mi.get("gml_ids") or [_mi.get("gml_id")]) if g]
        if not _gids:
            continue
        _pts_full = np.asarray(_inst["pcd_dev"].points)
        if len(_pts_full):
            for _gid in _gids:
                _matched_feature_pts.setdefault(_gid, []).append(_pts_full)
_matched_feature_pts = {gid: np.concatenate(v) for gid, v in _matched_feature_pts.items()}

# Instance points each layer is compared against: the union over all utility
# types whose LER match covers a segment in that layer. Only utility types with
# an explicit LER match contribute; unlabelled / unmatched instances (whose
# match mask would otherwise cover every segment) are skipped so that
# unsegmented LER layers, e.g. Gasledning or Foeringsroer, get no reference
# points and therefore no deviation. Instances already claimed by an exclusive
# match are excluded from this generic per-type pool (they feed
# _matched_feature_pts instead), so an unmatched nearby feature of the same
# type does not pick up an already-claimed instance as its neighbour.
_layer_ref_pts = {}
for ut, instances in class_instances.items():
    if UTILITY_TO_LER_MATCH.get(ut) is None:
        continue
    mask = _get_matching_segment_mask(ut)   # combined active + inactive
    if not mask.any() or not instances:
        continue
    unmatched = [inst for inst in instances if not inst.get("ler_match")]
    if not unmatched:
        continue
    pts_ut = np.concatenate(
        [np.asarray(inst["pcd_dev"].points) for inst in unmatched])
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
ler_pcd_xydev_klic = {}   # layer -> PointCloud, horizontal deviation, 2-class KLIC colours
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
    # Ledningstrace is a registered centerline with a width ("bredde"). In the
    # LER deviation modes it is drawn as a plain centerline (a dense line of
    # samples), not the full-width ribbon, so the deviation reads as a single
    # coloured line. Passing radius = half_width = 0 makes discretize_segment
    # return the centerline only. The solid-mesh (non-deviation) modes still use
    # the wide trace plane in ler_meshes, which this does not touch.
    _is_trace = ln.startswith("Ledningstrace")
    ref_list = _layer_ref_pts.get(ln)
    _generic_ref_pts = np.concatenate(ref_list) if ref_list else None
    _generic_tree = cKDTree(_generic_ref_pts) if _generic_ref_pts is not None else None
    _feature_trees = {}  # gml_id -> (cKDTree, ref_pts), built lazily per feature

    def _tree_for_gid(gid):
        """Exclusive per-feature tree when this gml_id has a matched instance,
        else the generic per-type tree shared by every unmatched feature."""
        if gid and gid in _matched_feature_pts:
            if gid not in _feature_trees:
                _fp = _matched_feature_pts[gid]
                _feature_trees[gid] = (cKDTree(_fp), _fp)
            return _feature_trees[gid]
        return (_generic_tree, _generic_ref_pts)

    samp_chunks = []
    col_chunks, col_cont_chunks = [], []
    zcol_chunks, zcol_cont_chunks = [], []
    xycol_chunks, xycol_cont_chunks = [], []
    xycol_klic_chunks = []
    dev_chunks, zdev_chunks, xydev_chunks = [], [], []
    for idx in seg_ids:
        if _is_trace:
            samp = discretize_segment(
                seg_p1[idx], seg_p2[idx], 0.0, 0.0,
                LER_LENGTH_STEP, LER_SURFACE_STEP)
        else:
            # Lower the axis by the tube radius so the sampled tube's crown sits on
            # the registered line (the pipe top, not the axis), for every pipe. This
            # makes the per-sample Z deviation a true top-to-top depth difference.
            # seg_crown_offset is 0 only for traces, which are handled above.
            _dz = np.array([0.0, 0.0, seg_crown_offset[idx]])
            samp = discretize_segment(
                seg_p1[idx] - _dz, seg_p2[idx] - _dz, seg_radius[idx], seg_half_width[idx],
                LER_LENGTH_STEP, LER_SURFACE_STEP)
        tree, ref_pts = _tree_for_gid(seg_gml_id[idx] if idx < len(seg_gml_id) else "")
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
            # Same horizontal distances, binned into the 2-class KLIC/WIBON
            # pass-fail scheme instead of the 5-class LER accuracy scheme.
            xycols_klic = deviation_to_color(xydev, KLIC_XY_THRESHOLDS, KLIC_XY_COLORS)
        else:
            cols = np.tile(_NO_DATA_COLOR, (len(samp), 1))
            # No measured neighbour: stays grey in every scheme. The KLIC cloud
            # must not fall back to its first class here, since green asserts
            # "within tolerance" and nothing was measured to support that.
            cols_cont = zcols = zcols_cont = xycols = xycols_cont = xycols_klic = cols
            dev = zdev = xydev = np.full(len(samp), np.nan)
        samp_chunks.append(samp)
        col_chunks.append(cols)
        col_cont_chunks.append(cols_cont)
        zcol_chunks.append(zcols)
        zcol_cont_chunks.append(zcols_cont)
        xycol_chunks.append(xycols)
        xycol_cont_chunks.append(xycols_cont)
        xycol_klic_chunks.append(xycols_klic)
        dev_chunks.append(dev)
        zdev_chunks.append(zdev)
        xydev_chunks.append(xydev)

    samp_pts = np.concatenate(samp_chunks)
    _n_samples_total += len(samp_pts)

    def _make_pc(color_chunks, pts=None):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(samp_pts if pts is None else pts)
        pc.colors = o3d.utility.Vector3dVector(np.concatenate(color_chunks))
        return pc

    # The KLIC cloud is drawn flat at street level (_FLAT_LER_MODES): KLIC
    # registers no depth, so any depth on screen there would be the Danish
    # register's, read as a claim KLIC does not make. Display only. The
    # deviations were measured from the registered samples above and the LAS
    # export reads those, so no number changes.
    _samp_flat = samp_pts.copy()
    _samp_flat[:, 2] = GROUND_Z

    ler_pcd_dev[ln] = _make_pc(col_chunks)
    ler_pcd_dev_cont[ln] = _make_pc(col_cont_chunks)
    ler_pcd_zdev[ln] = _make_pc(zcol_chunks)
    ler_pcd_zdev_cont[ln] = _make_pc(zcol_cont_chunks)
    ler_pcd_xydev[ln] = _make_pc(xycol_chunks)
    ler_pcd_xydev_cont[ln] = _make_pc(xycol_cont_chunks)
    ler_pcd_xydev_klic[ln] = _make_pc(xycol_klic_chunks, _samp_flat)
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

# ─────────────────────────────────────────────────────────────────────────────
# 5c.  Crown line geometry
# ─────────────────────────────────────────────────────────────────────────────
# One LineSet per crown line, built once per deviation metric so the colour-mode
# switch is a geometry swap rather than a recompute. The crown carries whichever
# metric the current point-cloud deviation mode shows, which keeps it readable
# against the same legend instead of adding six more entries to the mode list.
# Drawn as a thin line rather than a tube, like the registered centrelines in
# label_module: a tube wide enough to see is wide enough to hide the millimetre
# differences the deviation colours are there to show.
print("\n--- Building crown line geometry ---")
CROWN_LINE_WIDTH = 2.5         # px, matches label_module's centrelines
CROWN_SOLID = "solid"          # key for the non-deviation modes


def _crown_lineset(crown, colors):
    """Polyline through a crown, each segment taking its start station's colour.
    Segments bridging two parts are skipped, so nothing is drawn across a stretch
    that was never exposed."""
    pairs = []
    for p in range(crown.n_parts):
        idx = np.where(crown.part == p)[0]
        pairs.extend(zip(idx[:-1], idx[1:]))
    if not pairs:
        return None
    pts, lines, cols = [], [], []
    for k, (a, b) in enumerate(pairs):
        pts.extend([crown.points[a], crown.points[b]])
        lines.append([2 * k, 2 * k + 1])
        cols.append(colors[a])
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(pts, dtype=float)),
        lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(cols, dtype=float))
    return ls


_n_crown_total = 0
for _ut, _insts in class_instances.items():
    _ut_col = np.asarray(UTILITY_TYPE_COLORS.get(_ut, [0.5, 0.5, 0.5]), dtype=float)
    for _i, _inst in enumerate(_insts):
        _c = _inst["crown"]
        _inst["crown_linesets"] = {}
        if not _c.ok:
            continue
        _solid = np.tile(_ut_col, (_c.n_stations, 1))
        _inst["crown_linesets"][CROWN_SOLID] = _crown_lineset(_c, _solid)
        _n_crown_total += 1
        if len(_inst["crown_distances"]) != _c.n_stations:
            continue                      # no LER to compare: solid colour only
        for _mode, (_vals, _cont) in enumerate((
                (_inst["crown_distances"], False), (_inst["crown_distances"], True),
                (_inst["crown_xy"], False), (_inst["crown_xy"], True),
                (_inst["crown_z"], False), (_inst["crown_z"], True))):
            _fn = deviation_to_color_continuous if _cont else deviation_to_color
            _inst["crown_linesets"][_mode] = _crown_lineset(_c, _fn(_vals))

print(f"  {_n_crown_total} crown lines built "
      f"({sum(len(v) for v in class_instances.values())} instances)")

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
    "LER registered accuracy class",             # 8
    "LER XYZ deviation (discrete)",              # 9
    "LER XYZ deviation (continuous)",            # 10
    "LER XY deviation (discrete)",               # 11
    "LER XY deviation (continuous)",             # 12
    "LER Z deviation (discrete)",                # 13
    "LER Z deviation (continuous)",              # 14
    "KLIC XY deviation (discrete)",              # 15
]
# Instance point cloud shown per mode. In the LER modes the colouring lives on
# the registered geometry, so the instance points fall back to original RGB.
_MODE_INST_PCD = ["pcd_dev", "pcd_dev_cont",
                  "pcd_dev_xy", "pcd_dev_xy_cont",
                  "pcd_dev_z", "pcd_dev_z_cont",
                  "pcd_rgb", "pcd_class",
                  "pcd_rgb",
                  "pcd_rgb", "pcd_rgb", "pcd_rgb", "pcd_rgb", "pcd_rgb", "pcd_rgb",
                  "pcd_rgb"]
# The registered horizontal accuracy class (noejagtighedsklasse) painted onto
# the registered geometry itself. An LER mode, not a point-cloud one: it says
# what the register claims about its own geometry, and nothing about the
# measurement, so it never mixes with a measured deviation colouring.
ACC_CLASS_MODE = 8
# LER deviation modes: the LER layers become deviation-coloured point clouds.
# Each maps to the precomputed cloud carrying the right metric + colouring.
_LER_MODE_PCD = {
    9: ler_pcd_dev,           # XYZ deviation, discrete accuracy-class colours
    10: ler_pcd_dev_cont,     # XYZ deviation, continuous gradient
    11: ler_pcd_xydev,        # XY deviation, discrete accuracy-class colours
    12: ler_pcd_xydev_cont,   # XY deviation, continuous gradient
    13: ler_pcd_zdev,         # Z deviation, discrete accuracy-class colours
    14: ler_pcd_zdev_cont,    # Z deviation, continuous gradient
    15: ler_pcd_xydev_klic,   # XY deviation, 2-class KLIC/WIBON pass-fail colours
}
_LER_DEV_MODES = tuple(_LER_MODE_PCD)
# Instance-deviation modes: the measured points themselves are deviation
# coloured, and the crown line carries the same metric.
_PC_DEV_MODES = (0, 1, 2, 3, 4, 5)
# Modes that show the discrete accuracy-class heatmap legend (5-class LER scheme)
_HEATMAP_MODES = (0, 2, 4, 9, 11, 13)
# Modes that show the continuous deviation-gradient legend
_GRADIENT_MODES = (1, 3, 5, 10, 12, 14)
# Modes that show the 2-class KLIC/WIBON pass-fail legend
_KLIC_MODES = (15,)
# Modes that draw the registered geometry flat at street level. KLIC/WIBON
# registers no depth, so the only depth available to draw is the Danish
# register's, which in a KLIC comparison would read as a claim KLIC does not
# make. Separate from _KLIC_MODES, which is about the legend, not the geometry.
_FLAT_LER_MODES = (15,)

app = gui.Application.instance
app.initialize()

window = app.create_window(
    f"{_ply_path.stem}  |  Deviation: Instances vs LER  |  H for help",
    1460, 840,
)
em = window.theme.font_size

scene_widget = gui.SceneWidget()
scene_widget.scene = rendering.Open3DScene(window.renderer)
scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
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


def make_crown_mat():
    """Thin unlit line for the crown, as label_module draws the registered
    centrelines. Depth testing is disabled for the same reason it is there: the
    crown is recovered from the top of the measured cloud and so lies exactly in
    it, and the points it came from would otherwise swallow the line."""
    mat = line_material(CROWN_LINE_WIDTH)
    try:
        mat.depth_func = "always"
    except AttributeError:
        pass                       # older Open3D: the crown depth-tests normally
    return mat


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

# Crop-region wireframe at ground level (same style as base_module): the
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

# Colouring of the solid LER geometry: the utility colour of its layer, or the
# registered accuracy class of each feature (ACC_CLASS_MODE). The geometry names
# never change, so the layer toggles, the utility filter and the opacity slider
# are unaffected by which of the two is showing.
def _acc_class_active():
    return _color_mode[0] == ACC_CLASS_MODE


def _solid_ler_mesh(ln):
    return ler_meshes_acc.get(ln, ler_meshes[ln]) if _acc_class_active() else ler_meshes[ln]


def _flat_ler_active():
    return _color_mode[0] in _FLAT_LER_MODES


def _solid_comp_mesh(ln):
    if _acc_class_active():
        return comp_meshes_acc.get(ln, comp_meshes[ln])
    if _flat_ler_active():
        return comp_meshes_flat.get(ln, comp_meshes[ln])
    return comp_meshes[ln]


def _solid_trace_centerlines():
    return _trace_centerlines_acc if _acc_class_active() else _trace_centerlines


# Add LER pipe meshes
_ler_visible = {}
_ler_master_on = [True]   # Ledningspakke master checkbox state (legend section)
for ln, mesh in ler_meshes.items():
    gn = f"ler_{ln}"
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    scene_widget.scene.add_geometry(gn, mesh, make_mesh_mat(ribbon_alpha(ln, 0.6)))
    _ler_visible[ln] = True

# Trace centrelines, at the unscaled opacity so they read like the pipes.
add_trace_centerlines(scene_widget.scene, _trace_centerlines, 0.6, make_mesh_mat)

# LER signature overlays, at the unscaled opacity for the same reason. On by
# default, so this viewer reads like the ERR plan and like LER itself.
_sig_on = [True]
for _sln, _smesh in sig_meshes.items():
    scene_widget.scene.add_geometry(signature_gn(_sln), _smesh, make_mesh_mat(0.6))

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
#
# Type 0 (Unlabeled) has no LER counterpart by definition, so it contributes no
# deviation and starts hidden; it is still toggleable from the class panel.
UNLABELED_TYPE = 0

_inst_gnames = []
_inst_visible = {}
for ut, instances in class_instances.items():
    for i, inst in enumerate(instances):
        gn = f"inst_{ut}_{i}"
        _inst_gnames.append((ut, i, gn))
        _inst_visible[(ut, i)] = ut != UNLABELED_TYPE
        # Startup is mode 0, a deviation mode; the cloud carries that colouring
        # over its whole extent.
        scene_widget.scene.add_geometry(gn, inst["pcd_dev"], make_pt_mat(4.0))
        if not _inst_visible[(ut, i)]:
            scene_widget.scene.show_geometry(gn, False)

# Crown lines, drawn over the instance clouds. Shown by default: the crown line
# is the headline metric, and it is what the registered geometry is comparable to.
_crown_show = [True]


def _crown_gn(ut, i):
    return f"crown_{ut}_{i}"


def _apply_crown_mode(mode):
    """(Re)add each crown line with the LineSet matching the current colour mode.

    The crown carries the same metric as the instance points in the point-cloud
    deviation modes, and the flat class colour everywhere else."""
    key = mode if mode in _PC_DEV_MODES else CROWN_SOLID
    for ut, instances in class_instances.items():
        for i, inst in enumerate(instances):
            gn = _crown_gn(ut, i)
            scene_widget.scene.remove_geometry(gn)
            ls = inst["crown_linesets"].get(key) or inst["crown_linesets"].get(CROWN_SOLID)
            if ls is None:
                continue
            scene_widget.scene.add_geometry(gn, ls, make_crown_mat())
            scene_widget.scene.show_geometry(
                gn, _crown_show[0] and _inst_visible.get((ut, i), True))


_apply_crown_mode(0)

# Camera
bounds = scene_widget.scene.bounding_box
scene_widget.setup_camera(60, bounds, cloud_centroid.tolist())

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Colour-mode switch
# ─────────────────────────────────────────────────────────────────────────────
def _trace_centerlines_visible():
    """A trace's solid centreline tube belongs to the solid colour modes only:
    in the LER deviation modes the trace is itself drawn as a deviation-coloured
    centreline cloud, which the tube would cover."""
    return _color_mode[0] not in _LER_DEV_MODES


def _sync_trace_centerlines():
    """Match every trace centreline to the master, its layer, and the mode."""
    _on = _trace_centerlines_visible()
    for ln in _trace_centerlines:
        gn = trace_centerline_gn(ln)
        if scene_widget.scene.has_geometry(gn):
            scene_widget.scene.show_geometry(
                gn, _on and _ler_master_on[0] and _ler_visible.get(ln, True))


def _signatures_visible():
    """The LER signatures belong to the solid colour modes only. In a deviation
    mode the colour of the LER geometry carries a measurement, so a fixed red
    hazard triangle beside it would read as a deviation class."""
    return _sig_on[0] and _color_mode[0] not in _LER_DEV_MODES


def _sync_signatures():
    """Match every signature overlay to the master, its layer, and the mode."""
    _on = _signatures_visible()
    for ln in sig_meshes:
        show_signatures(scene_widget.scene, ln,
                        _ler_master_on[0] and _ler_visible.get(ln, True), _on)


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
            scene_widget.scene.add_geometry(
                gn, _solid_ler_mesh(ln), make_mesh_mat(ribbon_alpha(ln, _ler_opacity[0])))
        scene_widget.scene.show_geometry(gn, _ler_master_on[0] and _ler_visible.get(ln, True))
    _sync_trace_centerlines()
    _sync_signatures()


def _apply_ler_solid_colors():
    """Rebuild the LER geometry that only the mesh path draws: the trace
    centrelines and the component spheres. Both carry the registered accuracy
    class in ACC_CLASS_MODE and their layer's utility colour in every other
    mode, so they are re-added whenever the mode changes."""
    for ln in _trace_centerlines:
        gn = trace_centerline_gn(ln)
        if scene_widget.scene.has_geometry(gn):
            scene_widget.scene.remove_geometry(gn)
    add_trace_centerlines(scene_widget.scene, _solid_trace_centerlines(),
                          _ler_opacity[0], make_mesh_mat)
    for ln in comp_meshes:
        gn = f"comp_{ln}"
        scene_widget.scene.remove_geometry(gn)
        scene_widget.scene.add_geometry(gn, _solid_comp_mesh(ln),
                                        make_mesh_mat(_ler_opacity[0]))
        scene_widget.scene.show_geometry(
            gn, _ler_master_on[0] and _comp_visible.get(ln, False))


def _apply_color_mode(mode):
    _color_mode[0] = mode
    pcd_key = _MODE_INST_PCD[mode]
    for ut, instances in class_instances.items():
        for i, inst in enumerate(instances):
            gn = f"inst_{ut}_{i}"
            scene_widget.scene.remove_geometry(gn)
            scene_widget.scene.add_geometry(gn, inst[pcd_key], make_pt_mat(4.0))
            scene_widget.scene.show_geometry(gn, _inst_visible.get((ut, i), True))
    _apply_crown_mode(mode)
    _apply_ler_solid_colors()
    _apply_ler_color_mode(mode)
    _apply_orig_cloud_mode(mode)
    window.post_redraw()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Right panel
# ─────────────────────────────────────────────────────────────────────────────
PANEL_WIDTH = int(PANEL_WIDTH_EM * em)
PANEL_MARGIN_EM = 1.0
panel = gui.Vert(int(0.5 * em), gui.Margins(int(PANEL_MARGIN_EM * em), int(PANEL_MARGIN_EM * em),
                                            int(PANEL_MARGIN_EM * em), int(PANEL_MARGIN_EM * em)))

# Pixel budgets for panel text that is built from data and so has no bounded
# length: a crown summary, the list of LER layers an instance matched.
# _panel_fitter shortens each to its budget on the first layout pass, with the
# full string kept in the tooltip.
_TEXT_W_PLAIN = PANEL_WIDTH - int(2 * PANEL_MARGIN_EM * em)
# A combobox draws its own frame and drop-down arrow around the item text.
_TEXT_W_COMBO = _TEXT_W_PLAIN - int(2.5 * em)
_panel_fitter = PanelTextFitter()

panel.add_child(gui.Label(f"Original: {len(pts_orig):,} pts"))
total_inst = sum(inst["stats"]["n_pts"] for v in class_instances.values() for inst in v)
n_inst = sum(len(v) for v in class_instances.values())
panel.add_child(gui.Label(f"Instances: {n_inst} ({total_inst:,} pts)"))
panel.add_child(gui.Label(f"LER segments: {n_total_segs:,} ({n_active_segs}a, {n_inactive_segs}i)"))
if CROP_MODE == "rect":
    panel.add_child(gui.Label(f"Crop: cloud AABB + {UTILITY_RECT_BUFFER:.0f} m (rect)"))
else:
    panel.add_child(gui.Label(f"Crop radius: {CROP_RADIUS} m (circular)"))
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
    _heatmap_legend.add_child(make_legend_row(col, gui.Label(lbl), em))

# Continuous gradient legend: same anchor colours as the accuracy classes, but
# sampled at intermediate ticks to show the smooth interpolation between them.
_gradient_legend = gui.Vert(0)
_gradient_legend.add_child(gui.Label("Deviation (gradient):"))
_grad_ticks_m = DEVIATION_GRADIENT_TICKS
_grad_tick_cols = deviation_to_color_continuous(np.asarray(_grad_ticks_m, dtype=float))
for _tick_m, _col in zip(_grad_ticks_m, _grad_tick_cols):
    _lbl = f">= {_tick_m:.2f} m" if _tick_m == _grad_ticks_m[-1] else f"{_tick_m:.2f} m"
    _gradient_legend.add_child(make_legend_row(_col, gui.Label(_lbl), em))
_gradient_legend.visible = False

# Registered accuracy class legend (ACC_CLASS_MODE). Same five colours and
# bounds as the heatmap above, so the header has to say which of the two is on
# screen: here they are the class LER registers for its own geometry, not a
# measured deviation. The count is of features within the view, components
# included.
_acc_class_legend = gui.Vert(0)
_acc_class_legend.add_child(gui.Label(
    f"Registered accuracy class ({_acc_view_reg}/{_acc_view_total}):"))
for _col, _lbl in zip(ACCURACY_CLASS_COLORS, DEVIATION_CLASS_LABELS):
    _acc_class_legend.add_child(make_legend_row(_col, gui.Label(_lbl), em))
_acc_class_legend.add_child(make_legend_row(
    ACCURACY_UNREGISTERED_COLOR, gui.Label(ACCURACY_UNREGISTERED_LABEL), em))
_acc_class_legend.visible = False

# KLIC/WIBON pass-fail legend: separate 2-swatch widget so the 5-class
# heatmap legend is not reused (and shown incorrectly) for this 2-class scheme.
_klic_legend = gui.Vert(0)
_klic_legend.add_child(gui.Label("KLIC tolerance (1 m):"))
for i, (col, lbl) in enumerate(zip(KLIC_XY_COLORS, KLIC_XY_CLASS_LABELS)):
    _klic_legend.add_child(make_legend_row(col, gui.Label(lbl), em))
# Said on screen, so a screenshot of this mode cannot be read as a depth claim.
_klic_legend.add_child(gui.Label("  LER drawn at street level"))
_klic_legend.visible = False


def _on_mode(val, idx):
    _apply_color_mode(idx)
    _heatmap_legend.visible = (idx in _HEATMAP_MODES)
    _gradient_legend.visible = (idx in _GRADIENT_MODES)
    _klic_legend.visible = (idx in _KLIC_MODES)
    _acc_class_legend.visible = (idx == ACC_CLASS_MODE)
    window.set_needs_layout()


combo.set_on_selection_changed(_on_mode)
panel.add_child(combo)
panel.add_child(_heatmap_legend)
panel.add_child(_gradient_legend)
panel.add_child(_acc_class_legend)
panel.add_child(_klic_legend)
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

# Crown line toggle. The count says how many instances yielded one; the rest are
# shapes the recovery rejected (branches, risers, wells), listed at startup.
if _n_crown_total:
    crown_cb = gui.Checkbox(
        f"Crown line ({_n_crown_total}/{n_inst} instances)")
    crown_cb.checked = True

    def _on_crown(c):
        _crown_show[0] = c
        for _cu, _ci, _ in _inst_gnames:
            scene_widget.scene.show_geometry(
                _crown_gn(_cu, _ci), c and _inst_visible.get((_cu, _ci), True))
        window.post_redraw()

    crown_cb.set_on_checked(_on_crown)
    panel.add_child(crown_cb)
else:
    panel.add_child(gui.Label("Crown line: none recovered"))

# LER signature toggle: the cartographic signatures of the LER
# "Signaturforklaring", shown in the solid colour modes only (_signatures_visible).
sig_cb = gui.Checkbox("LER signatures")
sig_cb.checked = _sig_on[0]
if not sig_meshes:
    sig_cb.enabled = False


def _on_sig(c):
    _sig_on[0] = c
    _sync_signatures()
    window.post_redraw()


sig_cb.set_on_checked(_on_sig)
panel.add_child(sig_cb)

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
# The LER suffix lists every matched layer, traces included, so an entry can run
# several times the panel width. Items are shortened to fit; _filter_entries
# keeps the full text, which the tooltip shows for whichever is selected.
for _flbl, _ in _filter_entries:
    filter_combo.add_item(_flbl)
filter_combo.selected_index = 0
filter_combo.tooltip = _filter_entries[0][0]
_panel_fitter.add_combo(filter_combo, [_l for _l, _ in _filter_entries], _TEXT_W_COMBO)


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
            scene_widget.scene.show_geometry(_crown_gn(ut, i), vis and _crown_show[0])

    # LER layers: show only those matching the selected instance's type (or all)
    for ln in ler_meshes:
        if sel is None:
            vis = True
        else:
            vis = ln in matching_ler if matching_ler else False
        _ler_visible[ln] = vis
        scene_widget.scene.show_geometry(f"ler_{ln}", vis and _ler_master_on[0])
    _sync_trace_centerlines()
    _sync_signatures()

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
    _full, sel = _filter_entries[idx]
    filter_combo.tooltip = _full
    _apply_utility_filter(sel)


filter_combo.set_on_selection_changed(_on_filter)
panel.add_child(filter_combo)

# ── Utility Legend (uniform LerLegendSection, see core/gui_helpers.py) ───────
panel.add_fixed(int(0.3 * em))
_ler_opacity = [0.6]
_ler_section = LerLegendSection(em, LEDNINGSPAKKE_LABEL, opacity=0.6)


def _on_ler_opacity(val):
    _ler_opacity[0] = val
    # In the LER deviation modes the LER layers are deviation-coloured point
    # clouds, so the slider fades them via the point material; in the other
    # modes they are solid meshes faded via the mesh material.
    in_ler_dev = _color_mode[0] in _LER_DEV_MODES
    for ln in ler_meshes:
        if _ler_visible.get(ln, True):
            mat = (make_ler_pt_mat(6.0, val) if in_ler_dev
                   else make_mesh_mat(ribbon_alpha(ln, val)))
            scene_widget.scene.modify_geometry_material(f"ler_{ln}", mat)
    # Trace centrelines and signature overlays follow the slider at the
    # unscaled opacity
    for ln in _trace_centerlines:
        gn = trace_centerline_gn(ln)
        if scene_widget.scene.has_geometry(gn):
            scene_widget.scene.modify_geometry_material(gn, make_mesh_mat(val))
    for ln in sig_meshes:
        gn = signature_gn(ln)
        if scene_widget.scene.has_geometry(gn):
            scene_widget.scene.modify_geometry_material(gn, make_mesh_mat(val))
    for ln in comp_meshes:
        if _comp_visible.get(ln, False):
            scene_widget.scene.modify_geometry_material(f"comp_{ln}", make_mesh_mat(val))
    window.post_redraw()


_ler_section.set_on_opacity(_on_ler_opacity)


def _on_ler_master(checked):
    """Show/hide all LER geometry at once, remembering per-layer states."""
    _ler_master_on[0] = checked
    for ln in ler_meshes:
        scene_widget.scene.show_geometry(f"ler_{ln}",
                                         checked and _ler_visible.get(ln, True))
    for c_ln in comp_meshes:
        scene_widget.scene.show_geometry(f"comp_{c_ln}",
                                         checked and _comp_visible.get(c_ln, False))
    _sync_trace_centerlines()
    _sync_signatures()


_ler_section.set_on_master(window, _on_ler_master)
_ler_section.add_to(panel)

# -- LER signature legend ("Signaturforklaring", core/signature_legend.py) ----
# The utility legend above explains colour; this one explains form, which is
# the half a colour swatch cannot show. Collapsed by default, like LER's own.
_sig_legend = SignatureLegendSection(em, components="point")
_sig_legend.add_to(panel)

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

# LER layer toggles (rows live in the legend section)
_ler_layer_cbs = []
_comp_layer_cbs = []


def _on_toggle_all_ler(checked):
    for _ln, _cb in _ler_layer_cbs:
        _cb.checked = checked
        _ler_visible[_ln] = checked
        scene_widget.scene.show_geometry(f"ler_{_ln}", checked and _ler_master_on[0])
    _sync_trace_centerlines()
    _sync_signatures()
    window.post_redraw()


_ler_section.add_all_segments(True, _on_toggle_all_ler)

for ln in LINE_LAYERS:
    if ln not in ler_meshes:
        continue
    col = LINE_LAYERS[ln]["color"]
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

    def _make_ler_cb(layer):
        def _cb(checked):
            _ler_visible[layer] = checked
            scene_widget.scene.show_geometry(f"ler_{layer}", checked and _ler_master_on[0])
            _sync_trace_centerlines()
            _sync_signatures()
            window.post_redraw()
        return _cb

    cb = _ler_section.add_layer_row(col, f"{layer_display_name(ln)} ({count_str})",
                                    True, _make_ler_cb(ln))
    _ler_layer_cbs.append((ln, cb))


def _on_toggle_all_ler_comps(checked):
    for _ln, _cb in _comp_layer_cbs:
        _cb.checked = checked
        _comp_visible[_ln] = checked
        scene_widget.scene.show_geometry(f"comp_{_ln}", checked and _ler_master_on[0])
    window.post_redraw()


# LER component toggles
if comp_meshes:
    _ler_section.add_all_components(False, _on_toggle_all_ler_comps)
    for comp_ln in COMPONENT_LAYERS:
        if comp_ln not in comp_meshes:
            continue
        comp_col = COMPONENT_LAYERS[comp_ln]["color"]
        n_c = comp_stats.get(comp_ln, 0)

        def _make_comp_cb(layer):
            def _cb(checked):
                _comp_visible[layer] = checked
                scene_widget.scene.show_geometry(f"comp_{layer}", checked and _ler_master_on[0])
                window.post_redraw()
            return _cb

        ccb = _ler_section.add_layer_row(comp_col,
                                         f"{layer_display_name(comp_ln)} ({n_c})",
                                         False, _make_comp_cb(comp_ln))
        _comp_layer_cbs.append((comp_ln, ccb))

# Instance class toggles + stats
panel.add_fixed(int(0.8 * em))
panel.add_child(gui.Label("Instance classes:"))
panel.add_fixed(int(0.3 * em))

for ut in sorted(class_summaries.keys()):
    s = class_summaries[ut]
    col = UTILITY_TYPE_COLORS.get(ut, [0.5, 0.5, 0.5])

    def _make_cls_cb(u):
        def _cb(checked):
            for _u, _i, gn in _inst_gnames:
                if _u == u:
                    _inst_visible[(_u, _i)] = checked
                    scene_widget.scene.show_geometry(gn, checked)
                    scene_widget.scene.show_geometry(_crown_gn(_u, _i),
                                                     checked and _crown_show[0])
            window.post_redraw()
        return _cb

    # Which LER layer a class is reconciled against is fixed by
    # UTILITY_TO_LER_MATCH, so naming it here would only repeat the class name.
    # Name and count alone also keep the row inside the panel: a checkbox takes
    # its text in the constructor and Open3D exposes no way to change it later,
    # so this text cannot be measured and has to be short by construction. The
    # longest it can get is the longest UTILITY_TYPE_LABELS entry, which is fixed
    # in config, not data-driven.
    cb = gui.Checkbox(f"{s['label']} ({s['n_instances']})")
    cb.checked = any(_inst_visible[(ut, j)] for j in range(s["n_instances"]))
    cb.set_on_checked(_make_cls_cb(ut))
    _class_checkboxes[ut] = cb
    panel.add_child(make_legend_row(col, cb, em))

    # Detail lines under each class: whether the register holds a counterpart at
    # all (which varies per package, unlike the layer mapping), the headline
    # crown deviation, and the diameter the circle fits measured (comparable to
    # registered udvendigDiameter). Labels, so these can be measured and fitted.
    _cr = s["crown"]
    _dia = s["measured_diameter"]
    _rows = [] if s["has_ler"] else ["  no LER counterpart"]
    if _cr:
        _rows.append(f"  crown: mean {_cr['mean']*1000:.0f}, P95 {_cr['p95']*1000:.0f} mm")
        if _dia:
            _rows.append(f"  measured D: {_dia*1000:.0f} mm")
    elif s["n_crown_lines"]:
        _rows.append(f"  crown: {s['n_crown_lines']} line(s), not compared")
    else:
        _rows.append("  crown: not recoverable")
    for _row_txt in _rows:
        panel.add_child(_panel_fitter.add(gui.Label(""), _row_txt, _TEXT_W_PLAIN))
    panel.add_fixed(int(0.3 * em))

panel.add_stretch()

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Keys + layout
# ─────────────────────────────────────────────────────────────────────────────
HANDLED = gui.Widget.EventCallbackResult.HANDLED
IGNORED = gui.Widget.EventCallbackResult.IGNORED


def _pivot_to(pt):
    pivot_oblique(scene_widget, pt, np.linalg.norm(pc_max - pc_min))


def _top_view():
    """Bird's-eye view looking straight down, framed on the trench footprint
    when one is defined, otherwise on the whole scene."""
    top_view(scene_widget, *trench_or_scene_frame(_trench_path_obj, cloud_centroid,
                                                  pc_min, pc_max, trench_z=GROUND_Z))


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
    # Only place a LayoutContext exists, so the only place panel text can be
    # measured. Runs before the first draw, so nothing overflowing is shown.
    _panel_fitter.apply(ctx)
    r = window.content_rect
    scene_widget.frame = gui.Rect(r.x, r.y, r.width - PANEL_WIDTH, r.height)
    panel.frame = gui.Rect(r.x + r.width - PANEL_WIDTH, r.y, PANEL_WIDTH, r.height)


window.set_on_layout(on_layout)
window.add_child(scene_widget)
window.add_child(panel)

print(f"\nLaunching viewer ...\n")
app.run()
print("Viewer closed.")
