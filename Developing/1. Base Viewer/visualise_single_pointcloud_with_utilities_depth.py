# -*- coding: utf-8 -*-
"""
Single Point Cloud Viewer with Surrounding Utilities — Indicative Depth
========================================================================
Visualises one PLY point cloud together with all utility infrastructure
(pipes, cables, components) that falls within a configurable buffer
around the point cloud's bounding box.

Depth handling for Z = -99  (no reliable measurement)
------------------------------------------------------
  Before loading utilities, a VisualizerWithEditing window opens so you
  can manually pick one or more points on the ground surface:

      Shift + Left-Click   to select a ground-level vertex
      Q  or close window   when finished picking

  The average Z of the picked points becomes the ground level reference.
  For each utility vertex with Z = -99, depth is then estimated as:

      Z_estimated  =  ground_level  -  vejledendeDybde / 1000

  where vejledendeDybde is the indicative (suggested) depth below ground
  level stored as an attribute on the utility feature, in millimetres.

  If vejledendeDybde is not available the vertex falls back to the mean
  of valid Z values on the same feature, or the picked ground level.

  Components (Vandkomponent, etc.) do not carry vejledendeDybde.  For those
  the script uses the average depth of the corresponding pipe layer, or
  falls back to the picked ground level.

Usage
-----
  Set PLY_FILE below to any individual .ply site file, then run.

Picking
-------
  Ctrl + Left-Click  on any pipe segment or component sphere to show
  all its GML attribute fields in the "Selected Feature" panel.  The
  nearest segment midpoint / sphere centre is highlighted in yellow.

Keyboard shortcuts
------------------
  C   pivot to cloud centroid           P   pivot to pipe centroid
  0   pivot to world origin
  ]   increase opacity +0.05           [   decrease opacity -0.05
  H   help                             Esc quit
"""

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import geopandas as gpd
import numpy as np
from pathlib import Path
import re
import time

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PLY_FILE = (
    r"C:\Users\bosre\OneDrive - University of Twente\Documents\AAU UTwente thesis"
    r"\Python\Thesis\Data\OpenTrench3D\Water_Area_5\Area_5_Site_05.ply"
)

AREA_REF_GEOJSON = (
    r"C:\Users\bosre\OneDrive - University of Twente\Documents\AAU UTwente thesis"
    r"\Python\Thesis\Data\Translation_coordinates\area_points_utm32_etrs89.geojson"
)

GML_PATH = (
    r"C:\Users\bosre\OneDrive - University of Twente\Documents\AAU UTwente thesis"
    r"\Python\Thesis\Data\Ledningspakke_3383910\consolidated.gml"
)

# Buffer (metres) around the point cloud bbox when selecting utilities
BUFFER = 5.0

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY LAYER DEFINITIONS — colour per LER 2.0 / project convention
# ─────────────────────────────────────────────────────────────────────────────
LINE_LAYERS = {
    "Vandledning":                  {"color": [0.200, 0.500, 1.000], "fallback_radius": 0.025},
    "Afloebsledning":               {"color": [0.545, 0.271, 0.075], "fallback_radius": 0.050},
    "Gasledning":                   {"color": [1.000, 0.800, 0.000], "fallback_radius": 0.025},
    "Elledning":                    {"color": [0.900, 0.100, 0.100], "fallback_radius": 0.015},
    "Telekommunikationsledning":    {"color": [0.200, 0.800, 0.200], "fallback_radius": 0.015},
    "Foeringsroer":                 {"color": [0.500, 0.900, 0.500], "fallback_radius": 0.040},
    "LedningUkendtForsyningsart":   {"color": [0.300, 0.800, 0.800], "fallback_radius": 0.025},
    "Ledningstrace":                {"color": [1.000, 0.500, 0.500], "fallback_radius": 0.015},
}

COMPONENT_LAYERS = {
    "Vandkomponent":                  {"color": [0.000, 0.900, 0.900]},
    "Afloebskomponent":               {"color": [0.700, 0.400, 0.200]},
    "Gaskomponent":                   {"color": [1.000, 0.900, 0.300]},
    "Elkomponent":                    {"color": [1.000, 0.300, 0.300]},
    "Telekommunikationskomponent":    {"color": [0.400, 1.000, 0.400]},
}

COMPONENT_SPHERE_RADIUS = 0.05

# Vandledning diameter → colour mapping (LER 2.0)
DIAMETER_COLORS = {
    0:   [0.502, 0.502, 0.502],
    32:  [0.702, 0.851, 1.000],
    63:  [0.400, 0.698, 1.000],
    120: [0.102, 0.459, 1.000],
    150: [0.000, 0.278, 0.800],
    160: [0.000, 0.180, 0.522],
}

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
_required = {"PLY_FILE": PLY_FILE, "AREA_REF_GEOJSON": AREA_REF_GEOJSON, "GML_PATH": GML_PATH}
_missing = [(n, p) for n, p in _required.items() if not Path(p).exists()]
if _missing:
    print("\n[CONFIG ERROR] Missing paths:")
    for n, p in _missing:
        print(f"  {n:<20} = {p}")
    raise SystemExit(1)

print("Config paths OK.\n")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Auto-detect area from the PLY path
# ─────────────────────────────────────────────────────────────────────────────
_ply_path = Path(PLY_FILE)
_area_match = re.search(r"Area[_\s]*(\d+)", _ply_path.parent.name, re.IGNORECASE)
if not _area_match:
    _area_match = re.search(r"Area[_\s]*(\d+)", _ply_path.name, re.IGNORECASE)
if not _area_match:
    print("[ERROR] Cannot determine area number from PLY path.")
    raise SystemExit(1)

AREA_NUMBER = int(_area_match.group(1))
AREA_NAME   = f"Area{AREA_NUMBER}"
print(f"Detected area: {AREA_NAME}  (from path)")

ref   = gpd.read_file(AREA_REF_GEOJSON)
area  = ref[ref["name"] == AREA_NAME]
if area.empty:
    print(f"[ERROR] No origin for '{AREA_NAME}' in {AREA_REF_GEOJSON}")
    raise SystemExit(1)

area_row = area.iloc[0]
TX, TY, TZ = area_row.geometry.x, area_row.geometry.y, area_row.geometry.z
print(f"Origin -> TX={TX:.3f}  TY={TY:.3f}  TZ={TZ:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load the single point cloud
# ─────────────────────────────────────────────────────────────────────────────
_t0 = time.perf_counter()
print(f"\nLoading point cloud: {_ply_path.name} ...")
pcd = o3d.io.read_point_cloud(str(PLY_FILE))
pts = np.asarray(pcd.points)
print(f"  {len(pts):,} points loaded")

cloud_centroid = pts.mean(axis=0)

# Bounding box in local coordinates (already local in the PLY)
pc_min = pts.min(axis=0)
pc_max = pts.max(axis=0)

# Buffered bbox in UTM for filtering utilities
utm_min = pc_min.copy()
utm_max = pc_max.copy()
utm_min[:2] += np.array([TX, TY])
utm_max[:2] += np.array([TX, TY])

buf_min_x = utm_min[0] - BUFFER
buf_max_x = utm_max[0] + BUFFER
buf_min_y = utm_min[1] - BUFFER
buf_max_y = utm_max[1] + BUFFER

print(f"  Local bbox:  X[{pc_min[0]:.1f}, {pc_max[0]:.1f}]  "
      f"Y[{pc_min[1]:.1f}, {pc_max[1]:.1f}]  "
      f"Z[{pc_min[2]:.1f}, {pc_max[2]:.1f}]")
print(f"  UTM bbox + {BUFFER}m buffer:  "
      f"X[{buf_min_x:.1f}, {buf_max_x:.1f}]  "
      f"Y[{buf_min_y:.1f}, {buf_max_y:.1f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Pick ground-level points interactively
# ─────────────────────────────────────────────────────────────────────────────
def pick_points(pcd):
    """
    Open a VisualizerWithEditing window so the user can pick ground-level
    vertices with  Shift + Left-Click.

    Close the window (press Q or the X button) when done picking.
    Returns a list of picked point indices.
    """
    print("\n" + "=" * 62)
    print("  GROUND-LEVEL POINT PICKING")
    print("=" * 62)
    print("  Shift + Left-Click  to select points on the ground surface.")
    print("  Pick one or more points that represent the ground level.")
    print("  Press Q or close the window when finished.")
    print("=" * 62 + "\n")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick ground-level points  (Shift+Click, then Q to finish)",
                      width=1280, height=720)
    vis.add_geometry(pcd)
    vis.run()   # blocks until user closes window
    vis.destroy_window()

    picked = vis.get_picked_points()
    return picked


picked_indices = pick_points(pcd)

if len(picked_indices) == 0:
    print("[WARNING] No points picked!  Falling back to P95 of point cloud Z.")
    GROUND_Z = float(np.percentile(pts[:, 2], 95))
else:
    picked_pts = pts[picked_indices]
    GROUND_Z   = float(np.mean(picked_pts[:, 2]))
    print(f"\n  Picked {len(picked_indices)} ground-level point(s):")
    for i, idx in enumerate(picked_indices):
        p = pts[idx]
        print(f"    [{i+1}]  index {idx:>8,}  ->  "
              f"X={p[0]:.3f}  Y={p[1]:.3f}  Z={p[2]:.3f}")

print(f"\n  Ground level (local) = {GROUND_Z:.3f} m")
print(f"  Ground level (UTM)   = {GROUND_Z + TZ:.3f} m")

# Depth estimation counters
_depth_stats = {"estimated": 0, "fallback_feature_mean": 0, "fallback_global": 0}

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def segment_to_cylinder(p1, p2, radius, color, resolution=12):
    vec    = p2 - p1
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return None

    cyl = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=length, resolution=resolution, split=1
    )
    z_axis    = np.array([0.0, 0.0, 1.0])
    direction = vec / length
    cross     = np.cross(z_axis, direction)
    cross_norm = np.linalg.norm(cross)
    dot        = np.dot(z_axis, direction)

    if cross_norm > 1e-6:
        axis  = cross / cross_norm
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


def _clean_coords_with_depth(coords_raw, vejledende_dybde_mm):
    """
    Translate UTM -> local.  For vertices with Z = -99 (no reliable
    measurement), estimate depth using:
        Z = ground_level(XY) - vejledendeDybde / 1000
    If vejledendeDybde is not available, fall back to mean of valid Z
    on the same feature, or the global ground level.
    """
    coords = coords_raw.copy().astype(float)
    if coords.shape[1] == 2:
        coords = np.hstack([coords, np.zeros((len(coords), 1))])

    # Translate XY to local first (Z stays in absolute UTM for now)
    coords[:, 0] -= TX
    coords[:, 1] -= TY

    bad = coords[:, 2] == -99
    if bad.any():
        # Parse indicative depth (mm -> m)
        ind_depth_m = None
        if vejledende_dybde_mm is not None:
            try:
                d = float(vejledende_dybde_mm)
                if d > 0:
                    ind_depth_m = d / 1000.0
            except (ValueError, TypeError):
                pass

        # Mean of valid Z on this feature (absolute UTM Z)
        good_z = coords[~bad, 2]
        feature_mean_z = float(good_z.mean()) if len(good_z) > 0 else None

        for idx in np.where(bad)[0]:
            if ind_depth_m is not None:
                # Estimate: picked ground level (local) - indicative depth
                # GROUND_Z is local; coords Z is still absolute UTM → convert
                coords[idx, 2] = (GROUND_Z + TZ) - ind_depth_m
                _depth_stats["estimated"] += 1
            elif feature_mean_z is not None:
                # Fall back to mean of valid Z on this feature
                coords[idx, 2] = feature_mean_z
                _depth_stats["fallback_feature_mean"] += 1
            else:
                # Last resort: picked ground level (convert to absolute UTM)
                coords[idx, 2] = GROUND_Z + TZ
                _depth_stats["fallback_global"] += 1

    # Now translate Z to local
    coords[:, 2] -= TZ
    return coords


def _segments_in_bbox(coords_utm):
    """Check if any part of the line falls inside the buffered bbox (UTM)."""
    xs = coords_utm[:, 0]
    ys = coords_utm[:, 1]
    return (xs.max() >= buf_min_x and xs.min() <= buf_max_x and
            ys.max() >= buf_min_y and ys.min() <= buf_max_y)


def _point_in_bbox(x, y):
    return buf_min_x <= x <= buf_max_x and buf_min_y <= y <= buf_max_y


# Local-coordinate bbox limits for clipping (used after UTM -> local transform)
_local_buf_min_x = pc_min[0] - BUFFER
_local_buf_max_x = pc_max[0] + BUFFER
_local_buf_min_y = pc_min[1] - BUFFER
_local_buf_max_y = pc_max[1] + BUFFER


def _pt_in_local_bbox(x, y):
    return (_local_buf_min_x <= x <= _local_buf_max_x and
            _local_buf_min_y <= y <= _local_buf_max_y)


def _clip_segment_to_bbox(p1, p2):
    """
    Clip a 3D line segment (p1 -> p2) to the local buffered bbox in XY
    using the Liang-Barsky algorithm.
    Returns (clipped_p1, clipped_p2) or None if entirely outside.
    """
    x0, y0 = p1[0], p1[1]
    dx = p2[0] - x0
    dy = p2[1] - y0

    t0, t1 = 0.0, 1.0

    for p_val, q_val in [
        (-dx,  (x0 - _local_buf_min_x)),   # left
        ( dx, -( x0 - _local_buf_max_x)),   # right
        (-dy,  (y0 - _local_buf_min_y)),    # bottom
        ( dy, -(y0 - _local_buf_max_y)),    # top
    ]:
        if abs(p_val) < 1e-12:
            if q_val < 0:
                return None   # parallel and outside
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
# 5.  Load utility lines (pipes / cables) within bbox
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Loading utility lines within bbox ---")
all_pipe_meshes = []
layer_stats = {}
all_pipe_coords = []

# Picking data — segment midpoints and their GML attributes
pick_seg_midpoints = []   # list of np.array([x,y,z])
pick_seg_attrs     = []   # list of [(label, value), ...]
pick_seg_layer     = []   # layer name per segment

# Store per-utility-type average depth for component fallback
_layer_avg_depth_local = {}

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

        if layer_name == "Vandledning" and diam_mm > 0:
            color = DIAMETER_COLORS.get(int(diam_mm), default_color)
        else:
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

            coords = _clean_coords_with_depth(coords_raw, vejl_dybde)
            all_pipe_coords.append(coords)
            _layer_z_vals.extend(coords[:, 2].tolist())
            feature_hit = True

            for i in range(len(coords) - 1):
                clipped = _clip_segment_to_bbox(coords[i], coords[i + 1])
                if clipped is None:
                    continue
                cyl = segment_to_cylinder(clipped[0], clipped[1], radius, color)
                if cyl is not None:
                    all_pipe_meshes.append(cyl)
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
    if n_features > 0:
        print(f"  {layer_name:<35} {n_features:>4} features  {n_segments:>5} segments")

pick_seg_midpoints = np.array(pick_seg_midpoints) if pick_seg_midpoints else np.empty((0, 3))

print(f"\n  Total: {len(all_pipe_meshes):,} cylinder segments")
print(f"\n  Depth estimation stats:")
print(f"    Estimated from vejledendeDybde + ground model: {_depth_stats['estimated']}")
print(f"    Fallback to feature mean Z:                    {_depth_stats['fallback_feature_mean']}")
print(f"    Fallback to global ground level:               {_depth_stats['fallback_global']}")

# Merge pipe meshes
combined_pipe = None
if all_pipe_meshes:
    combined_pipe = all_pipe_meshes[0]
    for m in all_pipe_meshes[1:]:
        combined_pipe += m
    combined_pipe.compute_vertex_normals()

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
all_comp_meshes = []
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
                # Fall back to picked ground level
                pt[2] = GROUND_Z
                _comp_depth_stats["from_ground"] += 1

        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=COMPONENT_SPHERE_RADIUS, resolution=12
        )
        sphere.translate(pt)
        sphere.paint_uniform_color(color)
        all_comp_meshes.append(sphere)

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

# Merge component meshes
combined_comp = None
if all_comp_meshes:
    combined_comp = all_comp_meshes[0]
    for m in all_comp_meshes[1:]:
        combined_comp += m
    combined_comp.compute_vertex_normals()

_t_load = time.perf_counter()
print(f"\nData loaded in {_t_load - _t0:.2f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Coordinate frame + bounding box wireframe
# ─────────────────────────────────────────────────────────────────────────────
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5, origin=cloud_centroid
)

# Wireframe showing the buffered bbox at Z=0
bx0 = pc_min[0] - BUFFER
bx1 = pc_max[0] + BUFFER
by0 = pc_min[1] - BUFFER
by1 = pc_max[1] + BUFFER
bbox_wire_pts = np.array([
    [bx0, by0, GROUND_Z], [bx1, by0, GROUND_Z], [bx1, by1, GROUND_Z], [bx0, by1, GROUND_Z], [bx0, by0, GROUND_Z]
])
bbox_lines = [[i, i + 1] for i in range(4)]
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
    mat.shader     = "defaultUnlitTransparency"
    mat.base_color = [1.0, 1.0, 1.0, float(alpha)]
    return mat


def make_point_material() -> rendering.MaterialRecord:
    mat            = rendering.MaterialRecord()
    mat.shader     = "defaultUnlit"
    mat.point_size = 2.0
    return mat


def make_frame_material() -> rendering.MaterialRecord:
    mat        = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    return mat


def linear_to_srgb(c: float) -> float:
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Build GUI
# ─────────────────────────────────────────────────────────────────────────────
PIPE_GEOM      = "pipes"
COMP_GEOM      = "components"
FRAME_GEOM     = "frame"
BBOX_GEOM      = "bbox_wire"
HIGHLIGHT_GEOM = "highlight"

pipe_opacity = [1.0]
origin_pt    = np.array([0.0, 0.0, 0.0])
pick_active  = [False]

app = gui.Application.instance
app.initialize()

window = app.create_window(
    f"{_ply_path.stem}  |  Utilities + depth estimation  |  press H for help",
    1460, 840,
)
em = window.theme.font_size

scene_widget = gui.SceneWidget()
scene_widget.scene = rendering.Open3DScene(window.renderer)
scene_widget.scene.set_background([0.10, 0.10, 0.10, 1.0])

# Add point cloud
scene_widget.scene.add_geometry("point_cloud", pcd, make_point_material())

# Add pipes
if combined_pipe is not None:
    scene_widget.scene.add_geometry(PIPE_GEOM, combined_pipe, make_mesh_material(1.0))

# Add components
if combined_comp is not None:
    scene_widget.scene.add_geometry(COMP_GEOM, combined_comp, make_mesh_material(1.0))

# Add frame and bbox wireframe
scene_widget.scene.add_geometry(FRAME_GEOM, frame, make_frame_material())

line_mat            = rendering.MaterialRecord()
line_mat.shader     = "unlitLine"
line_mat.line_width = 3.0
scene_widget.scene.add_geometry(BBOX_GEOM, bbox_ls, line_mat)

bounds = scene_widget.scene.bounding_box
scene_widget.setup_camera(60, bounds, cloud_centroid.tolist())

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Right-side control panel
# ─────────────────────────────────────────────────────────────────────────────
PANEL_WIDTH = int(20 * em)
panel = gui.Vert(int(0.5 * em), gui.Margins(int(em), int(em), int(em), int(em)))

# Title
panel.add_child(gui.Label(f"--- {_ply_path.stem} ---"))
panel.add_fixed(int(0.4 * em))
panel.add_child(gui.Label(f"Points: {len(pts):,}"))
panel.add_child(gui.Label(f"Buffer: {BUFFER} m"))
panel.add_fixed(int(0.5 * em))

# Depth estimation info
panel.add_child(gui.Label("--- Depth Estimation ---"))
panel.add_fixed(int(0.3 * em))
_pick_method = f"picked from {len(picked_indices)} point(s)" if picked_indices else "fallback P95"
panel.add_child(gui.Label(f"Ground Z: {GROUND_Z:.3f} m ({_pick_method})"))
panel.add_fixed(int(0.15 * em))
_est_total = _depth_stats["estimated"]
_fb_total  = _depth_stats["fallback_feature_mean"] + _depth_stats["fallback_global"]
est_lbl = gui.Label(f"  Estimated from indicative depth: {_est_total}")
est_lbl.text_color = gui.Color(0.4, 1.0, 0.4, 1.0)
panel.add_child(est_lbl)
fb_lbl = gui.Label(f"  Fallback (no indicative depth):  {_fb_total}")
fb_lbl.text_color = gui.Color(1.0, 0.7, 0.3, 1.0)
panel.add_child(fb_lbl)
panel.add_fixed(int(0.8 * em))

# Opacity control
panel.add_child(gui.Label("--- Transparency ---"))
panel.add_fixed(int(0.3 * em))

opacity_value_label = gui.Label("1.00")
lbl_row = gui.Horiz(int(0.25 * em))
lbl_row.add_child(gui.Label("Pipes + Components"))
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
    mat = make_mesh_material(val)

    if combined_pipe is not None:
        scene_widget.scene.remove_geometry(PIPE_GEOM)
        scene_widget.scene.add_geometry(PIPE_GEOM, combined_pipe, mat)
    if combined_comp is not None:
        scene_widget.scene.remove_geometry(COMP_GEOM)
        scene_widget.scene.add_geometry(COMP_GEOM, combined_comp, mat)
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
panel.add_fixed(int(0.4 * em))
panel.add_child(gui.Label("Keys:  ]  +0.05    [  -0.05"))
panel.add_fixed(int(1.2 * em))

# Colour legend
panel.add_child(gui.Label("--- Utility Legend ---"))
panel.add_fixed(int(0.4 * em))

# Line layers
for layer_name, cfg in LINE_LAYERS.items():
    n_feat, n_seg = layer_stats.get(layer_name, (0, 0))
    if n_feat == 0:
        continue
    col = cfg["color"]
    sr, sg, sb = (linear_to_srgb(c) for c in col)

    row     = gui.Horiz(int(0.3 * em))
    swatch  = gui.Button("        ")
    swatch.background_color = gui.Color(sr, sg, sb, 1.0)
    swatch.toggleable = False
    row.add_child(swatch)
    row.add_fixed(int(0.5 * em))
    row.add_child(gui.Label(f"{layer_name} ({n_feat})"))
    panel.add_child(row)
    panel.add_fixed(int(0.15 * em))

# Component layers
for layer_name, cfg in COMPONENT_LAYERS.items():
    n_comp = comp_stats.get(layer_name, 0)
    if n_comp == 0:
        continue
    col = cfg["color"]
    sr, sg, sb = (linear_to_srgb(c) for c in col)

    row     = gui.Horiz(int(0.3 * em))
    swatch  = gui.Button("        ")
    swatch.background_color = gui.Color(sr, sg, sb, 1.0)
    swatch.toggleable = False
    row.add_child(swatch)
    row.add_fixed(int(0.5 * em))
    row.add_child(gui.Label(f"{layer_name} ({n_comp})"))
    panel.add_child(row)
    panel.add_fixed(int(0.15 * em))

# Bbox wireframe legend entry
panel.add_fixed(int(0.3 * em))
bbox_row    = gui.Horiz(int(0.3 * em))
bbox_swatch = gui.Button("        ")
bbox_swatch.background_color = gui.Color(1.0, 1.0, 0.0, 1.0)
bbox_swatch.toggleable = False
bbox_row.add_child(bbox_swatch)
bbox_row.add_fixed(int(0.5 * em))
bbox_row.add_child(gui.Label(f"Search bbox ({BUFFER}m buffer)"))
panel.add_child(bbox_row)

# ── Selected Feature panel ────────────────────────────────────────────────────
panel.add_fixed(int(0.8 * em))
panel.add_child(gui.Label("--- Selected Feature ---"))
panel.add_fixed(int(0.3 * em))

info_hint = gui.Label("Ctrl+click a pipe or component")
info_hint.text_color = gui.Color(0.55, 0.55, 0.55, 1.0)
panel.add_child(info_hint)
panel.add_fixed(int(0.3 * em))

info_scroll = gui.ScrollableVert(int(0.2 * em), gui.Margins(0, 0, 0, 0))
panel.add_child(info_scroll)

_info_type_lbl            = gui.Label("")
_info_type_lbl.text_color = gui.Color(0.85, 0.85, 0.20, 1.0)
info_scroll.add_child(_info_type_lbl)
info_scroll.add_fixed(int(0.25 * em))

# Pre-allocate attribute label rows (enough for the largest feature)
_MAX_ATTRS = 30
_attr_rows = []
for _ in range(_MAX_ATTRS):
    row_h = gui.Horiz(int(0.15 * em))
    k_lbl = gui.Label("")
    v_lbl = gui.Label("")
    k_lbl.text_color = gui.Color(0.65, 0.75, 1.00, 1.0)
    v_lbl.text_color = gui.Color(0.90, 0.90, 0.90, 1.0)
    row_h.add_child(k_lbl)
    row_h.add_fixed(int(0.3 * em))
    row_h.add_child(v_lbl)
    info_scroll.add_child(row_h)
    _attr_rows.append((k_lbl, v_lbl))

panel.add_stretch()


def _show_feature_attrs(feature_type: str, attrs: list):
    """Populate the Selected Feature panel with attribute key-value pairs."""
    info_hint.visible   = False
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
    window.set_needs_layout()
    window.post_redraw()


def _clear_highlight():
    if scene_widget.scene.has_geometry(HIGHLIGHT_GEOM):
        scene_widget.scene.remove_geometry(HIGHLIGHT_GEOM)
    pick_active[0] = False


def _place_highlight(centre: np.ndarray):
    _clear_highlight()
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.15, resolution=10)
    marker.translate(centre)
    marker.paint_uniform_color([1.0, 1.0, 0.0])
    marker.compute_vertex_normals()
    marker_mat        = rendering.MaterialRecord()
    marker_mat.shader = "defaultUnlit"
    scene_widget.scene.add_geometry(HIGHLIGHT_GEOM, marker, marker_mat)
    pick_active[0] = True
    window.post_redraw()

# ─────────────────────────────────────────────────────────────────────────────
# 11.  Mouse picking  (Ctrl + Left-Click)
# ─────────────────────────────────────────────────────────────────────────────
PICK_RADIUS_SEG  = 2.0   # max distance (m) for a pipe segment hit
PICK_RADIUS_COMP = 1.0   # max distance (m) for a component hit
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
        # Clicked on background — clear selection
        def _clear():
            _clear_highlight()
            info_hint.visible   = True
            _info_type_lbl.text = ""
            for k_lbl, v_lbl in _attr_rows:
                k_lbl.visible = False
                v_lbl.visible = False
            window.post_redraw()
        gui.Application.instance.post_to_main_thread(window, _clear)
        return

    world = scene_widget.scene.camera.unproject(
        ex, ey, depth,
        scene_widget.frame.width,
        scene_widget.frame.height,
    )
    hit = np.array(world[:3], dtype=float)

    # Find nearest segment midpoint
    best_seg_d = np.inf
    best_seg_i = -1
    if len(pick_seg_midpoints) > 0:
        dists = np.linalg.norm(pick_seg_midpoints - hit, axis=1)
        best_seg_i = int(np.argmin(dists))
        best_seg_d = float(dists[best_seg_i])

    # Find nearest component centre
    best_comp_d = np.inf
    best_comp_i = -1
    if len(pick_comp_centres) > 0:
        dists = np.linalg.norm(pick_comp_centres - hit, axis=1)
        best_comp_i = int(np.argmin(dists))
        best_comp_d = float(dists[best_comp_i])

    # Pick whichever is closer, with radius thresholds
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
        # Nothing close enough — clear
        def _clear():
            _clear_highlight()
            info_hint.visible   = True
            _info_type_lbl.text = ""
            for k_lbl, v_lbl in _attr_rows:
                k_lbl.visible = False
                v_lbl.visible = False
            window.post_redraw()
        gui.Application.instance.post_to_main_thread(window, _clear)
        return

    def _update():
        _place_highlight(centre)
        _show_feature_attrs(label, attrs)
        window.set_needs_layout()
        window.post_redraw()
    gui.Application.instance.post_to_main_thread(window, _update)


def on_mouse(event):
    if event.type != gui.MouseEvent.Type.BUTTON_DOWN:
        return gui.Widget.EventCallbackResult.IGNORED
    if not (int(event.buttons) & int(gui.MouseButton.LEFT)):
        return gui.Widget.EventCallbackResult.IGNORED
    if not event.is_modifier_down(gui.KeyModifier.CTRL):
        return gui.Widget.EventCallbackResult.IGNORED

    print(f"[pick] Ctrl+click at ({event.x}, {event.y})")
    _last_click[0] = (event.x, event.y)
    scene_widget.scene.scene.render_to_depth_image(_do_pick)
    return gui.Widget.EventCallbackResult.HANDLED


scene_widget.set_on_mouse(on_mouse)

# ───────────────��─────────────────────────────────────────────────────────────
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
    if k == ord('0'):
        print("Pivot -> world origin")
        _pivot_to(origin_pt)
        return HANDLED

    if k in (ord('H'), ord('h')):
        print("\n-- Shortcuts ---------------------------------------------------")
        print("  Ctrl+click     pick pipe segment or component (show attributes)")
        print("  C              pivot to point cloud centroid")
        print("  P              pivot to pipe centroid (all utilities)")
        print("  0              pivot to world origin (0, 0, 0)")
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
    scene_widget.frame = gui.Rect(r.x, r.y, r.width - PANEL_WIDTH, r.height)
    panel.frame        = gui.Rect(r.x + r.width - PANEL_WIDTH, r.y, PANEL_WIDTH, r.height)


window.set_on_layout(on_layout)
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
