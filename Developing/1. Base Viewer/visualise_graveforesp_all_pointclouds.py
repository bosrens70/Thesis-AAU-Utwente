"""
Graveforesp Viewer — All Point Clouds + All Utilities + Toggleable Layers
==========================================================================
Loads the Graveforesp polygon from the consolidated GML, applies a 2 m
buffer, and visualises:

  • Every PLY point cloud whose footprint overlaps the buffered polygon
  • The Graveforesp boundary as a semi-transparent surface mesh
  • All utility categories (pipes + components) within the buffered area

Layer controls
--------------
  Each utility category and the Graveforesp surface can be toggled on/off
  with a checkbox.  Each has its own opacity slider.

Ground-level picking
--------------------
  Before the main viewer opens, a VisualizerWithEditing window lets you
  pick ground-level points (Shift + Left-Click, then Q to close).

Attribute picking
-----------------
  Ctrl + Left-Click on any pipe segment or component sphere to show all
  its GML attribute fields in the "Selected Feature" panel.

Keyboard shortcuts
------------------
  C   pivot to cloud centroid           P   pivot to pipe centroid
  0   pivot to world origin
  ]   increase global utility opacity   [   decrease global utility opacity
  L   toggle class label colours
  H   help                             Esc quit
"""

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import geopandas as gpd
import numpy as np
import matplotlib.path as mpath
from shapely.geometry import Polygon
from shapely.ops import unary_union
from pathlib import Path
import re
import time

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
AREA_REF_GEOJSON = (
    r"C:\Users\bosre\OneDrive - University of Twente\Documents\AAU UTwente thesis"
    r"\Python\Thesis\Data\Translation_coordinates\area_points_utm32_etrs89.geojson"
)

GML_PATH = (
    r"C:\Users\bosre\OneDrive - University of Twente\Documents\AAU UTwente thesis"
    r"\Python\Thesis\Data\Ledningspakke_3383910\consolidated.gml"
)

PLY_BASE_DIR = (
    r"C:\Users\bosre\OneDrive - University of Twente\Documents\AAU UTwente thesis"
    r"\Python\Thesis\Data\OpenTrench3D"
)

# Buffer (metres) around the Graveforesp polygon
BUFFER = 2.0

# Voxel size for the fast bbox-screening pass (larger = faster but coarser)
SCREEN_VOXEL_SIZE = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# CLASS LABEL DEFINITIONS — OpenTrench3D semantic classes
# ─────────────────────────────────────────────────────────────────────────────
CLASS_LABELS = {
    0: {"name": "Main Utility",     "color": [0.00, 0.80, 0.00]},
    1: {"name": "Other Utility",    "color": [1.00, 1.00, 0.00]},
    2: {"name": "Trench",           "color": [0.55, 0.27, 0.07]},
    3: {"name": "Inactive Utility", "color": [0.00, 0.00, 0.00]},
    4: {"name": "Misc",             "color": [0.60, 0.60, 0.60]},
}
_DEFAULT_CLASS_COLOR = [1.0, 0.0, 1.0]

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY LAYER DEFINITIONS
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

_t0 = time.perf_counter()

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load Graveforesp polygon
# ─────────────────────────────────────────────────────────────────────────────
print("Loading Graveforesp polygon from GML ...")
gdf_grave  = gpd.read_file(GML_PATH, layer="Graveforesp")
grave_geom = gdf_grave.geometry.iloc[0]

# Buffer the polygon by BUFFER metres (in UTM coordinates)
grave_buffered = grave_geom.buffer(BUFFER)

grave_xy_utm   = np.array(grave_geom.exterior.coords)[:, :2]
buf_xy_utm     = np.array(grave_buffered.exterior.coords)[:, :2]

print(f"  Original polygon vertices: {len(grave_xy_utm)}")
print(f"  Buffered polygon vertices: {len(buf_xy_utm)}")

# Determine which area the Graveforesp falls in by checking overlap with
# the area reference points
ref = gpd.read_file(AREA_REF_GEOJSON)

# Find the centroid of the graveforesp in UTM
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

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Find all PLY folders for this area and scan for overlapping sites
# ─────────────────────────────────────────────────────────────────────────────
area_num = re.search(r"\d+", AREA_NAME).group()
ply_dirs = sorted(Path(PLY_BASE_DIR).glob(f"*Area_{area_num}*"))
if not ply_dirs:
    ply_dirs = sorted(Path(PLY_BASE_DIR).glob(f"*Area{area_num}*"))

# Collect all PLY files
all_ply_files = []
for d in ply_dirs:
    # Only use the base area folder (e.g. Water_Area_5), skip Finetuning variants
    if "Finetuning" in d.name:
        continue
    plys = sorted(d.glob("Area_*_Site_*.ply"))
    # Exclude UTM-transformed copies
    plys = [p for p in plys if "_utm" not in p.stem.lower()]
    all_ply_files.extend(plys)
    if plys:
        print(f"  Found {len(plys)} PLY files in {d.name}")

if not all_ply_files:
    print("[ERROR] No PLY files found for area. Check PLY_BASE_DIR.")
    raise SystemExit(1)

# Pass 1: fast bbox screening with downsampled clouds
print(f"\nPass 1 — screening {len(all_ply_files)} sites against buffered Graveforesp bbox ...")
sites_in_bbox = []

for ply_path in all_ply_files:
    pcd_down = o3d.io.read_point_cloud(str(ply_path))
    pcd_down = pcd_down.voxel_down_sample(voxel_size=SCREEN_VOXEL_SIZE)
    pts = np.asarray(pcd_down.points)
    mask = (
        (pts[:, 0] >= gx_min) & (pts[:, 0] <= gx_max) &
        (pts[:, 1] >= gy_min) & (pts[:, 1] <= gy_max)
    )
    if mask.any():
        sites_in_bbox.append(ply_path)
        print(f"  {ply_path.stem} OK  ({mask.sum():,} / {len(pts):,} voxel pts in bbox)")

print(f"\n  {len(sites_in_bbox)} sites overlap the buffered Graveforesp")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Pass 2 — full-resolution crop to buffered polygon
# ─────────────────────────────────────────────────────────────────────────────
print("\nPass 2 — loading full-resolution clouds, cropping to buffered Graveforesp ...")
all_pcd_filtered = []
total_pts_raw    = 0
total_pts_filt   = 0

# Store class labels across all clouds
all_class_labels = []

print(f"\n{'Site':>30}  {'Full res':>10}  {'In polygon':>10}")
print("-" * 56)

for ply_path in sites_in_bbox:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    total_pts_raw += len(pts)

    # Read class labels from PLY
    site_class_labels = None
    try:
        with open(str(ply_path), 'r') as f:
            property_names = []
            for line in f:
                line = line.strip()
                if line.startswith("property "):
                    property_names.append(line.split()[-1])
                if line == "end_header":
                    break
            if "class" in property_names:
                class_col_idx = property_names.index("class")
                labels = []
                for line in f:
                    parts = line.split()
                    if len(parts) > class_col_idx:
                        labels.append(int(parts[class_col_idx]))
                site_class_labels = np.array(labels, dtype=int)
    except Exception:
        pass

    # Bbox pre-filter then polygon crop
    bbox_mask = (
        (pts[:, 0] >= gx_min) & (pts[:, 0] <= gx_max) &
        (pts[:, 1] >= gy_min) & (pts[:, 1] <= gy_max)
    )
    candidates = pts[bbox_mask]
    poly_mask = buf_path.contains_points(candidates[:, :2])
    bbox_indices = np.where(bbox_mask)[0]
    final_indices = bbox_indices[poly_mask]

    pcd_filt = pcd.select_by_index(final_indices)
    n_filt = len(pcd_filt.points)
    total_pts_filt += n_filt

    if n_filt > 0:
        all_pcd_filtered.append(pcd_filt)
        # Store corresponding class labels
        if site_class_labels is not None and len(site_class_labels) == len(pts):
            all_class_labels.append(site_class_labels[final_indices])
        else:
            all_class_labels.append(None)

    print(f"  {ply_path.stem:>28}  {len(pts):>10,}  {n_filt:>10,}")

print("-" * 56)
print(f"  {'Total':>28}  {total_pts_raw:>10,}  {total_pts_filt:>10,}  "
      f"({len(all_pcd_filtered)} sites)\n")

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
    # Unknown / no-label points keep original colour
    unknown = ~np.isin(merged_class_labels, list(CLASS_LABELS.keys()))
    class_colors[unknown] = original_colors[unknown]
else:
    merged_class_labels = None
    class_colors = None

print(f"  Merged point cloud: {len(all_pts):,} points")
print(f"  Cloud centroid (local): [{cloud_centroid[0]:.2f}, {cloud_centroid[1]:.2f}, {cloud_centroid[2]:.2f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Pick ground-level points interactively
# ─────────────────────────────────────────────────────────────────────────────
def pick_points(pcd):
    print("\n" + "=" * 62)
    print("  GROUND-LEVEL POINT PICKING")
    print("=" * 62)
    print("  Shift + Left-Click  to select points on the ground surface.")
    print("  Pick one or more points that represent the ground level.")
    print("  Press Q or close the window when finished.")
    print("=" * 62 + "\n")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name="Pick ground-level points  (Shift+Click, then Q)",
        width=1280, height=720,
    )
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()


picked_indices = pick_points(merged_pcd)

if len(picked_indices) == 0:
    print("[WARNING] No points picked!  Falling back to P95 of point cloud Z.")
    GROUND_Z = float(np.percentile(all_pts[:, 2], 95))
else:
    picked_pts = all_pts[picked_indices]
    GROUND_Z = float(np.mean(picked_pts[:, 2]))
    print(f"\n  Picked {len(picked_indices)} ground-level point(s):")
    for i, idx in enumerate(picked_indices):
        p = all_pts[idx]
        print(f"    [{i+1}]  index {idx:>8,}  ->  X={p[0]:.3f}  Y={p[1]:.3f}  Z={p[2]:.3f}")

print(f"\n  Ground level (local) = {GROUND_Z:.3f} m")

_depth_stats = {"estimated": 0, "fallback_feature_mean": 0, "fallback_global": 0}

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def segment_to_cylinder(p1, p2, radius, color, resolution=12):
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


def _clean_coords_with_depth(coords_raw, vejledende_dybde_mm):
    coords = coords_raw.copy().astype(float)
    if coords.shape[1] == 2:
        coords = np.hstack([coords, np.zeros((len(coords), 1))])

    coords[:, 0] -= TX
    coords[:, 1] -= TY

    bad = coords[:, 2] == -99
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

        for idx in np.where(bad)[0]:
            if ind_depth_m is not None:
                coords[idx, 2] = (GROUND_Z + TZ) - ind_depth_m
                _depth_stats["estimated"] += 1
            elif feature_mean_z is not None:
                coords[idx, 2] = feature_mean_z
                _depth_stats["fallback_feature_mean"] += 1
            else:
                coords[idx, 2] = GROUND_Z + TZ
                _depth_stats["fallback_global"] += 1

    coords[:, 2] -= TZ
    return coords


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
# 6.  Build Graveforesp surface mesh (triangulated polygon)
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding Graveforesp surface mesh ...")

# Use the original (unbuffered) polygon for the surface
grave_verts_2d = grave_xy_local[:-1]  # drop closing vertex (duplicate of first)
n_grave = len(grave_verts_2d)

# Create 3D vertices at ground level
grave_verts_3d = np.zeros((n_grave, 3))
grave_verts_3d[:, 0] = grave_verts_2d[:, 0]
grave_verts_3d[:, 1] = grave_verts_2d[:, 1]
grave_verts_3d[:, 2] = GROUND_Z

# Fan triangulation from vertex 0
grave_triangles = []
for i in range(1, n_grave - 1):
    grave_triangles.append([0, i, i + 1])

grave_mesh = o3d.geometry.TriangleMesh()
grave_mesh.vertices = o3d.utility.Vector3dVector(grave_verts_3d)
grave_mesh.triangles = o3d.utility.Vector3iVector(np.array(grave_triangles, dtype=np.int32))
grave_mesh.paint_uniform_color([0.9, 0.9, 0.2])  # yellow-ish
grave_mesh.compute_vertex_normals()

print(f"  {n_grave} vertices, {len(grave_triangles)} triangles")

# Also build the boundary wireframe
grave_wire_pts = np.hstack([grave_xy_local, np.full((len(grave_xy_local), 1), GROUND_Z)])
grave_lines = [[i, i + 1] for i in range(len(grave_wire_pts) - 1)]
grave_ls = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(grave_wire_pts),
    lines=o3d.utility.Vector2iVector(grave_lines),
)
grave_ls.paint_uniform_color([1.0, 1.0, 0.0])

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Load utility lines (pipes / cables) within buffered Graveforesp
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Loading utility lines within buffered Graveforesp ---")

# Store meshes per layer for individual toggling
layer_meshes = {}       # layer_name -> combined TriangleMesh
layer_stats = {}        # layer_name -> (n_features, n_segments)
all_pipe_coords = []

# Picking data
pick_seg_midpoints = []
pick_seg_attrs = []
pick_seg_layer = []

# Per-layer average depth for component fallback
_layer_avg_depth_local = {}

for layer_name, cfg in LINE_LAYERS.items():
    try:
        gdf = gpd.read_file(GML_PATH, layer=layer_name)
    except Exception as e:
        print(f"  {layer_name}: skip ({e})")
        continue

    default_color = cfg["color"]
    fallback_radius = cfg["fallback_radius"]
    n_features = 0
    n_segments = 0
    _layer_z_vals = []
    layer_mesh_list = []

    for _, row in gdf.iterrows():
        geom = row.geometry

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

        if layer_name == "Vandledning" and diam_mm > 0:
            color = DIAMETER_COLORS.get(int(diam_mm), default_color)
        else:
            color = default_color

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
                    layer_mesh_list.append(cyl)
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

    # Merge layer meshes
    if layer_mesh_list:
        combined = layer_mesh_list[0]
        for m in layer_mesh_list[1:]:
            combined += m
        combined.compute_vertex_normals()
        layer_meshes[layer_name] = combined
        print(f"  {layer_name:<35} {n_features:>4} features  {n_segments:>5} segments")

pick_seg_midpoints = np.array(pick_seg_midpoints) if pick_seg_midpoints else np.empty((0, 3))

print(f"\n  Total line segments: {sum(s for _, s in layer_stats.values()):,}")
print(f"  Depth stats: estimated={_depth_stats['estimated']}, "
      f"fallback_mean={_depth_stats['fallback_feature_mean']}, "
      f"fallback_global={_depth_stats['fallback_global']}")

# Pipe centroid
pipe_centroid = np.array([0.0, 0.0, 0.0])
if all_pipe_coords:
    pipe_centroid = np.vstack(all_pipe_coords).mean(axis=0)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Load utility components (points) within buffered Graveforesp
# ─────────────────────────────────────────────────────────────────────────────
_COMP_TO_LINE = {
    "Vandkomponent":               "Vandledning",
    "Afloebskomponent":            "Afloebsledning",
    "Gaskomponent":                "Gasledning",
    "Elkomponent":                 "Elledning",
    "Telekommunikationskomponent": "Telekommunikationsledning",
}

print("\n--- Loading utility components within buffered Graveforesp ---")
comp_meshes = {}   # layer_name -> combined TriangleMesh
comp_stats = {}

pick_comp_centres = []
pick_comp_attrs = []
pick_comp_layer = []

for layer_name, cfg in COMPONENT_LAYERS.items():
    try:
        gdf_c = gpd.read_file(GML_PATH, layer=layer_name)
    except Exception:
        continue

    color = cfg["color"]
    n_comp = 0
    comp_mesh_list = []

    parent_line = _COMP_TO_LINE.get(layer_name)
    parent_avg_z = _layer_avg_depth_local.get(parent_line) if parent_line else None

    for _, row in gdf_c.iterrows():
        g = row.geometry
        if not _point_in_bbox(g.x, g.y):
            continue

        pt = np.array([g.x - TX, g.y - TY, g.z - TZ], dtype=float)
        if not _pt_in_local_bbox(pt[0], pt[1]):
            continue

        if g.z == -99 or pt[2] <= -98:
            if parent_avg_z is not None:
                pt[2] = parent_avg_z
            else:
                pt[2] = GROUND_Z

        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=COMPONENT_SPHERE_RADIUS, resolution=12
        )
        sphere.translate(pt)
        sphere.paint_uniform_color(color)
        comp_mesh_list.append(sphere)

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
    if comp_mesh_list:
        combined = comp_mesh_list[0]
        for m in comp_mesh_list[1:]:
            combined += m
        combined.compute_vertex_normals()
        comp_meshes[layer_name] = combined
        print(f"  {layer_name:<35} {n_comp:>4} components")

pick_comp_centres = np.array(pick_comp_centres) if pick_comp_centres else np.empty((0, 3))

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
    mat.shader = "defaultUnlitTransparency"
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
scene_widget.scene.set_background([0.10, 0.10, 0.10, 1.0])

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
# 12.  Right-side control panel
# ─────────────────────────────────────────────────────────────────────────────
PANEL_WIDTH = int(22 * em)
panel = gui.ScrollableVert(int(0.5 * em), gui.Margins(int(em), int(em), int(em), int(em)))

# Title
panel.add_child(gui.Label("--- Graveforesp Viewer ---"))
panel.add_fixed(int(0.3 * em))
panel.add_child(gui.Label(f"Area: {AREA_NAME}"))
panel.add_child(gui.Label(f"Sites: {len(all_pcd_filtered)}  |  Points: {total_pts_filt:,}"))
panel.add_child(gui.Label(f"Buffer: {BUFFER} m"))
panel.add_fixed(int(0.3 * em))

_pick_method = f"picked from {len(picked_indices)} pt(s)" if picked_indices else "fallback P95"
panel.add_child(gui.Label(f"Ground Z: {GROUND_Z:.3f} m ({_pick_method})"))
panel.add_fixed(int(0.8 * em))

# ── Class Label Toggle ─────────────────────────────────────────────────────
panel.add_child(gui.Label("--- Class Labels ---"))
panel.add_fixed(int(0.3 * em))

class_toggle_cb = gui.Checkbox("Show class label colours  (L)")
class_toggle_cb.checked = False
if class_colors is None:
    class_toggle_cb.enabled = False

def _on_class_toggle(checked):
    _toggle_class_labels(checked)

class_toggle_cb.set_on_checked(_on_class_toggle)
panel.add_child(class_toggle_cb)
panel.add_fixed(int(0.8 * em))

# ── Graveforesp Surface Controls ───────────────────────────────────────────
panel.add_child(gui.Label("--- Graveforesp Surface ---"))
panel.add_fixed(int(0.3 * em))

grave_visible_cb = gui.Checkbox("Show surface")
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
grave_opacity_row.add_stretch()
grave_opacity_row.add_child(grave_opacity_label)
panel.add_child(grave_opacity_row)

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
panel.add_child(grave_slider)
panel.add_fixed(int(0.8 * em))

# ── Per-Category Utility Controls ──────────────────────────────────────────
panel.add_child(gui.Label("--- Utility Layers ---"))
panel.add_fixed(int(0.3 * em))

# Build a unified list of all layers that have geometry
all_layer_names = []
for ln in LINE_LAYERS:
    if ln in layer_meshes:
        all_layer_names.append(ln)
for ln in COMPONENT_LAYERS:
    if ln in comp_meshes:
        all_layer_names.append(ln)

# Store UI widgets for each layer
_layer_sliders = {}
_layer_checkboxes = {}


def _make_layer_toggle(layer_name):
    """Create a checkbox + opacity slider for one utility layer."""
    is_line = layer_name in LINE_LAYERS
    is_comp = layer_name in COMPONENT_LAYERS
    cfg = LINE_LAYERS.get(layer_name) or COMPONENT_LAYERS.get(layer_name)
    col = cfg["color"]
    sr, sg, sb = (linear_to_srgb(c) for c in col)

    n_feat, n_seg = layer_stats.get(layer_name, (0, 0))
    n_comp_count = comp_stats.get(layer_name, 0)

    # Swatch + checkbox row
    row = gui.Horiz(int(0.3 * em))
    swatch = gui.Button("    ")
    swatch.background_color = gui.Color(sr, sg, sb, 1.0)
    swatch.toggleable = False
    row.add_child(swatch)
    row.add_fixed(int(0.3 * em))

    if is_line and layer_name in layer_meshes:
        label_text = f"{layer_name} ({n_feat}f/{n_seg}s)"
    elif is_comp:
        label_text = f"{layer_name} ({n_comp_count})"
    else:
        label_text = layer_name

    cb = gui.Checkbox(label_text)
    cb.checked = True
    row.add_child(cb)
    panel.add_child(row)

    # Opacity slider
    opacity_lbl = gui.Label("1.00")
    slider_row = gui.Horiz(int(0.25 * em))
    slider_row.add_child(gui.Label("  Opacity"))
    slider_row.add_stretch()
    slider_row.add_child(opacity_lbl)
    panel.add_child(slider_row)

    slider = gui.Slider(gui.Slider.DOUBLE)
    slider.set_limits(0.0, 1.0)
    slider.double_value = 1.0
    panel.add_child(slider)
    panel.add_fixed(int(0.3 * em))

    _layer_sliders[layer_name] = (slider, opacity_lbl)
    _layer_checkboxes[layer_name] = cb

    def _apply(val, _ln=layer_name):
        val = max(0.0, min(1.0, val))
        layer_opacity[_ln][0] = val
        _layer_sliders[_ln][0].double_value = val
        _layer_sliders[_ln][1].text = f"{val:.2f}"
        mat = make_mesh_material(val)
        if _ln in layer_meshes:
            gn = _line_geom_name(_ln)
            scene_widget.scene.remove_geometry(gn)
            scene_widget.scene.add_geometry(gn, layer_meshes[_ln], mat)
            if not _layer_checkboxes[_ln].checked:
                scene_widget.scene.show_geometry(gn, False)
        if _ln in comp_meshes:
            gn = _comp_geom_name(_ln)
            scene_widget.scene.remove_geometry(gn)
            scene_widget.scene.add_geometry(gn, comp_meshes[_ln], mat)
            if not _layer_checkboxes[_ln].checked:
                scene_widget.scene.show_geometry(gn, False)
        window.post_redraw()

    slider.set_on_value_changed(lambda v, _ln=layer_name: _apply(v, _ln))

    def _toggle(checked, _ln=layer_name):
        if _ln in layer_meshes:
            scene_widget.scene.show_geometry(_line_geom_name(_ln), checked)
        if _ln in comp_meshes:
            scene_widget.scene.show_geometry(_comp_geom_name(_ln), checked)
        window.post_redraw()

    cb.set_on_checked(lambda c, _ln=layer_name: _toggle(c, _ln))


# Group line layers, then component layers
# First: line layers
has_line_content = any(ln in layer_meshes for ln in LINE_LAYERS)
if has_line_content:
    panel.add_child(gui.Label("Pipes / Cables:"))
    panel.add_fixed(int(0.2 * em))
    for ln in LINE_LAYERS:
        if ln in layer_meshes:
            _make_layer_toggle(ln)

has_comp_content = any(ln in comp_meshes for ln in COMPONENT_LAYERS)
if has_comp_content:
    panel.add_fixed(int(0.3 * em))
    panel.add_child(gui.Label("Components:"))
    panel.add_fixed(int(0.2 * em))
    for ln in COMPONENT_LAYERS:
        if ln in comp_meshes:
            _make_layer_toggle(ln)

panel.add_fixed(int(0.8 * em))

# ── Selected Feature panel ────────────────────────────────────────────────
panel.add_child(gui.Label("--- Selected Feature ---"))
panel.add_fixed(int(0.3 * em))

info_hint = gui.Label("Ctrl+click a pipe or component")
info_hint.text_color = gui.Color(0.55, 0.55, 0.55, 1.0)
panel.add_child(info_hint)
panel.add_fixed(int(0.3 * em))

info_scroll = gui.ScrollableVert(int(0.2 * em), gui.Margins(0, 0, 0, 0))
panel.add_child(info_scroll)

_info_type_lbl = gui.Label("")
_info_type_lbl.text_color = gui.Color(0.85, 0.85, 0.20, 1.0)
info_scroll.add_child(_info_type_lbl)
info_scroll.add_fixed(int(0.25 * em))

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


def _show_feature_attrs(feature_type: str, attrs: list):
    info_hint.visible = False
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
_last_click = [None]


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

    if depth >= 1.0:
        def _clear():
            _clear_highlight()
            info_hint.visible = True
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

    best_seg_d = np.inf
    best_seg_i = -1
    if len(pick_seg_midpoints) > 0:
        dists = np.linalg.norm(pick_seg_midpoints - hit, axis=1)
        best_seg_i = int(np.argmin(dists))
        best_seg_d = float(dists[best_seg_i])

    best_comp_d = np.inf
    best_comp_i = -1
    if len(pick_comp_centres) > 0:
        dists = np.linalg.norm(pick_comp_centres - hit, axis=1)
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
    else:
        def _clear():
            _clear_highlight()
            info_hint.visible = True
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

    _last_click[0] = (event.x, event.y)
    scene_widget.scene.scene.render_to_depth_image(_do_pick)
    return gui.Widget.EventCallbackResult.HANDLED


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
        # Increase all utility layer opacities by 0.05
        for ln in layer_opacity:
            new_val = min(1.0, layer_opacity[ln][0] + 0.05)
            if ln in _layer_sliders:
                _layer_sliders[ln][0].double_value = new_val
                _layer_sliders[ln][1].text = f"{new_val:.2f}"
            layer_opacity[ln][0] = new_val
            mat = make_mesh_material(new_val)
            if ln in layer_meshes:
                scene_widget.scene.remove_geometry(_line_geom_name(ln))
                scene_widget.scene.add_geometry(_line_geom_name(ln), layer_meshes[ln], mat)
            if ln in comp_meshes:
                scene_widget.scene.remove_geometry(_comp_geom_name(ln))
                scene_widget.scene.add_geometry(_comp_geom_name(ln), comp_meshes[ln], mat)
        window.post_redraw()
        return HANDLED

    if k == ord('['):
        for ln in layer_opacity:
            new_val = max(0.0, layer_opacity[ln][0] - 0.05)
            if ln in _layer_sliders:
                _layer_sliders[ln][0].double_value = new_val
                _layer_sliders[ln][1].text = f"{new_val:.2f}"
            layer_opacity[ln][0] = new_val
            mat = make_mesh_material(new_val)
            if ln in layer_meshes:
                scene_widget.scene.remove_geometry(_line_geom_name(ln))
                scene_widget.scene.add_geometry(_line_geom_name(ln), layer_meshes[ln], mat)
            if ln in comp_meshes:
                scene_widget.scene.remove_geometry(_comp_geom_name(ln))
                scene_widget.scene.add_geometry(_comp_geom_name(ln), comp_meshes[ln], mat)
        window.post_redraw()
        return HANDLED

    if k in (ord('L'), ord('l')):
        new_state = not class_labels_active[0]
        class_toggle_cb.checked = new_state
        _toggle_class_labels(new_state)
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
        print("  Ctrl+click     pick pipe segment or component (show attributes)")
        print("  C              pivot to point cloud centroid")
        print("  P              pivot to pipe centroid (all utilities)")
        print("  0              pivot to world origin (0, 0, 0)")
        print("  ]              increase all utility opacities +0.05")
        print("  [              decrease all utility opacities -0.05")
        print("  L              toggle class label colours on/off")
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
    scene_widget.frame = gui.Rect(r.x, r.y, r.width - PANEL_WIDTH, r.height)
    panel.frame = gui.Rect(r.x + r.width - PANEL_WIDTH, r.y, PANEL_WIDTH, r.height)


window.set_on_layout(on_layout)
window.add_child(scene_widget)
window.add_child(panel)

n_total_segs = sum(s for _, s in layer_stats.values())
n_total_comps = sum(comp_stats.values())
print(f"\nRendering {total_pts_filt:,} points  +  {n_total_segs:,} pipe segments  "
      f"+  {n_total_comps} components")
print("Launching viewer ...\n")

app.run()
print("Viewer closed.")
