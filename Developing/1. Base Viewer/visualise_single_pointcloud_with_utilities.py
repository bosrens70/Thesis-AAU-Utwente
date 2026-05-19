# -*- coding: utf-8 -*-
"""
Single Point Cloud Viewer with Surrounding Utilities
=====================================================
Visualises one PLY point cloud together with all utility infrastructure
(pipes, cables, components) that falls within a configurable buffer
around the point cloud's bounding box.

Usage
-----
  Set PLY_FILE below to any individual .ply site file, then run.
  The script auto-detects which Area the file belongs to (by folder name)
  and loads the matching translation origin.

Supported utility layers (from consolidated.gml):
  Vandledning, Afloebsledning, Gasledning, Elledning,
  Telekommunikationsledning, Foeringsroer, LedningUkendtForsyningsart,
  Ledningstrace

Component layers:
  Vandkomponent, Afloebskomponent, Gaskomponent, Elkomponent,
  Telekommunikationskomponent

Keyboard shortcuts
------------------
  C   pivot to cloud centroid
  P   pivot to pipe centroid (all utilities)
  0   pivot to world origin
  ]   increase opacity +0.05
  [   decrease opacity -0.05
  H   help
  Esc quit
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
# 3.  Geometry helpers
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


def _clean_coords(coords_raw):
    """Translate UTM → local, fill -99 sentinel Z values."""
    coords = coords_raw.copy().astype(float)
    if coords.shape[1] == 2:
        coords = np.hstack([coords, np.zeros((len(coords), 1))])
    bad = coords[:, 2] == -99
    good_z = coords[~bad, 2]
    coords[bad, 2] = good_z.mean() if len(good_z) > 0 else 0.0
    coords[:, 0] -= TX
    coords[:, 1] -= TY
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

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Load utility lines (pipes / cables) within bbox
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Loading utility lines within bbox ---")
all_pipe_meshes = []
layer_stats = {}
all_pipe_coords = []

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

        feature_hit = False
        for sub_geom in sub_geoms:
            coords_raw = np.array(sub_geom.coords, dtype=float)
            if not _segments_in_bbox(coords_raw):
                continue

            coords = _clean_coords(coords_raw)
            all_pipe_coords.append(coords)
            feature_hit = True

            for i in range(len(coords) - 1):
                cyl = segment_to_cylinder(coords[i], coords[i + 1], radius, color)
                if cyl is not None:
                    all_pipe_meshes.append(cyl)
                    n_segments += 1

        if feature_hit:
            n_features += 1

    layer_stats[layer_name] = (n_features, n_segments)
    if n_features > 0:
        print(f"  {layer_name:<35} {n_features:>4} features  {n_segments:>5} segments")

print(f"\n  Total: {len(all_pipe_meshes):,} cylinder segments")

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
# 5.  Load utility components (points) within bbox
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Loading utility components within bbox ---")
all_comp_meshes = []
comp_stats = {}

for layer_name, cfg in COMPONENT_LAYERS.items():
    try:
        gdf_c = gpd.read_file(GML_PATH, layer=layer_name)
    except Exception:
        continue

    color = cfg["color"]
    n_comp = 0

    for _, row in gdf_c.iterrows():
        g = row.geometry
        if not _point_in_bbox(g.x, g.y):
            continue

        pt = np.array([g.x - TX, g.y - TY, g.z - TZ], dtype=float)
        if pt[2] <= -98:
            pt[2] = 0.0  # sentinel fill

        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=COMPONENT_SPHERE_RADIUS, resolution=12
        )
        sphere.translate(pt)
        sphere.paint_uniform_color(color)
        all_comp_meshes.append(sphere)
        n_comp += 1

    comp_stats[layer_name] = n_comp
    if n_comp > 0:
        print(f"  {layer_name:<35} {n_comp:>4} components")

print(f"\n  Total: {len(all_comp_meshes)} component spheres")

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
# 6.  Coordinate frame + bounding box wireframe
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
    [bx0, by0, 0], [bx1, by0, 0], [bx1, by1, 0], [bx0, by1, 0], [bx0, by0, 0]
])
bbox_lines = [[i, i + 1] for i in range(4)]
bbox_ls = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(bbox_wire_pts),
    lines=o3d.utility.Vector2iVector(bbox_lines),
)
bbox_ls.paint_uniform_color([1.0, 1.0, 0.0])

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Material helpers
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
# 8.  Build GUI
# ─────────────────────────────────────────────────────────────────────────────
PIPE_GEOM  = "pipes"
COMP_GEOM  = "components"
FRAME_GEOM = "frame"
BBOX_GEOM  = "bbox_wire"

pipe_opacity = [1.0]
origin_pt    = np.array([0.0, 0.0, 0.0])

app = gui.Application.instance
app.initialize()

window = app.create_window(
    f"{_ply_path.stem}  |  Utilities within {BUFFER}m  |  press H for help",
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
# 9.  Right-side control panel
# ─────────────────────────────────────────────────────────────────────────────
PANEL_WIDTH = int(20 * em)
panel = gui.Vert(int(0.5 * em), gui.Margins(int(em), int(em), int(em), int(em)))

# Title
panel.add_child(gui.Label(f"--- {_ply_path.stem} ---"))
panel.add_fixed(int(0.4 * em))
panel.add_child(gui.Label(f"Points: {len(pts):,}"))
panel.add_child(gui.Label(f"Buffer: {BUFFER} m"))
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

panel.add_stretch()

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Camera helpers
# ─────────────────────────────────────────────────────────────────────────────
def _pivot_to(point: np.ndarray):
    d   = max(1.0, np.linalg.norm(pc_max - pc_min) * 0.6)
    eye = point + np.array([d, -d, d * 0.6])
    scene_widget.look_at(point.tolist(), eye.tolist(), [0.0, 0.0, 1.0])
    print(f"  Pivot -> [{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 11.  Key callbacks
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
# 12.  Layout + run
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
