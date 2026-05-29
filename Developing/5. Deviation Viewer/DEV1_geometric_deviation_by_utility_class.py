# -*- coding: utf-8 -*-
"""
Geometric Deviation Viewer -Instances vs LER Utility Registry
===============================================================
Compares segmented point-cloud instances (from the Label Viewer)
against the official LER pipe / cable geometry (GML).

For each instance point the script computes the minimum 3D distance
to the nearest LER line segment, producing a per-point deviation
heatmap.  Statistics are reported per utility class.

Workflow
--------
1.  Load the original site PLY (background context).
2.  Auto-discover labeled instance PLY files (utility_type property).
3.  Load LER utility lines from the consolidated GML.
4.  Translate GML coordinates UTM -> local using the area reference.
5.  For each instance, compute point-to-segment distances to all
    LER line segments that fall within the crop radius.
6.  Visualise as a deviation heatmap in an Open3D GUI.

Usage
-----
  Set PLY_FILE below, then run.  The script expects the same
  GML / GeoJSON paths used by the Base Viewer.

Keyboard shortcuts
------------------
  C   pivot to cloud centroid       H   help
"""

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import geopandas as gpd
import numpy as np
from pathlib import Path
import re
import time
import json

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PLY_FILE = (
    r"C:\Users\bosre\OneDrive - University of Twente\Documents\AAU UTwente thesis"
    r"\Python\Thesis\Data\OpenTrench3D\Water_Area_5\Area_5_Site_11.ply"
)

AREA_REF_GEOJSON = (
    r"C:\Users\bosre\OneDrive - University of Twente\Documents\AAU UTwente thesis"
    r"\Python\Thesis\Data\Translation_coordinates\area_points_utm32_etrs89.geojson"
)

GML_PATH = (
    r"C:\Users\bosre\OneDrive - University of Twente\Documents\AAU UTwente thesis"
    r"\Python\Thesis\Data\Ledningspakke_2803288_Area_4_and_5\consolidated.gml"
)

CROP_RADIUS = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# DEVIATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
# LER accuracy classes (DS/EN 17609:2022)
DEVIATION_THRESHOLDS = [0.00, 0.25, 0.50, 1.00, 2.00]
DEVIATION_COLORS = [
    [0.0, 0.7, 0.2],   # Class 1: <= 0.25 m  - green
    [0.6, 0.9, 0.0],   # Class 2: <= 0.50 m  - yellow-green
    [1.0, 0.8, 0.0],   # Class 3: <= 1.00 m  - yellow
    [1.0, 0.4, 0.0],   # Class 4: <= 2.00 m  - orange
    [0.8, 0.0, 0.0],   # Class 5: >  2.00 m  - red
]
DEVIATION_CLASS_LABELS = [
    "Class 1:  ≤ 250 mm",
    "Class 2:  ≤ 500 mm",
    "Class 3:  ≤ 1000 mm",
    "Class 4:  ≤ 2000 mm",
    "Class 5:  > 2000 mm",
]

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
# Instance utility_type ID -> label
UTILITY_TYPE_LABELS = {
    0: "Unlabeled", 1: "PowerLine", 2: "DrainageLine", 3: "OilPipeLine",
    4: "GasLine", 5: "ThermalLine", 6: "Conduit", 7: "WaterLine",
    8: "TelecomunicationLine", 9: "OtherLine", 10: "LineUnknownServiceType",
}

# DLF-recommended colours (RGB 0-1), using the primary sub-type per utility
UTILITY_TYPE_COLORS = {
    0: [0.50, 0.50, 0.50],   # Unlabeled          - grey
    1: [1.00, 0.00, 0.00],   # PowerLine (EL MV)   - DLF red 255:0:0
    2: [1.00, 0.00, 0.00],   # DrainageLine         - DLF spildevand red 255:0:0
    3: [0.46, 0.46, 0.46],   # OilPipeLine          - DLF grey 118:118:118
    4: [1.00, 0.60, 0.00],   # GasLine              - DLF orange 255:153:0
    5: [1.00, 0.00, 1.00],   # ThermalLine          - DLF violet 255:0:255
    6: [0.88, 0.88, 0.88],   # Conduit              - DLF beskyttelsesroer grey 225:225:225
    7: [0.00, 0.00, 1.00],   # WaterLine            - DLF blue 0:0:255
    8: [0.98, 0.59, 0.27],   # TelecomunicationLine - DLF orange 250:150:70
    9: [0.80, 0.80, 0.80],   # OtherLine            - grey
    10: [0.30, 0.80, 0.80],  # LineUnknownServiceType
}

# LER GML layer definitions - DLF-recommended colours (RGB 0-1)
LINE_LAYERS = {
    "Vandledning":               {"color": [0.000, 0.000, 1.000], "fallback_radius": 0.005},  # DLF blue 0:0:255
    "Afloebsledning":            {"color": [1.000, 0.000, 0.000], "fallback_radius": 0.005},  # DLF red 255:0:0
    "Gasledning":                {"color": [1.000, 0.600, 0.000], "fallback_radius": 0.005},  # DLF orange 255:153:0
    "Elledning":                 {"color": [1.000, 0.000, 0.000], "fallback_radius": 0.005},  # DLF red 255:0:0
    "Telekommunikationsledning": {"color": [0.980, 0.588, 0.275], "fallback_radius": 0.005},  # DLF orange 250:150:70
    "Foeringsroer":              {"color": [0.882, 0.882, 0.882], "fallback_radius": 0.005},  # DLF grey 225:225:225
    "LedningUkendtForsyningsart":{"color": [0.300, 0.800, 0.800], "fallback_radius": 0.005},  # not in DLF, keep cyan
    "Ledningstrace":             {"color": [0.980, 0.588, 0.275], "fallback_radius": 0.005},  # DLF trace orange 250:150:70
}

# forsyningsart keyword -> colour for Ledningstrace sub-groups.
# Matches are substring-based so e.g. "telekommunikation" matches "tele".
# Unknown values fall back to the default Ledningstrace colour.
# DLF-recommended colours (RGB 0-1)
FORSYNINGSART_COLOR_HINTS = [
    ("vand",   [0.000, 0.000, 1.000]),   # DLF vand blue 0:0:255
    ("afloeb", [1.000, 0.000, 0.000]),   # DLF spildevand red 255:0:0
    ("spilde", [1.000, 0.000, 0.000]),   # DLF spildevand red 255:0:0
    ("gas",    [1.000, 0.600, 0.000]),   # DLF gas orange 255:153:0
    ("el",     [1.000, 0.000, 0.000]),   # DLF el red 255:0:0
    ("tele",   [0.980, 0.588, 0.275]),   # DLF tele orange 250:150:70
    ("kommu",  [0.980, 0.588, 0.275]),   # DLF kommunikation orange 250:150:70
    ("varme",  [1.000, 0.000, 1.000]),   # DLF varme violet 255:0:255
    ("fjern",  [1.000, 0.000, 1.000]),   # DLF fjernvarme violet 255:0:255
    ("olie",   [0.463, 0.463, 0.463]),   # DLF olie grey 118:118:118
]


def _forsyningsart_color(fa_value, fallback):
    """Return a colour for a forsyningsart value by substring matching."""
    fa_lower = fa_value.lower()
    for keyword, color in FORSYNINGSART_COLOR_HINTS:
        if keyword in fa_lower:
            return color
    return fallback


# Mapping: instance utility_type ID -> matching LER layer names
# Also includes keywords to match Ledningstrace sub-groups (by forsyningsart)
UTILITY_TO_LER_MATCH = {
    1:  {"layers": ["Elledning"],                   "trace_kw": ["el"]},
    2:  {"layers": ["Afloebsledning"],              "trace_kw": ["afloeb", "spilde"]},
    3:  {"layers": [],                              "trace_kw": ["olie"]},
    4:  {"layers": ["Gasledning"],                  "trace_kw": ["gas"]},
    5:  {"layers": [],                              "trace_kw": ["varme", "fjern"]},
    6:  {"layers": ["Foeringsroer"],                "trace_kw": []},
    7:  {"layers": ["Vandledning"],                 "trace_kw": ["vand"]},
    8:  {"layers": ["Telekommunikationsledning"],   "trace_kw": ["tele", "kommu"]},
    9:  {"layers": [],                              "trace_kw": []},
    10: {"layers": ["LedningUkendtForsyningsart"],  "trace_kw": []},
}

# ─────────────────────────────────────────────────────────────────────────────
# PLY READER (ASCII + binary, with utility_type)
# ─────────────────────────────────────────────────────────────────────────────
_PLY_TYPE_MAP = {
    "float": np.float32, "float32": np.float32,
    "double": np.float64, "float64": np.float64,
    "int": np.int32, "int32": np.int32,
    "uint": np.uint32, "uint32": np.uint32,
    "short": np.int16, "int16": np.int16,
    "ushort": np.uint16, "uint16": np.uint16,
    "char": np.int8, "int8": np.int8,
    "uchar": np.uint8, "uint8": np.uint8,
}


def read_ply_with_utility_type(filepath):
    filepath = str(filepath)
    prop_defs = []  # (name, ply_type_str)
    n_verts = 0
    ply_format = "ascii"

    with open(filepath, "rb") as f:
        while True:
            line = f.readline().decode("utf-8", errors="replace").strip()
            if line.startswith("format "):
                ply_format = line.split()[1]
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            if line.startswith("property "):
                parts = line.split()
                prop_defs.append((parts[-1], parts[1]))
            if line == "end_header":
                header_end = f.tell()
                break

    names = [p[0] for p in prop_defs]
    x_col, y_col, z_col = names.index("x"), names.index("y"), names.index("z")
    has_rgb = all(c in names for c in ("red", "green", "blue"))
    has_ut = "utility_type" in names
    r_col = names.index("red") if has_rgb else None
    g_col = names.index("green") if has_rgb else None
    b_col = names.index("blue") if has_rgb else None
    ut_col = names.index("utility_type") if has_ut else None

    if ply_format == "ascii":
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            while f.readline().strip() != "end_header":
                pass
            data = np.loadtxt(f, max_rows=n_verts)
    else:
        bo = "<" if "little" in ply_format else ">"
        dt = np.dtype([(name, bo + np.dtype(_PLY_TYPE_MAP[ptype]).str[1:])
                        for name, ptype in prop_defs])
        with open(filepath, "rb") as f:
            f.seek(header_end)
            raw = np.frombuffer(f.read(n_verts * dt.itemsize), dtype=dt)
        data = np.column_stack([raw[name].astype(np.float64) for name, _ in prop_defs])

    points = data[:, [x_col, y_col, z_col]]
    colors = data[:, [r_col, g_col, b_col]].astype(np.uint8) if has_rgb else None
    ut = data[:, ut_col].astype(int) if has_ut else np.zeros(len(points), dtype=int)
    return points, colors, ut


def utility_type_from_filename(filename):
    """Fallback: parse utility label from filename like '..._type_WaterLine.ply'."""
    m = re.search(r"_type_(\w+)\.ply$", filename)
    if m:
        label = m.group(1)
        for uid, name in UTILITY_TYPE_LABELS.items():
            if name == label:
                return uid
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY: vectorised point-to-segment distance
# ─────────────────────────────────────────────────────────────────────────────
def batch_point_to_segments(pts, seg_p1, seg_p2):
    """
    For each point in pts (N, 3), find the minimum distance to any of the
    M segments defined by seg_p1 (M, 3) -> seg_p2 (M, 3).
    Returns dists (N,).  Processes in batches to limit memory.
    """
    N = len(pts)
    M = len(seg_p1)
    if M == 0:
        return np.full(N, np.inf)

    BATCH = 2000
    min_dists = np.full(N, np.inf)

    d = seg_p2 - seg_p1                          # (M, 3)
    seg_len2 = np.einsum('ij,ij->i', d, d)       # (M,)
    safe = seg_len2 > 1e-12

    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        p = pts[start:end]                        # (B, 3)
        B = len(p)

        v = p[:, None, :] - seg_p1[None, :, :]   # (B, M, 3)
        dot_vd = np.einsum('ijk,jk->ij', v, d)   # (B, M)

        t = np.zeros((B, M), dtype=float)
        t[:, safe] = np.clip(dot_vd[:, safe] / seg_len2[None, safe], 0.0, 1.0)

        closest = seg_p1[None, :, :] + t[:, :, None] * d[None, :, :]  # (B, M, 3)
        diff = p[:, None, :] - closest                                  # (B, M, 3)
        dists2 = np.einsum('ijk,ijk->ij', diff, diff)                  # (B, M)
        min_dists[start:end] = np.sqrt(dists2.min(axis=1))

    return min_dists


# ─────────────────────────────────────────────────────────────────────────────
# HEATMAP COLOUR
# ─────────────────────────────────────────────────────────────────────────────
def deviation_to_color(distances):
    colors = np.zeros((len(distances), 3), dtype=float)
    thresholds = np.array(DEVIATION_THRESHOLDS)
    palette = np.array(DEVIATION_COLORS)

    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        mask = (distances >= lo) & (distances < hi)
        if mask.any():
            t = (distances[mask] - lo) / (hi - lo)
            colors[mask] = palette[i] * (1.0 - t[:, None]) + palette[i + 1] * t[:, None]

    colors[distances >= thresholds[-1]] = palette[-1]
    return colors


def linear_to_srgb(c):
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
_required = {"PLY_FILE": PLY_FILE, "AREA_REF_GEOJSON": AREA_REF_GEOJSON, "GML_PATH": GML_PATH}
_missing = [(n, p) for n, p in _required.items() if not Path(p).exists()]
if _missing:
    for n, p in _missing:
        print(f"  [MISSING] {n} = {p}")
    raise SystemExit(1)

print("Config paths OK.\n")
_t0 = time.perf_counter()

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Auto-detect area + coordinate translation
# ─────────────────────────────────────────────────────────────────────────────
_ply_path = Path(PLY_FILE)
_area_match = re.search(r"Area[_\s]*(\d+)", _ply_path.parent.name, re.IGNORECASE)
if not _area_match:
    _area_match = re.search(r"Area[_\s]*(\d+)", _ply_path.name, re.IGNORECASE)
if not _area_match:
    raise SystemExit("[ERROR] Cannot determine area number from PLY path.")

AREA_NUMBER = int(_area_match.group(1))
AREA_NAME = f"Area{AREA_NUMBER}"

ref = gpd.read_file(AREA_REF_GEOJSON)
area = ref[ref["name"] == AREA_NAME]
if area.empty:
    raise SystemExit(f"[ERROR] No origin for '{AREA_NAME}' in GeoJSON")

area_row = area.iloc[0]
TX, TY, TZ = area_row.geometry.x, area_row.geometry.y, area_row.geometry.z
print(f"Area: {AREA_NAME}  |  Origin TX={TX:.3f} TY={TY:.3f} TZ={TZ:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load original point cloud (background context)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nLoading point cloud: {_ply_path.name} ...")
pcd_orig = o3d.io.read_point_cloud(str(PLY_FILE))
pts_orig = np.asarray(pcd_orig.points)
original_colors = np.asarray(pcd_orig.colors).copy()
cloud_centroid_full = pts_orig.mean(axis=0)

_cx, _cy = float(cloud_centroid_full[0]), float(cloud_centroid_full[1])
_dxy2 = (pts_orig[:, 0] - _cx) ** 2 + (pts_orig[:, 1] - _cy) ** 2
_crop_mask = _dxy2 <= (CROP_RADIUS ** 2)

pts_orig = pts_orig[_crop_mask]
original_colors = original_colors[_crop_mask]

pcd_orig = o3d.geometry.PointCloud()
pcd_orig.points = o3d.utility.Vector3dVector(pts_orig)
pcd_orig.colors = o3d.utility.Vector3dVector(original_colors)

print(f"  {len(pts_orig):,} points (after crop r={CROP_RADIUS} m)")
cloud_centroid = pts_orig.mean(axis=0) if len(pts_orig) > 0 else cloud_centroid_full
pc_min, pc_max = pts_orig.min(axis=0), pts_orig.max(axis=0)

# Ground level estimate (P95 of Z)
GROUND_Z = float(np.percentile(pts_orig[:, 2], 95))
print(f"  Ground Z estimate (P95): {GROUND_Z:.3f} m")

# Crop centre in UTM
_cx_utm, _cy_utm = _cx + TX, _cy + TY
_crop_r2 = CROP_RADIUS ** 2

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Load LER utility line segments from GML
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Loading LER utility segments ---")
_t_ler0 = time.perf_counter()

all_seg_p1 = []
all_seg_p2 = []
all_seg_layer = []
all_seg_active = []       # True = "i drift", False = "permanent ude af drift"
ler_meshes = {}           # layer -> merged TriangleMesh (for visualisation)
ler_stats = {}            # layer -> (n_feat_active, n_seg_active, n_feat_inactive, n_seg_inactive)


def _in_crop_utm(coords):
    """Conservative check: any part of the polyline within the circular crop (UTM).
    First checks whether any vertex is inside the circle.
    Falls back to an AABB overlap test to catch segments that cross the disc
    but have no vertex inside it — the segment clipper makes the final call.
    """
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
    Clip a 3D segment to the circular crop disc in XY.
    Circle: centre (_cx, _cy), radius CROP_RADIUS.
    Returns (clipped_p1, clipped_p2) or None if entirely outside.
    """
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


def segment_to_plane(p1, p2, width, color):
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
    offset = side * (width / 2.0)
    verts = np.array([p1 - offset, p1 + offset, p2 + offset, p2 - offset], dtype=float)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(tris),
    )
    mesh.paint_uniform_color(color)
    return mesh


def segment_to_cylinder(p1, p2, radius, color, resolution=12):
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return None
    cyl = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=length, resolution=resolution, split=1)
    z_axis = np.array([0.0, 0.0, 1.0])
    direction = vec / length
    cross = np.cross(z_axis, direction)
    cross_norm = np.linalg.norm(cross)
    dot = np.dot(z_axis, direction)
    if cross_norm > 1e-6:
        axis = cross / cross_norm
        angle = np.arctan2(cross_norm, dot)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cyl.rotate(R, center=[0, 0, 0])
    elif dot < 0:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * np.pi)
        cyl.rotate(R, center=[0, 0, 0])
    cyl.translate((p1 + p2) / 2.0)
    cyl.paint_uniform_color(color)
    return cyl


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

        bredde_m = None
        if is_trace:
            bredde_m = 0.25
            if "bredde" in row.index:
                try:
                    b = float(row["bredde"] or 0)
                    if b > 0:
                        bredde_m = b / 1000.0
                except (ValueError, TypeError):
                    pass

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

seg_p1 = np.array(all_seg_p1) if all_seg_p1 else np.empty((0, 3))
seg_p2 = np.array(all_seg_p2) if all_seg_p2 else np.empty((0, 3))
seg_active = np.array(all_seg_active, dtype=bool) if all_seg_active else np.empty(0, dtype=bool)
n_total_segs = len(seg_p1)
n_active_segs = int(seg_active.sum()) if len(seg_active) else 0
n_inactive_segs = n_total_segs - n_active_segs

_t_ler1 = time.perf_counter()
print(f"\n  Total: {n_total_segs:,} LER segments loaded in {_t_ler1 - _t_ler0:.1f}s"
      f"  ({n_active_segs} active, {n_inactive_segs} inactive)")

if n_total_segs == 0:
    print("[WARNING] No LER segments found -deviations will be infinite.")


def _get_matching_segment_mask(utility_type, active_only=None):
    """Return a boolean mask over seg_p1/seg_p2 for segments matching this utility type.

    active_only: None = both, True = only active, False = only inactive.
    """
    match = UTILITY_TO_LER_MATCH.get(utility_type)
    if match is None:
        mask = np.ones(len(seg_p1), dtype=bool)
    else:
        mask = np.zeros(len(seg_p1), dtype=bool)
        for i, layer_name in enumerate(all_seg_layer):
            if layer_name in match["layers"]:
                mask[i] = True
            elif layer_name.startswith("Ledningstrace") and match["trace_kw"]:
                layer_lower = layer_name.lower()
                if any(kw in layer_lower for kw in match["trace_kw"]):
                    mask[i] = True

    # Filter by driftsstatus
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
    names = set()
    for layer_name in ler_meshes:
        if layer_name in match["layers"]:
            names.add(layer_name)
        elif layer_name.startswith("Ledningstrace") and match["trace_kw"]:
            layer_lower = layer_name.lower()
            if any(kw in layer_lower for kw in match["trace_kw"]):
                names.add(layer_name)
    return names


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Load instances + compute deviations against LER segments
# ─────────────────────────────────────────────────────────────────────────────
_ply_stem = _ply_path.stem
_inst_candidates = sorted(
    _ply_path.parent.glob(f"{_ply_stem}_instances_*"),
    key=lambda p: p.name, reverse=True,
)
if not _inst_candidates:
    raise SystemExit(f"[ERROR] No instance directories for {_ply_stem}")

_inst_dir = _inst_candidates[0]
_labeled_dir = _inst_dir / "labeled"
_inst_files = sorted(_labeled_dir.glob("*.ply")) if _labeled_dir.is_dir() else sorted(_inst_dir.glob("*.ply"))
print(f"\nInstance directory: {_inst_dir.name}/")
print(f"  {len(_inst_files)} PLY files ({'labeled/' if _labeled_dir.is_dir() else 'root'})")

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

    # Combined (active + inactive) for heatmap colouring
    if has_ler:
        dists = batch_point_to_segments(pts_inst, seg_p1[seg_mask_all], seg_p2[seg_mask_all])
        stats = _make_stats(dists)
    else:
        dists = np.full(len(pts_inst), np.nan)
        stats = dict(_nan_stats)

    # Separate stats for active / inactive
    if n_act > 0:
        dists_act = batch_point_to_segments(pts_inst, seg_p1[seg_mask_act], seg_p2[seg_mask_act])
        stats_act = _make_stats(dists_act)
    else:
        dists_act = None
        stats_act = dict(_nan_stats)

    if n_inact > 0:
        dists_inact = batch_point_to_segments(pts_inst, seg_p1[seg_mask_inact], seg_p2[seg_mask_inact])
        stats_inact = _make_stats(dists_inact)
    else:
        dists_inact = None
        stats_inact = dict(_nan_stats)

    # Deviation heatmap point cloud (grey if no matching LER)
    pcd_dev = o3d.geometry.PointCloud()
    pcd_dev.points = o3d.utility.Vector3dVector(pts_inst)
    if has_ler:
        pcd_dev.colors = o3d.utility.Vector3dVector(deviation_to_color(dists))
    else:
        pcd_dev.colors = o3d.utility.Vector3dVector(
            np.tile([0.5, 0.5, 0.5], (len(pts_inst), 1)))

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
        "pcd_rgb": pcd_rgb,
        "pcd_class": pcd_class,
        "distances": dists,
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

class_summaries = {}
for ut, instances in sorted(class_instances.items()):
    label = UTILITY_TYPE_LABELS.get(ut, f"Unknown({ut})")
    has_ler = any(inst["has_ler"] for inst in instances)
    total_pts = sum(inst["stats"]["n_pts"] for inst in instances)
    total_act = sum(inst["n_active_segs"] for inst in instances)
    total_inact = sum(inst["n_inactive_segs"] for inst in instances)

    def _agg_stats(key):
        """Aggregate per-instance stats arrays for a given stats key ('stats_active' or 'stats_inactive')."""
        # We need the raw distances, but we only stored summary stats per split.
        # Use the combined distances and filter isn't possible here,
        # so we report from the per-instance stats (weighted is close enough for display).
        vals = [inst[key] for inst in instances
                if not np.isnan(inst[key].get("mean", np.nan))]
        if not vals:
            return None
        # Re-aggregate from per-instance stats (approximate but representative)
        all_n = sum(v["n_pts"] for v in vals)
        w_mean = sum(v["mean"] * v["n_pts"] for v in vals) / all_n if all_n else np.nan
        return {
            "mean": w_mean,
            "p95": max(v["p95"] for v in vals),
            "max": max(v["max"] for v in vals),
        }

    if has_ler:
        matched_dists = np.concatenate([
            inst["distances"] for inst in instances if inst["has_ler"]])
        summary = {
            "label": label, "n_instances": len(instances), "n_points": total_pts,
            "has_ler": True,
            "n_active_segs": total_act, "n_inactive_segs": total_inact,
            "mean": float(np.mean(matched_dists)),
            "median": float(np.median(matched_dists)),
            "std": float(np.std(matched_dists)),
            "p95": float(np.percentile(matched_dists, 95)),
            "max": float(np.max(matched_dists)),
            "active_agg": _agg_stats("stats_active"),
            "inactive_agg": _agg_stats("stats_inactive"),
        }
    else:
        summary = {
            "label": label, "n_instances": len(instances), "n_points": total_pts,
            "has_ler": False,
            "n_active_segs": 0, "n_inactive_segs": 0,
            "mean": np.nan, "median": np.nan, "std": np.nan,
            "p95": np.nan, "max": np.nan,
            "active_agg": None, "inactive_agg": None,
        }
    class_summaries[ut] = summary

    print(f"\n  {label} (type {ut})")
    print(f"    Instances:  {len(instances)}")
    print(f"    Points:     {total_pts:,}")
    print(f"    LER segs:   {total_act} active, {total_inact} inactive")
    if has_ler:
        print(f"    ── Combined (all matching LER) ──")
        print(f"    Mean:       {summary['mean']*1000:>8.2f} mm")
        print(f"    Median:     {summary['median']*1000:>8.2f} mm")
        print(f"    Std dev:    {summary['std']*1000:>8.2f} mm")
        print(f"    P95:        {summary['p95']*1000:>8.2f} mm")
        print(f"    Max:        {summary['max']*1000:>8.2f} mm")
        if summary["active_agg"]:
            a = summary["active_agg"]
            print(f"    ── Active LER only ──")
            print(f"    Mean:       {a['mean']*1000:>8.2f} mm   "
                  f"P95: {a['p95']*1000:.2f} mm   Max: {a['max']*1000:.2f} mm")
        if summary["inactive_agg"]:
            ia = summary["inactive_agg"]
            print(f"    ── Inactive LER only ──")
            print(f"    Mean:       {ia['mean']*1000:>8.2f} mm   "
                  f"P95: {ia['p95']*1000:.2f} mm   Max: {ia['max']*1000:.2f} mm")
    else:
        print(f"    ** No matching LER utility — deviation not computed **")

print("\n" + "=" * 72)

# Normals for original cloud
try:
    pcd_orig.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=12))
    pcd_orig.orient_normals_towards_camera_location(cloud_centroid + np.array([0, 0, 5]))
except Exception:
    pass

_t_load = time.perf_counter()
print(f"\nTotal load + compute: {_t_load - _t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  GUI
# ─────────────────────────────────────────────────────────────────────────────
ORIG_GEOM = "original_cloud"
_color_mode = [0]
_MODE_NAMES = ["Deviation heatmap", "Original RGB", "Utility class"]

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
try:
    scene_widget.scene.view.set_post_processing(True)
except Exception:
    pass
scene_widget.scene.scene.set_sun_light([0, 0, -1], [1, 1, 1], 75000)
scene_widget.scene.scene.enable_sun_light(True)


def make_pt_mat(size=3.0):
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.point_size = size
    return mat


def make_mesh_mat(alpha=1.0):
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency"
    mat.base_color = [1, 1, 1, float(alpha)]
    return mat


# Add dimmed original cloud
_dim = original_colors * 0.35
pcd_dim = o3d.geometry.PointCloud()
pcd_dim.points = o3d.utility.Vector3dVector(pts_orig)
pcd_dim.colors = o3d.utility.Vector3dVector(_dim)
try:
    pcd_dim.normals = pcd_orig.normals
except Exception:
    pass
scene_widget.scene.add_geometry(ORIG_GEOM, pcd_dim, make_pt_mat(2.0))

# Add LER pipe meshes
_ler_visible = {}
for ln, mesh in ler_meshes.items():
    gn = f"ler_{ln}"
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    scene_widget.scene.add_geometry(gn, mesh, make_mesh_mat(0.6))
    _ler_visible[ln] = True

# Add instance geometries
_inst_gnames = []
_class_visible = {}
for ut, instances in class_instances.items():
    _class_visible[ut] = True
    for i, inst in enumerate(instances):
        gn = f"inst_{ut}_{i}"
        _inst_gnames.append((ut, i, gn))
        scene_widget.scene.add_geometry(gn, inst["pcd_dev"], make_pt_mat(4.0))

# Camera
bounds = scene_widget.scene.bounding_box
scene_widget.setup_camera(60, bounds, cloud_centroid.tolist())

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Colour-mode switch
# ─────────────────────────────────────────────────────────────────────────────
def _apply_color_mode(mode):
    _color_mode[0] = mode
    for ut, instances in class_instances.items():
        for i, inst in enumerate(instances):
            gn = f"inst_{ut}_{i}"
            pcd = [inst["pcd_dev"], inst["pcd_rgb"], inst["pcd_class"]][mode]
            scene_widget.scene.remove_geometry(gn)
            scene_widget.scene.add_geometry(gn, pcd, make_pt_mat(4.0))
            scene_widget.scene.show_geometry(gn, _class_visible.get(ut, True))
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
_heatmap_legend.add_child(gui.Label("LER Accuracy Class:"))
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


def _on_mode(val, idx):
    _apply_color_mode(idx)
    _heatmap_legend.visible = (idx == 0)
    window.set_needs_layout()


combo.set_on_selection_changed(_on_mode)
panel.add_child(combo)
panel.add_child(_heatmap_legend)
panel.add_fixed(int(0.5 * em))

# Original cloud toggle
orig_cb = gui.Checkbox("Original cloud")
orig_cb.checked = True
orig_cb.set_on_checked(lambda c: (scene_widget.scene.show_geometry(ORIG_GEOM, c), window.post_redraw()))
panel.add_child(orig_cb)

# ── Utility filter (per-class view) ──
panel.add_fixed(int(0.5 * em))
panel.add_child(gui.Label("Utility filter:"))

# Build filter entries: (label, utility_type or None for "all")
_filter_entries = [("All utilities", None)]
for _fut in sorted(class_summaries.keys()):
    _fs = class_summaries[_fut]
    _ler_names = _get_matching_ler_names(_fut)
    if _ler_names:
        _ler_short = ", ".join(sorted(_ler_names))
        _filter_entries.append((f"{_fs['label']}  ↔  {_ler_short}", _fut))
    else:
        _filter_entries.append((f"{_fs['label']}  (no LER)", _fut))

_active_filter = [None]   # None = show all

filter_combo = gui.Combobox()
for _flbl, _ in _filter_entries:
    filter_combo.add_item(_flbl)
filter_combo.selected_index = 0


def _apply_utility_filter(filter_ut):
    """Show/hide instances and LER layers to isolate one utility pair."""
    _active_filter[0] = filter_ut
    matching_ler = _get_matching_ler_names(filter_ut) if filter_ut is not None else None

    # Instances: show only the selected utility type (or all)
    for ut, instances in class_instances.items():
        vis = (filter_ut is None or ut == filter_ut)
        _class_visible[ut] = vis
        for i, inst in enumerate(instances):
            gn = f"inst_{ut}_{i}"
            scene_widget.scene.show_geometry(gn, vis)

    # LER layers: show only those matching the selected utility (or all)
    for ln in ler_meshes:
        if filter_ut is None:
            vis = True
        else:
            vis = ln in matching_ler if matching_ler else False
        _ler_visible[ln] = vis
        scene_widget.scene.show_geometry(f"ler_{ln}", vis)

    window.post_redraw()


def _on_filter(val, idx):
    _, filter_ut = _filter_entries[idx]
    _apply_utility_filter(filter_ut)


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
    for ln in ler_meshes:
        if _ler_visible.get(ln, True):
            scene_widget.scene.modify_geometry_material(f"ler_{ln}", make_mesh_mat(val))
    window.post_redraw()


ler_slider.set_on_value_changed(_on_ler_opacity)
ler_row.add_child(ler_slider)
panel.add_child(ler_row)

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
            _class_visible[u] = checked
            for _u, _i, gn in _inst_gnames:
                if _u == u:
                    scene_widget.scene.show_geometry(gn, checked)
            window.post_redraw()
        return _cb

    cb = gui.Checkbox(f"{s['label']} ({s['n_instances']})")
    cb.checked = True
    cb.set_on_checked(_make_cls_cb(ut))
    row.add_child(sw)
    row.add_fixed(int(0.4 * em))
    row.add_child(cb)
    panel.add_child(row)

    stat_box = gui.Vert(0, gui.Margins(int(2.5 * em), 0, 0, 0))
    n_a = s.get("n_active_segs", 0)
    n_i = s.get("n_inactive_segs", 0)
    if s.get("has_ler", True):
        stat_box.add_child(gui.Label(
            f"{s['n_points']:,} pts  |  LER: {n_a}a {n_i}i"))
        stat_box.add_child(gui.Label(
            f"mean {s['mean']*1000:.1f}  |  P95 {s['p95']*1000:.1f}  |  max {s['max']*1000:.1f} mm"))
        if s.get("active_agg") and s.get("inactive_agg"):
            a = s["active_agg"]
            ia = s["inactive_agg"]
            stat_box.add_child(gui.Label(
                f"active mean {a['mean']*1000:.0f}  |  inactive mean {ia['mean']*1000:.0f} mm"))
    else:
        stat_box.add_child(gui.Label(f"{s['n_points']:,} pts"))
        stat_box.add_child(gui.Label("No matching LER utility"))
    panel.add_child(stat_box)
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


def on_key(event):
    if event.type != gui.KeyEvent.DOWN:
        return IGNORED
    k = event.key
    if k in (ord('C'), ord('c')):
        _pivot_to(cloud_centroid)
        return HANDLED
    if k in (ord('H'), ord('h')):
        print("\n  C   pivot to centroid    H   help\n")
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

print(f"\nStartup: {time.perf_counter() - _t0:.1f}s  - Launching viewer ...\n")
app.run()
print("Viewer closed.")
