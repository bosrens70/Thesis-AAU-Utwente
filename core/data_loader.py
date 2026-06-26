# -*- coding: utf-8 -*-
"""
Centralised data loading — the init script.
=============================================
Every viewer calls functions from here instead of duplicating
data-loading boilerplate.  The master ``init_site()`` function
loads everything at once; individual functions can also be called
independently for viewers that only need part of the data.
"""

import open3d as o3d
import geopandas as gpd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
import re
import time
import json

from core.config import (
    PLY_FILE, GML_PATH, AREA_REF_GEOJSON, CROP_RADIUS, CROP_MODE,
    CLASS_LABELS, DEFAULT_CLASS_COLOR,
    LINE_LAYERS, COMPONENT_LAYERS, COMP_TO_LINE,
    COMPONENT_SPHERE_RADIUS,
)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT CONTAINERS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AreaInfo:
    """Translation offset and area identifier."""
    area_number: int
    area_name: str
    TX: float
    TY: float
    TZ: float


@dataclass
class PointCloudData:
    """Loaded and (optionally) cropped point cloud."""
    pcd: object                  # o3d.geometry.PointCloud
    pts: np.ndarray              # (N, 3)
    original_colors: np.ndarray  # (N, 3) float 0-1
    class_labels: np.ndarray     # (N,) int or None
    class_colors: np.ndarray     # (N, 3) or None
    cloud_centroid: np.ndarray   # (3,)
    cloud_centroid_full: np.ndarray  # (3,) before crop
    pc_min: np.ndarray           # (3,)
    pc_max: np.ndarray           # (3,)
    crop_center_local: tuple     # (cx, cy)
    crop_center_utm: tuple       # (cx, cy) — in UTM
    crop_radius: float
    ground_z: float = None       # ground level (local), set by pick_ground_level()
    ground_z_method: str = ""    # how ground_z was determined


@dataclass
class GMLData:
    """Raw GeoDataFrames loaded from GML, filtered to crop area."""
    line_gdfs: dict      # layer_name -> GeoDataFrame
    component_gdfs: dict # layer_name -> GeoDataFrame


@dataclass
class SiteData:
    """Master container: everything a viewer needs."""
    area: AreaInfo
    pc: PointCloudData
    gml: GMLData = None
    instance_dir: Path = None
    instance_files: list = field(default_factory=list)
    load_time: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. AREA DETECTION + OFFSET
# ─────────────────────────────────────────────────────────────────────────────

def detect_area(ply_path):
    """
    Parse area number from PLY path.
    Returns (area_number: int, area_name: str)  e.g.  (5, "Area5").
    """
    ply_path = Path(ply_path)
    match = re.search(r"Area[_\s]*(\d+)", ply_path.parent.name, re.IGNORECASE)
    if not match:
        match = re.search(r"Area[_\s]*(\d+)", ply_path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot determine area number from PLY path: {ply_path}")
    area_number = int(match.group(1))
    area_name = f"Area{area_number}"
    return area_number, area_name


def load_area_offset(geojson_path, area_name):
    """
    Read the area reference GeoJSON and return an AreaInfo with TX, TY, TZ.
    """
    geojson_path = str(geojson_path)
    t0 = time.perf_counter()
    ref = gpd.read_file(geojson_path)
    t1 = time.perf_counter()
    print(f"  [timer] Read area GeoJSON: {t1 - t0:.3f}s")

    area = ref[ref["name"] == area_name]
    if area.empty:
        raise ValueError(f"No origin for '{area_name}' in {geojson_path}")

    area_row = area.iloc[0]
    TX = area_row.geometry.x
    TY = area_row.geometry.y
    TZ = area_row.geometry.z

    area_number = int(area_name.replace("Area", ""))
    print(f"Detected area: {area_name}  |  Origin TX={TX:.3f} TY={TY:.3f} TZ={TZ:.3f}")
    return AreaInfo(area_number=area_number, area_name=area_name, TX=TX, TY=TY, TZ=TZ)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PLY LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_ply(ply_path, crop_radius=None, area_info=None, crop_mode="circle"):
    """
    Load a PLY point cloud.  Reads the Open3D PointCloud, parses class
    labels from the ASCII header, and optionally applies a crop.

    crop_mode : "circle" crops the cloud to a disc of radius crop_radius around
    the XY centroid; "rect" keeps the full cloud uncropped (its AABB drives
    utility selection in the viewers).

    Returns a PointCloudData.
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    print(f"\nLoading point cloud: {ply_path.name} ...")
    t0 = time.perf_counter()
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    t1 = time.perf_counter()
    print(f"  {len(pts):,} points loaded  [{t1 - t0:.3f}s]")

    # ── Parse class labels from PLY header ───────────────────────────────
    print("  Reading class labels from PLY ...")
    class_labels = None
    t_cls0 = time.perf_counter()

    header_lines = 0
    property_names = []
    with open(str(ply_path), 'r', errors='replace') as f:
        for line in f:
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("property "):
                property_names.append(stripped.split()[-1])
            if stripped == "end_header":
                break

    if "class" in property_names:
        class_col = property_names.index("class")
        class_labels = np.loadtxt(
            str(ply_path), dtype=int,
            skiprows=header_lines, usecols=class_col,
        )
        unique_classes = np.unique(class_labels)
        print(f"  Class labels found: {len(class_labels):,} values, "
              f"unique classes: {sorted(unique_classes.tolist())}")
    else:
        print(f"  [WARNING] No 'class' property in PLY (found: {property_names})"
              " — class colouring disabled")

    t_cls1 = time.perf_counter()
    print(f"  [timer] Class label parsing: {t_cls1 - t_cls0:.3f}s")

    # Store original RGB colours
    original_colors = np.asarray(pcd.colors).copy()

    # Build class-label colour array
    class_colors = None
    if class_labels is not None:
        class_colors = np.zeros_like(original_colors)
        for cls_id, cfg in CLASS_LABELS.items():
            mask = class_labels == cls_id
            class_colors[mask] = cfg["color"]
        known_mask = np.isin(class_labels, list(CLASS_LABELS.keys()))
        if not known_mask.all():
            class_colors[~known_mask] = DEFAULT_CLASS_COLOR
            n_unknown = (~known_mask).sum()
            print(f"  [WARNING] {n_unknown} points with unknown class IDs")

    cloud_centroid_full = pts.mean(axis=0)

    # ── Crop (circular disc, or none in rect mode) ───────────────────────
    crop_cx = float(cloud_centroid_full[0])
    crop_cy = float(cloud_centroid_full[1])

    if crop_mode == "rect":
        # Rect mode: keep the full point cloud.  Its AABB (pc_min/pc_max), grown
        # by the utility buffer, is what the viewers use to select utilities.
        crop_radius = 0.0
        print(f"  Rect mode: full point-cloud AABB, no crop ({len(pts):,} points)")
    elif crop_radius is not None and crop_radius > 0:
        dxy2 = (pts[:, 0] - crop_cx) ** 2 + (pts[:, 1] - crop_cy) ** 2
        crop_mask = dxy2 <= (crop_radius ** 2)

        n_full = len(pts)
        pts = pts[crop_mask]
        original_colors = original_colors[crop_mask]
        if class_labels is not None:
            class_labels = class_labels[crop_mask]
        if class_colors is not None:
            class_colors = class_colors[crop_mask]

        # Rebuild PointCloud with cropped data
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(original_colors)

        print(f"  Circular crop (r={crop_radius} m): {n_full:,} -> {len(pts):,} points")
    else:
        crop_radius = 0.0

    cloud_centroid = pts.mean(axis=0) if len(pts) > 0 else cloud_centroid_full
    pc_min = pts.min(axis=0) if len(pts) > 0 else cloud_centroid_full
    pc_max = pts.max(axis=0) if len(pts) > 0 else cloud_centroid_full

    # Crop center in UTM
    if area_info is not None:
        crop_cx_utm = crop_cx + area_info.TX
        crop_cy_utm = crop_cy + area_info.TY
    else:
        crop_cx_utm = crop_cx
        crop_cy_utm = crop_cy

    print(f"  Local bbox:  X[{pc_min[0]:.1f}, {pc_max[0]:.1f}]  "
          f"Y[{pc_min[1]:.1f}, {pc_max[1]:.1f}]  "
          f"Z[{pc_min[2]:.1f}, {pc_max[2]:.1f}]")
    if crop_radius > 0:
        print(f"  Crop center (UTM): ({crop_cx_utm:.1f}, {crop_cy_utm:.1f})  "
              f"r = {crop_radius} m")

    return PointCloudData(
        pcd=pcd, pts=pts,
        original_colors=original_colors,
        class_labels=class_labels,
        class_colors=class_colors,
        cloud_centroid=cloud_centroid,
        cloud_centroid_full=cloud_centroid_full,
        pc_min=pc_min, pc_max=pc_max,
        crop_center_local=(crop_cx, crop_cy),
        crop_center_utm=(crop_cx_utm, crop_cy_utm),
        crop_radius=crop_radius,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. GML LOADING (raw GeoDataFrames)
# ─────────────────────────────────────────────────────────────────────────────

def load_gml_layers(gml_path, line_layers=None, component_layers=None):
    """
    Load GML layers into GeoDataFrames.  Does NOT translate coordinates or
    create meshes — that is viewer-specific.

    Returns a GMLData with dicts of {layer_name: GeoDataFrame}.
    """
    gml_path = str(gml_path)
    if line_layers is None:
        line_layers = LINE_LAYERS
    if component_layers is None:
        component_layers = COMPONENT_LAYERS

    line_gdfs = {}
    component_gdfs = {}

    print("\n--- Loading GML line layers ---")
    t0 = time.perf_counter()
    for layer_name in line_layers:
        tl = time.perf_counter()
        try:
            gdf = gpd.read_file(gml_path, layer=layer_name)
            line_gdfs[layer_name] = gdf
            print(f"  {layer_name:<35} {len(gdf):>5} features  [{time.perf_counter() - tl:.2f}s]")
        except Exception as e:
            print(f"  {layer_name:<35} skip ({e})")
    t1 = time.perf_counter()
    print(f"  Line layers loaded in {t1 - t0:.2f}s")

    print("\n--- Loading GML component layers ---")
    t2 = time.perf_counter()
    for layer_name in component_layers:
        tc = time.perf_counter()
        try:
            gdf = gpd.read_file(gml_path, layer=layer_name)
            component_gdfs[layer_name] = gdf
            print(f"  {layer_name:<35} {len(gdf):>5} features  [{time.perf_counter() - tc:.2f}s]")
        except Exception:
            pass
    t3 = time.perf_counter()
    print(f"  Component layers loaded in {t3 - t2:.2f}s")

    return GMLData(line_gdfs=line_gdfs, component_gdfs=component_gdfs)


# ─────────────────────────────────────────────────────────────────────────────
# 4. INSTANCE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def instance_base_name(ply_path):
    """
    Derive the base name used for a PLY's instance directory.

    Strips a redundant leading ``Area_N_`` prefix from the PLY stem, since the
    instance directory already lives inside the ``Water_Area_N`` folder (e.g.
    ``Area_5_Site_11`` -> ``Site_11``). PLY stems without that prefix are
    returned unchanged.
    """
    return re.sub(r"^Area_\d+_", "", Path(ply_path).stem)


def discover_instances(ply_path):
    """
    Auto-discover the permanent instance directory for a PLY file.

    Directory layout::

        Water_Area_N/
          Site_XX_Instances/              <- permanent dir (returned as instance_dir)
            0_instance_0_type_7.ply       <- water instance from conversion script
            20260526_144758/              <- segment-viewer run (timestamp subfolder)
              1_instance_0.ply
              ...
            labeled_20260526_150000/      <- label-viewer session (timestamped)
              1_instance_0_type_4.ply

    Returns (instance_dir, instance_files) where instance_dir is the
    permanent ``Site_XX_Instances/`` folder and instance_files are the
    raw PLY files from the most recent timestamp subfolder.
    """
    ply_path = Path(ply_path)
    base = instance_base_name(ply_path)
    parent = ply_path.parent

    # New convention: permanent <base>_Instances/ directory
    perm_dir = parent / f"{base}_Instances"
    if perm_dir.is_dir():
        # Find the most recent timestamp subfolder with raw instance PLYs
        ts_dirs = sorted(
            [d for d in perm_dir.iterdir()
             if d.is_dir() and d.name[0].isdigit()],
            key=lambda p: p.name,
            reverse=True,
        )
        inst_files = []
        src_label = "none"
        for ts_dir in ts_dirs:
            files = sorted(ts_dir.glob("*.ply"))
            if files:
                inst_files = files
                src_label = ts_dir.name
                break

        # Fall back to most recent labeled_* subfolder
        if not inst_files:
            labeled_dirs = sorted(
                [d for d in perm_dir.iterdir()
                 if d.is_dir() and d.name.startswith("labeled_")],
                key=lambda p: p.name,
                reverse=True,
            )
            for ld in labeled_dirs:
                files = sorted(ld.glob("*.ply"))
                if files:
                    inst_files = files
                    src_label = ld.name
                    break

        # Legacy fall back to labeled/ (no timestamp)
        if not inst_files:
            labeled_dir = perm_dir / "labeled"
            if labeled_dir.is_dir():
                inst_files = sorted(labeled_dir.glob("*.ply"))
                src_label = "labeled/"

        print(f"\n  Instance directory: {perm_dir.name}/")
        print(f"  {len(inst_files)} PLY files ({src_label})")
        return perm_dir, inst_files

    # Legacy fallback: timestamped <base>_instances_<stamp>/ directories
    candidates = sorted(
        set(parent.glob(f"{base}_instances_*"))
        | set(parent.glob(f"{ply_path.stem}_instances_*")),
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        print(f"  [warn] No instance directories found for {base}")
        return None, []

    inst_dir = candidates[0]

    labeled_dir = inst_dir / "labeled"
    if labeled_dir.is_dir():
        inst_files = sorted(labeled_dir.glob("*.ply"))
        subdir_label = "labeled/"
    else:
        inst_files = sorted(inst_dir.glob("*.ply"))
        subdir_label = "root"

    print(f"\n  Instance directory: {inst_dir.name}/")
    print(f"  {len(inst_files)} PLY files ({subdir_label})")
    return inst_dir, inst_files


# ─────────────────────────────────────────────────────────────────────────────
# 5. ROBUST PLY READER (ASCII + binary, with utility_type)
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
    """
    Read a PLY file (ASCII or binary) and extract xyz, rgb, and
    utility_type columns.  Returns (points, colors, utility_types).
    """
    filepath = str(filepath)
    prop_defs = []   # (name, ply_type_str)
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
    """
    Fallback: parse the utility type from a labelled instance filename.

    Handles both the current numeric convention ('..._type_3.ply') and the
    legacy name convention ('..._type_WaterLine.ply').
    """
    from core.config import UTILITY_TYPE_LABELS
    m = re.search(r"_type_(\w+)\.ply$", filename)
    if not m:
        return 0
    token = m.group(1)
    # Numeric type id (current convention)
    if token.isdigit():
        return int(token)
    # Legacy name token
    for uid, name in UTILITY_TYPE_LABELS.items():
        if name == token:
            return uid
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. COORDINATE TRANSLATION
# ─────────────────────────────────────────────────────────────────────────────

def translate_coords_to_local(coords_raw, TX, TY, TZ):
    """
    Translate UTM coordinates to local coordinates.
    Returns coords (N, 3) with X -= TX, Y -= TY, Z -= TZ.
    """
    coords = coords_raw.copy().astype(float)
    if coords.shape[1] == 2:
        coords = np.hstack([coords, np.zeros((len(coords), 1))])
    coords[:, 0] -= TX
    coords[:, 1] -= TY
    coords[:, 2] -= TZ
    return coords


# ─────────────────────────────────────────────────────────────────────────────
# 7. EXTRACT GML ROW ATTRIBUTES (for picking display)
# ─────────────────────────────────────────────────────────────────────────────

def extract_row_attrs(row):
    """Extract all non-geometry attributes from a GeoDataFrame row as (label, value) pairs."""
    attrs = []
    for col in row.index:
        if col == "geometry":
            continue
        val = row[col]
        val_str = str(val) if (val is not None and str(val) != "nan") else "—"
        attrs.append((col, val_str))
    return attrs


# ─────────────────────────────────────────────────────────────────────────────
# 8. MASTER INIT
# ─────────────────────────────────────────────────────────────────────────────

def init_site(ply_file=None, geojson_path=None, gml_path=None,
              crop_radius=None, crop_mode=None, load_gml=True, load_instances=True):
    """
    Master initialisation: load area offset, point cloud, GML layers,
    and discover instances — everything a viewer needs to start.

    Parameters
    ----------
    ply_file      : str or Path, default config.PLY_FILE
    geojson_path  : str or Path, default config.AREA_REF_GEOJSON
    gml_path      : str or Path, default config.GML_PATH
    crop_radius   : float, default config.CROP_RADIUS
    crop_mode     : "circle" | "rect", default config.CROP_MODE
    load_gml      : bool — set False to skip GML loading (e.g. segment viewer)
    load_instances: bool — set False to skip instance discovery

    Returns
    -------
    SiteData with area, pc, gml, instance_dir, instance_files.
    """
    t_start = time.perf_counter()

    if ply_file is None:
        ply_file = PLY_FILE
    if geojson_path is None:
        geojson_path = AREA_REF_GEOJSON
    if gml_path is None:
        gml_path = GML_PATH
    if crop_radius is None:
        crop_radius = CROP_RADIUS
    if crop_mode is None:
        crop_mode = CROP_MODE

    # ── Validate paths ───────────────────────────────────────────────────
    required = {"PLY_FILE": ply_file, "AREA_REF_GEOJSON": geojson_path}
    if load_gml:
        required["GML_PATH"] = gml_path
    missing = [(n, p) for n, p in required.items() if not Path(p).exists()]
    if missing:
        print("\n[CONFIG ERROR] Missing paths:")
        for n, p in missing:
            print(f"  {n:<20} = {p}")
        raise SystemExit(1)
    print("Config paths OK.\n")

    # ── 1. Area detection + offset ───────────────────────────────────────
    area_num, area_name = detect_area(ply_file)
    area = load_area_offset(geojson_path, area_name)

    # ── 2. Point cloud ───────────────────────────────────────────────────
    pc = load_ply(ply_file, crop_radius=crop_radius, area_info=area,
                  crop_mode=crop_mode)

    # ── 3. GML layers ────────────────────────────────────────────────────
    gml = None
    if load_gml:
        gml = load_gml_layers(gml_path)

    # ── 4. Instance discovery ────────────────────────────────────────────
    inst_dir = None
    inst_files = []
    if load_instances:
        inst_dir, inst_files = discover_instances(ply_file)

    t_end = time.perf_counter()
    print(f"\nAll data loaded in {t_end - t_start:.2f}s")

    return SiteData(
        area=area, pc=pc, gml=gml,
        instance_dir=inst_dir, instance_files=inst_files,
        load_time=t_end - t_start,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GROUND-LEVEL ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────
def pick_ground_level(pc: PointCloudData) -> float:
    """
    Determine ground level for a loaded point cloud.

    Opens a VisualizerWithEditing window so the user can pick ground-level
    points with  Shift + Left-Click.  Close the window (Q or X) when done.

    If no points are picked, falls back to P95 of point cloud Z.

    Sets ``pc.ground_z`` and ``pc.ground_z_method`` in place and returns
    the ground Z value.
    """
    print("\n" + "=" * 62)
    print("  GROUND-LEVEL POINT PICKING")
    print("=" * 62)
    print("  Shift + Left-Click  to select points on the ground surface.")
    print("  Pick one or more points that represent the ground level.")
    print("  Press Q or close the window when finished.")
    print("=" * 62 + "\n")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name="Pick ground-level points  (Shift+Click, then Q to finish)",
        width=1280, height=720,
    )
    vis.add_geometry(pc.pcd)
    vis.run()
    picked_indices = vis.get_picked_points()
    vis.destroy_window()

    if len(picked_indices) == 0:
        ground_z = float(np.percentile(pc.pts[:, 2], 95))
        method = "fallback P95"
        print(f"[WARNING] No points picked!  Falling back to P95 = {ground_z:.3f} m")
    else:
        picked_pts = pc.pts[picked_indices]
        ground_z = float(np.mean(picked_pts[:, 2]))
        method = f"picked from {len(picked_indices)} point(s)"
        print(f"\n  Picked {len(picked_indices)} ground-level point(s):")
        for i, idx in enumerate(picked_indices):
            p = pc.pts[idx]
            print(f"    [{i+1}]  index {idx:>8,}  ->  "
                  f"X={p[0]:.3f}  Y={p[1]:.3f}  Z={p[2]:.3f}")

    pc.ground_z = ground_z
    pc.ground_z_method = method

    print(f"\n  Ground level (local) = {ground_z:.3f} m")

    return ground_z


def pick_trench_vertices(pc: PointCloudData) -> np.ndarray:
    """
    Let the user mark the trench outline by picking points.

    Opens a VisualizerWithEditing window (same mechanism as
    :func:`pick_ground_level`) so the user can Shift + Left-Click the corner
    points of the trench, then close the window (Q or X) when done.

    Returns an ``(N, 3)`` array of the picked point coordinates in the cloud's
    local frame (empty ``(0, 3)`` array if nothing was picked). The caller is
    responsible for turning these into a footprint polygon.
    """
    print("\n" + "=" * 62)
    print("  TRENCH OUTLINE POINT PICKING")
    print("=" * 62)
    print("  Shift + Left-Click  the corner points of the trench outline.")
    print("  Pick three or more points; close the window (Q) when finished.")
    print("  Pick nothing to leave the deviation colouring unrestricted.")
    print("=" * 62 + "\n")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name="Pick trench outline points  (Shift+Click, then Q to finish)",
        width=1280, height=720,
    )
    vis.add_geometry(pc.pcd)
    vis.run()
    picked_indices = vis.get_picked_points()
    vis.destroy_window()

    if len(picked_indices) == 0:
        print("  No trench points picked.")
        return np.empty((0, 3), dtype=float)

    picked_pts = pc.pts[picked_indices]
    print(f"\n  Picked {len(picked_indices)} trench-outline point(s):")
    for i, idx in enumerate(picked_indices):
        p = pc.pts[idx]
        print(f"    [{i+1}]  index {idx:>8,}  ->  "
              f"X={p[0]:.3f}  Y={p[1]:.3f}  Z={p[2]:.3f}")
    return np.asarray(picked_pts, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# SITE PICK PERSISTENCE (ground level + trench footprint)
# ─────────────────────────────────────────────────────────────────────────────
# Both the ground-level pick and the trench-outline pick are cached in a small
# JSON next to the site PLY (``<stem>_ground.json`` / ``<stem>_trench.json``) so
# the user only picks once per site. Delete the JSON or pass ``repick=True`` to
# pick again. Shared by every viewer that needs these picks.

def _site_sidecar(ply_path, suffix):
    """Path of the cache file ``<stem>_<suffix>.json`` next to the PLY."""
    p = Path(ply_path)
    return p.parent / f"{p.stem}_{suffix}.json"


def load_or_pick_ground_level(pc: "PointCloudData", ply_path, repick=False) -> float:
    """Ground Z for a site, cached in ``<stem>_ground.json`` next to the PLY.

    Loads the cached value when present (unless ``repick``); otherwise opens the
    picker (:func:`pick_ground_level`) and saves the result. Sets ``pc.ground_z``
    and ``pc.ground_z_method`` in place and returns the ground Z.
    """
    sidecar = _site_sidecar(ply_path, "ground")
    if not repick and sidecar.is_file():
        try:
            data = json.loads(sidecar.read_text())
            gz = float(data["ground_z"])
            pc.ground_z = gz
            pc.ground_z_method = data.get("method", "loaded from cache")
            print(f"\n  Loaded ground level {gz:.3f} m "
                  f"({pc.ground_z_method}) from {sidecar.name}")
            return gz
        except Exception as exc:
            print(f"[WARN] Could not read {sidecar.name}: {exc}")
    gz = pick_ground_level(pc)
    try:
        sidecar.write_text(json.dumps(
            {"ground_z": gz, "method": pc.ground_z_method}, indent=2))
        print(f"  Saved ground level to {sidecar.name}")
    except Exception as exc:
        print(f"[WARN] Could not save {sidecar.name}: {exc}")
    return gz


def load_or_pick_trench(pc: "PointCloudData", ply_path, mode="hull", repick=False):
    """Trench-outline vertices for a site, cached in ``<stem>_trench.json``.

    Returns ``(vertices, mode)`` where ``vertices`` is a list of ``[x, y, z]``
    picked points (or ``None`` when no trench is defined) and ``mode`` is
    ``"hull"`` or ``"order"``. Loads the cache when present (unless ``repick``);
    otherwise opens the picker (:func:`pick_trench_vertices`) and, if at least
    three points are picked, saves them. Use :func:`trench_path_from_vertices`
    to turn the result into a footprint polygon.
    """
    sidecar = _site_sidecar(ply_path, "trench")
    if not repick and sidecar.is_file():
        try:
            data = json.loads(sidecar.read_text())
            verts = data.get("vertices")
            saved_mode = data.get("mode", mode)
            n = len(verts) if verts else 0
            print(f"\n  Loaded trench: {n} vertices ({saved_mode}) "
                  f"from {sidecar.name}")
            return verts, saved_mode
        except Exception as exc:
            print(f"[WARN] Could not read {sidecar.name}: {exc}")
    picked = pick_trench_vertices(pc)
    if len(picked) >= 3:
        verts = picked.tolist()
        try:
            sidecar.write_text(json.dumps(
                {"vertices": verts, "mode": mode}, indent=2))
            print(f"  Saved trench to {sidecar.name}")
        except Exception as exc:
            print(f"[WARN] Could not save {sidecar.name}: {exc}")
        return verts, mode
    print("  No trench defined; restriction not applied.")
    return None, mode


def trench_path_from_vertices(vertices, mode="hull"):
    """Build a matplotlib ``Path`` footprint (XY) from picked trench vertices.

    ``mode="hull"`` uses the convex hull of the points; ``mode="order"`` keeps
    the pick order (for irregular / concave outlines). Returns ``None`` when
    fewer than three vertices are supplied.
    """
    if vertices is None or len(vertices) < 3:
        return None
    import matplotlib.path as mpath
    pts = np.asarray(vertices, dtype=float)[:, :2]
    if mode == "hull":
        from scipy.spatial import ConvexHull
        try:
            pts = pts[ConvexHull(pts).vertices]
        except Exception as exc:
            print(f"[WARN] Convex hull failed ({exc}); using pick order.")
    return mpath.Path(pts)
