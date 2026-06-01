# -*- coding: utf-8 -*-
"""
Convert Water_Area point clouds from local .ply to georeferenced .las
=====================================================================
Consolidated from five area-specific scripts into one configurable tool.

Translation: local origin -> UTM32 / ETRS89 (EPSG:25832) using
             area_points_utm32_etrs89.geojson

PLY attributes preserved:  x, y, z, red, green, blue, class -> LAS classification
Requires: plyfile, laspy, geopandas, numpy, pyproj

Usage
-----
  Set AREA_NUMBER below (1-5) and run, or use --area CLI argument:
      python tools/ply_to_las.py --area 3
"""

import sys
from pathlib import Path
import argparse

# Ensure the project root is on the path so `core` is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import geopandas as gpd
import laspy
from plyfile import PlyData
from pyproj import CRS

from core.config import AREA_REF_GEOJSON, PLY_BASE_DIR

CRS_UTM32 = CRS.from_epsg(25832)

# ── Configuration ─────────────────────────────────────────────────────────────
# Change this or pass --area on the command line
AREA_NUMBER = 1

# ── CLI argument parsing ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Convert Water_Area PLYs to LAS")
parser.add_argument("--area", type=int, default=AREA_NUMBER,
                    help="Area number to convert (1-5)")
args, _ = parser.parse_known_args()
AREA_NUMBER = args.area

AREA_NAME  = f"Area{AREA_NUMBER}"
INPUT_DIR  = Path(PLY_BASE_DIR) / f"Water_Area_{AREA_NUMBER}"
OUTPUT_DIR = Path(PLY_BASE_DIR) / f"Water_Area_{AREA_NUMBER}_LAS"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load UTM origin ───────────────────────────────────────────────────────
gdf = gpd.read_file(str(AREA_REF_GEOJSON))
area_row = gdf[gdf["name"] == AREA_NAME].iloc[0]
utm_origin = np.array([
    area_row.geometry.x,
    area_row.geometry.y,
    area_row.geometry.z,
])
print(f"{AREA_NAME} UTM origin: E={utm_origin[0]:.2f}, N={utm_origin[1]:.2f}, Z={utm_origin[2]:.2f}")

# ── 2. Collect input PLY files ────────────────────────────────────────────────
ply_files = sorted(INPUT_DIR.glob("*.ply"))
print(f"\nFound {len(ply_files)} PLY files to convert\n")

if not ply_files:
    print(f"[ERROR] No PLY files in {INPUT_DIR}")
    sys.exit(1)

# ── 3. Convert each file ──────────────────────────────────────────────────────
for ply_path in ply_files:
    plydata  = PlyData.read(str(ply_path))
    vertex   = plydata["vertex"]
    n_points = len(vertex)

    if n_points == 0:
        print(f"  [SKIP] {ply_path.name} — no points")
        continue

    # Translate coordinates
    x_utm = vertex["x"].astype(np.float64) + utm_origin[0]
    y_utm = vertex["y"].astype(np.float64) + utm_origin[1]
    z_utm = vertex["z"].astype(np.float64) + utm_origin[2]

    # Detect available attributes
    prop_names = {p.name for p in vertex.properties}
    has_colors = {"red", "green", "blue"}.issubset(prop_names)
    has_class  = "class" in prop_names

    # LAS point format 2 = XYZ + RGB (no GPS time); 0 = XYZ only
    point_format = 2 if has_colors else 0

    header = laspy.LasHeader(point_format=point_format, version="1.4")
    header.offsets = np.array([np.floor(x_utm.min()),
                                np.floor(y_utm.min()),
                                np.floor(z_utm.min())])
    header.scales  = np.array([0.001, 0.001, 0.001])  # 1 mm precision
    header.set_wkt_crs(CRS_UTM32)                      # embed EPSG:25832

    las = laspy.LasData(header=header)
    las.x = x_utm
    las.y = y_utm
    las.z = z_utm

    if has_colors:
        las.red   = vertex["red"].astype(np.uint16)   * 256
        las.green = vertex["green"].astype(np.uint16) * 256
        las.blue  = vertex["blue"].astype(np.uint16)  * 256

    if has_class:
        las.classification = vertex["class"].astype(np.uint8)

    out_path = OUTPUT_DIR / (ply_path.stem + ".las")
    las.write(str(out_path))

    print(f"  {ply_path.name:30s} -> {out_path.name}  ({n_points:,} pts)"
          f"  [RGB={'yes' if has_colors else 'no'}, class={'yes' if has_class else 'no'}]")

print(f"\nDone. LAS files written to:\n  {OUTPUT_DIR}")
