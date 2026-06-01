# -*- coding: utf-8 -*-
"""
Distribute a QGIS .qml style file to all point cloud files in target folders.

Steps:
  1. Style one point cloud layer in QGIS
  2. Right-click layer -> Export -> Save as QGIS Layer Style -> save as .qml
  3. Set QML_SOURCE below to that .qml file
  4. Add/remove folders in TARGET_DIRS
  5. Run the script — it copies and renames the .qml to match every .las / .copc.laz

When QGIS loads a point cloud it automatically picks up a .qml with the same stem
in the same folder, so the colour scheme will apply instantly.
"""

import sys
from pathlib import Path

# Ensure the project root is on the path so `core` is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import shutil

from core.config import PLY_BASE_DIR

# ── Configuration ──────────────────────────────────────────────────────────────

# Path to the .qml style file you exported from QGIS
QML_SOURCE = Path(_project_root) / "Developing" / "QML_Distributor" / "OpenTrench3D_class_color_scheme.qml"

# Folders containing the point cloud files
TARGET_DIRS = [
    Path(PLY_BASE_DIR) / "Water_Area_5_LAS",
    Path(PLY_BASE_DIR) / "Water_Area_4_LAS",
    Path(PLY_BASE_DIR) / "Water_Area_3_LAS",
    Path(PLY_BASE_DIR) / "Water_Area_2_LAS",
    Path(PLY_BASE_DIR) / "Water_Area_1_LAS",
]

# Extensions to match — QGIS reads .qml sidecars for all of these
EXTENSIONS = [".las", ".copc.laz"]

# ── Distribute ─────────────────────────────────────────────────────────────────

if not QML_SOURCE.exists():
    raise FileNotFoundError(f"QML source not found:\n  {QML_SOURCE}")

total = 0

for folder in TARGET_DIRS:
    if not folder.exists():
        print(f"[SKIP] Folder not found: {folder}")
        continue

    print(f"\n{folder.name}")

    for ext in EXTENSIONS:
        files = sorted(f for f in folder.iterdir() if f.name.endswith(ext))

        for pc_file in files:
            stem = pc_file.name[: -len(ext)]
            qml_dest = folder / (stem + ".qml")
            shutil.copy2(QML_SOURCE, qml_dest)
            print(f"  -> {qml_dest.name}")
            total += 1

print(f"\nDone. {total} .qml files written.")
