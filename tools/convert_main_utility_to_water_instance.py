# -*- coding: utf-8 -*-
"""
Convert "Main Utility" (class 0) points to WaterLine instances.
================================================================
For every PLY site in Water_Area_4 and Water_Area_5, extracts all
class-0 points and saves them as a single WaterLine instance using
the same directory layout and PLY format as label_viewer.py.

Usage:  python tools/convert_main_utility_to_water_instance.py
"""

import sys
from pathlib import Path
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.config import PLY_BASE_DIR
from core.data_loader import instance_base_name

WATER_LINE_TYPE_ID = 7
AREAS = ["Water_Area_4", "Water_Area_5"]


def parse_ply_header(ply_path):
    """Read PLY header and return (header_row_count, property_names)."""
    header_lines = 0
    property_names = []
    with open(ply_path, "r", errors="replace") as f:
        for line in f:
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("property "):
                property_names.append(stripped.split()[-1])
            if stripped == "end_header":
                break
    return header_lines, property_names


def process_site(ply_path):
    """Extract class-0 points from a site PLY and save as a WaterLine instance."""
    header_rows, props = parse_ply_header(ply_path)

    if "class" not in props:
        print(f"  SKIP {ply_path.name}: no 'class' property in PLY header")
        return False

    class_col = props.index("class")
    xyz_cols = [props.index("x"), props.index("y"), props.index("z")]
    rgb_cols = (
        [props.index("red"), props.index("green"), props.index("blue")]
        if all(c in props for c in ("red", "green", "blue"))
        else None
    )

    data = np.loadtxt(str(ply_path), skiprows=header_rows)
    classes = data[:, class_col].astype(int)
    mask = classes == 0
    n_class0 = int(mask.sum())

    if n_class0 == 0:
        print(f"  SKIP {ply_path.name}: 0 class-0 points")
        return False

    pts = data[mask][:, xyz_cols]
    rgb = data[mask][:, rgb_cols].astype(int) if rgb_cols else None

    inst_dir = ply_path.parent / f"{instance_base_name(ply_path)}_Instances"
    inst_dir.mkdir(parents=True, exist_ok=True)

    out_path = inst_dir / f"0_instance_0_type_{WATER_LINE_TYPE_ID}.ply"
    n = len(pts)

    with open(str(out_path), "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if rgb is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("property int utility_type\n")
        f.write("end_header\n")
        for i in range(n):
            parts = [f"{pts[i, 0]:.6f}", f"{pts[i, 1]:.6f}", f"{pts[i, 2]:.6f}"]
            if rgb is not None:
                parts.extend([str(rgb[i, 0]), str(rgb[i, 1]), str(rgb[i, 2])])
            parts.append(str(WATER_LINE_TYPE_ID))
            f.write(" ".join(parts) + "\n")

    print(f"  OK   {ply_path.name}: {n_class0:,} pts -> {inst_dir.name}/")
    return True


def main():
    base = Path(PLY_BASE_DIR)
    total_saved = 0
    total_skipped = 0

    for area_name in AREAS:
        area_dir = base / area_name
        if not area_dir.is_dir():
            print(f"\n[WARN] Directory not found: {area_dir}")
            continue

        ply_files = sorted(area_dir.glob("*.ply"))
        print(f"\n{'='*60}")
        print(f"  {area_name}  ({len(ply_files)} sites)")
        print(f"{'='*60}")

        for ply_path in ply_files:
            if process_site(ply_path):
                total_saved += 1
            else:
                total_skipped += 1

    print(f"\n{'='*60}")
    print(f"  Done.  Saved: {total_saved}   Skipped: {total_skipped}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
