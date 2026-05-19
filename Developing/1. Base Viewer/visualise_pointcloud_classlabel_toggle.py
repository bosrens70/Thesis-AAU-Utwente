# -*- coding: utf-8 -*-
"""
Point Cloud Viewer with Class Label Toggle
===========================================
Displays Water_Area_5_Site_05 point cloud in Open3D with a GUI checkbox
and keyboard shortcut (L) to toggle between original RGB and per-class
semantic colours.

Keyboard shortcuts:
    L   toggle class label colours
    R   reset camera view
"""

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
from pathlib import Path
import time

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PLY_FILE = (
    r"C:\Users\bosre\OneDrive - University of Twente\Documents\AAU UTwente thesis"
    r"\Python\Thesis\Data\OpenTrench3D\Water_Area_5\Area_5_Site_05.ply"
)

# OpenTrench3D class mapping
CLASS_LABELS = {
    0: {"name": "Main Utility",     "color": [0.00, 0.80, 0.00]},
    1: {"name": "Other Utility",    "color": [1.00, 1.00, 0.00]},
    2: {"name": "Trench",           "color": [0.55, 0.27, 0.07]},
    3: {"name": "Inactive Utility", "color": [0.00, 0.00, 0.00]},
    4: {"name": "Misc",             "color": [0.60, 0.60, 0.60]},
}

_DEFAULT_CLASS_COLOR = [1.0, 0.0, 1.0]  # magenta for unknown

POINT_SIZE = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# LOAD POINT CLOUD
# ─────────────────────────────────────────────────────────────────────────────
_ply_path = Path(PLY_FILE)
if not _ply_path.exists():
    print(f"[ERROR] PLY file not found: {PLY_FILE}")
    raise SystemExit(1)

print(f"Loading point cloud: {_ply_path.name} ...")
t0 = time.perf_counter()
pcd = o3d.io.read_point_cloud(str(PLY_FILE))
pts = np.asarray(pcd.points)
print(f"  {len(pts):,} points loaded in {time.perf_counter() - t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# READ CLASS LABELS FROM PLY
# ─────────────────────────────────────────────────────────────────────────────
print("  Reading class labels from PLY header ...")
class_labels = None
with open(str(PLY_FILE), 'r') as f:
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
        class_labels = np.array(labels, dtype=int)
        unique_classes = np.unique(class_labels)
        print(f"  Class labels found: {len(class_labels):,} values, "
              f"unique classes: {sorted(unique_classes.tolist())}")
    else:
        print(f"  [WARNING] No 'class' property found — class toggle disabled")

# Store original RGB
original_colors = np.asarray(pcd.colors).copy()

# Build class colour array
if class_labels is not None:
    class_colors = np.zeros_like(original_colors)
    for cls_id, cfg in CLASS_LABELS.items():
        mask = class_labels == cls_id
        class_colors[mask] = cfg["color"]
    known_mask = np.isin(class_labels, list(CLASS_LABELS.keys()))
    if not known_mask.all():
        class_colors[~known_mask] = _DEFAULT_CLASS_COLOR
else:
    class_colors = None

print(f"  Total load time: {time.perf_counter() - t0:.1f}s\n")

# ─────────────────────────────────────────────────────────────────────────────
# GUI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class PointCloudViewer:
    MENU_QUIT = 1

    def __init__(self):
        self.showing_classes = False

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "Point Cloud Viewer — Area 5 Site 05", 1400, 900)

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.1, 0.1, 0.1, 1.0])

        # Material
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = POINT_SIZE

        # Add point cloud
        self.scene.scene.add_geometry("pointcloud", pcd, self.mat)

        # Set up camera
        bounds = pcd.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60.0, bounds, bounds.get_center())

        # ── Side panel ────────────────────────────────────────────────────
        panel = gui.Vert(8, gui.Margins(10, 10, 10, 10))

        title = gui.Label("Class Label Toggle")
        title.text_color = gui.Color(1.0, 1.0, 1.0)
        panel.add_child(title)

        self.toggle_cb = gui.Checkbox("Show class labels (L)")
        self.toggle_cb.checked = False
        self.toggle_cb.set_on_checked(self._on_toggle)
        panel.add_child(self.toggle_cb)

        panel.add_child(gui.Label(""))  # spacer

        # Legend
        if class_labels is not None:
            legend_title = gui.Label("Legend:")
            legend_title.text_color = gui.Color(0.8, 0.8, 0.8)
            panel.add_child(legend_title)
            for cls_id in sorted(CLASS_LABELS.keys()):
                cfg = CLASS_LABELS[cls_id]
                count = int((class_labels == cls_id).sum())
                lbl = gui.Label(f"  {cls_id}: {cfg['name']} ({count:,} pts)")
                lbl.text_color = gui.Color(*cfg["color"])
                panel.add_child(lbl)

        panel.add_child(gui.Label(""))
        info = gui.Label(f"Total points: {len(pts):,}")
        info.text_color = gui.Color(0.7, 0.7, 0.7)
        panel.add_child(info)

        panel.add_child(gui.Label(""))
        help_lbl = gui.Label("Shortcuts: L=toggle, R=reset view")
        help_lbl.text_color = gui.Color(0.5, 0.5, 0.5)
        panel.add_child(help_lbl)

        # Layout
        self.window.add_child(self.scene)
        self.window.add_child(panel)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_key(self._on_key)

        self._panel = panel
        self._panel_width = 250

    def _on_layout(self, ctx):
        r = self.window.content_rect
        pw = min(self._panel_width, r.width // 3)
        self.scene.frame = gui.Rect(r.x, r.y, r.width - pw, r.height)
        self._panel.frame = gui.Rect(r.x + r.width - pw, r.y, pw, r.height)

    def _on_key(self, event):
        if event.type == gui.KeyEvent.DOWN:
            if event.key == ord('L') or event.key == ord('l'):
                self.toggle_cb.checked = not self.toggle_cb.checked
                self._on_toggle(self.toggle_cb.checked)
                return gui.Widget.EventCallbackResult.CONSUMED
            if event.key == ord('R') or event.key == ord('r'):
                bounds = pcd.get_axis_aligned_bounding_box()
                self.scene.setup_camera(60.0, bounds, bounds.get_center())
                return gui.Widget.EventCallbackResult.CONSUMED
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_toggle(self, is_checked):
        self.showing_classes = is_checked
        if is_checked and class_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(class_colors)
        else:
            pcd.colors = o3d.utility.Vector3dVector(original_colors)

        self.scene.scene.remove_geometry("pointcloud")
        self.scene.scene.add_geometry("pointcloud", pcd, self.mat)

    def run(self):
        gui.Application.instance.run()


if __name__ == "__main__":
    viewer = PointCloudViewer()
    viewer.run()
