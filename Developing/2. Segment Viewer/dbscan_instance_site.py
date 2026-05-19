# -*- coding: utf-8 -*-
"""
DBSCAN Instance Segmentation — Area 5 Site 05, Class 1 (Main Utility)
======================================================================
Opens an Open3D viewer with live EPS / MIN_SAMPLES controls and a
Re-run button so you can tune without restarting the script.

Key parameters:
    EPS         — neighbourhood radius in metres.
    MIN_SAMPLES — minimum points in a neighbourhood to form a core point.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from pathlib import Path
import time
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PLY_FILE = (
    r"C:\Users\bosre\OneDrive - University of Twente\Documents\AAU UTwente thesis"
    r"\Python\Thesis\Data\OpenTrench3D\Water_Area_5\Area_5_Site_26.ply"
)

PLY_HEADER_ROWS = 11
CLASS_COLUMN    = 6
TARGET_CLASS    = 1       # "Other Utility" (third-party utilities)

VOXEL_SIZE  = 0.02        # metres — downsample before clustering

# Initial DBSCAN parameters (editable in the GUI)
EPS         = 0.025
MIN_SAMPLES = 5

POINT_SIZE  = 2.0
MIN_INSTANCE_POINTS = 200   # instances smaller than this are reclassified as noise

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD PLY
# ─────────────────────────────────────────────────────────────────────────────
ply_path = Path(PLY_FILE)
if not ply_path.exists():
    raise FileNotFoundError(f"PLY not found: {PLY_FILE}")

print(f"Loading {ply_path.name} ...")
t0 = time.perf_counter()
data = np.loadtxt(str(PLY_FILE), skiprows=PLY_HEADER_ROWS)
print(f"  {len(data):,} points loaded in {time.perf_counter() - t0:.1f}s")

all_xyz     = data[:, :3]
all_classes = data[:, CLASS_COLUMN].astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXTRACT CLASS-1 POINTS & DOWNSAMPLE
# ─────────────────────────────────────────────────────────────────────────────
mask1 = all_classes == TARGET_CLASS
pts1  = all_xyz[mask1]
print(f"  Class {TARGET_CLASS} points: {len(pts1):,} "
      f"({100 * len(pts1) / len(data):.1f}% of total)")

if len(pts1) == 0:
    raise RuntimeError(f"No points with class={TARGET_CLASS} found.")

_pcd1 = o3d.geometry.PointCloud()
_pcd1.points = o3d.utility.Vector3dVector(pts1)
_pcd1_ds = _pcd1.voxel_down_sample(voxel_size=VOXEL_SIZE)
pts1_ds = np.asarray(_pcd1_ds.points)
print(f"  Voxel downsample ({VOXEL_SIZE} m): {len(pts1):,} → {len(pts1_ds):,} points\n")

# KDTree built once — reused for every re-run
_tree_ds = KDTree(pts1_ds)

# ─────────────────────────────────────────────────────────────────────────────
# 3. CLUSTERING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def run_dbscan(eps, min_samples):
    """Run DBSCAN on the downsampled cloud and propagate labels to full res."""
    t = time.perf_counter()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    ds_lbl = db.fit_predict(pts1_ds)
    print(f"  DBSCAN done in {time.perf_counter() - t:.1f}s")

    nn_idx = _tree_ds.query(pts1, k=1, return_distance=False).ravel()
    labels = ds_lbl[nn_idx]

    # Drop instances smaller than MIN_INSTANCE_POINTS → reclassify as noise
    uniq, cnts = np.unique(labels, return_counts=True)
    for u, c in zip(uniq, cnts):
        if u >= 0 and c < MIN_INSTANCE_POINTS:
            labels[labels == u] = -1

    # Reorder instance IDs by descending point count (0 = largest)
    uniq, cnts = np.unique(labels, return_counts=True)
    inst_mask  = uniq >= 0
    inst_ids   = uniq[inst_mask]
    inst_cnts  = cnts[inst_mask]
    order      = np.argsort(inst_cnts)[::-1]
    remap      = {int(old): new for new, old in enumerate(inst_ids[order])}
    new_labels = np.full_like(labels, -1)
    for old_id, new_id in remap.items():
        new_labels[labels == old_id] = new_id
    labels = new_labels

    uniq, cnts = np.unique(labels, return_counts=True)
    n_inst  = int((uniq >= 0).sum())
    n_noise = int(cnts[uniq == -1].sum()) if -1 in uniq else 0
    print(f"  {n_inst} instance(s), {n_noise:,} noise points "
          f"(min size filter: {MIN_INSTANCE_POINTS} pts)")
    for u, c in zip(uniq, cnts):
        print(f"    {'noise' if u == -1 else f'instance {u}'}: {c:,} pts")
    return labels, uniq, n_inst, n_noise


def make_colors(labels, unique, n_instances):
    """Build per-point RGB array from instance labels."""
    rng = np.random.default_rng(42)
    if n_instances > 0:
        hues = np.linspace(0.0, 1.0, n_instances, endpoint=False)
        rng.shuffle(hues)
        # Store as plain Python float lists so gui.Color receives exact values
        palette = {int(uid): [float(c) for c in mcolors.hsv_to_rgb([h, 0.85, 0.95])]
                   for uid, h in zip(sorted(unique[unique >= 0]), hues)}
    else:
        palette = {}

    colors = np.full((len(pts1), 3), 0.1)
    for uid, col in palette.items():
        colors[labels == uid] = col
    return colors, palette


# ─────────────────────────────────────────────────────────────────────────────
# 4. INITIAL CLUSTERING RUN
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nRunning DBSCAN (eps={EPS}, min_samples={MIN_SAMPLES}) ...")
instance_labels, unique, n_instances, n_noise = run_dbscan(EPS, MIN_SAMPLES)
init_colors, palette = make_colors(instance_labels, unique, n_instances)

# Build point clouds
pcd_instances = o3d.geometry.PointCloud() # TODO: add instance IDs to the point cloud
pcd_instances.points = o3d.utility.Vector3dVector(pts1) # TODO: add instance IDs to the point cloud
pcd_instances.colors = o3d.utility.Vector3dVector(init_colors) # TODO: add instance IDs to the point cloud 

pts_bg = all_xyz[~mask1]
rgb_bg = data[~mask1, 3:6] / 255.0 * 0.25
pcd_bg = o3d.geometry.PointCloud()
pcd_bg.points = o3d.utility.Vector3dVector(pts_bg)
pcd_bg.colors = o3d.utility.Vector3dVector(rgb_bg)

# ─────────────────────────────────────────────────────────────────────────────
# 5. GUI VIEWER
# ─────────────────────────────────────────────────────────────────────────────
class InstanceViewer:
    _PANEL_WIDTH  = 300
    _MAX_SLOTS    = 30   # max legend rows (instances + noise)

    def __init__(self):
        self._cur_labels  = instance_labels
        self._cur_palette = palette
        self._cur_n_inst  = n_instances
        self._cur_n_noise = n_noise

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            f"DBSCAN Instance Viewer — {ply_path.name} ", 1440, 900)

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.08, 0.08, 0.08, 1.0])

        self._mat = rendering.MaterialRecord()
        self._mat.shader     = "defaultUnlit"
        self._mat.point_size = POINT_SIZE

        self.scene.scene.add_geometry("bg",        pcd_bg,        self._mat)
        self.scene.scene.add_geometry("instances", pcd_instances, self._mat)
        bounds = pcd_instances.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60.0, bounds, bounds.get_center())

        # ── Panel ─────────────────────────────────────────────────────────
        panel = gui.Vert(8, gui.Margins(12, 12, 12, 12))

        title = gui.Label("DBSCAN parameters")
        title.text_color = gui.Color(1.0, 1.0, 1.0)
        panel.add_child(title)
        panel.add_child(gui.Label(""))

        panel.add_child(gui.Label("EPS (m):"))
        self._eps_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._eps_edit.double_value = EPS
        self._eps_edit.set_limits(0.001, 10.0)
        panel.add_child(self._eps_edit)

        panel.add_child(gui.Label("Min samples:"))
        self._ms_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self._ms_edit.int_value = MIN_SAMPLES
        self._ms_edit.set_limits(1, 500)
        panel.add_child(self._ms_edit)

        panel.add_child(gui.Label(""))

        rerun_btn = gui.Button("Re-run DBSCAN")
        rerun_btn.set_on_clicked(self._on_rerun)
        panel.add_child(rerun_btn)

        panel.add_child(gui.Label(""))

        self._bg_cb = gui.Checkbox("Show background (B)")
        self._bg_cb.checked = True
        self._bg_cb.set_on_checked(self._on_bg_toggle)
        panel.add_child(self._bg_cb)

        panel.add_child(gui.Label(""))

        self._stats_lbl = gui.Label("")
        self._stats_lbl.text_color = gui.Color(0.8, 0.8, 0.8)
        panel.add_child(self._stats_lbl)

        panel.add_child(gui.Label(""))

        # Pre-allocate fixed legend slots — text updated in place, no add/remove
        self._legend_slots = []
        for _ in range(self._MAX_SLOTS):
            lbl = gui.Label("")
            panel.add_child(lbl)
            self._legend_slots.append(lbl)

        panel.add_child(gui.Label(""))
        help_lbl = gui.Label("B = toggle background\nR = reset camera")
        help_lbl.text_color = gui.Color(0.5, 0.5, 0.5)
        panel.add_child(help_lbl)

        self.window.add_child(self.scene)
        self.window.add_child(panel)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_key(self._on_key)
        self._panel = panel

        self._update_stats()
        self._update_legend()

    # ── layout ────────────────────────────────────────────────────────────
    def _on_layout(self, ctx):
        r  = self.window.content_rect
        pw = self._PANEL_WIDTH
        self.scene.frame  = gui.Rect(r.x, r.y, r.width - pw, r.height)
        self._panel.frame = gui.Rect(r.x + r.width - pw, r.y, pw, r.height)

    # ── keyboard ──────────────────────────────────────────────────────────
    def _on_key(self, event):
        if event.type == gui.KeyEvent.DOWN:
            if event.key in (ord('b'), ord('B')):
                self._bg_cb.checked = not self._bg_cb.checked
                self._on_bg_toggle(self._bg_cb.checked)
                return gui.Widget.EventCallbackResult.CONSUMED
            if event.key in (ord('r'), ord('R')):
                bounds = pcd_instances.get_axis_aligned_bounding_box()
                self.scene.setup_camera(60.0, bounds, bounds.get_center())
                return gui.Widget.EventCallbackResult.CONSUMED
        return gui.Widget.EventCallbackResult.IGNORED

    # ── background toggle ─────────────────────────────────────────────────
    def _on_bg_toggle(self, checked):
        self.scene.scene.show_geometry("bg", checked)

    # ── re-run ────────────────────────────────────────────────────────────
    def _on_rerun(self):
        eps = self._eps_edit.double_value
        ms  = self._ms_edit.int_value
        print(f"\nRe-running DBSCAN (eps={eps}, min_samples={ms}) ...")

        labels, uniq, n_inst, n_noise = run_dbscan(eps, ms)
        colors, pal = make_colors(labels, uniq, n_inst)

        self._cur_labels  = labels
        self._cur_palette = pal
        self._cur_n_inst  = n_inst
        self._cur_n_noise = n_noise

        pcd_instances.colors = o3d.utility.Vector3dVector(colors)
        self.scene.scene.remove_geometry("instances")
        self.scene.scene.add_geometry("instances", pcd_instances, self._mat)

        self._update_stats()
        self._update_legend()

    # ── stats ─────────────────────────────────────────────────────────────
    def _update_stats(self):
        self._stats_lbl.text = (
            f"Total pts:       {len(data):,}\n"
            f"Class-{TARGET_CLASS} pts:    {len(pts1):,}\n"
            f"Instances found: {self._cur_n_inst}\n"
            f"Noise points:    {self._cur_n_noise:,}"
        )

    # ── legend (updates pre-allocated slots in place) ─────────────────────
    def _update_legend(self):
        rows = []
        for uid, col in sorted(self._cur_palette.items()):
            cnt = int((self._cur_labels == uid).sum())
            r, g, b = float(col[0]), float(col[1]), float(col[2])
            rows.append((f"■  inst {uid}: {cnt:,} pts", gui.Color(r, g, b)))
        if self._cur_n_noise:
            rows.append((f"■  noise: {self._cur_n_noise:,} pts",
                         gui.Color(0.4, 0.4, 0.4)))

        for i, slot in enumerate(self._legend_slots):
            if i < len(rows):
                slot.text       = rows[i][0]
                slot.text_color = rows[i][1]
            else:
                slot.text = ""   # hide unused slots

    def run(self):
        gui.Application.instance.run()


if __name__ == "__main__":
    viewer = InstanceViewer()
    viewer.run()
