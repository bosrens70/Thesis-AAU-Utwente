# -*- coding: utf-8 -*-
"""
HDBSCAN Instance Segmentation Viewer
=====================================
Opens an Open3D viewer with live MIN_CLUSTER_SIZE / MIN_SAMPLES controls
and a Re-run button so you can tune without restarting the script.

Refactored to use core/ for configuration.  Data loading is minimal
(PLY only — no GML needed).
"""

import sys
from pathlib import Path

# Ensure the project root is on the path so `core` is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import time
import matplotlib.colors as mcolors
from datetime import datetime

from core.config import (
    PLY_FILE, PLY_HEADER_ROWS, CLASS_COLUMN, TARGET_CLASS,
    VOXEL_SIZE, MIN_CLUSTER_SIZE, MIN_SAMPLES, POINT_SIZE,
    MIN_INSTANCE_POINTS,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD PLY (lightweight — no GML, no area offset needed)
# ─────────────────────────────────────────────────────────────────────────────
ply_path = Path(PLY_FILE)
if not ply_path.exists():
    raise FileNotFoundError(f"PLY not found: {PLY_FILE}")

print(f"Loading {ply_path.name} ...")
t0 = time.perf_counter()
data = np.loadtxt(str(PLY_FILE), skiprows=PLY_HEADER_ROWS)
print(f"  {len(data):,} points loaded in {time.perf_counter() - t0:.1f}s")

all_xyz     = data[:, :3]
all_rgb     = data[:, 3:6].astype(int)
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
print(f"  Voxel downsample ({VOXEL_SIZE} m): {len(pts1):,} -> {len(pts1_ds):,} points\n")

# KDTree built once — reused for every re-run
_tree_ds = KDTree(pts1_ds)

# ─────────────────────────────────────────────────────────────────────────────
# 3. CLUSTERING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def run_hdbscan(min_cluster_size, min_samples, min_instance_points):
    t = time.perf_counter()
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    ds_lbl = hdb.fit_predict(pts1_ds)
    print(f"  HDBSCAN done in {time.perf_counter() - t:.1f}s")

    nn_idx = _tree_ds.query(pts1, k=1, return_distance=False).ravel()
    labels = ds_lbl[nn_idx]

    uniq, cnts = np.unique(labels, return_counts=True)
    for u, c in zip(uniq, cnts):
        if u >= 0 and c < min_instance_points:
            labels[labels == u] = -1

    uniq, cnts = np.unique(labels, return_counts=True)
    inst_ids  = uniq[uniq >= 0]
    inst_cnts = cnts[uniq >= 0]
    order     = np.argsort(inst_cnts)[::-1]
    remap     = {int(old): new for new, old in enumerate(inst_ids[order])}
    new_labels = np.full_like(labels, -1)
    for old_id, new_id in remap.items():
        new_labels[labels == old_id] = new_id
    labels = new_labels

    uniq, cnts = np.unique(labels, return_counts=True)
    n_inst  = int((uniq >= 0).sum())
    n_noise = int(cnts[uniq == -1].sum()) if -1 in uniq else 0
    print(f"  {n_inst} instance(s), {n_noise:,} noise points "
          f"(min size filter: {min_instance_points} pts)")
    for u, c in zip(uniq, cnts):
        print(f"    {'noise' if u == -1 else f'instance {u}'}: {c:,} pts")
    return labels, uniq, n_inst, n_noise


def make_colors(labels, unique, n_instances):
    rng = np.random.default_rng(42)
    if n_instances > 0:
        hues = np.linspace(0.0, 1.0, n_instances, endpoint=False)
        rng.shuffle(hues)
        palette = {int(uid): [float(c) for c in mcolors.hsv_to_rgb([h, 0.85, 0.95])]
                   for uid, h in zip(sorted(unique[unique >= 0]), hues)}
    else:
        palette = {}

    colors = np.full((len(pts1), 3), 0.1)
    for uid, col in palette.items():
        colors[labels == uid] = col
    return colors, palette


def split_parallel_pipes(labels, eps_2d, min_pts_2d, min_split_size, target_instance):
    new_labels = labels.copy()
    next_id = int(labels.max()) + 1 if labels.max() >= 0 else 0

    for uid in [target_instance]:
        inst_mask = labels == uid
        inst_pts  = pts1[inst_mask]
        if len(inst_pts) < min_pts_2d * 2:
            continue

        pca = PCA(n_components=3)
        transformed = pca.fit_transform(inst_pts)
        cross_section = transformed[:, 1:3]

        db = DBSCAN(eps=eps_2d, min_samples=min_pts_2d)
        sub_labels = db.fit_predict(cross_section)

        for sid in set(sub_labels) - {-1}:
            if (sub_labels == sid).sum() < min_split_size:
                sub_labels[sub_labels == sid] = -1

        n_sub = len(set(sub_labels) - {-1})
        if n_sub <= 1:
            continue

        indices = np.where(inst_mask)[0]
        for sub_id in sorted(set(sub_labels) - {-1}):
            sub_mask = sub_labels == sub_id
            if sub_id == 0:
                new_labels[indices[sub_mask]] = uid
            else:
                new_labels[indices[sub_mask]] = next_id
                next_id += 1
        new_labels[indices[sub_labels == -1]] = -1

        print(f"  instance {uid} split into {n_sub} sub-instances")

    return _renumber_labels(new_labels)


def _renumber_labels(labels):
    uniq, cnts = np.unique(labels, return_counts=True)
    inst_ids  = uniq[uniq >= 0]
    inst_cnts = cnts[uniq >= 0]
    order     = np.argsort(inst_cnts)[::-1]
    remap     = {int(old): new for new, old in enumerate(inst_ids[order])}
    new_labels = np.full_like(labels, -1)
    for old_id, new_id in remap.items():
        new_labels[labels == old_id] = new_id

    uniq, cnts = np.unique(new_labels, return_counts=True)
    n_inst  = int((uniq >= 0).sum())
    n_noise = int(cnts[uniq == -1].sum()) if -1 in uniq else 0
    print(f"  After split: {n_inst} instance(s), {n_noise:,} noise points")
    for u, c in zip(uniq, cnts):
        print(f"    {'noise' if u == -1 else f'instance {u}'}: {c:,} pts")
    return new_labels, uniq, n_inst, n_noise


# ─────────────────────────────────────────────────────────────────────────────
# 4. INITIAL CLUSTERING RUN
# ─────────────────────────────────────────────────────────────────────────────
print(f"Running HDBSCAN (min_cluster_size={MIN_CLUSTER_SIZE}, "
      f"min_samples={MIN_SAMPLES}) ...")
instance_labels, unique, n_instances, n_noise = run_hdbscan(
    MIN_CLUSTER_SIZE, MIN_SAMPLES, MIN_INSTANCE_POINTS)
init_colors, palette = make_colors(instance_labels, unique, n_instances)

pcd_instances = o3d.geometry.PointCloud()
pcd_instances.points = o3d.utility.Vector3dVector(pts1)
pcd_instances.colors = o3d.utility.Vector3dVector(init_colors)

pts_bg = all_xyz[~mask1]
rgb_bg = data[~mask1, 3:6] / 255.0 * 0.25
pcd_bg = o3d.geometry.PointCloud()
pcd_bg.points = o3d.utility.Vector3dVector(pts_bg)
pcd_bg.colors = o3d.utility.Vector3dVector(rgb_bg)

# ─────────────────────────────────────────────────────────────────────────────
# 5. GUI VIEWER
# ─────────────────────────────────────────────────────────────────────────────
class InstanceViewer:
    _PANEL_WIDTH = 300
    _MAX_SLOTS   = 30

    def __init__(self):
        self._base_labels = instance_labels.copy()
        self._cur_labels  = instance_labels
        self._cur_palette = palette
        self._cur_n_inst  = n_instances
        self._cur_n_noise = n_noise

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            f"HDBSCAN Instance Viewer — {ply_path.name} ", 1440, 900)

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.08, 0.08, 0.08, 1.0])

        self._mat = rendering.MaterialRecord()
        self._mat.shader     = "defaultLit"
        self._mat.point_size = POINT_SIZE

        self.scene.scene.add_geometry("bg",        pcd_bg,        self._mat)
        self.scene.scene.add_geometry("instances", pcd_instances, self._mat)
        bounds = pcd_instances.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60.0, bounds, bounds.get_center())

        # ── Panel ─────────────────────────────────────────────────────────
        panel = gui.Vert(8, gui.Margins(12, 12, 12, 12))

        title = gui.Label("HDBSCAN parameters")
        title.text_color = gui.Color(1.0, 1.0, 1.0)
        panel.add_child(title)
        panel.add_child(gui.Label(""))

        panel.add_child(gui.Label("Min cluster size:"))
        self._mcs_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self._mcs_edit.int_value = MIN_CLUSTER_SIZE
        self._mcs_edit.set_limits(2, 10000)
        panel.add_child(self._mcs_edit)

        panel.add_child(gui.Label("Min samples:"))
        self._ms_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self._ms_edit.int_value = MIN_SAMPLES
        self._ms_edit.set_limits(1, 500)
        panel.add_child(self._ms_edit)

        panel.add_child(gui.Label("Min instance points:"))
        self._mip_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self._mip_edit.int_value = MIN_INSTANCE_POINTS
        self._mip_edit.set_limits(0, 100000)
        panel.add_child(self._mip_edit)

        panel.add_child(gui.Label(""))

        rerun_btn = gui.Button("Re-run HDBSCAN")
        rerun_btn.set_on_clicked(self._on_rerun)
        panel.add_child(rerun_btn)

        save_btn = gui.Button("Save instances")
        save_btn.set_on_clicked(self._on_save)
        panel.add_child(save_btn)

        panel.add_child(gui.Label(""))

        split_title = gui.Label("Split parallel pipes (PCA)")
        split_title.text_color = gui.Color(1.0, 1.0, 1.0)
        panel.add_child(split_title)

        panel.add_child(gui.Label("Instance to split:"))
        self._split_id_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self._split_id_edit.int_value = 0
        self._split_id_edit.set_limits(0, 999)
        panel.add_child(self._split_id_edit)

        panel.add_child(gui.Label("2D eps (m):"))
        self._eps_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._eps_edit.double_value = 0.03
        self._eps_edit.set_limits(0.001, 1.0)
        panel.add_child(self._eps_edit)

        panel.add_child(gui.Label("2D min points:"))
        self._mp2d_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self._mp2d_edit.int_value = 20
        self._mp2d_edit.set_limits(2, 1000)
        panel.add_child(self._mp2d_edit)

        panel.add_child(gui.Label("Min split size:"))
        self._mss_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self._mss_edit.int_value = 200
        self._mss_edit.set_limits(0, 100000)
        panel.add_child(self._mss_edit)

        split_btn = gui.Button("Split instance")
        split_btn.set_on_clicked(self._on_split)
        panel.add_child(split_btn)

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

    def _on_layout(self, ctx):
        r  = self.window.content_rect
        pw = self._PANEL_WIDTH
        self.scene.frame  = gui.Rect(r.x, r.y, r.width - pw, r.height)
        self._panel.frame = gui.Rect(r.x + r.width - pw, r.y, pw, r.height)

    def _on_key(self, event):
        if event.type == gui.KeyEvent.DOWN:
            if event.key in (ord('b'), ord('B')):
                self._bg_cb.checked = not self._bg_cb.checked
                self._on_bg_toggle(self._bg_cb.checked)
                return True
            if event.key in (ord('r'), ord('R')):
                bounds = pcd_instances.get_axis_aligned_bounding_box()
                self.scene.setup_camera(60.0, bounds, bounds.get_center())
                return True
        return False

    def _on_bg_toggle(self, checked):
        self.scene.scene.show_geometry("bg", checked)

    def _on_rerun(self):
        mcs = self._mcs_edit.int_value
        ms  = self._ms_edit.int_value
        mip = self._mip_edit.int_value
        print(f"\nRe-running HDBSCAN (min_cluster_size={mcs}, "
              f"min_samples={ms}, min_instance_points={mip}) ...")

        labels, uniq, n_inst, n_noise = run_hdbscan(mcs, ms, mip)
        colors, pal = make_colors(labels, uniq, n_inst)

        self._base_labels = labels.copy()
        self._cur_labels  = labels
        self._cur_palette = pal
        self._cur_n_inst  = n_inst
        self._cur_n_noise = n_noise

        pcd_instances.colors = o3d.utility.Vector3dVector(colors)
        self.scene.scene.remove_geometry("instances")
        self.scene.scene.add_geometry("instances", pcd_instances, self._mat)

        self._update_stats()
        self._update_legend()

    def _on_split(self):
        target   = self._split_id_edit.int_value
        eps_2d   = self._eps_edit.double_value
        mp2d     = self._mp2d_edit.int_value
        mss      = self._mss_edit.int_value

        if target not in set(self._cur_labels):
            print(f"\nInstance {target} does not exist.")
            return

        print(f"\nSplitting instance {target} (eps_2d={eps_2d:.3f}, "
              f"min_pts_2d={mp2d}, min_split_size={mss}) ...")

        labels, uniq, n_inst, n_noise = split_parallel_pipes(
            self._cur_labels, eps_2d, mp2d, mss, target)
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

    def _on_save(self):
        mcs = self._mcs_edit.int_value
        ms  = self._ms_edit.int_value
        mip = self._mip_edit.int_value

        stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ply_path.parent / f"{ply_path.stem}_instances_{stamp}"
        out_dir.mkdir(exist_ok=True)

        instance_ids = sorted(uid for uid in np.unique(self._cur_labels) if uid >= 0)
        if not instance_ids:
            print("\nNo instances to save.")
            return

        cls1 = all_classes[mask1]
        rgb1 = all_rgb[mask1]

        for uid in instance_ids:
            inst_mask = self._cur_labels == uid
            inst_pts  = pts1[inst_mask]
            inst_rgb  = rgb1[inst_mask]
            inst_cls  = cls1[inst_mask]
            n = len(inst_pts)

            fname = out_dir / f"{ply_path.stem}_{TARGET_CLASS}_instance_{uid}.ply"
            with open(fname, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {n}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("property int class\n")
                f.write("end_header\n")
                for i in range(n):
                    f.write(f"{inst_pts[i, 0]:.6f} {inst_pts[i, 1]:.6f} "
                            f"{inst_pts[i, 2]:.6f} {inst_rgb[i, 0]} "
                            f"{inst_rgb[i, 1]} {inst_rgb[i, 2]} {inst_cls[i]}\n")

        print(f"\nSaved {len(instance_ids)} instance PLY files -> {out_dir}")
        print(f"  (min_cluster_size={mcs}  min_samples={ms}  "
              f"min_instance_points={mip})")

    def _update_stats(self):
        self._stats_lbl.text = (
            f"Total pts:       {len(data):,}\n"
            f"Class-{TARGET_CLASS} pts:    {len(pts1):,}\n"
            f"Instances found: {self._cur_n_inst}\n"
            f"Noise points:    {self._cur_n_noise:,}"
        )

    def _update_legend(self):
        rows = []
        for uid, col in sorted(self._cur_palette.items()):
            cnt = int((self._cur_labels == uid).sum())
            r, g, b = float(col[0]), float(col[1]), float(col[2])
            rows.append((f" inst {uid}: {cnt:,} pts", gui.Color(r, g, b)))
        if self._cur_n_noise:
            rows.append((f" noise: {self._cur_n_noise:,} pts",
                         gui.Color(0.4, 0.4, 0.4)))

        for i, slot in enumerate(self._legend_slots):
            if i < len(rows):
                slot.text       = rows[i][0]
                slot.text_color = rows[i][1]
            else:
                slot.text = ""

    def run(self):
        gui.Application.instance.run()


if __name__ == "__main__":
    viewer = InstanceViewer()
    viewer.run()
