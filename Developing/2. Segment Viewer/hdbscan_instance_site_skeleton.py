# -*- coding: utf-8 -*-
"""
HDBSCAN Instance Segmentation — Skeleton-based split
=====================================================
Same HDBSCAN workflow as hdbscan_instance_site.py, but replaces the
PCA-based pipe split with 3D skeletonization:

  1. Voxelize the instance into a binary 3D grid.
  2. Apply morphological skeletonization (scikit-image skeletonize).
  3. Find connected components in the skeleton — each component is a
     separate pipe/cable.
  4. Assign every original point to its nearest skeleton component.

This handles curved, sagging, and arbitrarily shaped pipes without
any linearity assumption.

Tunable split parameters:
    Skel voxel size  — resolution of the voxel grid for skeletonization.
                       Smaller = more detail but slower. Must be large
                       enough that individual pipes don't merge in the grid.
    Min split size   — sub-instances smaller than this become noise.
"""

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import KDTree
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from pathlib import Path
import time
import matplotlib.colors as mcolors
from datetime import datetime
from skimage.morphology import skeletonize, ball, binary_closing
from scipy.ndimage import label as ndimage_label, binary_dilation

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PLY_FILE = (
    r"C:\Users\bosre\OneDrive - University of Twente\Documents\AAU UTwente thesis"
    r"\Python\Thesis\Data\OpenTrench3D\Water_Area_5\Area_5_Site_05.ply"
)

PLY_HEADER_ROWS = 11
CLASS_COLUMN    = 6
TARGET_CLASS    = 1       # "Other Utility" (third-party utilities)

VOXEL_SIZE  = 0.01        # metres — downsample before HDBSCAN clustering

# Initial HDBSCAN parameters (editable in the GUI)
MIN_CLUSTER_SIZE = 100
MIN_SAMPLES      = 5

POINT_SIZE          = 2.0
MIN_INSTANCE_POINTS = 250   # instances smaller than this become noise

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
print(f"  Voxel downsample ({VOXEL_SIZE} m): {len(pts1):,} → {len(pts1_ds):,} points\n")

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


def split_skeleton(labels, skel_voxel, dilation_r, min_split_size, target_instance):
    """Split a specific instance using 3D skeletonization.

    1. Build a binary voxel grid from the instance points.
    2. Dilate + close to fill hollow pipe interiors into solid volumes.
    3. Skeletonize the solid grid (morphological thinning).
    4. Find connected components in the skeleton.
    5. Assign each original point to its nearest skeleton component.
    """
    new_labels = labels.copy()
    next_id = int(labels.max()) + 1 if labels.max() >= 0 else 0

    uid = target_instance
    inst_mask = labels == uid
    inst_pts  = pts1[inst_mask]
    n_pts = len(inst_pts)
    if n_pts < 10:
        print(f"  instance {uid}: too few points ({n_pts}), skipping")
        return _renumber_labels(new_labels)

    t = time.perf_counter()

    # Build binary voxel grid with padding for dilation
    pad = dilation_r + 2
    origin = inst_pts.min(axis=0) - skel_voxel * pad
    ijk = ((inst_pts - origin) / skel_voxel).astype(int)
    grid_shape = ijk.max(axis=0) + 1 + pad
    grid = np.zeros(grid_shape, dtype=np.uint8)
    grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1

    print(f"  Voxel grid: {grid_shape} ({grid.sum():,} filled voxels, "
          f"voxel size {skel_voxel:.3f}m)")

    # Dilate to fill hollow pipe interiors, then close to smooth
    struct = ball(dilation_r)
    grid = binary_dilation(grid, structure=struct).astype(np.uint8)
    grid = binary_closing(grid, footprint=ball(max(1, dilation_r // 2))).astype(np.uint8)
    print(f"  After dilation (r={dilation_r}) + closing: {grid.sum():,} filled voxels")

    # Skeletonize
    skeleton = skeletonize(grid)
    n_skel = int(skeleton.sum())
    print(f"  Skeleton: {n_skel} voxels in {time.perf_counter() - t:.2f}s")

    if n_skel == 0:
        print(f"  Empty skeleton — try a smaller voxel size")
        return _renumber_labels(new_labels)

    # Connected components in the skeleton
    skel_labels, n_branches = ndimage_label(skeleton)
    print(f"  {n_branches} skeleton branch(es)")

    if n_branches <= 1:
        print(f"  Only 1 branch — nothing to split")
        return _renumber_labels(new_labels)

    # Get 3D coordinates of each skeleton branch
    skel_coords = np.argwhere(skeleton > 0).astype(float) * skel_voxel + origin
    skel_branch_ids = skel_labels[skeleton > 0]

    # Assign each original point to its nearest skeleton voxel,
    # then inherit that voxel's branch label
    skel_tree = KDTree(skel_coords)
    nn_idx = skel_tree.query(inst_pts, k=1, return_distance=False).ravel()
    point_branches = skel_branch_ids[nn_idx]

    # Drop branches smaller than min_split_size
    for bid in range(1, n_branches + 1):
        if (point_branches == bid).sum() < min_split_size:
            point_branches[point_branches == bid] = 0  # 0 = noise

    valid_branches = sorted(set(point_branches) - {0})
    n_valid = len(valid_branches)
    if n_valid <= 1:
        print(f"  After size filter: {n_valid} branch(es) — nothing to split")
        return _renumber_labels(new_labels)

    indices = np.where(inst_mask)[0]
    for rank, bid in enumerate(valid_branches):
        bmask = point_branches == bid
        if rank == 0:
            new_labels[indices[bmask]] = uid
        else:
            new_labels[indices[bmask]] = next_id
            next_id += 1
    new_labels[indices[point_branches == 0]] = -1

    print(f"  instance {uid} split into {n_valid} sub-instances")
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
            f"HDBSCAN Instance Viewer (skeleton) — {ply_path.name} ", 1440, 900)

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

        split_title = gui.Label("Split instance (skeleton)")
        split_title.text_color = gui.Color(1.0, 1.0, 1.0)
        panel.add_child(split_title)

        panel.add_child(gui.Label("Instance to split:"))
        self._split_id_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self._split_id_edit.int_value = 0
        self._split_id_edit.set_limits(0, 999)
        panel.add_child(self._split_id_edit)

        panel.add_child(gui.Label("Skel voxel size (m):"))
        self._skel_vox_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._skel_vox_edit.double_value = 0.02
        self._skel_vox_edit.set_limits(0.005, 0.5)
        panel.add_child(self._skel_vox_edit)

        panel.add_child(gui.Label("Dilation radius (voxels):"))
        self._dil_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self._dil_edit.int_value = 3
        self._dil_edit.set_limits(1, 20)
        panel.add_child(self._dil_edit)

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
        target    = self._split_id_edit.int_value
        skel_vox  = self._skel_vox_edit.double_value
        dil_r     = self._dil_edit.int_value
        mss       = self._mss_edit.int_value

        if target not in set(self._cur_labels):
            print(f"\nInstance {target} does not exist.")
            return

        print(f"\nSplitting instance {target} — skeleton "
              f"(voxel={skel_vox:.3f}m, dilation_r={dil_r}, "
              f"min_split_size={mss}) ...")

        labels, uniq, n_inst, n_noise = split_skeleton(
            self._cur_labels, skel_vox, dil_r, mss, target)
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

        print(f"\nSaved {len(instance_ids)} instance PLY files → {out_dir}")
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
