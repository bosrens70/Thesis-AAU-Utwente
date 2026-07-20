"""
core/gui_helpers.py
─────────────────────────────────────────────────────────────────────────────
Shared Open3D GUI widget builders and camera helpers.

Single source of truth for the LER utility legend look-and-feel (the little
coloured swatch boxes and the swatch+label/checkbox rows) so every viewer
renders the legend identically, and for the camera moves (oblique / top-down
pivot, trench-framed top view) that every viewer binds to its shortcut keys.

Usage
-----
    from core.gui_helpers import make_color_swatch, make_legend_row, make_master_pipe_toggle

    # Toggleable layer row (swatch + checkbox):
    cb  = gui.Checkbox(f"{layer_name} ({n_feat})")
    row = make_legend_row(cfg["color"], cb, em)
    container.add_child(row)

    # Static legend row (swatch + label):
    row = make_legend_row(cfg["color"], gui.Label(f"{cls_id}: {name}"), em)

    # Master toggle for "All segments":
    all_pipes_cb = gui.Checkbox("All segments")
    callback = make_master_pipe_toggle(pipe_checkboxes, layer_visible,
                                       pipe_layer_meshes, scene_widget,
                                       pipe_gn, make_mesh_material,
                                       pipe_opacity, window)
    all_pipes_cb.set_on_checked(callback)
"""

import numpy as np
import open3d.visualization.gui as gui

from core.geometry import linear_to_srgb

# ── Swatch / row geometry — the single styling source ────────────────────────
SWATCH_LABEL              = " "      # single space → compact square
SWATCH_VERTICAL_PADDING   = 0.0      # em
SWATCH_HORIZONTAL_PADDING = 0.3      # em
ROW_SPACING_EM            = 0.3      # gui.Horiz internal spacing
ROW_GAP_EM                = 0.4      # fixed gap between swatch and widget


def make_color_swatch(color, *, srgb_convert=True):
    """
    Build the small flat coloured square used in every legend.

    Parameters
    ----------
    color : sequence of float
        (r, g, b[, a]) layer colour. By default the colour is assumed to be in
        *linear* space (as stored in the config layer dicts) and is converted
        to sRGB for display. Pass ``srgb_convert=False`` if the colour is
        already in sRGB space.

    Returns
    -------
    gui.Button
        A non-toggleable, tightly-padded button acting as a colour swatch.
    """
    r, g, b = color[0], color[1], color[2]
    if srgb_convert:
        r, g, b = (linear_to_srgb(c) for c in (r, g, b))
    swatch = gui.Button(SWATCH_LABEL)
    swatch.background_color      = gui.Color(r, g, b, 1.0)
    swatch.toggleable            = False
    swatch.vertical_padding_em   = SWATCH_VERTICAL_PADDING
    swatch.horizontal_padding_em = SWATCH_HORIZONTAL_PADDING
    return swatch


def make_legend_row(color, widget, em, *, srgb_convert=True):
    """
    Build one legend row: ``[swatch] [gap] [widget]``.

    Parameters
    ----------
    color : sequence of float
        Layer colour (see :func:`make_color_swatch`).
    widget : gui.Widget
        The trailing widget — a ``gui.Checkbox`` for toggleable layers, or a
        ``gui.Label`` for static legends.
    em : int
        The window's em size (``window.theme.font_size``) for scaling.
    srgb_convert : bool
        Forwarded to :func:`make_color_swatch`.

    Returns
    -------
    gui.Horiz
        The assembled row, ready to add to a container.
    """
    row = gui.Horiz(int(ROW_SPACING_EM * em))
    row.add_child(make_color_swatch(color, srgb_convert=srgb_convert))
    row.add_fixed(int(ROW_GAP_EM * em))
    row.add_child(widget)
    return row


# ── Master toggle helpers for "All segments" / "All components" ───────────────

def make_master_pipe_toggle(pipe_checkboxes, layer_visible, pipe_layer_meshes,
                             scene_widget, pipe_gn, make_mesh_material,
                             pipe_opacity, window):
    """
    Create a callback for the "All segments" master checkbox.

    Toggles all pipe layer checkboxes and their visibility in the scene.

    Parameters
    ----------
    pipe_checkboxes : list of (str, gui.Checkbox)
        List of (layer_key, checkbox) tuples for all pipe layers.
    layer_visible : dict
        Shared visibility state dict for all layers.
    pipe_layer_meshes : dict
        Shared mesh dict for all pipe layers (layer_key -> mesh).
    scene_widget : gui.SceneWidget
        The 3D view to update.
    pipe_gn : callable
        Function to generate geometry names from layer keys.
    make_mesh_material : callable
        Function to create mesh materials given an alpha value.
    pipe_opacity : list of float
        List containing current pipe opacity [alpha].
    window : gui.Window
        The window to trigger redraws.

    Returns
    -------
    callable
        Callback suitable for checkbox.set_on_checked().
    """
    def _on_toggle_all_pipes(checked):
        for ln, cb in pipe_checkboxes:
            cb.checked = checked
            layer_visible[ln] = checked
            if ln in pipe_layer_meshes:
                alpha = pipe_opacity[0] if checked else 0.0
                scene_widget.scene.modify_geometry_material(pipe_gn(ln),
                                                             make_mesh_material(alpha))
        window.post_redraw()
    return _on_toggle_all_pipes


def make_master_comp_toggle(comp_checkboxes, layer_visible, comp_layer_meshes,
                             scene_widget, comp_gn, make_mesh_material,
                             pipe_opacity, window):
    """
    Create a callback for the "All components" master checkbox.

    Toggles all component layer checkboxes and their visibility in the scene.

    Parameters
    ----------
    comp_checkboxes : list of (str, gui.Checkbox)
        List of (layer_name, checkbox) tuples for all component layers.
    layer_visible : dict
        Shared visibility state dict for all layers.
    comp_layer_meshes : dict
        Shared mesh dict for all component layers (layer_name -> mesh).
    scene_widget : gui.SceneWidget
        The 3D view to update.
    comp_gn : callable
        Function to generate geometry names from layer names.
    make_mesh_material : callable
        Function to create mesh materials given an alpha value.
    pipe_opacity : list of float
        List containing current pipe opacity [alpha].
    window : gui.Window
        The window to trigger redraws.

    Returns
    -------
    callable
        Callback suitable for checkbox.set_on_checked().
    """
    def _on_toggle_all_comps(checked):
        for ln, cb in comp_checkboxes:
            cb.checked = checked
            layer_visible[ln] = checked
            if ln in comp_layer_meshes:
                alpha = pipe_opacity[0] if checked else 0.0
                scene_widget.scene.modify_geometry_material(comp_gn(ln),
                                                             make_mesh_material(alpha))
        window.post_redraw()
    return _on_toggle_all_comps


# ── LER legend section ───────────────────────────────────────────────────────

class LerLegendSection:
    """The uniform LER legend block shared by every viewer (base_module look).

    Structure: a Ledningspakke master checkbox above a collapsible container
    holding the "LER opacity" slider row, the "All segments" /
    "All components" master checkboxes, and swatch+checkbox layer rows, in
    that order. Owns the look and ordering only; every callback (per layer,
    master, opacity) stays with the calling module, so viewer semantics
    differ only where they mean to.
    """

    def __init__(self, em, title, *, master_checked=True, opacity=1.0):
        self._em = em
        self.master_cb = gui.Checkbox(title)
        self.master_cb.checked = master_checked
        self.container = gui.Vert(int(0.3 * em))
        self.opacity_slider = gui.Slider(gui.Slider.DOUBLE)
        self.opacity_slider.set_limits(0.0, 1.0)
        self.opacity_slider.double_value = float(opacity)
        row = gui.Horiz(int(0.25 * em))
        row.add_child(gui.Label("LER opacity"))
        row.add_child(self.opacity_slider)
        self.container.add_child(row)

    def set_on_master(self, window, callback):
        """Wire the master checkbox: collapse/expand the legend, then run the
        module's geometry callback."""
        def _cb(checked):
            self.container.visible = checked
            callback(checked)
            window.set_needs_layout()
            window.post_redraw()
        self.master_cb.set_on_checked(_cb)

    def set_on_opacity(self, callback):
        self.opacity_slider.set_on_value_changed(callback)

    def _add_master(self, text, checked, callback):
        cb = gui.Checkbox(text)
        cb.checked = checked
        cb.set_on_checked(callback)
        self.container.add_child(cb)
        return cb

    def add_all_segments(self, checked, callback):
        return self._add_master("All segments", checked, callback)

    def add_all_components(self, checked, callback):
        return self._add_master("All components", checked, callback)

    def add_layer_row(self, color, text, checked, callback):
        """One swatch+checkbox layer row inside the legend; returns the checkbox."""
        cb = gui.Checkbox(text)
        cb.checked = checked
        cb.set_on_checked(callback)
        self.container.add_child(make_legend_row(color, cb, self._em))
        return cb

    def add_to(self, panel):
        """Attach the section (master checkbox + legend container) to a panel."""
        panel.add_child(self.master_cb)
        panel.add_child(self.container)


# ── Camera helpers ───────────────────────────────────────────────────────────
# Shared camera moves for the viewers' shortcut keys. Logging stays in the
# module wrappers (each viewer prints its own message, or none).

def pivot_oblique(scene_widget, point, scene_diag):
    """Recentre the camera on ``point`` from the standard oblique angle.

    ``scene_diag`` is the scene's 3D diagonal (e.g. ``norm(pc_max - pc_min)``);
    the eye sits at ``point + (d, -d, 0.6 d)`` with ``d = max(1, 0.6 * diag)``,
    up = +Z. Used by the perspective viewers (base, deviation).
    """
    point = np.asarray(point, dtype=float)
    d = max(1.0, float(scene_diag) * 0.6)
    eye = point + np.array([d, -d, d * 0.6])
    scene_widget.look_at(point.tolist(), eye.tolist(), [0.0, 0.0, 1.0])


def pivot_top_down(scene_widget, point, height):
    """Recentre the camera straight above ``point`` at ``height`` metres,
    up = +Y, keeping a top-down orientation (label, ERR)."""
    cx, cy, cz = (float(point[0]), float(point[1]), float(point[2]))
    scene_widget.look_at([cx, cy, cz], [cx, cy, cz + float(height)], [0.0, 1.0, 0.0])


def trench_or_scene_frame(trench_path, cloud_centroid, pc_min, pc_max, *,
                          trench_z=None):
    """Return ``(cx, cy, cz, span)`` framing the trench footprint when one is
    defined, otherwise the whole scene.

    ``trench_path`` is the matplotlib Path from ``load_trench`` (or ``None``).
    ``trench_z`` overrides the look-at height in the trench branch (the
    deviation viewer passes the ground level); the default is the cloud
    centroid Z. Feed the result to :func:`top_view`.
    """
    if trench_path is not None:
        v = np.asarray(trench_path.vertices, dtype=float)
        cx, cy = float(v[:, 0].mean()), float(v[:, 1].mean())
        # np.ptp(): the ndarray .ptp() method was removed in NumPy 2.0
        span = max(float(np.ptp(v[:, 0])), float(np.ptp(v[:, 1])))
        cz = float(cloud_centroid[2]) if trench_z is None else float(trench_z)
    else:
        cx, cy = float(cloud_centroid[0]), float(cloud_centroid[1])
        span = max(float(pc_max[0] - pc_min[0]), float(pc_max[1] - pc_min[1]))
        cz = float(cloud_centroid[2])
    return cx, cy, cz, span


def top_view(scene_widget, cx, cy, cz, span):
    """Bird's-eye view looking straight down at ``(cx, cy, cz)`` from a height
    of ``1.2 * span`` (at least 1.2 m), up = +Y."""
    h = max(1.0, span) * 1.2
    scene_widget.look_at([cx, cy, cz], [cx, cy, cz + h], [0.0, 1.0, 0.0])
