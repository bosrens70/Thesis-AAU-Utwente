"""
core/gui_helpers.py
─────────────────────────────────────────────────────────────────────────────
Shared Open3D GUI widget builders.

Single source of truth for the LER utility legend look-and-feel (the little
coloured swatch boxes and the swatch+label/checkbox rows) so every viewer
renders the legend identically.

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
