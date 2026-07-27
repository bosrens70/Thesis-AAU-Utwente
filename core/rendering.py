# -*- coding: utf-8 -*-
"""
Shared Open3D rendering helpers.
================================
Centralises the material and scene-lighting setup that the viewers had been
duplicating. The two point-cloud helpers are intentionally separate because the
viewers make a deliberate choice based on how the cloud is coloured:

* ``point_material_shaded`` (defaultLit) for clouds coloured categorically
  (class / cluster / instance id). The sun light + SSAO act as an EDL-like depth
  cue that helps read 3D shape; colour fidelity does not matter for label colours.
* ``point_material_flat`` (defaultUnlit) for clouds whose colour carries meaning
  (raw RGB or a continuous metric). Colours render flat and faithful, the way a
  2D viewer such as QGIS shows them; lighting would distort the measured colour.

The remaining helpers (mesh / line / flat) cover geometry that every viewer
already rendered identically.
"""

import open3d.visualization.rendering as rendering


# ── Point clouds ─────────────────────────────────────────────────────────────
def point_material_shaded(point_size=3.0):
    """Lit point material (defaultLit) for categorically coloured clouds.

    Use when the cloud is coloured by class / cluster / instance id and you want
    the sun-light + SSAO depth cue rather than colour fidelity.
    """
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.point_size = float(point_size)
    return mat


def point_material_flat(point_size=3.0):
    """Unlit point material (defaultUnlit) for clouds whose colour carries meaning.

    Use for raw RGB or a continuous metric: colours render flat and faithful,
    with no sun-light shading or lighting-driven brightening.
    """
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = float(point_size)
    return mat


# ── Meshes and overlays ──────────────────────────────────────────────────────
def mesh_material(alpha=1.0):
    """Lit, transparent mesh material (defaultLitTransparency).

    Used for LER tubes / planes / buffers. The white base colour preserves the
    mesh's vertex colours; ``alpha`` drives the opacity sliders.

    The shader's default roughness of 1.0 is fully matte, so a tube shows no
    highlight and its round cross-section is hard to read. A moderate roughness
    with a little reflectance gives a soft directional sheen that, under the
    oblique sun of ``setup_scene_lighting``, makes the roundness and relative
    depth of the tubes visible. Metallic stays 0 and the base colour stays
    white, so the utility colours are not tinted.
    """
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency"
    mat.base_color = [1.0, 1.0, 1.0, float(alpha)]
    mat.base_metallic = 0.0
    mat.base_roughness = 0.65
    mat.base_reflectance = 0.4
    return mat


def line_material(width=2.0):
    """Unlit line material (unlitLine) for wireframe overlays: crop box, trench,
    bounding boxes, axes, pipe wireframes."""
    mat = rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.line_width = float(width)
    return mat


def flat_material():
    """Unlit material (defaultUnlit) for non-point geometry that should ignore
    lighting: coordinate frames and pick markers."""
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    return mat


# ── Scene lighting ───────────────────────────────────────────────────────────
def setup_scene_lighting(scene, post_processing=None,
                         sun_dir=(-0.304, -0.406, -0.862),
                         sun_color=(1.0, 1.0, 1.0),
                         sun_intensity=90000,
                         indirect_intensity=40000):
    """Configure consistent scene lighting for a viewer.

    ``scene`` is the viewer's ``rendering.Open3DScene``. The sun light is set and
    enabled (used by the lit materials). ``sun_dir`` points down but is tilted
    about 30 degrees from vertical, so a horizontal tube shows a light-to-dark
    gradient across its cross-section (rather than the uniform top a straight-down
    sun gives in a top-down view), which is what makes the roundness readable.
    The light is fixed in world space, so shading stays steady as the user orbits.

    ``indirect_intensity`` lowers the even ambient (image-based) fill relative to
    the directional sun, so the sun's shading is not washed out; both values are
    set explicitly so the balance does not depend on Open3D's default. Pass
    ``None`` to leave the indirect light untouched.

    ``post_processing`` toggles SSAO + tonemapping: pass ``True``/``False`` to set
    it explicitly, or leave it ``None`` to keep Open3D's default (so a viewer that
    never touched it is not changed). Safe to call before any geometry is added.
    """
    if post_processing is not None:
        try:
            scene.view.set_post_processing(bool(post_processing))
        except Exception:
            pass
    try:
        scene.scene.set_sun_light(list(sun_dir), list(sun_color), float(sun_intensity))
        scene.scene.enable_sun_light(True)
    except Exception:
        pass
    if indirect_intensity is not None:
        try:
            scene.scene.set_indirect_light_intensity(float(indirect_intensity))
        except Exception:
            pass
