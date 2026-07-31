# -*- coding: utf-8 -*-
"""
Ledningstrace rendering: transparent corridor + solid centreline.
=================================================================
A trace is a corridor, not a pipe: the registry gives a centreline and a width
("bredde"). Drawn as one opaque ribbon it hides whatever runs beneath it, so
every viewer draws a trace in two parts instead:

* the corridor ribbon, at ``LEDNINGSTRACE_ALPHA_SCALE`` of the current LER
  opacity, so the utilities below stay readable through it;
* the centreline, built as a thin tube and drawn through the same lit mesh
  material as every other utility, so it follows the LER opacity slider, the
  layer toggles, and the scene lighting with no special case.

The split lives here so all viewers behave identically. The pure-logic half
(``is_trace_key`` / ``ribbon_alpha``) stays in core/ledningstrace.py, which is
deliberately Open3D-free; this module builds geometry and therefore is not.
"""

import numpy as np

from core.config import TRACE_CENTERLINE_RADIUS
from core.geometry import segment_to_cylinder
from core.ledningstrace import is_trace_key, ribbon_alpha


def trace_centerline_gn(storage_key):
    """Scene geometry name for a trace's centreline. One convention for every
    viewer, so the material updates below can find it."""
    return f"trace_centerline_{storage_key}"


def build_trace_centerlines(seg_p1, seg_p2, seg_layer, color_of,
                            radius=TRACE_CENTERLINE_RADIUS,
                            color_of_index=None):
    """Build one merged centreline tube mesh per Ledningstrace storage key.

    Reads the per-segment arrays the viewers already keep for picking, so no
    build loop has to change:

    ``seg_p1`` / ``seg_p2``  (N, 3) segment endpoints in local coordinates.
    ``seg_layer``            length-N sequence of storage keys.
    ``color_of``             callable ``key -> [r, g, b]``.
    ``radius``               display radius; never derived from "bredde".
    ``color_of_index``       optional callable ``segment index -> [r, g, b]``,
                             overriding ``color_of``. For colourings that vary
                             per feature rather than per layer (e.g. by
                             registered accuracy class), where one layer's
                             centreline is no longer one colour.

    Returns ``{storage_key: TriangleMesh}``, empty when there are no traces.
    """
    if len(seg_p1) == 0 or len(seg_p2) == 0:
        return {}

    p1 = np.asarray(seg_p1, dtype=float)
    p2 = np.asarray(seg_p2, dtype=float)

    by_key = {}
    for i, key in enumerate(seg_layer):
        if not is_trace_key(key):
            continue
        if i >= len(p1) or i >= len(p2):
            break
        # The centreline sits on the registered line itself, exactly where the
        # ribbon plane sits, so the two coincide rather than one floating above
        # the other.
        col = color_of_index(i) if color_of_index is not None else color_of(key)
        cyl = segment_to_cylinder(p1[i], p2[i], radius, col)
        if cyl is not None:
            by_key.setdefault(key, []).append(cyl)

    meshes = {}
    for key, cyls in by_key.items():
        merged = cyls[0]
        for c in cyls[1:]:
            merged += c
        merged.compute_vertex_normals()
        meshes[key] = merged
    return meshes


def add_trace_centerlines(scene, centerlines, base_alpha, material_fn,
                          visible_of=None):
    """Add every centreline mesh to ``scene`` with the shared mesh material.

    ``visible_of`` is an optional callable ``key -> bool``; a key that is not
    visible is added at zero opacity, matching how the viewers hide a layer.
    """
    for key, mesh in centerlines.items():
        alpha = base_alpha
        if visible_of is not None and not visible_of(key):
            alpha = 0.0
        scene.add_geometry(trace_centerline_gn(key), mesh, material_fn(alpha))


def set_layer_material(scene, ribbon_gn, storage_key, base_alpha, material_fn):
    """Apply ``base_alpha`` to one LER layer, honouring the trace split.

    The filled geometry (pipe tube, or a trace's corridor ribbon) gets
    ``ribbon_alpha``, so only traces are made extra transparent; a trace's
    centreline gets the unscaled opacity, so it reads like any other utility.
    Geometry that a viewer does not have is skipped, which lets every viewer
    call this from each of its material-update sites.
    """
    if scene.has_geometry(ribbon_gn):
        scene.modify_geometry_material(
            ribbon_gn, material_fn(ribbon_alpha(storage_key, base_alpha)))
    cgn = trace_centerline_gn(storage_key)
    if scene.has_geometry(cgn):
        scene.modify_geometry_material(cgn, material_fn(base_alpha))
