# -*- coding: utf-8 -*-
"""
LER line signatures as Open3D geometry.
=======================================
The Open3D half of core/symbology.py, the way core/trace_render.py is the
Open3D half of core/ledningstrace.py. core/symbology.py places the signatures
of the LER "Signaturforklaring" (dashed line, danger triangles, El voltage
ticks) as plain numpy arrays and stays headless-testable; the meshes are built
here, once, for the top-down plan of ERR_module and for the 3D viewers alike.
The two must not fork: a signature that means one thing in the plan and another
in a 3D scene is worse than no signature.

The plan and a 3D viewer draw the same feature differently, so the signatures
have to be re-expressed rather than copied:

* the plan flattens every feature to one Z and builds the whole feature here,
  line and decorators together, as flat ribbons;
* a 3D viewer builds the line itself, one clipped segment at a time, keeping
  each one aligned with its picking arrays. It therefore takes the dash a
  segment at a time (``PolylineDash``, ``line_segment_mesh``) and the
  decorators per feature, lifted clear of the tube instead of sitting inside
  it.

Marker sizes come from core/config.py: SIGNATURE_* for the plan, SIGNATURE_3D_*
for the 3D viewers, where a marker tuned to a 0.10 m plan ribbon would swallow
a real ~0.01 m pipe.
"""

import numpy as np
import open3d as o3d

from core.config import (
    SIGNATURE_TICK_BAR_WIDTH_M, SIGNATURE_TICK_COLOR, SIGNATURE_HAZARD_COLOR,
    SIGNATURE_3D_DASH_LEN, SIGNATURE_3D_GAP_LEN,
    SIGNATURE_3D_TICK_SPACING, SIGNATURE_3D_TICK_LEN, SIGNATURE_3D_TICK_GAP,
    SIGNATURE_3D_TICK_BAR_WIDTH_M,
    SIGNATURE_3D_HAZARD_SPACING, SIGNATURE_3D_HAZARD_SIZE,
    SIGNATURE_Z_LIFT,
)
from core.geometry import segment_to_cylinder, segment_to_plane
from core import symbology as sym


# ─────────────────────────────────────────────────────────────────────────────
# MESH BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def segments_to_ribbon_mesh(points, lines, width, color):
    """Flat horizontal ribbon quads (two triangles per segment) for the line
    segments in (points, lines). ``width`` is the ribbon width in metres."""
    pts = np.asarray(points, dtype=float)
    lines = np.asarray(lines, dtype=int)
    if len(lines) == 0:
        return None
    hw = width / 2.0
    up = np.array([0.0, 0.0, 1.0])
    verts, tris = [], []
    for a, b in lines:
        p0 = pts[a]
        p1 = pts[b]
        fwd = p1 - p0
        n = np.linalg.norm(fwd)
        if n < 1e-9:
            continue
        side = np.cross(fwd / n, up)
        sn = np.linalg.norm(side)
        side = side / sn * hw if sn > 1e-9 else np.array([hw, 0.0, 0.0])
        k = len(verts)
        verts.extend([p0 + side, p0 - side, p1 - side, p1 + side])
        tris.extend([[k, k + 1, k + 2], [k, k + 2, k + 3]])
    if not verts:
        return None
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=float))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
    m.paint_uniform_color(color)
    m.compute_vertex_normals()
    return m


def faces_to_mesh(verts, faces, color):
    """Build a coloured TriangleMesh from vertex/face arrays (danger triangles)."""
    faces = np.asarray(faces, dtype=int)
    if len(faces) == 0:
        return None
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=float))
    m.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    m.paint_uniform_color(color)
    m.compute_vertex_normals()
    return m


def merge_meshes(meshes):
    """Merge a list of TriangleMesh into one, or None when the list is empty.

    Merging per layer keeps the scene at one draw call per layer, which is what
    every viewer's visibility and opacity handling assumes. Accumulates into a
    fresh mesh rather than into ``meshes[0]``: ``+=`` mutates its left operand,
    so seeding from the first element would leave the caller's list holding the
    whole layer in its first slot, and a second merge would double it.
    """
    meshes = [m for m in meshes if m is not None]
    if not meshes:
        return None
    merged = o3d.geometry.TriangleMesh()
    for m in meshes:
        merged += m
    merged.compute_vertex_normals()
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# PLAN VIEW (ERR_module top-down)
# ─────────────────────────────────────────────────────────────────────────────
def feature_signature_meshes_plan(coords, style, color, width, hazard, tick_count):
    """Ribbon + decorators for one polyline already flattened to the plan Z."""
    coords = np.asarray(coords, dtype=float)
    meshes = []
    if len(coords) < 2:
        return meshes
    if style == "dashed":
        p, l = sym.dash_segments(coords)
    else:
        p, l = sym.polyline_lines(coords)
    base = segments_to_ribbon_mesh(p, l, width, color)
    if base is not None:
        meshes.append(base)
    if tick_count > 0:
        tp, tl = sym.tick_bars(coords, tick_count)
        tm = segments_to_ribbon_mesh(tp, tl, SIGNATURE_TICK_BAR_WIDTH_M, SIGNATURE_TICK_COLOR)
        if tm is not None:
            meshes.append(tm)
    if hazard:
        hv, hf = sym.triangle_markers(coords)
        hm = faces_to_mesh(hv, hf, SIGNATURE_HAZARD_COLOR)
        if hm is not None:
            meshes.append(hm)
    return meshes


# ─────────────────────────────────────────────────────────────────────────────
# 3D VIEWERS (base / label / deviation / agent)
# ─────────────────────────────────────────────────────────────────────────────
class PolylineDash:
    """The dash pattern of one polyline, for driftsstatus "under etablering".

    Built once per polyline: resolved per segment the pattern would restart at
    every vertex, and a dash that straddles a vertex would be lost. The phase
    lives in arc length along the whole line, and each drawn segment takes only
    the part of it that falls on a dash.

    ``coords`` is the axis the viewer draws on, so a pipe passes its polyline
    already lowered by one radius (the registered Z is the crown, not the axis)
    and a Ledningstrace passes its corridor centreline unchanged.
    """

    def __init__(self, coords, dash_len=SIGNATURE_3D_DASH_LEN,
                 gap_len=SIGNATURE_3D_GAP_LEN):
        self.coords = np.asarray(coords, dtype=float)
        self.cum, self.spans = sym.polyline_dash_spans(self.coords, dash_len, gap_len)

    def segment_mesh(self, index, p1, p2, color, radius=0.0, width=None):
        """One mesh for one drawn segment, holding only its dashed parts.

        ``index`` is the segment's position in the polyline and ``p1``/``p2``
        its endpoints, already clipped to the viewer's crop. A segment that
        falls entirely in a gap returns an EMPTY mesh rather than None, so the
        caller's mesh list stays one to one with its picking and depth-source
        arrays; None is reserved for a degenerate segment, matching what
        segment_to_cylinder and segment_to_plane return for one.
        """
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        d = p2 - p1
        seg = float(np.linalg.norm(d))
        if seg < 1e-6:
            return None
        mesh = o3d.geometry.TriangleMesh()
        if index < 0 or index + 1 >= len(self.cum) or len(self.spans) == 0:
            return mesh
        # The chord lies on segment `index`, so its Euclidean length is also its
        # arc length and the two map onto each other linearly.
        s0 = float(self.cum[index]) + float(np.linalg.norm(p1 - self.coords[index]))
        s1 = s0 + seg
        for a, b in self.spans:
            lo, hi = max(a, s0), min(b, s1)
            if hi - lo <= 1e-9:
                continue
            q1 = p1 + d * ((lo - s0) / seg)
            q2 = p1 + d * ((hi - s0) / seg)
            part = _line_chord_mesh(q1, q2, color, radius, width)
            if part is not None:
                mesh += part
        return mesh


def _line_chord_mesh(p1, p2, color, radius, width):
    """One chord of a line, built the way the viewer builds the utility itself:
    a flat corridor ribbon where a width is registered ("bredde"), a tube
    otherwise."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    if width is not None:
        return segment_to_plane(p1, p2, width, color)
    return segment_to_cylinder(p1, p2, radius, color)


def line_segment_mesh(p1, p2, color, radius=0.0, width=None, dash=None, index=0):
    """The drawn form of one clipped segment of an LER line.

    Solid: exactly the tube or corridor ribbon the viewer would build anyway.
    Dashed: the same segment with its gaps cut out, as one mesh, so the caller
    keeps emitting one mesh per segment and its parallel arrays stay aligned.
    """
    if dash is not None:
        return dash.segment_mesh(index, p1, p2, color, radius=radius, width=width)
    return _line_chord_mesh(p1, p2, color, radius, width)


def stitch_clipped_segments(chords, tol=1e-6):
    """Chain crop-clipped chords back into polylines.

    Marker spacing is measured along the line, so a feature the crop cut into
    pieces has to be reassembled per surviving piece: measured on each chord
    separately, the pattern would restart at every vertex.
    """
    pieces, cur = [], []
    for a, b in chords:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if not cur:
            cur = [a, b]
        elif np.allclose(cur[-1], a, atol=tol):
            cur.append(b)
        else:
            pieces.append(np.asarray(cur, dtype=float))
            cur = [a, b]
    if cur:
        pieces.append(np.asarray(cur, dtype=float))
    return pieces


def feature_signature_meshes_3d(coords, color, hazard=False, tick_count=0,
                                radius=0.0, width=None):
    """Signature decorators for one polyline of a 3D scene.

    ``coords`` is the axis the viewer drew the utility on, in local viewer
    coordinates: the tube axis of a pipe (already lowered by its radius, since
    the registered Z is the crown, not the axis) or the centreline of a
    Ledningstrace corridor. ``width`` is the corridor width for a trace and
    ``None`` for a pipe, and only decides the lift: a flat ribbon has no body
    above its axis, a tube has one radius of it.

    Everything is placed on the registered crown line itself, which is where LER
    draws it: the lift cancels the body the viewer drew below that line, leaving
    only SIGNATURE_Z_LIFT to keep the marker off it. Open3D 0.19 has no
    depth_func to force an overlay to the front, so that margin has to be
    geometric.

    Returns a list of meshes. The utility's own line is never among them, the
    dashed style included: a dash is a property of the line, so the viewer gets
    it from ``line_segment_mesh`` while it builds that line. Keeping the two
    apart is what lets the decorators hold their fixed legend colours through
    every recolouring of the utilities.
    """
    coords = np.asarray(coords, dtype=float)
    meshes = []
    if len(coords) < 2:
        return meshes

    lift = (0.0 if width is not None else float(radius)) + SIGNATURE_Z_LIFT

    if tick_count > 0:
        tp, tl = sym.tick_bars(coords, tick_count,
                               spacing=SIGNATURE_3D_TICK_SPACING,
                               tick_len=SIGNATURE_3D_TICK_LEN,
                               tick_gap=SIGNATURE_3D_TICK_GAP,
                               z_lift=lift)
        tm = segments_to_ribbon_mesh(tp, tl, SIGNATURE_3D_TICK_BAR_WIDTH_M,
                                     SIGNATURE_TICK_COLOR)
        if tm is not None:
            meshes.append(tm)
    if hazard:
        hv, hf = sym.triangle_markers(coords, spacing=SIGNATURE_3D_HAZARD_SPACING,
                                      size=SIGNATURE_3D_HAZARD_SIZE, z_lift=lift)
        hm = faces_to_mesh(hv, hf, SIGNATURE_HAZARD_COLOR)
        if hm is not None:
            meshes.append(hm)
    return meshes


# ─────────────────────────────────────────────────────────────────────────────
# SCENE PLUMBING (one convention for every 3D viewer)
# ─────────────────────────────────────────────────────────────────────────────
def signature_gn(storage_key):
    """Scene geometry name for a layer's signature overlay. One convention for
    every viewer, so the material and visibility updates below can find it."""
    return f"ler_signature_{storage_key}"


def add_signature_meshes(scene, meshes, base_alpha, material_fn,
                         visible_of=None, signatures_on=True):
    """Add every layer's signature overlay to ``scene``.

    ``visible_of`` is an optional callable ``key -> bool``; a key that is not
    visible is added at zero opacity, matching how the viewers hide a layer.
    A trace's overlay is added at the unscaled opacity, like its centreline
    tube: the transparency of a corridor ribbon is there to stop it hiding what
    runs beneath, which a legend marker does not do.
    """
    for key, mesh in meshes.items():
        if mesh is None:
            continue
        alpha = base_alpha
        if visible_of is not None and not visible_of(key):
            alpha = 0.0
        if not signatures_on:
            alpha = 0.0
        scene.add_geometry(signature_gn(key), mesh, material_fn(alpha))


def set_signature_material(scene, storage_key, base_alpha, material_fn,
                           signatures_on=True):
    """Apply ``base_alpha`` to one layer's signature overlay, or hide it when
    the signatures are switched off. Silently skips a layer that has none, which
    lets every viewer call this from each of its material-update sites."""
    gn = signature_gn(storage_key)
    if scene.has_geometry(gn):
        scene.modify_geometry_material(
            gn, material_fn(base_alpha if signatures_on else 0.0))


def show_signatures(scene, storage_key, visible, signatures_on=True):
    """Show or hide one layer's signature overlay, for the viewers that switch
    visibility with show_geometry rather than with opacity."""
    gn = signature_gn(storage_key)
    if scene.has_geometry(gn):
        scene.show_geometry(gn, bool(visible and signatures_on))
