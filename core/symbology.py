# -*- coding: utf-8 -*-
"""
LER line signatures as plain geometry arrays.
=============================================
Reproduces the cartographic line styles of the LER "Signaturforklaring"
(solid / dashed lines, danger-class triangles, El voltage tick marks) as plain
numpy point / line / triangle arrays. The graveforesp top-view builds Open3D
``LineSet`` / ``TriangleMesh`` objects from these arrays.

This module is numpy only (no open3d import) so the geometry generators and the
attribute rules can be unit-tested headless, the way ``core.ler_las_export`` is.

Sizes are in metres (world-scaled): the signatures grow and shrink with the
scene, matching the utilities they annotate.
"""

import numpy as np

from core.config import (
    SIGNATURE_DASH_LEN, SIGNATURE_GAP_LEN,
    VOLTAGE_KV_THRESHOLDS,
    SIGNATURE_TICK_SPACING, SIGNATURE_TICK_LEN, SIGNATURE_TICK_GAP,
    SIGNATURE_HAZARD_VALUES, SIGNATURE_HAZARD_SPACING, SIGNATURE_HAZARD_SIZE,
    SIGNATURE_Z_LIFT,
)

_UP = np.array([0.0, 0.0, 1.0])


# ─────────────────────────────────────────────────────────────────────────────
# ATTRIBUTE RULES: LER columns -> signature choice
# ─────────────────────────────────────────────────────────────────────────────
def line_style_from_driftsstatus(value):
    """Return ``"dashed"`` for driftsstatus "under etablering", else ``"solid"``."""
    return "dashed" if "under etablering" in str(value or "").strip().lower() else "solid"


def is_hazard(fareklasse_value, trigger=SIGNATURE_HAZARD_VALUES):
    """True when the fareklasse value is one of the danger classes that get the
    red triangle signature (by default only "meget farlig")."""
    s = str(fareklasse_value or "").strip().lower()
    return s in {t.strip().lower() for t in trigger}


def voltage_tick_count(spaendingsniveau_kv, thresholds=VOLTAGE_KV_THRESHOLDS):
    """Number of tick-mark groups (0..3) for an El voltage in kV.

    With the default thresholds ``[1, 30, 131]`` the legend bins map as
    ``< 1 -> 0``, ``1-29 -> 1``, ``30-130 -> 2``, ``>= 131 -> 3``. Missing /
    non-numeric values give 0.
    """
    if spaendingsniveau_kv is None:
        return 0
    try:
        v = float(spaendingsniveau_kv)
    except (TypeError, ValueError):
        return 0
    if np.isnan(v):
        return 0
    return int(np.searchsorted(np.asarray(thresholds, dtype=float), v, side="right"))


# ─────────────────────────────────────────────────────────────────────────────
# POLYLINE ARC-LENGTH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _arc_length(coords):
    """Return (cumulative arc length per vertex, total length)."""
    d = np.diff(coords, axis=0)
    seglen = np.linalg.norm(d, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    return cum, float(cum[-1])


def _point_and_dir(coords, cum, s):
    """Position and unit forward direction at arc length ``s`` along the polyline."""
    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = min(max(i, 0), len(coords) - 2)
    seg = cum[i + 1] - cum[i]
    t = 0.0 if seg <= 1e-12 else (s - cum[i]) / seg
    p = coords[i] * (1.0 - t) + coords[i + 1] * t
    fwd = coords[i + 1] - coords[i]
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
    return p, fwd


def _stations(total, spacing):
    """Evenly spaced arc-length stations centred within the polyline (first at
    half a spacing so short lines still get one marker)."""
    if total <= 1e-9 or spacing <= 1e-9:
        return np.empty(0)
    return np.arange(spacing * 0.5, total, spacing)


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY GENERATORS  (return plain numpy arrays)
# ─────────────────────────────────────────────────────────────────────────────
def polyline_lines(coords):
    """Solid line: (points (N,3), lines (N-1,2)) for a polyline."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return np.empty((0, 3)), np.empty((0, 2), dtype=int)
    lines = np.column_stack([np.arange(len(coords) - 1), np.arange(1, len(coords))])
    return coords, lines.astype(int)


def dash_segments(coords, dash_len=SIGNATURE_DASH_LEN, gap_len=SIGNATURE_GAP_LEN):
    """Dashed line ("under etablering"): (points (2K,3), lines (K,2)).

    Dash chords are sampled by arc length, so a dash may slightly cut a bend;
    dashes are short, so the error is negligible.
    """
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return np.empty((0, 3)), np.empty((0, 2), dtype=int)
    cum, total = _arc_length(coords)
    if total <= 1e-9:
        return np.empty((0, 3)), np.empty((0, 2), dtype=int)
    period = max(dash_len + gap_len, 1e-6)
    pts, lines, s = [], [], 0.0
    while s < total:
        s0, s1 = s, min(s + dash_len, total)
        if s1 - s0 > 1e-6:
            p0, _ = _point_and_dir(coords, cum, s0)
            p1, _ = _point_and_dir(coords, cum, s1)
            k = len(pts)
            pts.extend([p0, p1])
            lines.append([k, k + 1])
        s += period
    if not pts:
        return np.empty((0, 3)), np.empty((0, 2), dtype=int)
    return np.asarray(pts, dtype=float), np.asarray(lines, dtype=int)


def tick_bars(coords, n_ticks, spacing=SIGNATURE_TICK_SPACING,
              tick_len=SIGNATURE_TICK_LEN, tick_gap=SIGNATURE_TICK_GAP,
              z_lift=SIGNATURE_Z_LIFT):
    """El voltage ticks: groups of ``n_ticks`` short perpendicular bars along the
    line. Returns (points (2T,3), lines (T,2)). ``n_ticks <= 0`` -> empty.
    """
    coords = np.asarray(coords, dtype=float)
    if n_ticks <= 0 or len(coords) < 2:
        return np.empty((0, 3)), np.empty((0, 2), dtype=int)
    cum, total = _arc_length(coords)
    offs = (np.arange(n_ticks) - (n_ticks - 1) / 2.0) * tick_gap
    pts, lines = [], []
    for s in _stations(total, spacing):
        for o in offs:
            sc = min(max(s + o, 0.0), total)
            p, fwd = _point_and_dir(coords, cum, sc)
            side = np.cross(fwd, _UP)
            n = np.linalg.norm(side)
            side = side / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
            a = p + side * (tick_len / 2.0)
            b = p - side * (tick_len / 2.0)
            a[2] += z_lift
            b[2] += z_lift
            k = len(pts)
            pts.extend([a, b])
            lines.append([k, k + 1])
    if not pts:
        return np.empty((0, 3)), np.empty((0, 2), dtype=int)
    return np.asarray(pts, dtype=float), np.asarray(lines, dtype=int)


def triangle_markers(coords, spacing=SIGNATURE_HAZARD_SPACING,
                     size=SIGNATURE_HAZARD_SIZE, z_lift=SIGNATURE_Z_LIFT):
    """Danger-class triangles: flat triangles (apex along the line) at intervals.
    Returns (vertices (3M,3), faces (M,3)).
    """
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    cum, total = _arc_length(coords)
    verts, faces = [], []
    for s in _stations(total, spacing):
        p, fwd = _point_and_dir(coords, cum, s)
        side = np.cross(fwd, _UP)
        n = np.linalg.norm(side)
        side = side / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        apex = p + fwd * (size * 0.6)
        base1 = p - fwd * (size * 0.3) + side * (size * 0.5)
        base2 = p - fwd * (size * 0.3) - side * (size * 0.5)
        k = len(verts)
        for v in (apex, base1, base2):
            v = v.copy()
            v[2] += z_lift
            verts.append(v)
        faces.append([k, k + 1, k + 2])
    if not verts:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)
