# -*- coding: utf-8 -*-
"""
The LER "Signaturforklaring" as a collapsible panel section.
============================================================
The utility legend in core/gui_helpers.py is a colour legend: one flat coloured
square per layer. It can say what an Elledning is coloured and nothing else, so
the signatures this project draws (a dashed line for driftsstatus
"under etablering", red triangles for fareklasse "meget farlig", El voltage tick
bars) have no entry there. They cannot be given one either, because they are not
per layer: "meget farlig" appears across Gas, El, Foeringsroer, Trace and
Termisk, so it is a property of a feature rather than of a layer. That is why
LER itself gives them a second, collapsible legend, and why this module builds
one instead of extending the first.

Every symbol is rasterised from core/symbology.py, the same generators that
build the geometry in the scene, and every label is derived from the rule
constants in core/config.py. So the legend cannot drift from the rendering:
change VOLTAGE_KV_THRESHOLDS or SIGNATURE_HAZARD_VALUES and the rows follow. A
legend drawn by hand would keep claiming the old bins.

The strip is drawn on a light card, the way LER prints it, which is also what
lets the neutral rows read as black on a dark panel.
"""

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui

from core.config import (
    LINE_LAYERS,
    SIGNATURE_DASH_LEN, SIGNATURE_GAP_LEN,
    SIGNATURE_TICK_SPACING, SIGNATURE_TICK_LEN, SIGNATURE_TICK_GAP,
    SIGNATURE_HAZARD_SPACING, SIGNATURE_HAZARD_SIZE,
    SIGNATURE_3D_DASH_LEN, SIGNATURE_3D_GAP_LEN,
    SIGNATURE_3D_TICK_SPACING, SIGNATURE_3D_TICK_LEN, SIGNATURE_3D_TICK_GAP,
    SIGNATURE_3D_HAZARD_SPACING, SIGNATURE_3D_HAZARD_SIZE,
    SIGNATURE_TICK_BAR_WIDTH_M, SIGNATURE_3D_TICK_BAR_WIDTH_M,
    SIGNATURE_TICK_COLOR, SIGNATURE_HAZARD_COLOR,
    SIGNATURE_HAZARD_VALUES, SIGNATURE_DASH_DRIFTSSTATUS,
    DRIFTSSTATUS_DISPLAY_EN, FAREKLASSE_DISPLAY_EN,
    VOLTAGE_KV_THRESHOLDS,
    SIGNATURE_LEGEND_TITLE, SIGNATURE_LEGEND_BG, SIGNATURE_LEGEND_INK,
    SIGNATURE_LEGEND_TRACE_INK,
    SIGNATURE_LEGEND_SWATCH_W_EM, SIGNATURE_LEGEND_SWATCH_H_EM,
)
from core.geometry import linear_to_srgb
from core import symbology as sym


# The two size sets the project draws with. A swatch is a schematic, so what it
# has to preserve is the proportion between mark and gap, not the metre values.
STYLE_3D = {
    "dash_len": SIGNATURE_3D_DASH_LEN,   "gap_len": SIGNATURE_3D_GAP_LEN,
    "tick_spacing": SIGNATURE_3D_TICK_SPACING, "tick_len": SIGNATURE_3D_TICK_LEN,
    "tick_gap": SIGNATURE_3D_TICK_GAP,   "tick_bar_w": SIGNATURE_3D_TICK_BAR_WIDTH_M,
    "hazard_spacing": SIGNATURE_3D_HAZARD_SPACING, "hazard_size": SIGNATURE_3D_HAZARD_SIZE,
}
STYLE_PLAN = {
    "dash_len": SIGNATURE_DASH_LEN,      "gap_len": SIGNATURE_GAP_LEN,
    "tick_spacing": SIGNATURE_TICK_SPACING, "tick_len": SIGNATURE_TICK_LEN,
    "tick_gap": SIGNATURE_TICK_GAP,      "tick_bar_w": SIGNATURE_TICK_BAR_WIDTH_M,
    "hazard_spacing": SIGNATURE_HAZARD_SPACING, "hazard_size": SIGNATURE_HAZARD_SIZE,
}


def _srgb8(color, *, convert=False):
    """A colour as a uint8 RGB triple. ``convert`` for the linear layer colours."""
    c = [linear_to_srgb(v) for v in color[:3]] if convert else list(color[:3])
    return np.asarray([int(round(255 * min(max(v, 0.0), 1.0))) for v in c], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# RASTERISER
# ─────────────────────────────────────────────────────────────────────────────
# Small enough (a few thousand pixels per swatch) that a plain per-primitive
# pass over the pixel grid is cheaper to read than a scanline fill.
class _Swatch:
    """A pixel canvas with a world-to-pixel mapping, so the symbology arrays can
    be drawn straight into it in the metre units they are generated in."""

    def __init__(self, w, h, length, bg):
        self.img = np.empty((h, w, 3), dtype=np.uint8)
        self.img[:, :] = _srgb8(bg)
        self.w, self.h = w, h
        pad = 1.0
        self.scale = (w - 2 * pad) / float(length)
        self.x0 = pad
        self.ys, self.xs = np.mgrid[0:h, 0:w]
        self.px = self.xs + 0.5
        self.py = self.ys + 0.5

    def to_px(self, xy):
        """World (x, y) to pixel (x, y); y is centred and points up."""
        x, y = float(xy[0]), float(xy[1])
        return self.x0 + x * self.scale, self.h / 2.0 - y * self.scale

    def stroke(self, p0, p1, width_px, color):
        ax, ay = self.to_px(p0)
        bx, by = self.to_px(p1)
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        if l2 < 1e-12:
            t = np.zeros_like(self.px)
        else:
            t = np.clip(((self.px - ax) * dx + (self.py - ay) * dy) / l2, 0.0, 1.0)
        d = np.hypot(self.px - (ax + t * dx), self.py - (ay + t * dy))
        self.img[d <= max(width_px, 1.0) / 2.0] = color

    def fill_triangle(self, tri, color):
        (ax, ay), (bx, by), (cx, cy) = (self.to_px(v) for v in tri)
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            return
        u = ((by - cy) * (self.px - cx) + (cx - bx) * (self.py - cy)) / den
        v = ((cy - ay) * (self.px - cx) + (ax - cx) * (self.py - cy)) / den
        self.img[(u >= 0) & (v >= 0) & (u + v <= 1)] = color

    def fill_rect(self, x0, y0, x1, y1, color):
        (ax, ay), (bx, by) = self.to_px((x0, y0)), self.to_px((x1, y1))
        lo_x, hi_x = sorted((ax, bx))
        lo_y, hi_y = sorted((ay, by))
        self.img[(self.px >= lo_x) & (self.px <= hi_x) &
                 (self.py >= lo_y) & (self.py <= hi_y)] = color

    def fill_disc(self, centre, radius_px, color):
        cx, cy = self.to_px(centre)
        self.img[np.hypot(self.px - cx, self.py - cy) <= radius_px] = color


def _polyline(length):
    """The straight sample line every symbol is generated along."""
    return np.array([[0.0, 0.0, 0.0], [float(length), 0.0, 0.0]])


def render_swatch(kind, w, h, style, *, ticks=0, line_px=2.0, bg=SIGNATURE_LEGEND_BG):
    """Rasterise one legend symbol. ``kind`` names the row, not the layer."""
    # Three marker intervals across the swatch: enough repeats to read as a
    # pattern, few enough that each mark stays legible at panel size.
    length = 3.0 * style["hazard_spacing"]
    sw = _Swatch(w, h, length, bg)
    coords = _polyline(length)
    ink = _srgb8(SIGNATURE_LEGEND_INK)

    if kind == "line":
        sw.stroke((0, 0), (length, 0), line_px, ink)
    elif kind == "dashed":
        p, l = sym.dash_segments(coords, dash_len=style["dash_len"],
                                 gap_len=style["gap_len"])
        for a, b in l:
            sw.stroke(p[a][:2], p[b][:2], line_px, ink)
    elif kind == "trace":
        sw.stroke((0, 0), (length, 0), max(h * 0.45, line_px),
                  _srgb8(SIGNATURE_LEGEND_TRACE_INK))
    elif kind == "hazard":
        # Triangles only, with no line under them: the row explains the marker,
        # and the line it rides on is already the "Ledning" row above.
        hv, hf = sym.triangle_markers(coords, spacing=style["hazard_spacing"],
                                      size=style["hazard_size"], z_lift=0.0)
        for f in hf:
            sw.fill_triangle([hv[i][:2] for i in f], _srgb8(SIGNATURE_HAZARD_COLOR))
    elif kind == "el":
        sw.stroke((0, 0), (length, 0), line_px,
                  _srgb8(LINE_LAYERS["Elledning"]["color"], convert=True))
        if ticks > 0:
            tp, tl = sym.tick_bars(coords, ticks, spacing=style["tick_spacing"],
                                   tick_len=style["tick_len"],
                                   tick_gap=style["tick_gap"], z_lift=0.0)
            # Bar thickness comes from the same constant the scene draws with,
            # not from the generic line width: at the generic width a 2- or
            # 3-tick group closes up into one blob and stops being countable,
            # which is the only thing these rows exist to show.
            bar_px = style["tick_bar_w"] * sw.scale
            for a, b in tl:
                sw.stroke(tp[a][:2], tp[b][:2], bar_px, _srgb8(SIGNATURE_TICK_COLOR))
    elif kind == "comp_point":
        sw.fill_disc((length / 2.0, 0.0), max(h * 0.18, 2.0), ink)
    elif kind == "comp_polygon":
        q = length / 4.0
        y = length * 0.055
        sw.fill_rect(q, -y, length - q, y, ink)          # border
        sw.fill_rect(q + 1.0 / sw.scale, -y + 1.0 / sw.scale,
                     length - q - 1.0 / sw.scale, y - 1.0 / sw.scale,
                     _srgb8(SIGNATURE_LEGEND_TRACE_INK))  # fill
    elif kind == "comp_line":
        sw.stroke((0, 0), (length, 0), line_px * 2.0, ink)
    else:
        raise ValueError(f"unknown legend row kind: {kind}")
    return sw.img


# ─────────────────────────────────────────────────────────────────────────────
# ROWS
# ─────────────────────────────────────────────────────────────────────────────
# The El rows name the layer in short form: LAYER_DISPLAY_EN's "Electricity
# Cable" plus a voltage band overruns the panel at its 20 em width.
EL_ROW_NAME = "Electricity"


def voltage_row_labels(thresholds=VOLTAGE_KV_THRESHOLDS):
    """The El legend bins, written out from the thresholds that classify them.

    ``[1, 30, 131]`` gives "Electricity  < 1 kV", "Electricity  1 kV - 29 kV",
    "Electricity  30 kV - 130 kV", "Electricity  > 131 kV", which is what the
    LER legend prints. Derived rather than typed out, so retuning the
    thresholds relabels the legend.
    """
    t = [float(x) for x in thresholds]
    if not t:
        return [EL_ROW_NAME]
    out = [f"{EL_ROW_NAME}  < {t[0]:g} kV"]
    for i in range(1, len(t)):
        out.append(f"{EL_ROW_NAME}  {t[i - 1]:g} kV - {t[i] - 1:g} kV")
    out.append(f"{EL_ROW_NAME}  > {t[-1]:g} kV")
    return out


def legend_rows(components="point"):
    """The rows of the legend, as ``(kind, ticks, label)``.

    The labels are English, but translated from the rule constants through the
    display tables in core/config.py rather than typed out, so a retuned rule
    still cannot disagree with the label that explains it.

    ``components`` is what the viewer actually draws for a component: "plan"
    for the ERR top view, which draws a point, a polygon and a line, three
    forms the colour legend cannot tell apart. The 3D viewers draw every
    component as a sphere, so their single row would explain no form that the
    utility legend above does not already carry in colour, and they get none.
    """
    dash_label = DRIFTSSTATUS_DISPLAY_EN.get(SIGNATURE_DASH_DRIFTSSTATUS,
                                             SIGNATURE_DASH_DRIFTSSTATUS)
    hazard_label = " / ".join(FAREKLASSE_DISPLAY_EN.get(v, v)
                              for v in SIGNATURE_HAZARD_VALUES)
    rows = [
        ("line",   0, "Utility line"),
        ("dashed", 0, dash_label.capitalize()),
        ("trace",  0, "Trace"),
        ("hazard", 0, f"Hazard class: {hazard_label}"),
    ]
    if components == "plan":
        rows += [("comp_point",   0, "Component point"),
                 ("comp_polygon", 0, "Component polygon"),
                 ("comp_line",    0, "Component line")]
    rows += [("el", i, lbl) for i, lbl in enumerate(voltage_row_labels())]
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# PANEL SECTION
# ─────────────────────────────────────────────────────────────────────────────
class SignatureLegendSection:
    """The "Signaturforklaring" block: a collapsible list of symbol + label rows.

    Collapsed by default, like the LER legend it reproduces, so it costs no
    panel height until the user opens it.
    """

    def __init__(self, em, *, components="point", style=None, is_open=False,
                 title=SIGNATURE_LEGEND_TITLE):
        style = STYLE_3D if style is None else style
        self._em = em
        w = int(SIGNATURE_LEGEND_SWATCH_W_EM * em)
        h = int(SIGNATURE_LEGEND_SWATCH_H_EM * em)
        line_px = max(round(h * 0.13), 2.0)
        # Zero spacing between rows, so the light cards abut into one strip
        # rather than reading as a column of separate chips.
        self.widget = gui.CollapsableVert(title, 0,
                                          gui.Margins(int(0.25 * em), 0, 0, 0))
        self.widget.set_is_open(bool(is_open))
        self._images = []
        for kind, ticks, label in legend_rows(components):
            img = o3d.geometry.Image(
                render_swatch(kind, w, h, style, ticks=ticks, line_px=line_px))
            self._images.append(img)          # keep alive: the widget does not own it
            row = gui.Horiz(0)
            row.add_child(gui.ImageWidget(img))
            row.add_fixed(int(0.5 * em))
            row.add_child(gui.Label(label))
            self.widget.add_child(row)

    def add_to(self, panel):
        panel.add_child(self.widget)
        return self.widget
