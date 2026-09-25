# -*- coding: utf-8 -*-
"""
Report figures for the area-wide deviation analysis.
====================================================
Reads only the CSV written by ``tools/deviation_sweep.py``, so a figure can be
redrawn without re-running the sweep, and whatever is handed to someone else is
exactly what the figures are drawn from.

Deliberately free of Open3D and of ``core.config``'s site resolution: the
palettes are imported from ``core.config`` (colours must mean the same thing here
as on screen) but no point cloud is touched, so this runs in the base env, where
matplotlib's savefig is known to work.

The measurement being plotted, in one paragraph. LER registers the horizontal
centreline of a utility carrying the Z of its **top**. The crown line is the
measured top of the same utility. Comparing the two is therefore top against
top, with no pipe-radius bias, which is why every figure here uses the crown
columns and not the raw per-point ones.

**The horizontal component is the primary metric.** Two reasons, and both are
about what the numbers can support rather than about presentation:

1. ``noejagtighedsklasse`` is defined as the *horizontal* accuracy of the
   centreline. ``noejagtighedsklasseVertikal`` exists in the datamodel but is
   absent from this package's data. Testing a 3D Euclidean deviation against a
   horizontal class compares a quantity against a bound that does not govern it.
2. Most features here carry no registered Z, so ``core/depth.py`` synthesises one.
   A 3D distance to such a feature contains a vertical component that was derived
   rather than registered, and in the ``GROUND_PLANE`` case derived from this very
   point cloud. The horizontal component is free of that.

So the 3D and vertical figures are gated on ``z_provenance``, while the horizontal
figures use every analysed instance.

Groups that are never pooled:

* **utility lines** (``Vandledning``, ``Elledning``, ...), where the registered
  geometry is the utility's own centreline;
* **Ledningstrace corridors**, where the registered geometry is a corridor with a
  width. Deviation to a corridor means "how far outside the registered corridor",
  a different quantity from distance to a registered centreline, and ``bredde`` is
  a corridor width rather than a utility cross-section;
* **same-campaign features**, whose registered Z was surveyed by the project that
  captured these point clouds. They agree with the cloud to within tens of
  micrometres, which is not independent agreement but the same survey appearing on
  both sides. Reported as a photogrammetry check, never as register accuracy.
  See SAME_SURVEY_CAMPAIGN in core/config.py.

Usage (from the project root, base env):
    python tools/build_deviation_figures.py
    python tools/build_deviation_figures.py --csv figures/Water_Area_5_deviation.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.config import (DEVIATION_THRESHOLDS, DEVIATION_COLORS,
                         DEVIATION_CLASS_LABELS, LINE_LAYERS, KLIC_XY_THRESHOLDS,
                         layer_display_name,
                         LEDNINGSTRACE_LAYER, SAME_SURVEY_CAMPAIGN,
                         UTILITY_TYPE_COLORS, UTILITY_TYPE_LABELS)

MM = 1000.0          # metres -> millimetres, for display only
DPI = 200
GREY = "#888888"

# Same typography as the other report figures (tools/plot_ler_depth.py,
# tools/plot_prescribed_depths.py), so a reader turning the page does not
# change typeface.
FONT = "Arial"
plt.rcParams.update({"font.family": FONT, "font.size": 10})

# No figure here carries a title or an explanatory note. Every figure in the
# report is described by its LaTeX caption, and a caption baked into the
# image cannot be edited with the text, sets its own typeface and size, and
# prints twice when the float is captioned as well. Panel labels and axis
# labels stay: they identify what is plotted rather than describe it.

# Which (owner, establishment date) pairs are not independent of the capture is
# a property of the package, not of this script, so it lives in config.py with
# the reasoning that identified it.

# A vertical statement needs a registered Z coordinate. VEJLEDENDE is weaker but
# not circular: vejledendeDybde is a registered indicative depth to the top, so it
# tests a registered claim, only anchored to a ground level taken from the cloud.
# The featurekatalog notes the accuracy class does not apply to it, so it is
# reported apart from REGISTERED and never merged into it.
Z_INDEPENDENT = "registered"
Z_INDICATIVE = "fallback"     # with depth_sources == VEJLEDENDE


def is_same_campaign(row):
    return (row.get("owner", ""), row.get("etableringstidspunkt", "")) in SAME_SURVEY_CAMPAIGN


# ─────────────────────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _f(row, key):
    """A float from a CSV cell, NaN for the empty 'not measured' cell."""
    v = (row.get(key) or "").strip()
    if not v:
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def load(csv_path):
    """The analysis set: instances with a human-confirmed exclusive match and a
    recoverable crown line.

    Both filters are about what a row can support rather than about tidiness. A
    ``nearest_of_type`` row was never confirmed by anyone, so it may be measured
    against an unrelated utility that merely sits nearby. A row with no crown
    line has no measured top to compare against the registered top. Non-pipe
    fragments (a valve, a tiewrap) fail the crown test on their own, because they
    are not tubes.
    """
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    keep, dropped = [], {"not exclusive": 0, "no crown line": 0}
    for r in rows:
        if r.get("match_kind") != "exclusive":
            dropped["not exclusive"] += 1
            continue
        if str(r.get("crown_ok")) != "1":
            dropped["no crown line"] += 1
            continue
        r["_is_trace"] = str(r.get("is_trace")) == "1"
        r["_cls"] = int(r.get("declared_class_idx") or 0)
        r["_bound"] = _f(r, "declared_bound_m")
        for k in ("crown_mean_m", "crown_median_m", "crown_p95_m", "crown_max_m",
                  "crown_xy_mean_m", "crown_xy_p95_m", "crown_xy_max_m",
                  "crown_z_max_m", "crown_z_signed_median_m",
                  "crown_z_signed_p5_m", "crown_z_signed_p95_m",
                  "measured_diameter_m", "run_length_m"):
            r["_" + k] = _f(r, k)
        # Vertical eligibility, decided once here so no figure has to re-derive it.
        r["_z_reg"] = r.get("z_provenance") == Z_INDEPENDENT
        r["_z_indicative"] = (r.get("z_provenance") == Z_INDICATIVE
                              and r.get("depth_sources") == "VEJLEDENDE")
        r["_same_campaign"] = is_same_campaign(r)
        keep.append(r)
    return rows, keep, dropped


def split(keep):
    """(utility-line rows, corridor rows). Never pooled: see the module docstring."""
    return ([r for r in keep if not r["_is_trace"]],
            [r for r in keep if r["_is_trace"]])


def _save(fig, out_dir, name):
    path = Path(out_dir) / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# F1  DISTRIBUTION OF THE CROWN DEVIATION
# ─────────────────────────────────────────────────────────────────────────────

def _class_axis(ax, lim):
    """A dotted line at each accuracy-class bound inside the x-range, and ticks
    every 250 mm. Replaces the coloured bands, so the class of a bar stays
    readable in the plain style."""
    for t in DEVIATION_THRESHOLDS[1:]:
        if 0 < t * MM < lim:
            ax.axvline(t * MM, color=GREY, ls=":", lw=0.9, zorder=1)
    ax.set_xticks(np.arange(0, lim + 1, 250))


def _panel_distribution(rows, lim, bins, out_dir, name):
    """One group of the distribution figure, drawn as a figure of its own.

    ``lim`` and ``bins`` are handed in rather than derived here: the two files
    are read against one another as subfigures of one float, so a bar in one
    must cover the same span of millimetres as a bar in the other. Nothing in
    the image names the group, because the subcaption does.
    """
    fig, ax = plt.subplots(figsize=(9.2, 3.3))
    v = np.array([r["_crown_xy_max_m"] for r in rows]) * MM
    v = v[np.isfinite(v)]
    ax.set_xlim(0, lim)
    _class_axis(ax, lim)
    ax.set_ylabel("instances")
    ax.set_xlabel("horizontal (XY) crown deviation from the registered "
                  "geometry, max per instance (mm)")
    ax.grid(axis="y", alpha=0.25, zorder=1)
    if v.size == 0:
        ax.text(0.5, 0.45, "no instances", ha="center", color=GREY,
                transform=ax.transAxes, fontsize=10)
    else:
        ax.hist(v, bins=bins, color="#1f6fb4", edgecolor="white", lw=0.5, zorder=2)
        med = float(np.median(v))
        ax.axvline(med, color="black", ls="--", lw=1.2, zorder=3)
        # White backing: the corridor median is 0 mm, which puts the label on
        # top of the tallest bar.
        ax.annotate(f"median {med:.0f} mm", (med, ax.get_ylim()[1] * 0.92),
                    xytext=(6, 0), textcoords="offset points", fontsize=9,
                    va="top", bbox=dict(fc="white", ec="none", pad=1.5))
    # No class bands behind the bars: the report's histograms share one plain
    # style with the signed-Z figure, and the classes are read off the
    # conformance figure instead.
    fig.tight_layout()
    return _save(fig, out_dir, name)


def fig_distribution(lines, traces, out_dir):
    """How the measured crown deviations are distributed.

    Two files: utility lines against a registered centreline, corridors against
    a registered corridor whose half-width is subtracted first. Never pooled,
    for the reason given in ``split``. They are the two subfigures of one float
    in the report.

    The horizontal component is plotted because noejagtighedsklasse bounds the
    horizontal accuracy of the centreline and no vertical class is registered
    in this package.
    """
    groups = [(lines, "deviation_distribution_lines.png"),
              (traces, "deviation_distribution_traces.png")]
    top = max([r["_crown_xy_max_m"] * MM for g, _ in groups for r in g
               if np.isfinite(r["_crown_xy_max_m"])] or [100])
    # End the axis on the class bound that holds the largest value, so no class
    # appears as a sliver at the edge, and bin in 25 mm steps so no bar
    # straddles a class bound.
    bounds = [t * MM for t in DEVIATION_THRESHOLDS[1:]]
    lim = next((b for b in bounds if b >= top), top * 1.02)
    bins = np.arange(0, lim + 25, 25)
    return [_panel_distribution(rows, lim, bins, out_dir, name)
            for rows, name in groups]


# Legend names for the observed utility types. Presentation only: the data
# carries the integer code from UTILITY_TYPE_LABELS.
_TYPE_NAME = {1: "Power", 2: "Drainage", 3: "Oil", 4: "Gas", 5: "Thermal",
              6: "Conduit", 7: "Water", 8: "Telecommunication", 9: "Other",
              10: "Unknown service type", 0: "Unlabelled"}


def _panel_distribution_by_type(rows, lim, bins, types, out_dir, name):
    """The distribution panel with each bar split into the observed utility
    types of the instances in it. ``types`` fixes the stacking order and the
    colours across both files, so a colour means the same type in each."""
    fig, ax = plt.subplots(figsize=(9.2, 3.3))
    ax.set_xlim(0, lim)
    _class_axis(ax, lim)
    ax.set_ylabel("instances")
    ax.set_xlabel("horizontal (XY) crown deviation from the registered "
                  "geometry, max per instance (mm)")
    ax.grid(axis="y", alpha=0.25, zorder=1)
    stacks, colors, labels = [], [], []
    for t in types:
        v = np.array([r["_crown_xy_max_m"] for r in rows
                      if int(r["utility_type"]) == t]) * MM
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        stacks.append(v)
        colors.append(UTILITY_TYPE_COLORS[t])
        labels.append(f"{_TYPE_NAME.get(t, UTILITY_TYPE_LABELS[t])} ({v.size})")
    ax.hist(stacks, bins=bins, stacked=True, color=colors, label=labels,
            edgecolor="white", lw=0.5, zorder=2)
    allv = np.concatenate(stacks)
    med = float(np.median(allv))
    ax.axvline(med, color="black", ls="--", lw=1.2, zorder=3)
    ax.annotate(f"median {med:.0f} mm", (med, ax.get_ylim()[1] * 0.92),
                xytext=(6, 0), textcoords="offset points", fontsize=9,
                va="top", bbox=dict(fc="white", ec="none", pad=1.5))
    ax.legend(fontsize=8, loc="upper right", title="Observed utility type",
              alignment="left", title_fontsize=8, framealpha=0.95)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def fig_distribution_by_type(lines, traces, out_dir):
    """The distribution figure again, with each bar split by observed utility
    type. Same x-range and bins as ``fig_distribution``, so the two sets can be
    read against each other."""
    groups = [(lines, "deviation_distribution_lines_by_type.png"),
              (traces, "deviation_distribution_traces_by_type.png")]
    top = max([r["_crown_xy_max_m"] * MM for g, _ in groups for r in g
               if np.isfinite(r["_crown_xy_max_m"])] or [100])
    bounds = [t * MM for t in DEVIATION_THRESHOLDS[1:]]
    lim = next((b for b in bounds if b >= top), top * 1.02)
    bins = np.arange(0, lim + 25, 25)
    counts = {}
    for g, _ in groups:
        for r in g:
            counts[int(r["utility_type"])] = counts.get(int(r["utility_type"]), 0) + 1
    types = sorted(counts, key=lambda t: -counts[t])
    return [_panel_distribution_by_type(rows, lim, bins, types, out_dir, name)
            for rows, name in groups]


# ─────────────────────────────────────────────────────────────────────────────
# F2  MEASURED DEVIATION AGAINST THE DECLARED ACCURACY CLASS
# ─────────────────────────────────────────────────────────────────────────────

def fig_conformance(lines, traces, out_dir):
    """The conformance figure: what the owner promised against what the trench
    shows. One row per declared class, the declared bound drawn as the bar the
    measurement has to stay inside."""
    rows = lines + traces
    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    classes = [c for c in range(1, len(DEVIATION_COLORS) + 1)
               if any(r["_cls"] == c for r in rows)]
    unreg = [r for r in rows if r["_cls"] == 0]
    ypos, labels, n_bust = [], [], 0

    for i, c in enumerate(classes):
        group = [r for r in rows if r["_cls"] == c]
        y = len(classes) - i
        ypos.append(y)
        bound = DEVIATION_THRESHOLDS[c] * MM if c < len(DEVIATION_THRESHOLDS) else None
        is_open = (c == len(DEVIATION_COLORS))
        labels.append(f"{DEVIATION_CLASS_LABELS[c - 1]}\n(n = {len(group)})")

        if is_open:
            # Right-hand margin in axes fractions: the class has points across the
            # whole left of the row, so a data-space label would sit on top of them.
            ax.annotate("no upper bound registered", (0.985, y),
                        xycoords=("axes fraction", "data"), fontsize=8.5,
                        color=GREY, va="center", ha="right", style="italic")
        else:
            # Neutral band: colour is reserved for the utility type of a point.
            ax.barh(y, bound, height=0.62, color=GREY, alpha=0.15, lw=0, zorder=1)
            ax.plot([bound, bound], [y - 0.31, y + 0.31], color="black",
                    lw=1.6, ls="--", zorder=4)
            ax.annotate(f"{bound:.0f}", (bound, y + 0.34), fontsize=8,
                        ha="center", va="bottom")

        v = np.array([r["_crown_xy_max_m"] for r in group]) * MM
        col = np.array([UTILITY_TYPE_COLORS[int(r["utility_type"])] for r in group])
        ok = np.isfinite(v)
        v, col = v[ok], col[ok]
        jitter = (np.random.default_rng(7 + c).random(v.size) - 0.5) * 0.34
        over = np.zeros(v.size, dtype=bool) if bound is None else v > bound
        n_bust += int(over.sum())
        ax.scatter(v[~over], y + jitter[~over], s=17, c=col[~over],
                   edgecolor="white", lw=0.4, zorder=5)
        if over.any():
            ax.scatter(v[over], y + jitter[over], s=42, marker="X",
                       color="black", zorder=6,
                       label="exceeds its declared class" if n_bust else None)
        if v.size:
            ax.plot([np.median(v)] * 2, [y - 0.33, y + 0.33], color="black",
                    lw=2.2, zorder=7)

    if unreg:
        y = 0
        ypos.append(y)
        labels.append(f"not registered\n(n = {len(unreg)})")
        v = np.array([r["_crown_xy_max_m"] for r in unreg]) * MM
        col = np.array([UTILITY_TYPE_COLORS[int(r["utility_type"])] for r in unreg])
        ok = np.isfinite(v)
        v, col = v[ok], col[ok]
        jitter = (np.random.default_rng(3).random(v.size) - 0.5) * 0.34
        ax.scatter(v, y + jitter, s=17, c=col, edgecolor="white", lw=0.4, zorder=5)

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("measured horizontal (XY) crown deviation, max per instance (mm)")
    ax.set_ylabel("accuracy class declared in LER (noejagtighedsklasse)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    counts = {}
    for r in rows:
        counts[int(r["utility_type"])] = counts.get(int(r["utility_type"]), 0) + 1
    type_handles = [Line2D([], [], marker="o", ls="", markersize=5,
                           color=UTILITY_TYPE_COLORS[t],
                           label=f"{_TYPE_NAME.get(t, UTILITY_TYPE_LABELS[t])} ({counts[t]})")
                    for t in sorted(counts, key=lambda t: -counts[t])]
    handles = type_handles + [
        Line2D([], [], color="black", lw=2.2, label="Group median"),
        Line2D([], [], color="black", lw=1.6, ls="--", label="Declared bound")]
    if n_bust:
        handles.append(Line2D([], [], marker="X", ls="", color="black",
                              markersize=7, label="Exceeds its declared class"))
    # Upper right, because the open ">2.00 m" class sits on the bottom row and
    # carries the "no upper bound" note in that margin; the empty right halves
    # of the class-2 and class-3 rows leave room for the full key.
    ax.legend(handles=handles, fontsize=8, loc="upper right",
              title="Observed utility type", alignment="left",
              title_fontsize=8, framealpha=0.95)
    # For the caption: the instance count and the number exceeding the class
    # they declare are both printed by report(), so the caption can quote them
    # without reading them off the image. noejagtighedsklasse bounds
    # horizontal accuracy, so the XY component is the quantity it governs.
    return _save(fig, out_dir, "deviation_vs_declared_class.png")


# ─────────────────────────────────────────────────────────────────────────────
# F3  DEVIATION BY UTILITY TYPE
# ─────────────────────────────────────────────────────────────────────────────

def _layer_color(layer):
    """The project colour for an LER layer, corridors included."""
    base = layer.split(" (")[0]
    cfg = LINE_LAYERS.get(base)
    if cfg and base != LEDNINGSTRACE_LAYER:
        return cfg["color"]
    # A corridor takes the colour of the utility it carries, so a telecom
    # corridor reads as telecom rather than as a generic trace.
    from core.config import forsyningsart_color, trace_forsyningsart
    fa = trace_forsyningsart(layer)
    return forsyningsart_color(fa or "", LINE_LAYERS[LEDNINGSTRACE_LAYER]["color"])


def fig_by_layer(lines, traces, out_dir):
    groups = []
    for rows, kind in ((lines, "line"), (traces, "trace")):
        by = {}
        for row in rows:
            by.setdefault(row["ler_layer"], []).append(row["_crown_xy_max_m"] * MM)
        for layer, vals in by.items():
            v = np.array([x for x in vals if np.isfinite(x)])
            if v.size:
                groups.append((layer, v, kind))
    # Utility lines first, then corridors; within each, worst median at the top.
    groups.sort(key=lambda g: (g[2] != "line", -np.median(g[1])))

    fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(groups) + 2.2))
    for i, (layer, v, kind) in enumerate(groups):
        y = len(groups) - i
        col = _layer_color(layer)
        bp = ax.boxplot([v], positions=[y], vert=False, widths=0.56,
                        patch_artist=True, showfliers=False, zorder=3,
                        medianprops=dict(color="black", lw=1.8))
        # Neutral box, as the class bands of the conformance figure: colour
        # is carried by the points. Corridors keep the hatch.
        bp["boxes"][0].set(facecolor=(*matplotlib.colors.to_rgb(GREY), 0.15),
                           edgecolor=GREY, lw=1.0,
                           hatch="///" if kind == "trace" else None)
        jitter = (np.random.default_rng(11 + i).random(v.size) - 0.5) * 0.3
        ax.scatter(v, y + jitter, s=11, color=col, edgecolor="white", lw=0.3,
                   zorder=4)
    ax.set_yticks(range(1, len(groups) + 1))
    ax.set_yticklabels([f"{layer_display_name(g[0])}  (n = {g[1].size})"
                        for g in reversed(groups)], fontsize=8.5)
    for t in DEVIATION_THRESHOLDS[1:]:
        ax.axvline(t * MM, color=GREY, ls=":", lw=0.8, zorder=1)
    ax.set_xlabel("measured horizontal (XY) crown deviation, max per instance (mm)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    # No line/corridor legend: the row label already names the group, so the
    # hatch and the lighter fill only reinforce it.
    # For the caption: dotted lines mark the deviation-class bounds.
    return _save(fig, out_dir, "deviation_by_layer.png")


# ─────────────────────────────────────────────────────────────────────────────
# F4  SIGNED VERTICAL OFFSET
# ─────────────────────────────────────────────────────────────────────────────

def _panel_signed_z(rows, lim, out_dir, name):
    """One group of the signed-Z figure, drawn as a figure of its own.

    ``lim`` is handed in rather than derived here: the three files are read
    against one another as subfigures of one float, so they must share an
    x-scale. Nothing in the image names the group, because the subcaption does.
    """
    fig, ax = plt.subplots(figsize=(9.2, 2.9))
    v = np.array([r["_crown_z_signed_median_m"] for r in rows]) * MM
    v = v[np.isfinite(v)]
    ax.set_ylabel("instances")
    ax.set_xlabel("signed vertical offset of the measured crown from the "
                  "registered top (mm)")
    ax.grid(axis="y", alpha=0.25, zorder=1)
    ax.set_xlim(-lim, lim)
    if v.size == 0:
        ax.text(0.5, 0.45, "no instances qualify", ha="center", color=GREY,
                transform=ax.transAxes, fontsize=10)
    else:
        bins = np.linspace(-lim, lim, 45)
        n_up = int((v > 0).sum())
        ax.hist(v[v <= 0], bins=bins, color="#1f6fb4", edgecolor="white", lw=0.5,
                label=f"deeper than registered  ({v.size - n_up})", zorder=3)
        ax.hist(v[v > 0], bins=bins, color="#c0392b", edgecolor="white", lw=0.5,
                label=f"shallower than registered  ({n_up})", zorder=3)
        ax.axvline(0, color="black", lw=1.3, zorder=4)
        med = float(np.median(v))
        ax.axvline(med, color="black", ls="--", lw=1.2, zorder=4)
        ax.annotate(f"median {med:+.0f} mm\n|median| {np.median(np.abs(v)):.0f} mm",
                    (med, ax.get_ylim()[1] * 0.95), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, va="top")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def _panel_signed_z_by_type(rows, lim, types, out_dir, name):
    """The signed-Z panel with each bar split by observed utility type. The
    sign is carried by the zero line and the deeper/shallower counts in the
    note, since colour now means type."""
    fig, ax = plt.subplots(figsize=(9.2, 2.9))
    ax.set_ylabel("instances")
    ax.set_xlabel("signed vertical offset of the measured crown from the "
                  "registered top (mm)")
    ax.grid(axis="y", alpha=0.25, zorder=1)
    ax.set_xlim(-lim, lim)
    stacks, colors, labels = [], [], []
    for t in types:
        v = np.array([r["_crown_z_signed_median_m"] for r in rows
                      if int(r["utility_type"]) == t]) * MM
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        stacks.append(v)
        colors.append(UTILITY_TYPE_COLORS[t])
        labels.append(f"{_TYPE_NAME.get(t, UTILITY_TYPE_LABELS[t])} ({v.size})")
    if not stacks:
        ax.text(0.5, 0.45, "no instances qualify", ha="center", color=GREY,
                transform=ax.transAxes, fontsize=10)
    else:
        ax.hist(stacks, bins=np.linspace(-lim, lim, 45), stacked=True,
                color=colors, label=labels, edgecolor="white", lw=0.5, zorder=3)
        v = np.concatenate(stacks)
        n_up = int((v > 0).sum())
        ax.axvline(0, color="black", lw=1.3, zorder=4)
        med = float(np.median(v))
        ax.axvline(med, color="black", ls="--", lw=1.2, zorder=4)
        # The deeper/shallower counts go in the subcaption, not the image.
        print(f"  {name}: {v.size - n_up} deeper, {n_up} shallower")
        ax.annotate(f"median {med:+.0f} mm\n|median| {np.median(np.abs(v)):.0f} mm",
                    (med, ax.get_ylim()[1] * 0.95), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, va="top",
                    bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8, loc="upper left", title="Observed utility type",
                  alignment="left", title_fontsize=8, framealpha=0.95)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def fig_signed_z_by_type(lines, traces, out_dir):
    """``fig_signed_z`` with each bar split by observed utility type. Same
    groups, same shared x-scale, same bins."""
    keep = lines + traces
    groups = [
        ([r for r in keep if r["_z_reg"] and not r["_same_campaign"]],
         "deviation_signed_z_registered_by_type.png"),
        ([r for r in keep if r["_z_reg"] and r["_same_campaign"]],
         "deviation_signed_z_same_campaign_by_type.png"),
        ([r for r in keep if r["_z_indicative"]],
         "deviation_signed_z_vejledende_by_type.png"),
    ]
    lim = max([abs(r["_crown_z_signed_median_m"]) * MM
               for g, _ in groups for r in g
               if np.isfinite(r["_crown_z_signed_median_m"])] or [100]) * 1.08
    counts = {}
    for g, _ in groups:
        for r in g:
            counts[int(r["utility_type"])] = counts.get(int(r["utility_type"]), 0) + 1
    types = sorted(counts, key=lambda t: -counts[t])
    return [_panel_signed_z_by_type(rows, lim, types, out_dir, name)
            for rows, name in groups]


def fig_signed_z(lines, traces, out_dir):
    """Does the register sit above or below the utility? A magnitude cannot say;
    this is the figure the signed vertical component exists for.

    Three files, because the instances answer three different questions and
    pooling them would answer none. Only the first is evidence about how
    accurately LER records depth. They are the three subfigures of one float in
    the report.
    """
    keep = lines + traces
    # What each file is evidence of, for the subcaption rather than the image:
    #   1. independently surveyed Z: the only group that measures how
    #      accurately LER records depth;
    #   2. same survey campaign as the capture: photogrammetry against the
    #      as-built survey, not register accuracy;
    #   3. vejledendeDybde only: tests a registered indicative depth, but
    #      anchored to a ground level taken from the point cloud.
    groups = [
        ([r for r in keep if r["_z_reg"] and not r["_same_campaign"]],
         "deviation_signed_z_registered.png"),
        ([r for r in keep if r["_z_reg"] and r["_same_campaign"]],
         "deviation_signed_z_same_campaign.png"),
        ([r for r in keep if r["_z_indicative"]],
         "deviation_signed_z_vejledende.png"),
    ]
    lim = max([abs(r["_crown_z_signed_median_m"]) * MM
               for g, _ in groups for r in g
               if np.isfinite(r["_crown_z_signed_median_m"])] or [100]) * 1.08
    return [_panel_signed_z(rows, lim, out_dir, name) for rows, name in groups]


# ─────────────────────────────────────────────────────────────────────────────
# F5  HORIZONTAL AGAINST VERTICAL
# ─────────────────────────────────────────────────────────────────────────────

def fig_xy_vs_z(lines, traces, out_dir):
    """Separates a horizontally wrong registration from a vertically wrong one.
    The KLIC line is the Dutch 1 m horizontal excavation tolerance, the practical
    reading of the horizontal axis.

    Restricted to instances carrying a registered Z, and excluding the
    same-campaign features. Everywhere else the vertical axis would plot a value
    core/depth.py synthesised, so a point's height above the diagonal would be an
    artefact of the fallback rather than a property of the register.
    """
    eligible = [r for r in lines + traces if r["_z_reg"] and not r["_same_campaign"]]
    lines = [r for r in eligible if not r["_is_trace"]]
    traces = [r for r in eligible if r["_is_trace"]]

    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    for rows, marker, kind in ((lines, "o", "utility line"),
                               (traces, "^", f"{LEDNINGSTRACE_LAYER} corridor")):
        for r in rows:
            x, y = r["_crown_xy_max_m"] * MM, r["_crown_z_max_m"] * MM
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            ax.scatter(x, y, s=30, marker=marker, color=_layer_color(r["ler_layer"]),
                       edgecolor="white", lw=0.5, alpha=0.9, zorder=3)
    klic = KLIC_XY_THRESHOLDS[-1] * MM
    ax.axvline(klic, color="#c0392b", ls="--", lw=1.4, zorder=2)
    ax.annotate(f"KLIC horizontal tolerance {klic:.0f} mm", (klic, ax.get_ylim()[1]),
                xytext=(-6, -8), textcoords="offset points", rotation=90,
                ha="right", va="top", fontsize=8, color="#c0392b")
    top = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, top], [0, top], color=GREY, ls=":", lw=1, zorder=1)
    ax.annotate("equal horizontal and vertical error", (top * 0.62, top * 0.62),
                rotation=45, fontsize=8, color=GREY, ha="center", va="bottom")
    ax.set_xlabel("horizontal (XY) crown deviation, max per instance (mm)")
    ax.set_ylabel("vertical (Z) crown deviation, max per instance (mm)")
    ax.grid(alpha=0.22)
    ax.set_axisbelow(True)
    layers = sorted({r["ler_layer"] for r in lines + traces})
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=_layer_color(l),
                              markersize=6, label=layer_display_name(l))
                       for l in layers]
                      + [Line2D([], [], marker="^", ls="", color=GREY,
                                markersize=6, label="corridor (triangle)")],
              fontsize=7.5, loc="upper right", framealpha=0.95, ncol=1)
    # For the caption: points above the diagonal are registered worse in depth
    # than in plan, and only instances carrying an independently registered Z
    # are eligible. report() prints how many that is.
    return _save(fig, out_dir, "deviation_xy_vs_z.png")


# ─────────────────────────────────────────────────────────────────────────────
# HEADLINE NUMBERS
# ─────────────────────────────────────────────────────────────────────────────

def report(all_rows, keep, dropped, lines, traces):
    print("\n" + "=" * 74)
    print("  HEADLINE NUMBERS")
    print("=" * 74)
    print(f"{len(all_rows)} instances swept; {len(keep)} in the analysis set "
          f"({len(lines)} utility lines, {len(traces)} corridors)")
    for why, n in dropped.items():
        print(f"  {n:4d} excluded: {why}")

    print("\n--- HORIZONTAL: XY deviation vs noejagtighedsklasse, all instances ---")
    for name, rows in (("utility lines", lines), ("corridors", traces)):
        if not rows:
            continue
        v = np.array([r["_crown_xy_max_m"] for r in rows])
        v = v[np.isfinite(v)] * MM
        print(f"\n{name} (n = {len(rows)})")
        print(f"  XY deviation, max per instance: median {np.median(v):.0f} mm, "
              f"P95 {np.percentile(v, 95):.0f} mm, worst {v.max():.0f} mm")

        graded = [r for r in rows if np.isfinite(r["_bound"])]
        bust = [r for r in graded if r["_crown_xy_max_m"] > r["_bound"]]
        print(f"  {len(bust)} of {len(graded)} with a bounded class exceed it")
        if graded:
            ratio = sorted(r["_crown_xy_max_m"] / r["_bound"] for r in graded)
            print(f"  median bounded instance uses "
                  f"{ratio[len(ratio) // 2] * 100:.0f}% of its declared tolerance")
        n_open = sum(1 for r in rows if r["_cls"] == len(DEVIATION_COLORS))
        n_unreg = sum(1 for r in rows if r["_cls"] == 0)
        print(f"  {n_open} ({n_open / len(rows) * 100:.0f}%) sit in the open "
              f"'> 2.00 m' class, which sets no bound; {n_unreg} register no class")

        for r in sorted(bust, key=lambda r: -(r["_crown_xy_max_m"] - r["_bound"]))[:5]:
            print(f"    EXCEEDS  {r['site']:16s} {r['label']:22s} "
                  f"{r['ler_layer']:28s} declared {r['declared_class_text']:18s} "
                  f"measured {r['_crown_xy_max_m'] * MM:.0f} mm")
        # Identity, not equality: two rows can hold the same values.
        _bust_ids = {id(r) for r in bust}
        tight = sorted((r for r in graded if id(r) not in _bust_ids),
                       key=lambda r: r["_bound"] - r["_crown_xy_max_m"])[:3]
        for r in tight:
            print(f"    tightest {r['site']:16s} {r['label']:22s} "
                  f"declared {r['declared_class_text']:18s} "
                  f"measured {r['_crown_xy_max_m'] * MM:.0f} mm, "
                  f"{(r['_bound'] - r['_crown_xy_max_m']) * MM:.0f} mm to spare")

    print("\n--- VERTICAL: only where the register carries a Z ---")
    tiers = [
        ("independently registered Z",
         [r for r in keep if r["_z_reg"] and not r["_same_campaign"]]),
        ("registered Z, same survey campaign (NOT register accuracy)",
         [r for r in keep if r["_z_reg"] and r["_same_campaign"]]),
        ("vejledendeDybde only (registered depth, measured ground)",
         [r for r in keep if r["_z_indicative"]]),
    ]
    accounted = sum(len(g) for _, g in tiers)
    for name, grp in tiers:
        print(f"\n{name} (n = {len(grp)})")
        if not grp:
            print("  no instances qualify, so no vertical statement is possible")
            continue
        v = np.array([r["_crown_z_signed_median_m"] for r in grp])
        v = v[np.isfinite(v)] * MM
        n_up = int((v > 0).sum())
        print(f"  signed offset: median {np.median(v):+.0f} mm  "
              f"(P5 {np.percentile(v, 5):+.0f} .. P95 {np.percentile(v, 95):+.0f} mm)")
        print(f"  absolute error: median {np.median(np.abs(v)):.0f} mm, "
              f"P95 {np.percentile(np.abs(v), 95):.0f} mm, "
              f"worst {np.abs(v).max():.0f} mm")
        print(f"  {n_up} shallower / {v.size - n_up} deeper than registered")
        n_tr = sum(1 for r in grp if r["_is_trace"])
        print(f"  {len(grp) - n_tr} utility lines, {n_tr} corridors")
    print(f"\n{len(keep) - accounted} of {len(keep)} analysed instances support no "
          f"vertical statement at all (Z synthesised from the point cloud)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--csv", default=str(_project_root / "figures"
                                             / "Water_Area_5_deviation.csv"),
                        help="sweep CSV to read")
    parser.add_argument("--out-dir", default=str(_project_root / "figures"),
                        help="where the PNGs go")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        parser.error(f"no such CSV: {csv_path}\n  run tools/deviation_sweep.py first")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows, keep, dropped = load(csv_path)
    lines, traces = split(keep)
    print(f"Read {len(all_rows)} rows from {csv_path}\n")
    fig_distribution(lines, traces, out_dir)
    fig_distribution_by_type(lines, traces, out_dir)
    fig_conformance(lines, traces, out_dir)
    fig_by_layer(lines, traces, out_dir)
    fig_signed_z(lines, traces, out_dir)
    fig_signed_z_by_type(lines, traces, out_dir)
    fig_xy_vs_z(lines, traces, out_dir)
    report(all_rows, keep, dropped, lines, traces)
    return 0


if __name__ == "__main__":
    sys.exit(main())
