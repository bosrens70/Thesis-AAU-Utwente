# -*- coding: utf-8 -*-
"""
Export trench-restricted LER deviation point clouds to georeferenced LAS
========================================================================
Writes the discrete LER deviation modes (XYZ, XY, Z) of the deviation viewer
as LAS files that load directly in QGIS as an overlay on the LER utilities.

Each exported LAS carries, per LER surface sample inside the picked trench:
  - XYZ in UTM32 / ETRS89 (EPSG:25832), embedded WKT CRS;
  - RGB = the baked accuracy-class colour (so the palette shows with no setup);
  - classification = accuracy class 1..5 (0 = no measured neighbour / no data);
  - extra dimension deviation_m = the raw deviation value in metres.

Only the three discrete metrics are exported. The continuous gradient modes do
not map to the five accuracy classes and are intentionally left out.

Requires: laspy, pyproj, numpy.
"""

from pathlib import Path

import numpy as np
import laspy
from pyproj import CRS

from core.config import (
    DEVIATION_THRESHOLDS, DEVIATION_COLORS, DEVIATION_CLASS_LABELS,
)

CRS_UTM32 = CRS.from_epsg(25832)


def linear_to_srgb(c):
    """Linear RGB in [0,1] -> sRGB in [0,1] (matches core.geometry, kept local
    so this module does not pull in open3d)."""
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.clip(c, 0, None), 1 / 2.4) - 0.055)

# The three discrete metrics that get exported. Each maps to a label used in
# the output file names and the QGIS colour-table header.
EXPORT_METRICS = ("xyz", "xy", "z")
_METRIC_LABELS = {"xyz": "XYZ", "xy": "XY", "z": "Z"}


def deviation_to_class(deviations):
    """Accuracy class 1..5 for each deviation (NaN -> 0 = no data).

    Mirrors :func:`core.geometry.deviation_to_color` binning: the class upper
    bounds are ``DEVIATION_THRESHOLDS[1:]`` with ``right=True`` so a boundary
    value belongs to the lower (better) class.
    """
    dev = np.asarray(deviations, dtype=float)
    edges = np.asarray(DEVIATION_THRESHOLDS, dtype=float)[1:]
    cls = np.digitize(dev, edges, right=True) + 1
    cls = np.clip(cls, 1, len(DEVIATION_THRESHOLDS))
    cls[np.isnan(dev)] = 0
    return cls.astype(np.uint8)


def _write_colour_table(out_dir):
    """Write a QGIS colour map (.txt) mapping classification 1..5 to the baked
    sRGB accuracy-class colours with their labels. Loadable in the QGIS point
    cloud / raster "Palette / Unique values" classification styling."""
    path = out_dir / "deviation_accuracy_class_colours.txt"
    lines = ["# QGIS colour table: accuracy classification 1..5",
             "# value,red,green,blue,alpha,label"]
    for i, (col, label) in enumerate(zip(DEVIATION_COLORS, DEVIATION_CLASS_LABELS), start=1):
        r, g, b = (int(round(linear_to_srgb(c) * 255)) for c in col)
        lines.append(f"{i},{r},{g},{b},255,{label}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build_metric_arrays(samples_by_layer, raw_by_layer, inside_by_layer):
    """Concatenate the inside-trench samples + deviations across all LER layers.

    Returns ``(pts_local, dev)`` where ``pts_local`` is (N,3) local XYZ and
    ``dev`` is (N,) raw deviation in metres. Only labelled utilities that were
    part of the deviation detection are kept: samples with no measured
    neighbour (NaN deviation) are dropped, so unmatched LER layers do not
    appear in the export. Layers with no points are skipped. When a layer has
    no inside mask (no trench) all its samples are kept.
    """
    pts_parts, dev_parts = [], []
    for ln, pts in samples_by_layer.items():
        pts = np.asarray(pts)
        if len(pts) == 0:
            continue
        dev = np.asarray(raw_by_layer.get(ln))
        mask = inside_by_layer.get(ln)
        if mask is not None:
            pts = pts[mask]
            dev = dev[mask]
        # Keep only samples with a computed deviation (labelled / detected).
        valid = ~np.isnan(dev)
        pts = pts[valid]
        dev = dev[valid]
        if len(pts) == 0:
            continue
        pts_parts.append(pts)
        dev_parts.append(dev)
    if not pts_parts:
        return np.empty((0, 3)), np.empty((0,))
    return np.concatenate(pts_parts), np.concatenate(dev_parts)


def _write_las(pts_local, dev, offset, out_path):
    """Write one LAS: local XYZ + offset -> UTM, baked class RGB, classification,
    a ``deviation_m`` extra dimension, and the deviation mirrored into the
    standard Intensity attribute (in millimetres) as a QGIS-reliable ramp field.
    Returns the point count written."""
    tx, ty, tz = offset
    x = pts_local[:, 0].astype(np.float64) + tx
    y = pts_local[:, 1].astype(np.float64) + ty
    z = pts_local[:, 2].astype(np.float64) + tz

    cls = deviation_to_class(dev)
    # Baked class colours (no-data class 0 -> grey). DEVIATION_COLORS are linear
    # RGB in [0,1]; bake the sRGB values so QGIS matches the viewer legend.
    palette = np.asarray(DEVIATION_COLORS, dtype=float)
    cols = palette[np.clip(cls - 1, 0, len(palette) - 1)]
    srgb = np.clip(linear_to_srgb(cols), 0.0, 1.0)
    srgb[cls == 0] = 0.5  # grey for samples with no measured neighbour

    # Deviation in millimetres for the Intensity mirror. QGIS always exposes
    # Intensity in the "Attribute by Ramp" renderer, whereas float extra
    # dimensions are not shown by every QGIS/PDAL build. uint16 caps at 65535,
    # i.e. 65.5 m, well beyond any realistic deviation.
    dev_mm = np.rint(np.nan_to_num(dev, nan=0.0) * 1000.0)
    intensity = np.clip(dev_mm, 0, 65535).astype(np.uint16)

    header = laspy.LasHeader(point_format=2, version="1.4")
    header.offsets = np.array([np.floor(x.min()), np.floor(y.min()), np.floor(z.min())])
    header.scales = np.array([0.001, 0.001, 0.001])  # 1 mm precision
    header.add_crs(CRS_UTM32)
    header.add_extra_dim(laspy.ExtraBytesParams(name="deviation_m", type=np.float32))

    las = laspy.LasData(header=header)
    las.x, las.y, las.z = x, y, z
    las.red = (srgb[:, 0] * 65535).astype(np.uint16)
    las.green = (srgb[:, 1] * 65535).astype(np.uint16)
    las.blue = (srgb[:, 2] * 65535).astype(np.uint16)
    las.classification = cls
    las.deviation_m = dev.astype(np.float32)
    las.intensity = intensity

    out_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out_path))
    return len(x)


def export_ler_deviation_las(
    site_stem, out_dir, offset,
    samples_by_layer,
    raw_by_metric,
    inside_by_layer,
):
    """Export the three discrete LER deviation metrics to LAS for QGIS.

    Parameters
    ----------
    site_stem : str
        Site name used in output file names.
    out_dir : Path
        Directory to write into (created if missing).
    offset : (float, float, float)
        (TX, TY, TZ) added to local coordinates to reach UTM32.
    samples_by_layer : dict[str, np.ndarray]
        Layer -> (N,3) local sample points (shared across all three metrics).
    raw_by_metric : dict[str, dict[str, np.ndarray]]
        Metric ("xyz"/"xy"/"z") -> {layer -> (N,) raw deviation in metres}.
    inside_by_layer : dict[str, np.ndarray]
        Layer -> boolean inside-trench mask. Missing key = keep all samples.

    Returns
    -------
    list[Path]
        The written LAS files (plus the QGIS colour table).
    """
    out_dir = Path(out_dir)
    written = []
    for metric in EXPORT_METRICS:
        pts_local, dev = _build_metric_arrays(
            samples_by_layer, raw_by_metric[metric], inside_by_layer)
        if len(pts_local) == 0:
            print(f"  [SKIP] LER {_METRIC_LABELS[metric]} deviation: no in-trench samples")
            continue
        out_path = out_dir / f"{site_stem}_LER_deviation_{metric}.las"
        n = _write_las(pts_local, dev, offset, out_path)
        written.append(out_path)
        print(f"  LER {_METRIC_LABELS[metric]:>3s} deviation -> {out_path.name}  ({n:,} pts)")
    if written:
        written.append(_write_colour_table(out_dir))
    return written
