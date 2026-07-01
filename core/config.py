# -*- coding: utf-8 -*-
"""
Centralised configuration for all viewers and tools.
=====================================================
Change the site by editing PLY_FILE (and optionally GML_PATH /
AREA_REF_GEOJSON if working with a different Ledningspakke).
All other scripts import from here, nothing is duplicated.
"""

import os
from pathlib import Path
from enum import IntEnum
from dataclasses import dataclass
import warnings

# Suppress noisy pyogrio/GDAL warnings from GML files with null geometries
# and non-numeric values in integer fields. Applied globally so every script
# that imports core.config gets the filter automatically.
warnings.filterwarnings("ignore", message="Unrecognized geometry type", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="parsed incompletely", category=RuntimeWarning)
warnings.filterwarnings("ignore", module="pyogrio", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOCATION: portable, project-root-relative
# ─────────────────────────────────────────────────────────────────────────────
# PROJECT_ROOT is the Thesis/ folder (this file lives in Thesis/core/config.py).
# DATA_DIR defaults to Thesis/Data/, but can be pointed elsewhere by setting the
# THESIS_DATA_DIR environment variable, so no source edits are needed to relocate data.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = Path(os.environ.get("THESIS_DATA_DIR", PROJECT_ROOT / "Data"))

PLY_BASE_DIR = DATA_DIR / "OpenTrench3D"

# ─────────────────────────────────────────────────────────────────────────────
# SITE SELECTION: change these to switch site
# ─────────────────────────────────────────────────────────────────────────────
PLY_FILE         = PLY_BASE_DIR / "Water_Area_5" / "Area_5_Site_37.ply"
AREA_REF_GEOJSON = DATA_DIR / "Translation_coordinates" / "area_points_utm32_etrs89.geojson"
GML_PATH         = DATA_DIR / "Ledningspakke_2803288_Area_4_and_5" / "consolidated.gml"



# Crop region shape (load-time switch).
#   "circle" — disc of radius CROP_RADIUS around the cloud XY centroid (the cloud
#              is cropped to that disc).
#   "rect"   — the cloud is kept in full; its 3D axis-aligned bounding box (AABB)
#              is expanded by UTILITY_RECT_BUFFER in X, Y and Z, and utilities are
#              selected and clipped to that box.  CROP_RADIUS is not used here.
# Honoured by the point-cloud crop (every init_site viewer) and by utility
# selection in base_viewer / label_viewer / deviation_viewer.  Other viewers keep
# their own selection logic.  Set to "circle" to restore the legacy disc crop.
CROP_MODE = "rect"

# Margin (metres) added around the point-cloud AABB in every dimension (X, Y, Z)
# when selecting utilities in "rect" mode (the "additional crop distance").
UTILITY_RECT_BUFFER = 2.0
# Circular crop radius (metres) around the point cloud centroid (XY).
CROP_RADIUS = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# CLASS LABEL DEFINITIONS: OpenTrench3D semantic classes
# ─────────────────────────────────────────────────────────────────────────────
CLASS_LABELS = {
    0: {"name": "Main Utility",     "color": [0.00, 0.80, 0.00]},
    1: {"name": "Other Utility",    "color": [1.00, 1.00, 0.00]},
    2: {"name": "Trench",           "color": [0.55, 0.27, 0.07]},
    3: {"name": "Inactive Utility", "color": [0.00, 0.00, 0.00]},
    4: {"name": "Misc",             "color": [0.60, 0.60, 0.60]},
}

DEFAULT_CLASS_COLOR = [1.0, 0.0, 1.0]  # magenta, unknown class IDs

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY LAYER DEFINITIONS: DLF-recommended colours (RGB 0-1)
# ─────────────────────────────────────────────────────────────────────────────
LINE_LAYERS = {
    "Vandledning":               {"color": [0.000, 0.000, 1.000], "fallback_radius": 0.010},  # DLF blue
    "Afloebsledning":            {"color": [1.000, 0.000, 0.000], "fallback_radius": 0.010},  # DLF red
    "Gasledning":                {"color": [1.000, 0.600, 0.000], "fallback_radius": 0.010},  # DLF orange
    "Elledning":                 {"color": [1.000, 0.000, 0.000], "fallback_radius": 0.010},  # DLF red
    "Telekommunikationsledning": {"color": [0.980, 0.588, 0.275], "fallback_radius": 0.010},  # DLF orange
    "Foeringsroer":              {"color": [0.882, 0.882, 0.882], "fallback_radius": 0.010},  # DLF grey
    "LedningUkendtForsyningsart":{"color": [0.300, 0.800, 0.800], "fallback_radius": 0.010},  # cyan
    "Ledningstrace":             {"color": [0.980, 0.588, 0.275], "fallback_radius": 0.010},  # DLF trace orange
    "TermiskLedning":            {"color": [1.000, 0.000, 1.000], "fallback_radius": 0.010},  # DLF violet
    "Olieledning":               {"color": [0.463, 0.463, 0.463], "fallback_radius": 0.010},  # DLF grey
    "AndenLedning":              {"color": [0.800, 0.800, 0.800], "fallback_radius": 0.010},  # grey
}

# Right-panel legend order: keep Ledningstrace last (dense / low-priority visually)
PIPE_LEGEND_UI_ORDER = [ln for ln in LINE_LAYERS if ln != "Ledningstrace"]
if "Ledningstrace" in LINE_LAYERS:
    PIPE_LEGEND_UI_ORDER.append("Ledningstrace")

COMPONENT_LAYERS = {
    "Vandkomponent":                  {"color": [0.000, 0.900, 0.900]},
    "Afloebskomponent":               {"color": [0.700, 0.400, 0.200]},
    "Gaskomponent":                   {"color": [1.000, 0.900, 0.300]},
    "Elkomponent":                    {"color": [1.000, 0.300, 0.300]},
    "Telekommunikationskomponent":    {"color": [0.400, 1.000, 0.400]},
    "TermiskKomponent":               {"color": [1.000, 0.500, 0.900]},
    "Oliekomponent":                  {"color": [0.463, 0.463, 0.463]},
    "AndenKomponent":                 {"color": [0.800, 0.800, 0.800]},
}

COMPONENT_SPHERE_RADIUS = 0.05

# Map component layer -> corresponding line layer for depth estimation
COMP_TO_LINE = {
    "Vandkomponent":               "Vandledning",
    "Afloebskomponent":            "Afloebsledning",
    "Gaskomponent":                "Gasledning",
    "Elkomponent":                 "Elledning",
    "Telekommunikationskomponent": "Telekommunikationsledning",
    "TermiskKomponent":            "TermiskLedning",
    "Oliekomponent":               "Olieledning",
    "AndenKomponent":              "AndenLedning",
}

# ─────────────────────────────────────────────────────────────────────────────
# FORSYNINGSART keyword -> colour for Ledningstrace sub-groups
# ─────────────────────────────────────────────────────────────────────────────
FORSYNINGSART_COLOR_HINTS = [
    # Check longer/more specific keywords first to avoid substring conflicts
    # (e.g., "tele" must come before "el" since "telekommunikation" contains "el")
    ("fjern",  [1.000, 0.000, 1.000]),   # DLF fjernvarme violet
    ("varme",  [1.000, 0.000, 1.000]),   # DLF varme violet
    ("tele",   [0.980, 0.588, 0.275]),   # DLF tele orange
    ("kommu",  [0.980, 0.588, 0.275]),   # DLF kommunikation orange
    ("afloeb", [1.000, 0.000, 0.000]),   # DLF spildevand red
    ("spilde", [1.000, 0.000, 0.000]),   # DLF spildevand red
    ("vejafv", [1.000, 0.000, 0.000]),   # DLF vejafvanding red
    ("vand",   [0.000, 0.000, 1.000]),   # DLF vand blue
    ("gas",    [1.000, 0.600, 0.000]),   # DLF gas orange
    ("el",     [1.000, 0.000, 0.000]),   # DLF el red (checked last to avoid matching "tele")
    ("olie",   [0.463, 0.463, 0.463]),   # DLF olie grey
    ("anden",  [0.800, 0.800, 0.800]),   # anden/andet grey
    ("andet",  [0.800, 0.800, 0.800]),   # anden/andet grey
]


def forsyningsart_color(fa_value, fallback):
    """Return a colour for a forsyningsart value by substring matching."""
    fa_lower = fa_value.lower()
    for keyword, color in FORSYNINGSART_COLOR_HINTS:
        if keyword in fa_lower:
            return color
    return fallback


# Direct mapping: forsyningsart value -> corresponding line layer name
# LER 2025 datamodel: forsyningsart value -> line layer
FORSYNINGSART_TO_LINE = {
    "vand":                "Vandledning",
    "afloeb":              "Afloebsledning",
    "spildevand":          "Afloebsledning",
    "vejafvanding":        "Afloebsledning",
    "gas":                 "Gasledning",
    "el":                  "Elledning",
    "telekommunikation":   "Telekommunikationsledning",
    "fjernvarme":          "TermiskLedning",
    "fjernkoeling":        "TermiskLedning",
    "varme":               "TermiskLedning",
    "termisk":             "TermiskLedning",
    "olie":                "Olieledning",
    "anden":               "AndenLedning",
    "andet":               "AndenLedning",
}


# ─────────────────────────────────────────────────────────────────────────────
# INSTANCE / UTILITY TYPE DEFINITIONS (for labelled instance PLY files)
# ─────────────────────────────────────────────────────────────────────────────
UTILITY_TYPE_LABELS = {
    0: "Unlabeled", 1: "PowerLine", 2: "DrainageLine", 3: "OilPipeLine",
    4: "GasLine", 5: "ThermalLine", 6: "Conduit", 7: "WaterLine",
    8: "TelecomunicationLine", 9: "OtherLine", 10: "LineUnknownServiceType",
}

# DLF-recommended colours (RGB 0-1), using the primary sub-type per utility
UTILITY_TYPE_COLORS = {
    0: [0.50, 0.50, 0.50],   # Unlabeled          - grey
    1: [1.00, 0.00, 0.00],   # PowerLine           - DLF red
    2: [1.00, 0.00, 0.00],   # DrainageLine        - DLF red
    3: [0.46, 0.46, 0.46],   # OilPipeLine         - DLF grey
    4: [1.00, 0.60, 0.00],   # GasLine             - DLF orange
    5: [1.00, 0.00, 1.00],   # ThermalLine         - DLF violet
    6: [0.88, 0.88, 0.88],   # Conduit             - DLF grey
    7: [0.00, 0.00, 1.00],   # WaterLine           - DLF blue
    8: [0.98, 0.59, 0.27],   # TelecomunicationLine - DLF orange
    9: [0.80, 0.80, 0.80],   # OtherLine           - grey
    10: [0.30, 0.80, 0.80],  # LineUnknownServiceType
}

INSTANCE_LABEL_OPTIONS = [
    "PowerLine",
    "DrainageLine",
    "OilPipeLine",
    "GasLine",
    "ThermalLine",
    "Conduit",
    "WaterLine",
    "TelecomunicationLine",
    "OtherLine",
    "LineUnknownServiceType",
]

INSTANCE_COLORS = [
    [1.00, 0.20, 0.20],  # red
    [0.20, 0.60, 1.00],  # blue
    [0.20, 0.90, 0.20],  # green
    [1.00, 0.60, 0.00],  # orange
    [0.80, 0.20, 1.00],  # purple
    [0.00, 0.90, 0.90],  # cyan
    [1.00, 1.00, 0.20],  # yellow
    [1.00, 0.40, 0.70],  # pink
    [0.60, 0.40, 0.20],  # brown
    [0.50, 1.00, 0.50],  # lime
]

# Vandledning diameter -> colour mapping
DIAMETER_COLORS = {
    0:   [0.502, 0.502, 0.502],
    32:  [0.702, 0.851, 1.000],
    63:  [0.400, 0.698, 1.000],
    120: [0.102, 0.459, 1.000],
    150: [0.000, 0.278, 0.800],
    160: [0.000, 0.180, 0.522],
}

# ─────────────────────────────────────────────────────────────────────────────
# DEVIATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
DEVIATION_THRESHOLDS = [0.00, 0.25, 0.50, 1.00, 2.00]
DEVIATION_COLORS = [
    [0.0, 0.7, 0.2],   # Class 1: ≤ 250 mm  - green
    [0.6, 0.9, 0.0],   # Class 2: ≤ 500 mm  - yellow-green
    [1.0, 0.8, 0.0],   # Class 3: ≤ 1000 mm - yellow
    [1.0, 0.4, 0.0],   # Class 4: ≤ 2000 mm - orange
    [0.8, 0.0, 0.0],   # Class 5: > 2000 mm - red
]

DEVIATION_CLASS_LABELS = [
    "Class 1:  <= 250 mm",
    "Class 2:  <= 500 mm",
    "Class 3:  <= 1000 mm",
    "Class 4:  <= 2000 mm",
    "Class 5:  > 2000 mm",
]

# ─────────────────────────────────────────────────────────────────────────────
# REGISTERED ACCURACY CLASS (noejagtighedsklasse) -> 2D buffer half-width
# ─────────────────────────────────────────────────────────────────────────────
# The LER registers a horizontal (planimetric) accuracy class per feature. It is
# text such as "<= 0.50 m" or "> 2.00 m" and maps onto the same class bounds as
# DEVIATION_THRESHOLDS, so the class directly gives a 2D buffer half-width.
ACCURACY_CLASS_FIELD = "noejagtighedsklasse"

# Display half-width (metres) for the open top class ("> 2.00 m"), which has no
# registered upper bound. A display convention only, not a registered value.
ACCURACY_OPEN_CLASS_WIDTH = 2.00


def accuracy_class_halfwidth(value):
    """Map a registered ``noejagtighedsklasse`` value to a 2D buffer half-width.

    Parses the numeric bound out of the class text (tolerating a decimal comma)
    and snaps it to the matching DEVIATION_THRESHOLDS class. Returns
    ``(half_width_m, class_idx)`` with ``class_idx`` in 1..5, or ``None`` when the
    value is missing or unparseable. The half-width equals the class upper bound
    (the horizontal tolerance); the open top class uses ACCURACY_OPEN_CLASS_WIDTH.
    """
    import re as _re
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    m = _re.search(r"(\d+(?:[.,]\d+)?)", s)
    if not m:
        return None
    num = float(m.group(1).replace(",", "."))
    edges = DEVIATION_THRESHOLDS[1:]        # class upper bounds: 0.25 .. 2.00
    n_classes = len(DEVIATION_THRESHOLDS)   # 5
    # "> X" with X at/above the last bound is the open top class.
    if ">" in s and num >= edges[-1] - 1e-9:
        return ACCURACY_OPEN_CLASS_WIDTH, n_classes
    j = min(range(len(edges)), key=lambda k: abs(num - edges[k]))
    return edges[j], j + 1

# ─────────────────────────────────────────────────────────────────────────────
# DEPTH HIERARCHY: enum, config (used by BASE1 and LABEL1)
# ─────────────────────────────────────────────────────────────────────────────
class DepthSource(IntEnum):
    REGISTERED   = 1
    VEJLEDENDE   = 2
    FEATURE_MEAN = 3   # pipes only
    LAYER_MEAN   = 4   # components only (parent pipe layer average)
    GROUND_PLANE = 5
    NONE         = 99


@dataclass(frozen=True)
class DepthConfig:
    enabled_levels: frozenset
    track_per_vertex: bool = True


PIPE_DEPTH_CONFIG = DepthConfig(
    enabled_levels=frozenset({
        DepthSource.REGISTERED,
        DepthSource.VEJLEDENDE,
        DepthSource.FEATURE_MEAN,
        DepthSource.GROUND_PLANE,
    })
)

COMPONENT_DEPTH_CONFIG = DepthConfig(
    enabled_levels=frozenset({
        DepthSource.REGISTERED,
        DepthSource.LAYER_MEAN,
        DepthSource.GROUND_PLANE,
    })
)

DEPTH_STATS_KEY = {
    DepthSource.VEJLEDENDE:   "estimated",
    DepthSource.FEATURE_MEAN: "fallback_feature_mean",
    DepthSource.LAYER_MEAN:   "fallback_layer_mean",
    DepthSource.GROUND_PLANE: "fallback_global",
}

# ─────────────────────────────────────────────────────────────────────────────
# SEGMENT VIEWER DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
PLY_HEADER_ROWS    = 11
CLASS_COLUMN       = 6
TARGET_CLASS       = 1       # "Other Utility"
VOXEL_SIZE         = 0.01    # metres, downsample before clustering
MIN_CLUSTER_SIZE   = 100
MIN_SAMPLES        = 5
POINT_SIZE         = 2.0
MIN_INSTANCE_POINTS = 250

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY-TO-LER MATCHING (for deviation viewer)
# ─────────────────────────────────────────────────────────────────────────────
UTILITY_TO_LER_MATCH = {
    1: {"layers": {"Elledning"}},
    2: {"layers": {"Afloebsledning"}},
    3: {"layers": {"Olieledning"}},
    4: {"layers": {"Gasledning"}},
    5: {"layers": {"TermiskLedning"}},
    6: {"layers": {"Foeringsroer"}},
    7: {"layers": {"Vandledning"}},
    8: {"layers": {"Telekommunikationsledning"}},
    9: {"layers": {"AndenLedning"}},
    10: {"layers": {"LedningUkendtForsyningsart"}},
}
