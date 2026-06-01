# -*- coding: utf-8 -*-
"""
Centralised configuration for all viewers and tools.
=====================================================
Change the site by editing PLY_FILE (and optionally GML_PATH /
AREA_REF_GEOJSON if working with a different Ledningspakke).
All other scripts import from here — nothing is duplicated.
"""

from pathlib import Path
from enum import IntEnum
from dataclasses import dataclass
import warnings

# Suppress noisy pyogrio/GDAL warnings from GML files with null geometries
# and non-numeric values in integer fields. Applied globally so every script
# that imports core.config gets the filter automatically.
warnings.filterwarnings("ignore", message="Unrecognized geometry type", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="parsed incompletely", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# SITE SELECTION — change these to switch site
# ─────────────────────────────────────────────────────────────────────────────
PLY_FILE = (
    r"C:\Users\bosre\OneDrive - University of Twente\Documents\AAU UTwente thesis"
    r"\Python\Thesis\Data\OpenTrench3D\Water_Area_5\Area_5_Site_11.ply"
)

AREA_REF_GEOJSON = (
    r"C:\Users\bosre\OneDrive - University of Twente\Documents\AAU UTwente thesis"
    r"\Python\Thesis\Data\Translation_coordinates\area_points_utm32_etrs89.geojson"
)

GML_PATH = (
    r"C:\Users\bosre\OneDrive - University of Twente\Documents\AAU UTwente thesis"
    r"\Python\Thesis\Data\Ledningspakke_2803288_Area_4_and_5\consolidated.gml"
)

PLY_BASE_DIR = (
    r"C:\Users\bosre\OneDrive - University of Twente\Documents\AAU UTwente thesis"
    r"\Python\Thesis\Data\OpenTrench3D"
)

# Circular crop radius (metres) around the point cloud centroid (XY).
CROP_RADIUS = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# CLASS LABEL DEFINITIONS — OpenTrench3D semantic classes
# ─────────────────────────────────────────────────────────────────────────────
CLASS_LABELS = {
    0: {"name": "Main Utility",     "color": [0.00, 0.80, 0.00]},
    1: {"name": "Other Utility",    "color": [1.00, 1.00, 0.00]},
    2: {"name": "Trench",           "color": [0.55, 0.27, 0.07]},
    3: {"name": "Inactive Utility", "color": [0.00, 0.00, 0.00]},
    4: {"name": "Misc",             "color": [0.60, 0.60, 0.60]},
}

DEFAULT_CLASS_COLOR = [1.0, 0.0, 1.0]  # magenta — unknown class IDs

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY LAYER DEFINITIONS — DLF-recommended colours (RGB 0-1)
# ─────────────────────────────────────────────────────────────────────────────
LINE_LAYERS = {
    "Vandledning":               {"color": [0.000, 0.000, 1.000], "fallback_radius": 0.005},  # DLF blue
    "Afloebsledning":            {"color": [1.000, 0.000, 0.000], "fallback_radius": 0.005},  # DLF red
    "Gasledning":                {"color": [1.000, 0.600, 0.000], "fallback_radius": 0.005},  # DLF orange
    "Elledning":                 {"color": [1.000, 0.000, 0.000], "fallback_radius": 0.005},  # DLF red
    "Telekommunikationsledning": {"color": [0.980, 0.588, 0.275], "fallback_radius": 0.005},  # DLF orange
    "Foeringsroer":              {"color": [0.882, 0.882, 0.882], "fallback_radius": 0.005},  # DLF grey
    "LedningUkendtForsyningsart":{"color": [0.300, 0.800, 0.800], "fallback_radius": 0.005},  # cyan
    "Ledningstrace":             {"color": [0.980, 0.588, 0.275], "fallback_radius": 0.005},  # DLF trace orange
    "TermiskLedning":            {"color": [1.000, 0.000, 1.000], "fallback_radius": 0.005},  # DLF violet
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
}

# ─────────────────────────────────────────────────────────────────────────────
# FORSYNINGSART keyword -> colour for Ledningstrace sub-groups
# ─────────────────────────────────────────────────────────────────────────────
FORSYNINGSART_COLOR_HINTS = [
    ("vand",   [0.000, 0.000, 1.000]),   # DLF vand blue
    ("afloeb", [1.000, 0.000, 0.000]),   # DLF spildevand red
    ("spilde", [1.000, 0.000, 0.000]),   # DLF spildevand red
    ("gas",    [1.000, 0.600, 0.000]),   # DLF gas orange
    ("el",     [1.000, 0.000, 0.000]),   # DLF el red
    ("tele",   [0.980, 0.588, 0.275]),   # DLF tele orange
    ("kommu",  [0.980, 0.588, 0.275]),   # DLF kommunikation orange
    ("varme",  [1.000, 0.000, 1.000]),   # DLF varme violet
    ("fjern",  [1.000, 0.000, 1.000]),   # DLF fjernvarme violet
    ("olie",   [0.463, 0.463, 0.463]),   # DLF olie grey
]


def forsyningsart_color(fa_value, fallback):
    """Return a colour for a forsyningsart value by substring matching."""
    fa_lower = fa_value.lower()
    for keyword, color in FORSYNINGSART_COLOR_HINTS:
        if keyword in fa_lower:
            return color
    return fallback


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
    "Class 1:  ≤ 250 mm",
    "Class 2:  ≤ 500 mm",
    "Class 3:  ≤ 1000 mm",
    "Class 4:  ≤ 2000 mm",
    "Class 5:  > 2000 mm",
]

# ─────────────────────────────────────────────────────────────────────────────
# DEPTH HIERARCHY — enum, config (used by BASE1 and LABEL1)
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
VOXEL_SIZE         = 0.01    # metres — downsample before clustering
MIN_CLUSTER_SIZE   = 100
MIN_SAMPLES        = 5
POINT_SIZE         = 2.0
MIN_INSTANCE_POINTS = 250

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY-TO-LER MATCHING (for deviation viewer)
# ─────────────────────────────────────────────────────────────────────────────
UTILITY_TO_LER_MATCH = {
    1: {"layers": {"Elledning"},                                   "trace_kw": ["el"]},
    2: {"layers": {"Afloebsledning"},                              "trace_kw": ["afloeb", "spilde"]},
    3: {"layers": set(),                                           "trace_kw": ["olie"]},
    4: {"layers": {"Gasledning"},                                  "trace_kw": ["gas"]},
    5: {"layers": {"TermiskLedning"},                              "trace_kw": ["varme", "fjern"]},
    6: {"layers": {"Foeringsroer"},                                "trace_kw": []},
    7: {"layers": {"Vandledning"},                                 "trace_kw": ["vand"]},
    8: {"layers": {"Telekommunikationsledning"},                   "trace_kw": ["tele", "kommu"]},
    9: {"layers": set(),                                           "trace_kw": []},
    10: {"layers": {"LedningUkendtForsyningsart"},                 "trace_kw": []},
}
