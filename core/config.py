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
# Switching sites while working (a routine, frequent action) should not dirty
# a tracked file. The default below is the committed fallback; day-to-day
# switching instead goes in core/site_local.py, a gitignored override (fields
# documented in that file's docstring). If that file is absent or missing a
# field, the corresponding default here is used.
_DEFAULT_SITE_REL          = "Water_Area_5/Area_5_Site_37.ply"
_DEFAULT_LEDNINGSPAKKE_DIR = "Ledningspakke_2803288_Area_4_and_5"

_site_rel = _DEFAULT_SITE_REL
_ledningspakke_dir = _DEFAULT_LEDNINGSPAKKE_DIR
try:
    from core import site_local as _site_local
    _site_rel = getattr(_site_local, "SITE", _site_rel)
    _ledningspakke_dir = getattr(_site_local, "LEDNINGSPAKKE_DIR", _ledningspakke_dir)
except ImportError:
    pass

PLY_FILE         = PLY_BASE_DIR / _site_rel
AREA_REF_GEOJSON = DATA_DIR / "Translation_coordinates" / "area_points_utm32_etrs89.geojson"
GML_PATH         = DATA_DIR / _ledningspakke_dir / "consolidated.gml"



# Crop region shape (load-time switch).
#   "circle" — disc of radius CROP_RADIUS around the cloud XY centroid (the cloud
#              is cropped to that disc).
#   "rect"   — the cloud is kept in full; its 3D axis-aligned bounding box (AABB)
#              is expanded by UTILITY_RECT_BUFFER in X, Y and Z, and utilities are
#              selected and clipped to that box.  CROP_RADIUS is not used here.
# Honoured by the point-cloud crop (every init_site viewer) and by utility
# selection in base_module / label_module / deviation_module.  Other modules keep
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
    "Vandledning":               {"color": [0.016, 0.184, 0.723], "fallback_radius": 0.010},  # screenshot blue #2277DD
    "Afloebsledning":            {"color": [0.068, 0.042, 0.005], "fallback_radius": 0.010},  # screenshot dark brown #4A3A0F
    "Gasledning":                {"color": [0.913, 0.651, 0.000], "fallback_radius": 0.010},  # screenshot yellow #F5D300
    "Elledning":                 {"color": [0.768, 0.002, 0.007], "fallback_radius": 0.010},  # screenshot red #E30613
    "Telekommunikationsledning": {"color": [0.038, 0.376, 0.038], "fallback_radius": 0.010},  # screenshot green #37A537
    "Foeringsroer":              {"color": [0.905, 0.352, 0.254], "fallback_radius": 0.010},  # screenshot salmon #F4A08A
    "LedningUkendtForsyningsart":{"color": [0.855, 0.184, 0.006], "fallback_radius": 0.010},  # screenshot orange #EE7711
    "Ledningstrace":             {"color": [0.980, 0.588, 0.275], "fallback_radius": 0.010},  # DLF trace orange
    "TermiskLedning":            {"color": [0.445, 0.042, 0.445], "fallback_radius": 0.010},  # screenshot magenta #B23AB2
    "Olieledning":               {"color": [0.463, 0.463, 0.463], "fallback_radius": 0.010},  # DLF grey
    "AndenLedning":              {"color": [0.800, 0.800, 0.800], "fallback_radius": 0.010},  # grey
}

# Right-panel legend order: keep Ledningstrace last (dense / low-priority visually)
PIPE_LEGEND_UI_ORDER = [ln for ln in LINE_LAYERS if ln != "Ledningstrace"]
if "Ledningstrace" in LINE_LAYERS:
    PIPE_LEGEND_UI_ORDER.append("Ledningstrace")

COMPONENT_LAYERS = {
    "Vandkomponent":                  {"color": [0.016, 0.184, 0.723]},  # screenshot blue #2277DD
    "Afloebskomponent":               {"color": [0.068, 0.042, 0.005]},  # screenshot dark brown #4A3A0F
    "Gaskomponent":                   {"color": [0.913, 0.651, 0.000]},  # screenshot yellow #F5D300
    "Elkomponent":                    {"color": [0.768, 0.002, 0.007]},  # screenshot red #E30613
    "Telekommunikationskomponent":    {"color": [0.038, 0.376, 0.038]},  # screenshot green #37A537
    "TermiskKomponent":               {"color": [0.445, 0.042, 0.445]},  # screenshot magenta #B23AB2
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
    # Screenshot utility colours (linear RGB). Longer/more specific keywords come
    # first so substrings do not shadow (e.g. "tele" before "el", since
    # "telekommunikation" contains "el").
    ("fjern",   [0.445, 0.042, 0.445]),   # termisk / fjernvarme magenta #B23AB2
    ("varme",   [0.445, 0.042, 0.445]),   # termisk / varme magenta #B23AB2
    ("termisk", [0.445, 0.042, 0.445]),   # termisk magenta #B23AB2
    ("tele",    [0.038, 0.376, 0.038]),   # telekom green #37A537
    ("kommu",   [0.038, 0.376, 0.038]),   # kommunikation green #37A537
    ("foering", [0.905, 0.352, 0.254]),   # foeringsroer salmon #F4A08A
    ("foring",  [0.905, 0.352, 0.254]),   # foringsror salmon #F4A08A
    ("føring",  [0.905, 0.352, 0.254]),   # føringsrør salmon #F4A08A
    ("afloeb",  [0.068, 0.042, 0.005]),   # afloeb dark brown #4A3A0F
    ("afl",     [0.068, 0.042, 0.005]),   # afloeb / afløb dark brown #4A3A0F
    ("spilde",  [0.068, 0.042, 0.005]),   # spildevand dark brown #4A3A0F
    ("vejafv",  [0.068, 0.042, 0.005]),   # vejafvanding dark brown #4A3A0F
    ("vand",    [0.016, 0.184, 0.723]),   # vand blue #2277DD
    ("gas",     [0.913, 0.651, 0.000]),   # gas yellow #F5D300
    ("el",      [0.768, 0.002, 0.007]),   # el red #E30613 (after "tele")
    ("olie",    [0.463, 0.463, 0.463]),   # olie grey
    ("ukendt",  [0.855, 0.184, 0.006]),   # ukendt orange #EE7711
    ("anden",   [0.800, 0.800, 0.800]),   # anden / andet grey
    ("andet",   [0.800, 0.800, 0.800]),   # anden / andet grey
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
    0: [0.50, 0.50, 0.50],   # Unlabeled           - grey
    1: [0.768, 0.002, 0.007],# PowerLine            - screenshot red #E30613
    2: [0.068, 0.042, 0.005],# DrainageLine         - screenshot dark brown #4A3A0F
    3: [0.46, 0.46, 0.46],   # OilPipeLine          - DLF grey
    4: [0.913, 0.651, 0.000],# GasLine              - screenshot yellow #F5D300
    5: [0.445, 0.042, 0.445],# ThermalLine          - screenshot magenta #B23AB2
    6: [0.905, 0.352, 0.254],# Conduit              - screenshot salmon #F4A08A
    7: [0.016, 0.184, 0.723],# WaterLine            - screenshot blue #2277DD
    8: [0.038, 0.376, 0.038],# TelecomunicationLine - screenshot green #37A537
    9: [0.80, 0.80, 0.80],   # OtherLine            - grey
    10: [0.855, 0.184, 0.006],# LineUnknownServiceType - screenshot orange #EE7711
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
# LER LINE SIGNATURES (ERR_module top-view plan)
# ─────────────────────────────────────────────────────────────────────────────
# Cartographic line styles reproduced from the LER "Signaturforklaring".
# All sizes are in metres and world-scaled: they zoom with the scene. Consumed
# by core.symbology and the ERR_module top-view.
SIGNATURE_DASH_LEN          = 0.60   # m, dash length for driftsstatus "under etablering"
SIGNATURE_GAP_LEN           = 0.40   # m, gap between dashes
# Signatures are drawn as flat horizontal ribbons, so widths are in metres
# (world-scaled: they zoom with the scene).
SIGNATURE_LINE_WIDTH_M      = 0.10   # m, ordinary utility line ribbon ("Ledning")
SIGNATURE_TRACE_WIDTH_M     = 0.30   # m, Ledningstrace ribbon ("Trace")
SIGNATURE_COMP_LINE_WIDTH_M = 0.20   # m, component line ribbon ("Komponent linje")
SIGNATURE_TICK_BAR_WIDTH_M  = 0.05   # m, thickness of an El voltage tick bar
SIGNATURE_TICK_COLOR        = [0.0, 0.0, 0.0]   # linear black (voltage ticks)
SIGNATURE_Z_LIFT            = 0.03   # m, lift decorators above the base line (avoid z-fight)

# El voltage classes -> number of tick-mark groups (spaendingsniveau in kV).
# Legend bins: < 1, 1-29, 30-130, > 131 kV  ->  0, 1, 2, 3 ticks.
VOLTAGE_KV_THRESHOLDS     = [1.0, 30.0, 131.0]
SIGNATURE_TICK_SPACING    = 1.50   # m between tick groups along the line
SIGNATURE_TICK_LEN        = 0.50   # m, full width of a tick bar across the line
SIGNATURE_TICK_GAP        = 0.14   # m between ticks within a 2- or 3-tick group

# Danger class ("fareklasse") that gets the red triangle signature.
SIGNATURE_HAZARD_VALUES   = ("meget farlig",)
SIGNATURE_HAZARD_SPACING  = 1.50   # m between triangles along the line
SIGNATURE_HAZARD_SIZE     = 0.40   # m, triangle edge scale
SIGNATURE_HAZARD_COLOR    = [0.80, 0.0, 0.0]   # linear red (legend triangles)

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
