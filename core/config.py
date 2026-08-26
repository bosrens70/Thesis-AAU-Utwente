# -*- coding: utf-8 -*-
"""
Centralised configuration for all viewers and tools.
=====================================================
The active site and Ledningspakke are set in core/site_local.py (gitignored);
everything else is defined here. All other scripts import from here, nothing
is duplicated.
"""

import os
import re
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
# SITE SELECTION: set in core/site_local.py (gitignored)
# ─────────────────────────────────────────────────────────────────────────────
# The active site and Ledningspakke are the single responsibility of
# core/site_local.py, a gitignored file, so switching sites never dirties a
# tracked file. There is deliberately no committed default here: site_local.py
# is the one source of truth, and its absence (or a missing field) raises a
# clear error rather than silently falling back to some other site.
try:
    from core import site_local as _site_local
except ImportError as _e:
    raise RuntimeError(
        "core/site_local.py is missing. It is gitignored, so each machine sets "
        "its own active site there. Create core/site_local.py with:\n"
        '    SITE = "Water_Area_5/Area_5_Site_11.ply"\n'
        '    LEDNINGSPAKKE_DIR = "Ledningspakke_2803288_Area_4_and_5"'
    ) from _e

try:
    _site_rel = _site_local.SITE
    _ledningspakke_dir = _site_local.LEDNINGSPAKKE_DIR
except AttributeError as _e:
    raise RuntimeError(
        f"core/site_local.py must define both SITE and LEDNINGSPAKKE_DIR ({_e})."
    ) from _e

PLY_FILE          = PLY_BASE_DIR / _site_rel
AREA_REF_GEOJSON  = DATA_DIR / "Translation_coordinates" / "area_points_utm32_etrs89.geojson"
GML_PATH          = DATA_DIR / _ledningspakke_dir / "consolidated.gml"
# Name of the active Ledningspakke, for display in the viewers. Derived from the
# site selection above, so a GUI label can never disagree with the loaded data.
LEDNINGSPAKKE_NAME = _ledningspakke_dir

# Short display form ("Ledningspakke 2803288") for the legend headers, so every
# viewer shows the same compact title. Falls back to the raw directory name.
_lp_match = re.match(r"(Ledningspakke)[_\s]*(\d+)", _ledningspakke_dir, re.IGNORECASE)
LEDNINGSPAKKE_LABEL = (f"{_lp_match.group(1)} {_lp_match.group(2)}"
                       if _lp_match else _ledningspakke_dir)



# Crop region shape (load-time switch).
#   "circle" — disc of radius CROP_RADIUS around the cloud XY centroid (the cloud
#              is cropped to that disc).
#   "rect"   — the cloud is kept in full; its XY bounding box is expanded by
#              UTILITY_RECT_BUFFER, and utilities are selected and clipped to that
#              box.  Selection is XY-only, so a utility passing through the
#              footprint is kept whatever its depth.  CROP_RADIUS is not used here.
# Honoured by the point-cloud crop (every init_site viewer) and by utility
# selection in base_module / label_module / deviation_module.  Other modules keep
# their own selection logic.  Set to "circle" to restore the legacy disc crop.
CROP_MODE = "rect"

# Margin (metres) added around the point-cloud XY bounding box when selecting
# utilities in "rect" mode (the "additional crop distance").  CropRegion applies
# it in X and Y only; there is no Z bound.
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

# Ledningstrace rendering. A trace is a corridor: a registered centreline plus a
# width ("bredde"). Drawn at full opacity the corridor ribbon hides the utilities
# beneath it, so the ribbon is rendered at this fraction of the current LER
# opacity while the centreline is drawn like any other utility (see
# core/trace_render.py).
LEDNINGSTRACE_ALPHA_SCALE = 0.7

# Display radius of the trace centreline tube, in metres. Deliberately a fixed
# display value (the same fallback pipes use), never derived from "bredde":
# bredde is a corridor width, not a utility cross-section, so the drawn tube
# makes no claim about the physical size of anything in the corridor.
TRACE_CENTERLINE_RADIUS = 0.01

# Width of the right-hand control panel in every GUI module, in multiples of the
# theme font size (em). Kept here so the panels line up across modules; each
# module derives its pixel width as int(PANEL_WIDTH_EM * em) rather than
# hardcoding one, so the panel scales with the font/DPI.
PANEL_WIDTH_EM = 20

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
# ENGLISH DISPLAY NAMES (GUI legends; the thesis report is in English)
# ─────────────────────────────────────────────────────────────────────────────
# Keys are the real GML layer names, which stay Danish everywhere in code and
# data access (gpd.read_file layer names, dict keys, geometry names, JSON);
# only the text shown in the GUI is translated.
#
# The values are the names Section 4.1 of the manuscript introduces, each one
# checked against the class definition in the LER 2025 featurekatalog, which
# carries no English of its own. Two of them are not the obvious translation:
#   * TermiskLedning is "hvorigennem varme eller kulde foeres", heat *or* cold,
#     and forsyningsart carries both fjernvarme and fjernkoeling, so it is not
#     a district heating pipe;
#   * Foeringsroer is "roer hvori der kan foeres en eller flere ledninger", a
#     conduit lines *can* run in. Whether any do is what indeholderLedninger
#     records, so the type itself is not an empty conduit.
# "Ledningstrace" is the one value that is not the name Section 4.1 uses: that
# section introduces it as "Utility Trace", but Chapter 6 and its figures call
# it "Trace", so the short form is what is shown.
LAYER_DISPLAY_EN = {
    # Utility lines
    "Vandledning":                 "Water Pipe",
    "Afloebsledning":              "Drainage Pipe",
    "Gasledning":                  "Gas Pipe",
    "Olieledning":                 "Oil Pipe",
    "Elledning":                   "Electricity Cable",
    "Telekommunikationsledning":   "Telecommunication Cable",
    "TermiskLedning":              "Thermal Pipe",
    "Foeringsroer":                "Conduit",
    "LedningUkendtForsyningsart":  "Utility Line of Unknown Service Type",
    "AndenLedning":                "Other Utility Line",
    "Ledningstrace":               "Trace",
    # Components, each defined as a component belonging to the matching line
    "Vandkomponent":               "Water Component",
    "Afloebskomponent":            "Drainage Component",
    "Gaskomponent":                "Gas Component",
    "Oliekomponent":               "Oil Component",
    "Elkomponent":                 "Electricity Component",
    "Telekommunikationskomponent": "Telecommunication Component",
    "TermiskKomponent":            "Thermal Component",
    "AndenKomponent":              "Other Component",
}

# forsyningsart attribute values (Danish, straight from the GML) -> English,
# for the "Ledningstrace (<forsyningsart>)" legend variants.
FORSYNINGSART_DISPLAY_EN = {
    "vand":              "water",
    "afloeb":            "drainage",
    "afløb":             "drainage",
    "spildevand":        "wastewater",
    "vejafvanding":      "road drainage",
    "gas":               "gas",
    "el":                "electricity",
    "telekommunikation": "telecommunication",
    "fjernvarme":        "district heating",
    "fjernkoeling":      "district cooling",
    "fjernkøling":       "district cooling",
    "varme":             "heating",
    "termisk":           "thermal",
    "olie":              "oil",
    "anden":             "other",
    "andet":             "other",
    "ukendt":            "unknown",
}


def layer_display_name(key):
    """English display text for a layer key, including compound
    ``"Ledningstrace (<forsyningsart>)"`` storage keys. Unknown keys are
    returned unchanged, so data-driven values never crash the GUI."""
    key = str(key)
    m = re.match(r"^Ledningstrace \((.+)\)$", key)
    if m:
        fa = m.group(1)
        fa_en = FORSYNINGSART_DISPLAY_EN.get(fa.strip().lower(), fa)
        return f"{LAYER_DISPLAY_EN['Ledningstrace']} ({fa_en})"
    return LAYER_DISPLAY_EN.get(key, key)


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
# The five-class palette of the whole project: a measured deviation binned into
# these classes and a registered accuracy class (ACCURACY_CLASS_COLORS below)
# share it, so a colour always means the same class bound wherever it appears.
DEVIATION_COLORS = [
    [0.00, 0.70, 0.20],   # Class 1: <= 0.25 m - green
    [1.00, 0.95, 0.00],   # Class 2: <= 0.50 m - yellow
    [1.00, 0.25, 0.00],   # Class 3: <= 1.00 m - orange
    [0.60, 0.00, 0.00],   # Class 4: <= 2.00 m - dark red
    [0.55, 0.15, 0.75],   # Class 5: > 2.00 m  - purple
]

DEVIATION_CLASS_LABELS = [
    "Class 1:  <= 0.25 m",
    "Class 2:  <= 0.50 m",
    "Class 3:  <= 1.00 m",
    "Class 4:  <= 2.00 m",
    "Class 5:  > 2.00 m",
]

# Continuous gradient legend ticks (metres): the DEVIATION_THRESHOLDS anchors
# plus interpolated midpoints, so the gradient legend shows the smooth
# interpolation between the five accuracy-class anchor colours.
DEVIATION_GRADIENT_TICKS = [0.00, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]

# Two-class pass/fail scheme for the "KLIC XY deviation (discrete)" colour
# mode, matching Dutch KLIC/WIBON excavation practice: a 1 m horizontal
# deviation is treated as a single pass/fail threshold rather than a graded
# accuracy scale (see the WIBON minimum-tolerance discussion in the thesis
# background chapter). Its own pass/fail pair, not two colours borrowed from
# DEVIATION_COLORS: green/red carries "within tolerance" directly, which a rank
# on a five-class scale does not.
KLIC_XY_THRESHOLDS = [0.00, 1.00]
KLIC_XY_COLORS = [
    [0.0, 0.7, 0.2],   # Class 1: <= 1.00 m - green (within KLIC tolerance)
    [0.8, 0.0, 0.0],   # Class 2: > 1.00 m  - red (exceeds KLIC tolerance)
]
KLIC_XY_CLASS_LABELS = [
    "Class 1:  <= 1.00 m",
    "Class 2:  > 1.00 m",
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

# Display colours for the five registered classes. Deliberately the same palette
# as the deviation classes: the bounds are identical, so a colour reads as one
# class throughout. Named separately because the two are different quantities (a
# registered claim, not a measured deviation), and only the legend header and the
# colour mode say which is on screen. Give this its own list to break the tie.
ACCURACY_CLASS_COLORS = DEVIATION_COLORS

# Colour for a feature that registers no usable accuracy class, used wherever
# geometry is painted by class (ACCURACY_CLASS_COLORS gives the five classes).
# Kept distinct from every class colour: an unregistered class is missing data,
# not a coarse one, and how often it happens is itself a result.
ACCURACY_UNREGISTERED_COLOR = [0.55, 0.55, 0.55]
ACCURACY_UNREGISTERED_LABEL = "not registered"


def accuracy_class_color(class_idx):
    """Display colour for a registered accuracy class index (1..5), or the
    unregistered grey for 0 / None / anything outside that range."""
    if class_idx and 1 <= int(class_idx) <= len(ACCURACY_CLASS_COLORS):
        return ACCURACY_CLASS_COLORS[int(class_idx) - 1]
    return ACCURACY_UNREGISTERED_COLOR


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

# Features whose registered Z came from the same survey campaign that captured
# the point clouds, so the two sides of the comparison are not independent and
# a vertical agreement between them measures the capture, not the register.
# Identified from the evidence rather than assumed: this owner's water main is
# registered on one date across 45 sites and its measured crown agrees with the
# registered top to within 75 micrometres, which no pair of independent
# measurements achieves. Keyed on (ledningsejer, etableringstidspunkt) so a
# later package cannot silently inherit the exemption.
SAME_SURVEY_CAMPAIGN = {("31884993", "2022-06-09")}

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
# UTILITY LINE GROUPING (see core/ler_lines.py)
# ─────────────────────────────────────────────────────────────────────────────
# One physical utility is often registered as a chain of separate features, so
# an instance is linked to the whole chain rather than to one gml_id. Features
# that share a geometry node are one chain, so this distance is the only
# geometric threshold in the grouping; everything else it demands is an
# attribute agreement (see core/ler_lines.py).

# Metres. How close two geometry nodes must be to count as the same node. The
# junctions observed in this data are exact (0.000 m) while the nearest
# non-junction pair, two cables leaving one cabinet, is 0.200 m apart, so the
# threshold sits well inside that gap.
LINE_JOIN_TOL = 0.01

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

# Base GML layer name of a trace. The viewers store one mesh group per
# forsyningsart under a compound key, "Ledningstrace (vand)", so any code that
# has to recognise a trace from its storage key tests this prefix.
LEDNINGSTRACE_LAYER = "Ledningstrace"

# Corridor width (metres) assumed for a Ledningstrace that registers no bredde.
# 186 of the 516 traces in package 2803288 carry none; the 330 that do all carry
# 500 mm. This is deliberately half that, because the half-width is subtracted
# from the measured offset and clamped at zero, so a narrower assumed corridor
# can only increase the deviation reported. Erring towards reporting a deviation
# is the safe direction, but it makes any deviation measured against this value
# a departure from an assumption rather than from a registered corridor, which
# is why the reported figures state how many instances reached it.
LEDNINGSTRACE_FALLBACK_WIDTH = 0.25


def trace_forsyningsart(layer_key):
    """The forsyningsart carried by a Ledningstrace storage key, lowercased.

    ``"Ledningstrace (Vand)"`` gives ``"vand"``. A trace recorded without a
    forsyningsart gives ``""``, and a key that is not a trace at all gives
    ``None``, so the return value doubles as the is-a-trace test.
    """
    key = str(layer_key)
    if not key.startswith(LEDNINGSTRACE_LAYER):
        return None
    if "(" not in key:
        return ""
    return key.split("(")[-1].rstrip(")").strip().lower()


def ler_layers_for_type(utility_type, available_layers=(), *,
                        include_components=False):
    """The LER layers a given utility type may legitimately be matched against.

    A type maps to its own line layer through UTILITY_TO_LER_MATCH, but a trace
    is registered under a layer of its own, so ``Ledningstrace (telekommunikation)``
    is just as valid a counterpart for a telecom instance as
    ``Telekommunikationsledning`` is. This resolves both, plus optionally the
    component layers belonging to those lines, against the layer names actually
    present in ``available_layers``.

    Returns ``None`` when the type carries no mapping at all (an unlabelled
    instance). ``None`` means "this type says nothing about which layers apply",
    not "every layer applies": each caller decides what to do with it, and the
    deviation module treats it as matching no layer so that a missing label
    cannot produce a deviation against arbitrary geometry.
    """
    match = UTILITY_TO_LER_MATCH.get(utility_type)
    if match is None:
        return None
    line_layers = set(match["layers"])
    comp_layers = ({c for c, parent in COMP_TO_LINE.items() if parent in line_layers}
                   if include_components else set())
    allowed = set()
    for name in available_layers:
        if name in line_layers or name in comp_layers:
            allowed.add(name)
            continue
        fa = trace_forsyningsart(name)
        if fa and FORSYNINGSART_TO_LINE.get(fa) in line_layers:
            allowed.add(name)
    return allowed
