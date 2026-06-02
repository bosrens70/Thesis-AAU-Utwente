# -*- coding: utf-8 -*-
"""
Ledningstrace utility helpers for consistent variant tracking and coloring.
=====================================================================
Provides shared logic for detecting, tracking, and rendering Ledningstrace
features by forsyningsart (utility type) across all viewers.
"""

from core.config import forsyningsart_color


def get_ledningstrace_display_info(layer_name, row, default_color):
    """
    Extract Ledningstrace display information (color and forsyningsart).

    Returns:
        tuple: (is_trace, display_fa, color)
            is_trace: bool - True if this is a Ledningstrace feature
            display_fa: str or None - forsyningsart value if present and valid
            color: list - RGB color [r, g, b] for rendering
    """
    is_trace = (layer_name == "Ledningstrace")
    display_fa = None
    color = default_color

    if is_trace and "forsyningsart" in row.index:
        fa = str(row.get("forsyningsart", "") or "").strip()
        if fa:
            color = forsyningsart_color(fa, default_color)
            display_fa = fa

    return is_trace, display_fa, color


def get_storage_key(layer_name, display_fa):
    """
    Generate storage key for mesh groups.

    For Ledningstrace features with forsyningsart, returns compound key
    like "Ledningstrace (vand)"; otherwise returns layer_name.
    """
    if layer_name == "Ledningstrace" and display_fa:
        return f"Ledningstrace ({display_fa})"
    return layer_name


def get_bredde_width(row):
    """
    Extract and convert bredde (width) attribute from GML row.

    Ledningstrace widths are specified in millimeters in GML;
    returns value in meters.

    Returns:
        float or None: Width in meters, or None if not available
    """
    if "bredde" not in row.index:
        return None

    try:
        b = float(row["bredde"] or 0)
        if b > 0:
            return b / 1000.0  # mm to m
    except (ValueError, TypeError):
        pass

    return None


def is_ledningstrace(layer_name):
    """Check if a layer is Ledningstrace."""
    return layer_name == "Ledningstrace"
