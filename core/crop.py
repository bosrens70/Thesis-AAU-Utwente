# -*- coding: utf-8 -*-
"""
Crop-region selection and clipping.
===================================
Pure-numpy primitives for the circular and rectangular crop regions, plus the
CropRegion wrapper that bundles the CROP_MODE dispatch and the local/UTM frame
offset. Deliberately Open3D-free so it stays headless-testable (core.geometry
re-exports the primitives for existing importers).
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CIRCLE PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def clip_segment_to_circle(p1, p2, center_x, center_y, radius):
    """
    Clip a 3D line segment (p1 -> p2) to a circular crop disc in XY.
    Circle defined by (center_x, center_y) and radius.
    Returns (clipped_p1, clipped_p2) or None if entirely outside.
    Z is linearly interpolated along the segment parameter.
    """
    r2 = radius * radius
    x1 = p1[0] - center_x
    y1 = p1[1] - center_y
    x2 = p2[0] - center_x
    y2 = p2[1] - center_y

    dx = x2 - x1
    dy = y2 - y1
    a  = dx * dx + dy * dy

    if a < 1e-12:
        # Degenerate — segment is a single point
        if x1 * x1 + y1 * y1 <= r2:
            return p1, p2
        return None

    b = 2.0 * (x1 * dx + y1 * dy)
    c = x1 * x1 + y1 * y1 - r2
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None

    sq      = np.sqrt(disc)
    t_enter = (-b - sq) / (2.0 * a)
    t_exit  = (-b + sq) / (2.0 * a)

    t0 = max(0.0, t_enter)
    t1 = min(1.0, t_exit)
    if t0 > t1:
        return None

    c1 = p1 + t0 * (p2 - p1)
    c2 = p1 + t1 * (p2 - p1)
    return c1, c2


def segments_in_crop(coords_utm, center_x_utm, center_y_utm, crop_radius):
    """
    Conservative check: does any part of the polyline (coords_utm) fall
    within the circular crop (in UTM)?  First checks vertex-in-circle,
    then falls back to AABB overlap.
    """
    r2 = crop_radius * crop_radius
    dx = coords_utm[:, 0] - center_x_utm
    dy = coords_utm[:, 1] - center_y_utm
    d2 = dx * dx + dy * dy
    if (d2 <= r2).any():
        return True
    # AABB fallback
    xs, ys = coords_utm[:, 0], coords_utm[:, 1]
    if xs.max() < center_x_utm - crop_radius: return False
    if xs.min() > center_x_utm + crop_radius: return False
    if ys.max() < center_y_utm - crop_radius: return False
    if ys.min() > center_y_utm + crop_radius: return False
    return True


def point_in_crop(x, y, center_x, center_y, crop_radius):
    """Check if a single point (x, y) lies within the circular crop."""
    dx = x - center_x
    dy = y - center_y
    return (dx * dx + dy * dy) <= (crop_radius * crop_radius)


# ─────────────────────────────────────────────────────────────────────────────
# RECTANGLE PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def segments_in_rect(coords, min_x, min_y, max_x, max_y):
    """
    Conservative AABB-overlap test: does the polyline ``coords`` (N, 2+) fall
    within the axis-aligned rectangle [min_x, max_x] x [min_y, max_y]?
    The segment clipper makes the final call for crossing segments.
    """
    xs, ys = coords[:, 0], coords[:, 1]
    if xs.max() < min_x or xs.min() > max_x:
        return False
    if ys.max() < min_y or ys.min() > max_y:
        return False
    return True


def point_in_rect(x, y, min_x, min_y, max_x, max_y):
    """Check if a single point (x, y) lies within the axis-aligned rectangle."""
    return (min_x <= x <= max_x) and (min_y <= y <= max_y)


def clip_segment_to_rect(p1, p2, min_x, min_y, max_x, max_y):
    """
    Liang-Barsky clip of a 3D segment (p1 -> p2) to an axis-aligned XY
    rectangle.  Z is linearly interpolated along the segment parameter.
    Returns (clipped_p1, clipped_p2) or None if entirely outside.
    """
    x0, y0 = p1[0], p1[1]
    dx = p2[0] - x0
    dy = p2[1] - y0

    t0, t1 = 0.0, 1.0
    for p_val, q_val in [
        (-dx, x0 - min_x),
        (dx,  max_x - x0),
        (-dy, y0 - min_y),
        (dy,  max_y - y0),
    ]:
        if abs(p_val) < 1e-12:
            # Segment parallel to this edge — reject if it starts outside
            if q_val < 0:
                return None
        else:
            r = q_val / p_val
            if p_val < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return None

    c1 = p1 + t0 * (p2 - p1)
    c2 = p1 + t1 * (p2 - p1)
    return c1, c2


# ─────────────────────────────────────────────────────────────────────────────
# CROP REGION — one object per viewer instead of per-module copies
# ─────────────────────────────────────────────────────────────────────────────

class CropRegion:
    """Crop-region selection and clipping shared by the viewers.

    Bundles the CROP_MODE dispatch ("rect": the point cloud's XY AABB grown by
    a buffer; "circle": a disc around the cloud centroid) with the local/UTM
    frame offset, so the containment tests and the segment clipper exist once
    instead of being copied into every module.

    Selection tests run in UTM (raw GML coordinates); clipping and the local
    containment test run in local coordinates (after translation by TX/TY).
    """

    def __init__(self, mode, TX, TY, *, rect_bounds_local=None,
                 center_local=None, radius=None):
        self.mode = mode
        self.TX = float(TX)
        self.TY = float(TY)
        if mode == "rect":
            if rect_bounds_local is None:
                raise ValueError("rect mode requires rect_bounds_local")
            (self.min_x, self.min_y,
             self.max_x, self.max_y) = (float(v) for v in rect_bounds_local)
            self.min_x_utm = self.min_x + self.TX
            self.max_x_utm = self.max_x + self.TX
            self.min_y_utm = self.min_y + self.TY
            self.max_y_utm = self.max_y + self.TY
        elif mode == "circle":
            if center_local is None or radius is None:
                raise ValueError("circle mode requires center_local and radius")
            self.cx = float(center_local[0])
            self.cy = float(center_local[1])
            self.cx_utm = self.cx + self.TX
            self.cy_utm = self.cy + self.TY
            self.radius = float(radius)
        else:
            raise ValueError(f"unknown crop mode {mode!r}")

    @classmethod
    def from_pointcloud(cls, pc, TX, TY, mode=None, rect_buffer=None,
                        radius=None):
        """Build the region for a loaded ``PointCloudData``.

        ``mode`` / ``rect_buffer`` / ``radius`` default to the config values
        (CROP_MODE / UTILITY_RECT_BUFFER / CROP_RADIUS), matching how every
        viewer derived its region from ``pc_min`` / ``pc_max`` / the crop
        centre before.
        """
        from core.config import CROP_MODE, UTILITY_RECT_BUFFER, CROP_RADIUS
        if mode is None:
            mode = CROP_MODE
        if mode == "rect":
            buf = UTILITY_RECT_BUFFER if rect_buffer is None else rect_buffer
            return cls("rect", TX, TY, rect_bounds_local=(
                pc.pc_min[0] - buf, pc.pc_min[1] - buf,
                pc.pc_max[0] + buf, pc.pc_max[1] + buf))
        r = CROP_RADIUS if radius is None else radius
        return cls("circle", TX, TY,
                   center_local=pc.crop_center_local, radius=r)

    # ── Selection (UTM frame) ────────────────────────────────────────────────
    def contains_utm(self, x, y):
        """Is the UTM point (x, y) inside the crop region?"""
        if self.mode == "rect":
            return point_in_rect(x, y, self.min_x_utm, self.min_y_utm,
                                 self.max_x_utm, self.max_y_utm)
        return point_in_crop(x, y, self.cx_utm, self.cy_utm, self.radius)

    def polyline_in_region_utm(self, coords_utm):
        """Conservative check: any part of the polyline within the region.
        The segment clipper makes the final call for crossing segments."""
        if self.mode == "rect":
            return segments_in_rect(coords_utm, self.min_x_utm, self.min_y_utm,
                                    self.max_x_utm, self.max_y_utm)
        return segments_in_crop(coords_utm, self.cx_utm, self.cy_utm,
                                self.radius)

    # ── Containment / clipping (local frame) ─────────────────────────────────
    def contains_local(self, x, y):
        """Is the local point (x, y) inside the crop region?"""
        if self.mode == "rect":
            return point_in_rect(x, y, self.min_x, self.min_y,
                                 self.max_x, self.max_y)
        return point_in_crop(x, y, self.cx, self.cy, self.radius)

    def clip_local(self, p1, p2):
        """Clip a local 3D segment to the region in XY (Z interpolated).
        Returns (clipped_p1, clipped_p2) or None if entirely outside."""
        if self.mode == "rect":
            return clip_segment_to_rect(p1, p2, self.min_x, self.min_y,
                                        self.max_x, self.max_y)
        return clip_segment_to_circle(p1, p2, self.cx, self.cy, self.radius)
