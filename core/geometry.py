# -*- coding: utf-8 -*-
"""
Shared geometry helpers for Open3D mesh creation and spatial operations.
========================================================================
All functions are stateless — they take coordinates / parameters and
return Open3D geometry objects or numpy arrays.
"""

import open3d as o3d
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# MESH PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def segment_to_cylinder(p1, p2, radius, color, resolution=12):
    """Create an Open3D cylinder mesh between two 3D points."""
    vec    = p2 - p1
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return None

    cyl = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=length, resolution=resolution, split=1)
    z_axis    = np.array([0.0, 0.0, 1.0])
    direction = vec / length
    cross     = np.cross(z_axis, direction)
    cross_norm = np.linalg.norm(cross)
    dot        = np.dot(z_axis, direction)

    if cross_norm > 1e-6:
        axis  = cross / cross_norm
        angle = np.arctan2(cross_norm, dot)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cyl.rotate(R, center=[0.0, 0.0, 0.0])
    elif dot < 0:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([1.0, 0.0, 0.0]) * np.pi)
        cyl.rotate(R, center=[0.0, 0.0, 0.0])

    cyl.translate((p1 + p2) / 2.0)
    cyl.paint_uniform_color(color)
    return cyl


def segment_to_plane(p1, p2, width, color):
    """Create a flat horizontal quad between two 3D points with the given width."""
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return None
    fwd = vec / length
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(fwd, up)
    side_len = np.linalg.norm(side)
    if side_len < 1e-6:
        side = np.array([1.0, 0.0, 0.0])
    else:
        side = side / side_len
    offset = side * (width / 2.0)
    verts = np.array([p1 - offset, p1 + offset, p2 + offset, p2 - offset], dtype=float)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(tris),
    )
    mesh.paint_uniform_color(color)
    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# POINT-TO-SEGMENT DISTANCE
# ─────────────────────────────────────────────────────────────────────────────

def point_to_segment_dists(p, p1s, p2s):
    """
    Vectorised minimum distances from a single point p (3,) to each
    segment p1s[i] -> p2s[i].  Returns dists (N,).
    """
    d     = p2s - p1s                        # (N, 3)
    denom = np.einsum('ij,ij->i', d, d)      # (N,)  squared lengths
    v     = p - p1s                          # (N, 3)
    t     = np.einsum('ij,ij->i', v, d)     # (N,)  dot products
    safe  = denom > 1e-12
    t_clamped = np.where(safe, np.clip(t / np.where(safe, denom, 1.0), 0.0, 1.0), 0.0)
    closest = p1s + t_clamped[:, None] * d
    diff    = p - closest
    dists   = np.sqrt(np.einsum('ij,ij->i', diff, diff))
    return dists


def batch_point_to_segments(pts, seg_p1, seg_p2, batch_size=2000):
    """
    For each point in pts (N, 3), find the minimum distance to any of the
    M segments defined by seg_p1 (M, 3) -> seg_p2 (M, 3).
    Returns dists (N,).  Processes in batches to limit memory.
    """
    N = len(pts)
    M = len(seg_p1)
    if M == 0:
        return np.full(N, np.inf)

    min_dists = np.full(N, np.inf)

    d = seg_p2 - seg_p1                          # (M, 3)
    seg_len2 = np.einsum('ij,ij->i', d, d)       # (M,)
    safe = seg_len2 > 1e-12

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        p = pts[start:end]                        # (B, 3)
        B = len(p)

        v = p[:, None, :] - seg_p1[None, :, :]   # (B, M, 3)
        dot_vd = np.einsum('ijk,jk->ij', v, d)   # (B, M)

        t = np.zeros((B, M), dtype=float)
        t[:, safe] = np.clip(dot_vd[:, safe] / seg_len2[None, safe], 0.0, 1.0)

        closest = seg_p1[None, :, :] + t[:, :, None] * d[None, :, :]  # (B, M, 3)
        diff = p[:, None, :] - closest                                  # (B, M, 3)
        dists2 = np.einsum('ijk,ijk->ij', diff, diff)                  # (B, M)
        min_dists[start:end] = np.sqrt(dists2.min(axis=1))

    return min_dists


def batch_point_to_plane_segments(pts, seg_p1, seg_p2, seg_half_width,
                                  batch_size=2000):
    """
    Like batch_point_to_segments, but for segments that represent horizontal
    planes with a given half-width.  The distance is measured to the nearest
    edge of the plane surface rather than the centerline.

    For each point, the offset from the closest point on the centerline is
    decomposed into a lateral component (perpendicular to the segment in XY)
    and a vertical component (Z).  The lateral distance is reduced by the
    half-width (clamped to zero), and the final distance is
    sqrt(lateral_eff^2 + dz^2).

    seg_half_width : (M,) — half-width per segment.  Segments with
                     half_width == 0 fall back to normal centerline distance.
    """
    N = len(pts)
    M = len(seg_p1)
    if M == 0:
        return np.full(N, np.inf)

    min_dists = np.full(N, np.inf)

    d = seg_p2 - seg_p1                          # (M, 3)
    seg_len2 = np.einsum('ij,ij->i', d, d)       # (M,)
    safe = seg_len2 > 1e-12

    # Lateral unit vector for each segment (perpendicular to d in XY)
    d_xy = d.copy()
    d_xy[:, 2] = 0.0
    d_xy_len = np.sqrt(np.einsum('ij,ij->i', d_xy, d_xy))
    has_lateral = d_xy_len > 1e-12
    lat_dir = np.zeros_like(d)  # (M, 3)
    lat_dir[has_lateral, 0] = -d_xy[has_lateral, 1] / d_xy_len[has_lateral]
    lat_dir[has_lateral, 1] =  d_xy[has_lateral, 0] / d_xy_len[has_lateral]

    hw = seg_half_width  # (M,)
    is_plane = hw > 1e-9  # (M,) bool

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        p = pts[start:end]                        # (B, 3)
        B = len(p)

        v = p[:, None, :] - seg_p1[None, :, :]   # (B, M, 3)
        dot_vd = np.einsum('ijk,jk->ij', v, d)   # (B, M)

        t = np.zeros((B, M), dtype=float)
        t[:, safe] = np.clip(dot_vd[:, safe] / seg_len2[None, safe], 0.0, 1.0)

        closest = seg_p1[None, :, :] + t[:, :, None] * d[None, :, :]  # (B, M, 3)
        diff = p[:, None, :] - closest                                  # (B, M, 3)

        # For non-plane segments: standard Euclidean distance
        dists2 = np.einsum('ijk,ijk->ij', diff, diff)                  # (B, M)

        # For plane segments: decompose into lateral and vertical
        if is_plane.any():
            lat_comp = np.abs(np.einsum('ijk,jk->ij', diff, lat_dir))  # (B, M)
            dz = diff[:, :, 2]                                          # (B, M)
            lat_eff = np.maximum(0.0, lat_comp - hw[None, :])           # (B, M)
            plane_dists2 = lat_eff ** 2 + dz ** 2                       # (B, M)
            dists2[:, is_plane] = plane_dists2[:, is_plane]

        min_dists[start:end] = np.sqrt(dists2.min(axis=1))

    return min_dists


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL CLIPPING / FILTERING
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
# COLOUR-SPACE CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def srgb_to_linear(c: float) -> float:
    """Convert a single sRGB component to linear."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    """Convert a single linear component to sRGB."""
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


# ─────────────────────────────────────────────────────────────────────────────
# DEVIATION HEATMAP COLOUR
# ─────────────────────────────────────────────────────────────────────────────

def deviation_to_color(distances, thresholds=None, palette=None):
    """
    Map an array of deviation distances to RGB colours via piecewise
    linear interpolation between threshold/colour pairs.
    """
    from core.config import DEVIATION_THRESHOLDS, DEVIATION_COLORS
    if thresholds is None:
        thresholds = DEVIATION_THRESHOLDS
    if palette is None:
        palette = DEVIATION_COLORS

    thresholds = np.array(thresholds)
    palette = np.array(palette)
    colors = np.zeros((len(distances), 3), dtype=float)

    for i in range(len(thresholds) - 1):
        lo, hi = thresholds[i], thresholds[i + 1]
        mask = (distances >= lo) & (distances < hi)
        if mask.any():
            t = (distances[mask] - lo) / (hi - lo)
            colors[mask] = palette[i] * (1.0 - t[:, None]) + palette[i + 1] * t[:, None]

    colors[distances >= thresholds[-1]] = palette[-1]
    return colors


# ─────────────────────────────────────────────────────────────────────────────
# PLANE FITTING
# ─────────────────────────────────────────────────────────────────────────────

def fit_plane_z(points, n_robust_iters=3, reject_sigma=2.0):
    """
    Least-squares fit of a height plane ``z = a*x + b*y + c`` to an (N, 3)
    array of points.

    The fit is made robust by iteratively rejecting points whose residual
    exceeds ``reject_sigma`` times the residual standard deviation and
    re-fitting on the surviving inliers.

    Parameters
    ----------
    points : array-like, shape (N, 3)
        XYZ coordinates.
    n_robust_iters : int
        Maximum number of robust re-fit iterations (>= 1).
    reject_sigma : float
        Outlier rejection threshold in residual sigmas.

    Returns
    -------
    coeffs : np.ndarray, shape (3,) | None
        ``(a, b, c)`` such that ``z ≈ a*x + b*y + c``. ``None`` if fewer
        than 3 points were supplied.
    inlier_mask : np.ndarray of bool, shape (N,) | None
        Boolean mask of points kept by the final fit. ``None`` when the
        fit could not be performed.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return None, None

    mask = np.ones(len(pts), dtype=bool)
    coeffs = None
    for _ in range(max(1, n_robust_iters)):
        P = pts[mask]
        if len(P) < 3:
            break
        A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
        coeffs, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)

        # Residuals over ALL points relative to the current plane
        pred = pts[:, 0] * coeffs[0] + pts[:, 1] * coeffs[1] + coeffs[2]
        resid = pts[:, 2] - pred
        s = float(np.std(resid[mask]))
        if s < 1e-9:
            break
        new_mask = np.abs(resid) <= reject_sigma * s
        if new_mask.sum() < 3 or np.array_equal(new_mask, mask):
            break
        mask = new_mask

    return coeffs, mask
