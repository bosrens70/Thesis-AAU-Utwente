# -*- coding: utf-8 -*-
"""
Depth resolution for LER vertices without a usable registered Z.
================================================================
Single implementation of the UTM -> local translation plus the ordered
DepthSource fallback hierarchy (registered Z, vejledendeDybde, feature mean,
parent-layer mean, ground level). A vertex enters the fallback when its Z is
the -99 sentinel or fails the below-ground plausibility gate. Previously copied, and drifted, across
base / label / ERR / deviation. Deliberately Open3D-free so it stays
headless-testable.
"""

import numpy as np

from core.config import DepthSource, PIPE_DEPTH_CONFIG

# The -99 sentinel is compared with a tolerance so float imprecision (or a
# -99.0000001 written by an exporter) is still caught. Any real registered Z
# is far above this in the Danish height datum.
SENTINEL_MAX = -98.0

# Plausibility gate: a registered Z more than this many metres below the local
# ground level is treated as unregistered and routed through the fallback
# hierarchy. Catches placeholder elevations that pass the sentinel check: some
# owners register Z = 0.0 DVR90 where the terrain sits at ~30 m, and a few
# vertices carry corrupted near-sentinel values (-97.5 .. -71.3).
MAX_DEPTH_BELOW_GROUND = 15.0


def clean_coords_with_depth(coords_raw, vejledende_dybde_mm, *, TX, TY, TZ,
                            ground_z_at, cfg=PIPE_DEPTH_CONFIG,
                            parent_avg_z=None, clamp_z=None,
                            max_below_ground=MAX_DEPTH_BELOW_GROUND):
    """
    Translate UTM -> local.  For vertices whose Z is the -99 sentinel or lies
    implausibly far below the local ground (``max_below_ground``), resolve the
    depth using the ordered DepthSource hierarchy defined in *cfg*.

    Parameters
    ----------
    coords_raw : (N, 2) or (N, 3) array — raw GML vertices in UTM.
    vejledende_dybde_mm : the feature's vejledendeDybde attribute (mm), or None.
    TX, TY, TZ : UTM -> local translation offsets.
    ground_z_at : callable ``f(x_local, y_local) -> float`` — local ground Z
        (flat value, fitted plane, or IDW surface, depending on the viewer).
    cfg : DepthConfig — which DepthSource levels are enabled.
    parent_avg_z : local mean Z of the parent pipe layer (components only).
    clamp_z : optional ``(lo, hi)`` — clamp final local Z into this range
        (catches unresolved sentinels and wildly wrong estimates).
    max_below_ground : registered Z more than this many metres below the local
        ground is treated as unregistered (placeholder / corrupted values).

    Returns
    -------
    (coords, sources) where ``sources`` is a DepthSource int8 array (one entry
    per vertex) when ``cfg.track_per_vertex`` is True, else just ``coords``.
    """
    coords = coords_raw.copy().astype(float)
    if coords.shape[1] == 2:
        coords = np.hstack([coords, np.zeros((len(coords), 1))])

    coords[:, 0] -= TX
    coords[:, 1] -= TY

    n = len(coords)
    sources = np.full(n, DepthSource.NONE, dtype=np.int8)

    # ground_z_at takes local XY (already translated above); +TZ lifts the
    # returned local ground back to the absolute datum the raw Z values use.
    ground_utm = np.array([ground_z_at(x, y)
                           for x, y in coords[:, :2]], dtype=float) + TZ
    bad = ((coords[:, 2] <= SENTINEL_MAX)
           | (coords[:, 2] < ground_utm - max_below_ground))
    sources[~bad] = DepthSource.REGISTERED

    if bad.any():
        # Pre-compute resolver inputs once per feature
        ind_depth_m = None
        if vejledende_dybde_mm is not None:
            try:
                d = float(vejledende_dybde_mm)
                if d > 0:
                    ind_depth_m = d / 1000.0
            except (ValueError, TypeError):
                pass

        good_z = coords[~bad, 2]
        feature_mean_z = float(good_z.mean()) if len(good_z) > 0 else None

        # Resolver table: level -> callable(idx) -> float | None (absolute UTM Z)
        def _resolve_vejledende(idx):
            if ind_depth_m is None:
                return None
            g = ground_z_at(coords[idx, 0], coords[idx, 1])
            return (g + TZ) - ind_depth_m

        def _resolve_feature_mean(idx):
            return feature_mean_z

        def _resolve_layer_mean(idx):
            # parent_avg_z is local; convert to absolute UTM so the final
            # coords[:, 2] -= TZ brings it back to local.
            if parent_avg_z is None:
                return None
            return parent_avg_z + TZ

        def _resolve_ground_plane(idx):
            return ground_z_at(coords[idx, 0], coords[idx, 1]) + TZ

        resolvers = {
            DepthSource.VEJLEDENDE:   _resolve_vejledende,
            DepthSource.FEATURE_MEAN: _resolve_feature_mean,
            DepthSource.LAYER_MEAN:   _resolve_layer_mean,
            DepthSource.GROUND_PLANE: _resolve_ground_plane,
        }

        ordered_levels = sorted(
            lv for lv in cfg.enabled_levels if lv != DepthSource.REGISTERED
        )

        for idx in np.where(bad)[0]:
            for level in ordered_levels:
                resolver = resolvers.get(level)
                if resolver is None:
                    continue
                z = resolver(idx)
                if z is not None:
                    coords[idx, 2] = z
                    sources[idx] = level
                    break

    # Translate Z to local
    coords[:, 2] -= TZ

    if clamp_z is not None:
        coords[:, 2] = np.clip(coords[:, 2], clamp_z[0], clamp_z[1])

    if cfg.track_per_vertex:
        return coords, sources
    return coords
