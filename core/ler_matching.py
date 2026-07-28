# -*- coding: utf-8 -*-
"""
Heuristic instance -> LER feature match suggestion.

Ranks nearby LER line features against a segmented instance's point cloud
using proximity, direction alignment, registered vs. measured diameter, and
(weakly) registered vs. measured colour, so label_module.py can propose a
"most likely" match instead of requiring the user to find and click the
exact feature themselves. The user still confirms or overrides the result.

Uses only numpy and scikit-learn (no Open3D import), so this stays
headless-testable, matching the convention used by core/ler_las_export.py
and core/symbology.py. The point-to-segments distance routine is duplicated
(not imported) from core/geometry.py for a related reason: core/geometry.py
imports Open3D, which fails to load headless in this environment.

Note: the scikit-learn PCA used here (like numpy's own matrix routines)
needs the conda environment activated so its BLAS DLLs are on PATH. The
modules are always run that way, and segment_module.py uses the same PCA to
split parallel pipes, so this adds no new requirement.
"""

import numpy as np
from sklearn.decomposition import PCA

# Danish "udvendigFarve" (exterior colour) attribute -> approximate RGB (0-1).
# Best-effort only: LER text values are free-form, and this dataset has known
# encoding issues with non-ASCII Danish letters (æ/ø/å), so lookups strip
# all non-letter characters and match case-insensitively against ASCII keys.
_DANISH_COLOR_RGB = {
    "sort":   (0.05, 0.05, 0.05),
    "hvid":   (0.95, 0.95, 0.95),
    "graa":   (0.55, 0.55, 0.55),
    "gra":    (0.55, 0.55, 0.55),
    "roed":   (0.75, 0.10, 0.10),
    "rod":    (0.75, 0.10, 0.10),
    "blaa":   (0.10, 0.30, 0.75),
    "bla":    (0.10, 0.30, 0.75),
    "groen":  (0.10, 0.55, 0.20),
    "gron":   (0.10, 0.55, 0.20),
    "gul":    (0.90, 0.80, 0.10),
    "orange": (0.90, 0.50, 0.10),
    "brun":   (0.40, 0.25, 0.10),
    "soelv":  (0.75, 0.75, 0.75),
    "solv":   (0.75, 0.75, 0.75),
    "kobber": (0.72, 0.45, 0.20),
}

DEFAULT_WEIGHTS = {"proximity": 0.5, "direction": 0.2, "diameter": 0.2, "color": 0.1}


def parse_diameter_m(attrs):
    """Registered pipe outer diameter, in metres, from GML attrs (a list of
    (key, str-value) pairs, as stored per picking segment). Uses only
    udvendigDiameter (mm), the physical outer diameter of the utility.

    Deliberately does NOT fall back to bredde: bredde is the width of a
    Ledningstrace corridor, not a utility's cross-section, so it cannot be
    compared against an instance's measured radius. When a feature has no
    udvendigDiameter (e.g. a trace), this returns None and the diameter term
    is simply left out of that candidate's score."""
    d = dict(attrs)
    val = d.get("udvendigDiameter")
    if val is None or val in ("—", "nan", "None", ""):
        return None
    try:
        v = float(val)
        if v > 0:
            return v / 1000.0
    except (ValueError, TypeError):
        pass
    return None


def parse_color_rgb(attrs):
    """Registered exterior colour as RGB (0-1) from GML attrs, or None."""
    d = dict(attrs)
    val = d.get("udvendigFarve")
    if not val or val in ("—", "nan", "None"):
        return None
    key = "".join(c for c in val.strip().lower() if c.isalpha())
    for name, rgb in _DANISH_COLOR_RGB.items():
        if name in key:
            return rgb
    return None


def _point_to_segments_min_dist(pts, seg_p1, seg_p2):
    """Per-point minimum distance to any of the given segments. Mirrors
    core.geometry.batch_point_to_segments (duplicated, not imported — see
    module docstring)."""
    M = len(seg_p1)
    if M == 0:
        return np.full(len(pts), np.inf)
    d = seg_p2 - seg_p1
    seg_len2 = np.einsum('ij,ij->i', d, d)
    safe = seg_len2 > 1e-12
    v = pts[:, None, :] - seg_p1[None, :, :]
    dot_vd = np.einsum('ijk,jk->ij', v, d)
    t = np.zeros((len(pts), M), dtype=float)
    t[:, safe] = np.clip(dot_vd[:, safe] / seg_len2[None, safe], 0.0, 1.0)
    closest = seg_p1[None, :, :] + t[:, :, None] * d[None, :, :]
    diff = pts[:, None, :] - closest
    dists2 = np.einsum('ijk,ijk->ij', diff, diff)
    return np.sqrt(dists2.min(axis=1))


def _principal_direction(pts):
    """Elongation axis (unit vector) of a roughly pipe-shaped point cloud,
    taken as its first principal component.

    Uses scikit-learn PCA, the same routine segment_module.py uses to split
    parallel pipes, so both stages of the pipeline estimate axes the same way.
    Requires the conda environment activated so numpy's BLAS DLLs are on PATH,
    which is how the modules are always run.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return np.array([1.0, 0.0, 0.0])
    direction = PCA(n_components=1).fit(pts).components_[0]
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])


def _instance_radius(pts, direction):
    """Robust perpendicular half-width estimate: median offset from the
    principal axis, in the plane perpendicular to it."""
    if len(pts) == 0:
        return 0.0
    centered = pts - pts.mean(axis=0)
    along = centered @ direction
    perp = centered - np.outer(along, direction)
    r = np.linalg.norm(perp, axis=1)
    return float(np.median(r))


def build_feature_index(seg_p1, seg_p2, seg_layer, seg_gml_id, seg_attrs):
    """Group picking segments (as built by label_module.py / deviation_module.py
    while loading LER line layers) by gml_id into one record per LER feature.

    seg_p1, seg_p2 : (M, 3) arrays — clipped segment endpoints, local coords.
    seg_layer      : length-M sequence of display layer names.
    seg_gml_id     : length-M sequence of GML gml_id strings (may be empty).
    seg_attrs      : length-M sequence of [(key, str-value), ...] GML attrs.

    Returns dict: gml_id -> {
        "layer": str, "p1": (k,3) array, "p2": (k,3) array,
        "centroid": (3,) array, "direction": (3,) unit vector,
        "diameter_m": float or None, "color_rgb": (r,g,b) or None,
    }
    """
    groups = {}
    for i, gid in enumerate(seg_gml_id):
        if not gid:
            continue
        groups.setdefault(gid, []).append(i)

    index = {}
    for gid, idxs in groups.items():
        p1s = np.asarray([seg_p1[i] for i in idxs], dtype=float)
        p2s = np.asarray([seg_p2[i] for i in idxs], dtype=float)
        pts = np.vstack([p1s, p2s])
        index[gid] = {
            "layer": seg_layer[idxs[0]],
            "p1": p1s,
            "p2": p2s,
            "centroid": pts.mean(axis=0),
            "direction": _principal_direction(pts),
            "diameter_m": parse_diameter_m(seg_attrs[idxs[0]]),
            "color_rgb": parse_color_rgb(seg_attrs[idxs[0]]),
        }
    return index


def merge_index_by_line(feature_index, line_of):
    """Collapse a per-feature index into a per-utility-line one.

    ``line_of`` maps gml_id -> line_id (see core/ler_lines.py). Records of the
    features on one line are concatenated and their centroid and principal
    direction recomputed over the whole run, so a candidate is scored on the
    utility's real extent rather than on whichever fragment it was registered
    in. Keys of the returned index are line_ids, and each record additionally
    carries ``gml_ids``, the features it covers.

    Diameter and colour are taken from the first member that has one: they
    describe the physical utility, so any member that records them speaks for
    the line.
    """
    groups = {}
    for gid, rec in feature_index.items():
        groups.setdefault(line_of.get(gid, gid), []).append((gid, rec))

    merged = {}
    for line_id, members in groups.items():
        members.sort(key=lambda gr: gr[0])
        p1s = np.vstack([r["p1"] for _g, r in members])
        p2s = np.vstack([r["p2"] for _g, r in members])
        pts = np.vstack([p1s, p2s])
        merged[line_id] = {
            "layer": members[0][1]["layer"],
            "gml_ids": [g for g, _r in members],
            "p1": p1s,
            "p2": p2s,
            "centroid": pts.mean(axis=0),
            "direction": _principal_direction(pts),
            "diameter_m": next((r["diameter_m"] for _g, r in members
                                if r["diameter_m"] is not None), None),
            "color_rgb": next((r["color_rgb"] for _g, r in members
                               if r["color_rgb"] is not None), None),
        }
    return merged


def score_candidates(inst_pts, inst_colors, feature_index, allowed_layers=None,
                     max_dist=8.0, top_k=5, weights=None, max_query_pts=300,
                     rng_seed=0):
    """Rank LER features in feature_index by likely correspondence to an
    instance's point cloud.

    inst_pts    : (N, 3) instance points, in the same local coordinate frame
                  as feature_index (i.e. as built from pick_seg_p1/p2).
    inst_colors : (N, 3) instance RGB in 0-1, or None if unavailable.
    allowed_layers : optional set/iterable restricting candidates to these
                  layer names (e.g. once the instance's utility type is
                  known via UTILITY_TO_LER_MATCH); None searches every layer.
    max_dist    : centroid-proximity pre-filter (metres). LER accuracy classes
                  in this dataset go beyond 2 m, so this defaults generously
                  rather than to a tight "obviously nearby" radius.

    Returns a list of dicts sorted by descending score (best first), each:
        {"gml_id", "layer", "score", "breakdown": {...},
         "rep_segment": (p1, p2)}  — the feature's own segment nearest the
                                      instance, for placing a UI highlight.
    """
    if len(inst_pts) == 0 or not feature_index:
        return []

    weights = weights or DEFAULT_WEIGHTS
    inst_pts = np.asarray(inst_pts, dtype=float)
    inst_centroid = inst_pts.mean(axis=0)

    if len(inst_pts) > max_query_pts:
        rng = np.random.default_rng(rng_seed)
        sel = rng.choice(len(inst_pts), max_query_pts, replace=False)
        query_pts = inst_pts[sel]
        query_colors = inst_colors[sel] if inst_colors is not None else None
    else:
        query_pts = inst_pts
        query_colors = inst_colors

    inst_dir = _principal_direction(inst_pts)
    inst_radius = _instance_radius(inst_pts, inst_dir)
    inst_color = (np.mean(query_colors, axis=0)
                 if query_colors is not None and len(query_colors) else None)

    results = []
    for gid, feat in feature_index.items():
        if allowed_layers is not None and feat["layer"] not in allowed_layers:
            continue
        # Proximity pre-filter. A long feature, or a whole utility line merged
        # from several, can have its centroid far away while still running
        # straight through the instance, so being near any of its segments
        # counts as well as being near its centroid.
        mids = (feat["p1"] + feat["p2"]) / 2.0
        if (np.linalg.norm(feat["centroid"] - inst_centroid) > max_dist
                and np.min(np.linalg.norm(mids - inst_centroid, axis=1)) > max_dist):
            continue

        dists = _point_to_segments_min_dist(query_pts, feat["p1"], feat["p2"])
        mean_dist = float(np.mean(dists))
        proximity = float(np.exp(-mean_dist / 0.5))  # 0.5 m e-folding scale

        cos_align = abs(float(np.dot(inst_dir, feat["direction"])))

        parts = {"proximity": proximity, "direction": cos_align}
        used_weights = {"proximity": weights["proximity"], "direction": weights["direction"]}

        if feat["diameter_m"] is not None and inst_radius > 0:
            cand_radius = feat["diameter_m"] / 2.0
            rel_err = abs(inst_radius - cand_radius) / max(cand_radius, 0.01)
            parts["diameter"] = 1.0 / (1.0 + rel_err)
            used_weights["diameter"] = weights["diameter"]

        if feat["color_rgb"] is not None and inst_color is not None:
            rgb_dist = float(np.linalg.norm(np.array(feat["color_rgb"]) - inst_color))
            parts["color"] = max(0.0, 1.0 - rgb_dist / np.sqrt(3.0))
            used_weights["color"] = weights["color"]

        wsum = sum(used_weights.values())
        score = sum(parts[k] * used_weights[k] for k in parts) / wsum if wsum > 0 else 0.0

        rep_i = int(np.argmin(np.linalg.norm(mids - inst_centroid, axis=1)))
        results.append({
            "gml_id": gid,
            # Every feature this candidate covers. One entry for a plain
            # feature index, the whole run for a line-merged one.
            "gml_ids": list(feat.get("gml_ids", [gid])),
            "layer": feat["layer"],
            "score": score,
            "breakdown": parts,
            "rep_segment": (feat["p1"][rep_i], feat["p2"][rep_i]),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]
