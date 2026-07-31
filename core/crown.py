# -*- coding: utf-8 -*-
"""
Measured top centreline ("crown line") of an instance point cloud.
==================================================================
LER registers a utility as its horizontal centreline carrying the vertical
coordinate of the **top** ("Vertikale koordinater af geometrien angives for
overkanten af ledningen"), not the axis. Photogrammetry reconstructs only the
exposed upper arc, also never the axis. Measuring the instance points straight
against the registered line therefore compares a surface against a line and
carries a lateral bias of roughly one pipe radius.

The two registered components are recovered by two different means, because no
single operation gives both:

* **Horizontal centreline** from the skeleton of the plan-view footprint. These
  utilities lie within a few degrees of horizontal, so the footprint is a
  faithful projection and the middle of it is the centreline the register holds.
  The skeleton also carries the topology, so a tee comes back as its main run
  plus its branch instead of being refused as "not a single run".
* **Crown height** by rolling a ball **upwards** through the cloud. The ball
  passes freely through the unscanned underside of the pipe and settles inside
  the top arc; the point it touches is the crown.

The ball halts at the *first* surface above it, not the highest one. That is
what separates the pipe from anything resting on top of it: over a valve seated
on a main, a highest-point rule returns the valve at every station, the rising
ball returns the pipe. It is also why the skeleton alone will not do: a skeleton
runs through the middle of the exposed arc, which sits roughly 0.36 of a radius
below the crown by an amount that varies with how much of the pipe is exposed.

A circle is still fitted to each cross-section, but only to measure the radius,
an independent reading of udvendigDiameter. It does not position the crown.

Deliberately Open3D-free so it stays headless-testable.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, label as nd_label
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

# ── Plan-view skeleton (horizontal centreline + topology) ────────────────────
SKELETON_CELL = 0.02       # m, occupancy cell of the footprint
SKELETON_CLOSING = 5       # cells, bridges sampling gaps before skeletonising
# Two different jobs, so two thresholds. A short chain hanging off a junction is
# an artefact of a ragged footprint boundary and is pruned. A short chain that
# stands alone is simply a short exposure, and is kept if it can carry three
# stations. Skeletonisation also shrinks a chain relative to its footprint,
# roughly one pipe width at each end, so the acceptance threshold has to sit
# below the shortest run worth measuring rather than at it.
MIN_SPUR_LENGTH = 0.20     # m, dead-end chain next to a junction: an artefact
MIN_ARM_LENGTH = 3 * 0.05  # m, three stations at STATION_STEP
MIN_BRANCH_PIXELS = 4
PATH_SMOOTH_WINDOW = 5     # samples, takes the pixel staircase out of a branch
# Splitting at a junction is a means, not an end: it separates a branch leaving
# the run from the run itself. Two arms carrying straight on through the node are
# one pipe and are rejoined, so a tee no longer cuts its own main line in two.
# The turn angles at the tee in this data are 8 deg for the straight pair against
# 82 and 90 for the branch, so the threshold has a wide margin on either side.
MAX_JUNCTION_TURN = 40.0    # deg, deviation from straight a rejoin may carry
JUNCTION_SNAP_CELLS = 4     # cells, how close an arm end must sit to the node
JUNCTION_DIR_LENGTH = 0.12  # m, arm length the direction at the node is taken over

# ── Rising ball (crown height) ───────────────────────────────────────────────
# The ball must be small enough to enter the top arc of the narrowest utility
# (the 45 mm telecom ducts here have a 22 mm radius) and large enough that it
# cannot slip between neighbouring points (spacing is about 3 mm).
BALL_RADIUS = 0.012        # m
CEILING_GRID = 0.005       # m, lateral sampling of the ceiling across a station
# The crown sits near the middle of the exposed cross-section: measured lateral
# offsets from the centre run 4 to 32 mm across these sites. Scanning wider lets
# the ridge jump to whatever is highest anywhere in the slab.
MAX_SCAN_HALF_WIDTH = 0.15  # m

# ── Stationing ───────────────────────────────────────────────────────────────
STATION_STEP = 0.05        # m, spacing of crown stations along a branch
STATION_WINDOW = 0.10      # m, slab thickness contributing to one station
# A crown vertex is a measured contact point, so it is free to hop laterally
# from one station to the next by the few tens of millimetres of offset noted at
# MAX_SCAN_HALF_WIDTH. Over one STATION_STEP that reads as a kink, and a window
# of a few stations is not enough to take it out. Seven spans 0.30 m, still well
# under the scale of a bend the register would hold.
SMOOTH_WINDOW = 7          # stations in the moving average applied to the crown

# ── Acceptance gates ─────────────────────────────────────────────────────────
# Stations further apart than this leave a hole in the exposure, so the polyline
# is split rather than drawn straight across ground that was never measured.
MAX_STATION_GAP_STEPS = 3
MIN_COVERAGE = 0.5         # share of the branch length the crown parts must span
# A near-horizontal branch. A riser has no meaningful crown datum (the register's
# vertical datum is the top of the utility, which for a vertical pipe is its end
# cap) and its footprint collapses to a blob, so it is refused rather than fitted.
MAX_RISE_RATIO = 0.35
MIN_STATION_PTS = 30       # points in the slab before a station is attempted
MIN_GOOD_FRACTION = 0.5    # share of a branch's stations that must yield a crown

# ── Radius fit (measurement only, does not position the crown) ───────────────
ENVELOPE_BIN = 0.005       # m, lateral bin width of the cross-section upper envelope
MIN_ENVELOPE_PTS = 6       # lateral bins holding a sample
MIN_ARC_DEG = 60.0         # angular span of the exposed arc the fit is based on
# An upper envelope cannot span more than half of a correctly fitted circle, so a
# wider span means the fit has shrunk onto something that is not one pipe.
MAX_ARC_DEG = 200.0
# The apex must stand about a full radius above the fitted centre, which is what
# separates an upward-convex pipe top from a fit that has curled the wrong way.
MIN_APEX_RISE_FRAC = 0.75
MIN_RADIUS = 0.010         # m, below this the fit is noise, not a pipe
MAX_RADIUS = 0.750         # m, above this the arc is too flat to constrain a radius
MAX_FIT_RESIDUAL = 0.010   # m, circle-fit residual sigma

_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class CrownLine:
    """Crown polyline of one instance, in the same local frame as its points.

    ``ok`` is False when the instance was rejected; ``reason`` then says why and
    every array is empty. Callers must check ``ok`` before using ``points``.

    One instance can hold several parts: one per skeleton run (a tee gives two,
    its main line and its branch), and a further split wherever the exposure
    breaks. ``station`` is arc length along the run a point belongs to, so it
    restarts at each run.
    ``radius`` is NaN where the cross-section did not support a fit, which
    affects the diameter reading only, never the crown position.
    """
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    radius: np.ndarray = field(default_factory=lambda: np.empty(0))
    station: np.ndarray = field(default_factory=lambda: np.empty(0))
    part: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    run_length: float = 0.0
    n_branches: int = 0
    ok: bool = False
    reason: str = ""
    # Arms the footprint found but the crown does not cover, with the reason for
    # each. Populated on success too: an instance can yield a good line for its
    # main run while a service connection diving away from it is refused, and
    # that has to be visible rather than silently missing from the statistics.
    notes: list = field(default_factory=list)

    @property
    def n_stations(self):
        return len(self.points)

    @property
    def n_radius(self):
        """Stations where the cross-section yielded a radius measurement."""
        return int(np.count_nonzero(np.isfinite(self.radius)))

    @property
    def n_parts(self):
        return int(self.part.max()) + 1 if len(self.part) else 0

    @property
    def coverage(self):
        """Share of the accepted branch length the crown parts actually span. A
        pipe exposed in two short windows with rubble between them covers less
        than one exposed end to end, and the deviation is only a sample of it."""
        if not len(self.station) or self.run_length <= 0:
            return 0.0
        spanned = sum(float(self.station[m].max() - self.station[m].min())
                      for m in self._part_masks())
        return spanned / self.run_length

    @property
    def median_radius(self):
        """Median fitted radius, or None when no station supported a fit."""
        r = self.radius[np.isfinite(self.radius)]
        return float(np.median(r)) if len(r) else None

    def _part_masks(self):
        return [self.part == p for p in range(self.n_parts)]

    @property
    def segments(self):
        """``(p1, p2)`` arrays of the polyline's segments. Segments bridging two
        parts are omitted, so nothing is drawn or measured across a gap in the
        exposure or between two arms of a junction."""
        p1, p2 = [], []
        for m in self._part_masks():
            pp = self.points[m]
            if len(pp) >= 2:
                p1.append(pp[:-1])
                p2.append(pp[1:])
        if not p1:
            return np.empty((0, 3)), np.empty((0, 3))
        return np.vstack(p1), np.vstack(p2)

    def resample(self, step):
        """The crown resampled at roughly ``step`` metres, part by part, for
        rendering or for a deviation profile denser than the station spacing."""
        out = []
        for m in self._part_masks():
            pp = self.points[m]
            if len(pp) < 2:
                out.append(pp)
                continue
            out.append(_resample_polyline(pp, step)[0])
        return np.vstack(out) if out else self.points.copy()


# ─────────────────────────────────────────────────────────────────────────────
# PLAN-VIEW SKELETON
# ─────────────────────────────────────────────────────────────────────────────

def footprint_skeleton(xy, cell=SKELETON_CELL, closing=SKELETON_CLOSING):
    """Plan-view skeleton of a cloud's XY footprint.

    Holes are closed and filled first, so a sampling gap in the middle of a pipe
    does not fragment the skeleton or open a false loop around itself.

    Returns ``(skel, origin)`` with ``skel`` a boolean image indexed
    ``[row, col]`` (row is Y, column is X) and ``origin`` the XY of the corner of
    cell (0, 0).
    """
    xy = np.asarray(xy, dtype=float)
    origin = xy.min(axis=0)
    ij = np.floor((xy - origin) / cell).astype(int)
    mask = np.zeros((ij[:, 1].max() + 1, ij[:, 0].max() + 1), dtype=bool)
    mask[ij[:, 1], ij[:, 0]] = True
    if closing >= 2:
        mask = binary_closing(mask, np.ones((closing, closing), dtype=bool))
    mask = binary_fill_holes(mask)
    return skeletonize(mask), origin


def _skeleton_degree(skel):
    """Number of skeleton neighbours of each skeleton pixel (0 elsewhere)."""
    padded = np.pad(skel, 1).astype(np.int8)
    h, w = skel.shape
    deg = np.zeros((h, w), dtype=np.int8)
    for dy, dx in _NEIGHBOURS:
        deg += padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return deg * skel


def _order_path(coords):
    """Order a set of 8-connected pixel coordinates into a single chain."""
    pts = {(int(r), int(c)) for r, c in coords}

    def neighbours(p):
        return [q for q in ((p[0] + dy, p[1] + dx) for dy, dx in _NEIGHBOURS)
                if q in pts]

    ends = [p for p in pts if len(neighbours(p)) <= 1]
    path = [min(ends) if ends else min(pts)]
    seen = set(path)
    while True:
        nxt = [q for q in neighbours(path[-1]) if q not in seen]
        if not nxt:
            break
        # Prefer the 4-connected step so the chain does not cut a corner
        nxt.sort(key=lambda q: abs(q[0] - path[-1][0]) + abs(q[1] - path[-1][1]))
        path.append(nxt[0])
        seen.add(nxt[0])
    return np.asarray(path)


def _chain_length(coords, cell):
    """Polyline length of an ordered pixel chain, in metres."""
    path = _order_path(coords)
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path.astype(float), axis=0),
                                axis=1).sum() * cell)


def _prune_spurs(skel, cell, min_length, max_rounds=8):
    """Iteratively drop short dead-end chains.

    A ragged footprint boundary makes skeletonize sprout small side chains. Left
    in place they register as junctions and cut a straight pipe into several
    arms, so they are removed before the skeleton is split. A chain qualifies
    only if it carries a free end; interior links between two junctions are
    kept however short they are.
    """
    skel = skel.copy()
    structure = np.ones((3, 3), dtype=int)
    for _ in range(max_rounds):
        deg = _skeleton_degree(skel)
        if not (deg >= 3).any():
            break
        lab, n = nd_label(skel & (deg < 3), structure=structure)
        drop = np.zeros_like(skel)
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            if not (deg[coords[:, 0], coords[:, 1]] == 1).any():
                continue
            if _chain_length(coords, cell) < min_length:
                drop[coords[:, 0], coords[:, 1]] = True
        if not drop.any():
            break
        skel = skel & ~drop
    return skel


def _polyline_length(poly):
    return float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())


def _junction_nodes(skel, origin, cell):
    """XY centroids of the skeleton's junction pixel clusters.

    ``skel`` must already be pruned, so that a spur no longer reads as a node.
    """
    junc = skel & (_skeleton_degree(skel) >= 3)
    lab, n = nd_label(junc, structure=np.ones((3, 3), dtype=int))
    nodes = []
    for k in range(1, n + 1):
        co = np.argwhere(lab == k).astype(float)
        nodes.append(np.array([origin[0] + (co[:, 1].mean() + 0.5) * cell,
                               origin[1] + (co[:, 0].mean() + 0.5) * cell]))
    return nodes


def _end_direction(poly, at_start, span):
    """Unit direction at one end of a polyline, pointing away from its body.
    Taken over ``span`` vertices, so the pixel staircase does not set it."""
    m = min(span, len(poly) - 1)
    v = poly[0] - poly[m] if at_start else poly[-1] - poly[-1 - m]
    n = float(np.linalg.norm(v))
    return v / n if n else np.array([1.0, 0.0])


def _merge_through_junctions(arms, nodes, cell, max_turn=MAX_JUNCTION_TURN):
    """Rejoin the arms that carry straight on through a junction.

    At each node the arm ends meeting it are paired straightest first, and a
    pair is joined only while it stays under ``max_turn``. An end takes part in
    one join at most, so a crossing gives two runs through it and a tee keeps
    its branch separate. The node itself is inserted between the two arms:
    skeletonisation removed those pixels and left a hole roughly two cells wide.

    Returns ``(run, arms)`` pairs. ``arms`` are the pieces the run was built
    from, kept because the test that decides whether a run is a riser can only
    be applied once the crown is known, and a run that fails it has to be able
    to fall back to the arms the skeleton would have handed over on its own.
    """
    span = max(2, int(round(JUNCTION_DIR_LENGTH / cell)))
    snap = JUNCTION_SNAP_CELLS * cell
    ends = {}
    for bi, a in enumerate(arms):
        ends[(bi, 0)] = (a[0], _end_direction(a, True, span))
        ends[(bi, 1)] = (a[-1], _end_direction(a, False, span))

    link, used = {}, set()
    for node in nodes:
        near = [k for k, (p, _) in ends.items()
                if k not in used and np.linalg.norm(p - node) <= snap]
        cand = []
        for i in range(len(near)):
            for j in range(i + 1, len(near)):
                a, b = near[i], near[j]
                if a[0] == b[0]:
                    continue        # both ends of one arm: a loop, not a run
                # Both directions point away from the node, so a straight
                # continuation has them opposed.
                turn = 180.0 - np.degrees(np.arccos(
                    np.clip(float(ends[a][1] @ ends[b][1]), -1.0, 1.0)))
                cand.append((turn, a, b))
        for turn, a, b in sorted(cand, key=lambda t: t[0]):
            if turn > max_turn or a in used or b in used:
                continue
            link[a], link[b] = (b, node), (a, node)
            used.update((a, b))

    # Every end carries at most one link, so the arms form paths, plus cycles
    # where a run closes on itself. Walk each path from a free end; a cycle has
    # none, so it is picked up by the second pass and closed where it repeats.
    runs, done = [], set()
    free = [(bi, s) for bi in range(len(arms)) for s in (0, 1)
            if (bi, s) not in link]
    for bi, s in free + [(bi, 0) for bi in range(len(arms))]:
        if bi in done:
            continue
        poly = arms[bi] if s == 0 else arms[bi][::-1]
        made_of = [arms[bi]]
        done.add(bi)
        cur = (bi, 1 - s)
        while cur in link:
            nxt, node = link[cur]
            if nxt[0] in done:
                break
            nb = arms[nxt[0]] if nxt[1] == 0 else arms[nxt[0]][::-1]
            poly = np.vstack([poly, node[None, :], nb])
            made_of.append(arms[nxt[0]])
            done.add(nxt[0])
            cur = (nxt[0], 1 - nxt[1])
        runs.append((poly, made_of))
    return runs


def skeleton_branches(skel, origin, cell=SKELETON_CELL,
                      min_length=MIN_ARM_LENGTH,
                      spur_length=MIN_SPUR_LENGTH,
                      min_pixels=MIN_BRANCH_PIXELS,
                      max_junction_turn=MAX_JUNCTION_TURN):
    """Ordered runs of a skeleton, as ``(run, arms)`` pairs in the cloud's XY
    frame.

    Spurs are pruned first, so a pixel of boundary noise does not cut a straight
    pipe in two. Junction pixels are then removed to separate the arms, the
    remainder is split into connected chains, each chain is walked end to end,
    and the arms that carry straight on through a node are rejoined. A tee
    therefore returns its main run whole, with only its branch separate.

    ``min_length`` applies to the rejoined run, not to the arm: a short link
    between two nodes is part of a longer pipe rather than a short one.
    """
    skel = _prune_spurs(skel, cell, spur_length)
    body = skel & (_skeleton_degree(skel) < 3)
    lab, n = nd_label(body, structure=np.ones((3, 3), dtype=int))
    arms = []
    for k in range(1, n + 1):
        coords = np.argwhere(lab == k)
        if len(coords) < min_pixels:
            continue
        path = _order_path(coords)
        arms.append(np.column_stack([origin[0] + (path[:, 1] + 0.5) * cell,
                                     origin[1] + (path[:, 0] + 0.5) * cell]))
    runs = _merge_through_junctions(arms, _junction_nodes(skel, origin, cell),
                                    cell, max_junction_turn)
    return [(r, made_of) for r, made_of in runs
            if _polyline_length(r) >= min_length]


def _footprint_branches(xy, cell, min_length, spur_length=MIN_SPUR_LENGTH):
    """Plan-view runs of a cloud footprint: skeletonise, prune, split, rejoin.
    Returns ``(run, arms)`` pairs, see :func:`skeleton_branches`."""
    skel, origin = footprint_skeleton(xy, cell)
    return skeleton_branches(skel, origin, cell, min_length, spur_length)


# ─────────────────────────────────────────────────────────────────────────────
# RISING BALL
# ─────────────────────────────────────────────────────────────────────────────

def ceiling_field(points, query_xy=None, ball_radius=BALL_RADIUS,
                  grid=CEILING_GRID, tree=None):
    """Rising-ball ceiling of a point cloud.

    A ball of ``ball_radius`` rises along the vertical through each query XY and
    stops at first contact. With the ball centre at height ``zc`` a point ``p``
    is touched when ``dxy**2 + (zp - zc)**2 == ball_radius**2``, so the stop
    height is ``min over p of (zp - sqrt(ball_radius**2 - dxy**2))`` and the
    contact is the point attaining it.

    Because the ball halts at the first surface above it rather than the highest
    one, an object resting on the utility does not capture it: the ball rises
    through the unscanned underside and settles inside the pipe's top arc.

    ``query_xy`` defaults to a regular ``grid`` over the cloud's XY extent. Pass
    ``tree`` (a ``cKDTree`` over ``points[:, :2]``) to reuse one across calls.

    Returns ``(query_xy, contact)`` where ``contact`` indexes ``points`` and is
    -1 for a query whose column holds nothing within ``ball_radius``.
    """
    pts = np.asarray(points, dtype=float)
    if tree is None:
        tree = cKDTree(pts[:, :2])
    if query_xy is None:
        xs = np.arange(pts[:, 0].min(), pts[:, 0].max() + grid, grid)
        ys = np.arange(pts[:, 1].min(), pts[:, 1].max() + grid, grid)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        query_xy = np.column_stack([gx.ravel(), gy.ravel()])
    query_xy = np.atleast_2d(np.asarray(query_xy, dtype=float))

    contact = np.full(len(query_xy), -1, dtype=np.int64)
    r2 = float(ball_radius) ** 2
    for k, idx in enumerate(tree.query_ball_point(query_xy, ball_radius)):
        if not idx:
            continue
        idx = np.asarray(idx)
        d = pts[idx, :2] - query_xy[k]
        d2 = np.einsum("ij,ij->i", d, d)
        inside = d2 < r2
        if not inside.any():
            continue
        idx, d2 = idx[inside], d2[inside]
        zc = pts[idx, 2] - np.sqrt(r2 - d2)
        contact[k] = idx[int(np.argmin(zc))]
    return query_xy, contact


# ─────────────────────────────────────────────────────────────────────────────
# RADIUS FIT (measurement only)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_circle(u, v):
    """Algebraic (Kasa) circle fit in a cross-section plane.

    Returns ``(cu, cv, r, residual_sigma)``, or None when the system is
    degenerate. The fit is linear, so it is biased towards larger radii on short
    arcs; the arc-coverage gate is what keeps that in check.
    """
    if len(u) < 3:
        return None
    A = np.column_stack([u, v, np.ones(len(u))])
    b = u ** 2 + v ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cu, cv = sol[0] / 2.0, sol[1] / 2.0
    disc = sol[2] + cu ** 2 + cv ** 2
    if not np.isfinite(disc) or disc <= 0:
        return None
    r = float(np.sqrt(disc))
    resid = np.sqrt((u - cu) ** 2 + (v - cv) ** 2) - r
    return float(cu), float(cv), r, float(np.std(resid))


def _fit_circle_robust(u, v, n_iters=2, reject_sigma=2.5):
    """Circle fit with residual rejection, mirroring core.geometry.fit_plane_z."""
    keep = np.ones(len(u), dtype=bool)
    out = _fit_circle(u, v)
    for _ in range(n_iters):
        if out is None or keep.sum() < MIN_ENVELOPE_PTS:
            break
        cu, cv, r, sd = out
        if sd < 1e-9:
            break
        resid = np.abs(np.sqrt((u - cu) ** 2 + (v - cv) ** 2) - r)
        new_keep = resid <= reject_sigma * sd
        if new_keep.sum() < MIN_ENVELOPE_PTS or np.array_equal(new_keep, keep):
            break
        keep = new_keep
        refit = _fit_circle(u[keep], v[keep])
        if refit is None:
            break
        out = refit
    return out, keep


def _upper_envelope(t, z, bin_width=ENVELOPE_BIN):
    """Highest point per lateral bin of a cross-section: the exposed top arc.

    Used for the radius fit only. Reducing the slab to its upper envelope weights
    every part of the arc equally regardless of how densely it was reconstructed.
    """
    if len(t) == 0:
        return np.empty(0), np.empty(0)
    key = np.floor((t - t.min()) / bin_width).astype(np.int64)
    order = np.lexsort((-z, key))
    key_s, t_s, z_s = key[order], t[order], z[order]
    first = np.ones(len(key_s), dtype=bool)
    first[1:] = key_s[1:] != key_s[:-1]
    return t_s[first], z_s[first]


def _station_radius(t, z, radius_range, min_arc_deg, max_residual):
    """Radius of the pipe at one station, or NaN when the arc does not support
    a fit. Never used to place the crown, only to measure udvendigDiameter."""
    tu, zu = _upper_envelope(t, z)
    if len(tu) < MIN_ENVELOPE_PTS:
        return np.nan
    fit, keep = _fit_circle_robust(tu, zu)
    if fit is None:
        return np.nan
    cu, cv, r, sd = fit
    lo_r, hi_r = radius_range
    ang = np.degrees(np.arctan2(zu[keep] - cv, tu[keep] - cu))
    coverage = float(ang.max() - ang.min())
    convex_up = (zu[keep].max() - cv) >= MIN_APEX_RISE_FRAC * r
    if (lo_r <= r <= hi_r and sd <= max_residual
            and min_arc_deg <= coverage <= MAX_ARC_DEG and convex_up):
        return r
    return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# POLYLINE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _moving_average(arr, window):
    """Moving average along an array's first axis, endpoints held fixed."""
    n = len(arr)
    if n < 3 or window < 3:
        return arr.copy()
    half = window // 2
    out = arr.copy()
    for i in range(1, n - 1):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = arr[lo:hi].mean(axis=0)
    return out


def _resample_polyline(poly, step):
    """Resample a polyline at roughly ``step`` metres.

    Returns ``(points, arc_length)``, both with the same leading dimension.
    """
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 2:
        return poly.copy(), np.zeros(len(poly))
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-9:
        return poly[:1].copy(), np.zeros(1)
    n = max(2, int(np.round(total / max(step, 1e-6))) + 1)
    want = np.linspace(0.0, total, n)
    pts = np.column_stack([np.interp(want, cum, poly[:, k])
                           for k in range(poly.shape[1])])
    return pts, want


def _smooth_polyline(pts, part, window=SMOOTH_WINDOW):
    """Moving average along each part of a polyline. Parts are smoothed
    separately so a gap in the exposure, or a junction between two arms, is
    never averaged across."""
    out = pts.copy()
    for p in np.unique(part):
        idx = np.where(part == p)[0]
        if len(idx) >= 3:
            out[idx] = _moving_average(pts[idx], window)
    return out


def _tangents(centres):
    """Unit XY tangent at each sample of a polyline."""
    d = np.gradient(centres, axis=0) if len(centres) > 1 else np.array([[1.0, 0.0]])
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    return np.divide(d, np.where(norm > 1e-9, norm, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# CROWN LINE
# ─────────────────────────────────────────────────────────────────────────────

def crown_line(points, *, step=STATION_STEP, window=STATION_WINDOW,
               ball_radius=BALL_RADIUS, ceiling_grid=CEILING_GRID,
               max_scan_half_width=MAX_SCAN_HALF_WIDTH,
               cell=SKELETON_CELL, min_branch_length=MIN_ARM_LENGTH,
               min_arc_deg=MIN_ARC_DEG, max_residual=MAX_FIT_RESIDUAL,
               radius_range=(MIN_RADIUS, MAX_RADIUS),
               max_rise_ratio=MAX_RISE_RATIO,
               min_good_fraction=MIN_GOOD_FRACTION, smooth=SMOOTH_WINDOW):
    """Recover the crown polyline of one instance cloud.

    ``points`` is an (N, 3) array in the viewer's local frame. The plan-view
    skeleton supplies the horizontal centreline and separates a branch from the
    run it leaves; the rising ball supplies the crown height along each. Returns a
    :class:`CrownLine`; check ``ok`` before using it, since a branch that is not
    near-horizontal, or an instance whose footprint carries no usable branch, is
    rejected with a reason rather than forced into a line.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or len(pts) < MIN_STATION_PTS:
        return CrownLine(reason=f"too few points ({len(pts)})")

    origin = pts.mean(axis=0)
    P = pts - origin

    runs = _footprint_branches(P[:, :2], cell, min_branch_length)
    if not runs:
        return CrownLine(reason="footprint holds no branch longer than "
                                f"{min_branch_length:.2f} m")

    tree = cKDTree(P[:, :2])

    def _measure(polys):
        """Crown stations along a list of branch polylines.

        Also returns the branches the rise gate refused, which is what tells the
        caller a rejoin has to be taken back."""
        # Every measured point is assigned to its nearest branch, so a station's
        # slab and its radius fit see only the arm they belong to. The ball still
        # queries the whole cloud, so anything resting above a pipe still blocks it.
        branch_xy = np.vstack(polys)
        branch_id = np.concatenate([np.full(len(b), i, dtype=int)
                                    for i, b in enumerate(polys)])
        owner = branch_id[cKDTree(branch_xy).query(P[:, :2])[1]]

        crown_pts, radii, st_out, part_out = [], [], [], []
        next_part = 0
        accepted_length = 0.0
        notes, risers = [], set()

        for bi, poly in enumerate(polys):
            mine = owner == bi
            blen = _polyline_length(poly)
            if mine.sum() < MIN_STATION_PTS:
                notes.append(f"arm {bi + 1} holds {int(mine.sum())} points")
                continue
            Pm = P[mine]
            centres, arc = _resample_polyline(poly, step)
            centres = _moving_average(centres, PATH_SMOOTH_WINDOW)
            tangents = _tangents(centres)

            got_arc, got_pt, got_r = [], [], []
            for k in range(len(centres)):
                c = centres[k]
                lat = np.array([-tangents[k][1], tangents[k][0]])
                v = Pm[:, :2] - c
                m = ((np.abs(v @ tangents[k]) <= window / 2.0)
                     & (np.abs(v @ lat) <= max_scan_half_width))
                if m.sum() < MIN_STATION_PTS:
                    continue
                Q = Pm[m]
                t = (Q[:, :2] - c) @ lat
                t_lo = max(t.min(), -max_scan_half_width)
                t_hi = min(t.max(), max_scan_half_width)
                if t_hi <= t_lo:
                    continue
                # Roll the ball up across the cross-section and take the ridge:
                # the highest of those first contacts is the crown. The scan is
                # bounded so the ridge cannot wander off this arm onto something
                # else.
                n_scan = max(2, int(np.ceil((t_hi - t_lo) / ceiling_grid)) + 1)
                offs = np.linspace(t_lo, t_hi, n_scan)
                _, hit = ceiling_field(P, c[None, :] + offs[:, None] * lat[None, :],
                                       ball_radius, tree=tree)
                hit = hit[hit >= 0]
                if len(hit) == 0:
                    continue
                got_arc.append(arc[k])
                got_pt.append(P[hit[int(np.argmax(P[hit, 2]))]])
                got_r.append(_station_radius(t, Q[:, 2], radius_range,
                                             min_arc_deg, max_residual))

            if len(got_pt) < 2 or len(got_pt) < min_good_fraction * len(centres):
                notes.append(f"arm {bi + 1} yielded {len(got_pt)}/{len(centres)} "
                             f"stations")
                continue

            a = np.asarray(got_arc, dtype=float)
            # Slope is measured on the crown, not on the point spread. A short arm
            # spans about one pipe diameter in Z whatever its slope, so testing the
            # raw spread would call a level 0.16 m fragment a riser.
            span = max(float(a.max() - a.min()), step)
            crown_rise = float(max(p[2] for p in got_pt) - min(p[2] for p in got_pt))
            if crown_rise > max_rise_ratio * span:
                notes.append(f"arm {bi + 1} crown rises {crown_rise:.2f} m over "
                             f"{span:.2f} m")
                risers.add(bi)
                continue

            # A gap in the exposure splits the run further; each run always starts
            # a new part, so nothing is ever drawn from one run to another.
            sub = np.concatenate([[0],
                                  np.cumsum(np.diff(a) > MAX_STATION_GAP_STEPS * step)])
            st_out.extend(got_arc)
            crown_pts.extend(got_pt)
            radii.extend(got_r)
            part_out.extend(next_part + sub.astype(int))
            next_part += int(sub.max()) + 1
            accepted_length += blen

        return crown_pts, radii, st_out, part_out, accepted_length, notes, risers

    polys = [r for r, _ in runs]
    measured = _measure(polys)

    # Rejoining a junction is an improvement only while it costs nothing. A valve
    # riser standing on a main is a blob in plan view, so nothing in the footprint
    # separates it from the pipe it sits on, and rejoining can put it in the same
    # run as a level stretch. The rise gate then refuses the two together, where
    # the arms on their own would have cost only the riser. So a refused run goes
    # back to its arms, and the rejoin is kept everywhere it is free.
    # Taking a run back can leave a single arm, where the other was below
    # min_branch_length and the rejoin was the only thing carrying it, so
    # whether anything was taken back is tracked rather than inferred from the
    # number of branches.
    fallback, took_back = [], False
    for bi, (run, arms) in enumerate(runs):
        if bi in measured[-1] and len(arms) > 1:
            kept = [a for a in arms if _polyline_length(a) >= min_branch_length]
            if kept:
                fallback.extend(kept)
                took_back = True
                continue
        fallback.append(run)
    if took_back:
        polys = fallback
        measured = _measure(polys)

    crown_pts, radii, st_out, part_out, accepted_length, notes, _ = measured

    if len(crown_pts) < 2:
        detail = "; ".join(notes) if notes else "no arm yielded a crown"
        return CrownLine(reason=detail, n_branches=len(polys), notes=notes)

    part = np.asarray(part_out, dtype=np.int32)
    crown = _smooth_polyline(np.asarray(crown_pts, dtype=float), part, smooth)
    line = CrownLine(points=crown + origin,
                     radius=np.asarray(radii, dtype=float),
                     station=np.asarray(st_out, dtype=float),
                     part=part,
                     run_length=accepted_length,
                     n_branches=len(polys),
                     ok=True,
                     notes=notes)
    if line.coverage < MIN_COVERAGE:
        return CrownLine(reason=f"crown covers only {line.coverage*100:.0f}% "
                                f"of the {accepted_length:.2f} m of arm",
                         n_branches=len(polys), notes=notes)
    return line
