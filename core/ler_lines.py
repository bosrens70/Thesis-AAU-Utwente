# -*- coding: utf-8 -*-
"""
Grouping LER features into utility lines.
=========================================
The registry splits one physical utility into several features. A water main
crossing a single trench is routinely registered as a chain of separate
``Vandledning`` features that meet end to end, each with its own ``gml_id``.
Linking a measured instance to one of them would cover a fragment of the run
rather than the run itself, so this module defines what counts as one *line*:
every feature reachable from another through a shared geometry node.

The grouping is a plain connected-component merge. Where three or more features
meet, all of them join the same line, so a service branch belongs to the main it
taps into. That is deliberate: inside one trench the arms of a junction are the
same utility, and an instance measured there is reconciled against all of them.

What keeps distinct utilities apart is therefore not the shape of the junction
but what a join requires:

* the same storage key, so a layer or forsyningsart variant is never crossed.
  Without this the data merges an Elledning into a Telekommunikationsledning
  that shares a node;
* the same ledningsejer, driftsstatus and udvendigDiameter wherever both
  features record one, so a run never continues into a differently registered
  utility. A value only one side records cannot block a join, because the
  registry leaves these fields empty often enough that treating a blank as a
  mismatch would split runs that are plainly continuous;
* geometry nodes within ``LINE_JOIN_TOL``. The junctions observed in this data
  are exact (0.000 m) while the nearest non-junction pair, two cables leaving
  one cabinet, is 0.200 m apart, so the threshold sits well inside that gap.

Nothing is ever joined on proximity or on running parallel: two lines side by
side share no node, so they stay separate however close they run. The same holds
for the duplicate reports in this data, where one geometry appears under several
indberetningsNr: those overlap without meeting at a node.

A node is any vertex, not only an end. A service pipe often taps a main partway
along it, and the main is then one feature carrying the junction at an interior
vertex. At least one side of a join must still be an end, though: two interior
vertices meeting is a crossing, not a junction.

Everything is decided in plan (XY). A registered Z is often an estimate or a
placeholder (see core/depth.py), so letting it into a distance would break
junctions that are real.

Deliberately Open3D-free, like core/ler_matching.py, so it stays importable in a
headless shell.
"""

import numpy as np
from scipy.spatial import cKDTree

from core.config import LINE_JOIN_TOL

# GML attributes that must agree before two features can be one line.
JOIN_ATTRS = ("ledningsejer", "driftsstatus", "udvendigDiameter")

# How the loaders spell "no value": label_module renders a missing attribute as
# an em dash, geopandas leaves the string "nan".
_MISSING = {"", "-", "—", "nan", "none", "null"}


def _attr(table, name):
    """One attribute of a feature, normalised, or "" when it records none."""
    val = table.get(name)
    text = "" if val is None else str(val).strip()
    return "" if text.lower() in _MISSING else text


def _may_join(a, b):
    """Whether two feature parts are allowed to belong to the same line."""
    if a["key"] != b["key"]:
        return False
    for name in JOIN_ATTRS:
        va, vb = a["attrs"][name], b["attrs"][name]
        if va and vb and va != vb:
            return False
    return True


class _UnionFind:
    def __init__(self):
        self._parent = {}

    def add(self, x):
        self._parent.setdefault(x, x)

    def find(self, x):
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:      # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def group_features_into_lines(features, tol=None):
    """Group LER features into utility lines.

    ``features`` is an iterable of ``(storage_key, gml_id, coords, attrs)``,
    where coords is the feature's own polyline, straight from the GML rather
    than clipped to a crop box: a clipping artefact at the crop boundary must
    never be able to fabricate a junction. ``attrs`` is that feature's GML
    attributes as (key, value) pairs or a mapping. A feature split into several
    parts (a MultiLineString) is passed once per part, under the same gml_id.

    Returns ``(line_of, lines)``:
      line_of : gml_id -> line_id
      lines   : line_id -> sorted list of gml_ids
    ``line_id`` is the smallest gml_id in the group, so it is stable between
    runs and legible in a saved match file. A feature that joins nothing is a
    line of its own.
    """
    tol = LINE_JOIN_TOL if tol is None else float(tol)

    uf = _UnionFind()
    parts = []
    for storage_key, gml_id, coords, attrs in features:
        gml_id = str(gml_id or "")
        if not gml_id:
            continue                     # unidentifiable, cannot be linked
        uf.add(gml_id)                   # every feature is at least its own line
        xy = np.asarray(coords, dtype=float)[:, :2]
        if len(xy) < 2:
            continue
        table = dict(attrs)
        parts.append({
            "key": storage_key,
            "gml_id": gml_id,
            "xy": xy,
            "attrs": {name: _attr(table, name) for name in JOIN_ATTRS},
        })

    if parts:
        node_xy = np.vstack([p["xy"] for p in parts])
        owner = np.concatenate(
            [np.full(len(p["xy"]), i, dtype=int) for i, p in enumerate(parts)])
        is_end = np.zeros(len(node_xy), dtype=bool)
        at = 0
        for p in parts:
            is_end[at] = True
            is_end[at + len(p["xy"]) - 1] = True
            at += len(p["xy"])

        for i, j in cKDTree(node_xy).query_pairs(tol):
            if not (is_end[i] or is_end[j]):
                continue                 # two interior vertices: a crossing
            pa, pb = parts[owner[i]], parts[owner[j]]
            if pa["gml_id"] != pb["gml_id"] and _may_join(pa, pb):
                uf.union(pa["gml_id"], pb["gml_id"])

    groups = {}
    for gml_id in list(uf._parent):
        groups.setdefault(uf.find(gml_id), []).append(gml_id)

    lines, line_of = {}, {}
    for members in groups.values():
        members = sorted(set(members))
        line_id = members[0]
        lines[line_id] = members
        for gml_id in members:
            line_of[gml_id] = line_id
    return line_of, lines


def line_members(line_of, lines, gml_id):
    """Every gml_id on the same line as ``gml_id``.

    Falls back to the feature itself, so a match that names something this site
    no longer loads still resolves to a usable single-feature link.
    """
    gml_id = str(gml_id or "")
    if not gml_id:
        return []
    return list(lines.get(line_of.get(gml_id), [gml_id]))
