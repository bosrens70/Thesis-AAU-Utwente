# -*- coding: utf-8 -*-
"""
Processing status of a site, derived from the files it left on disk.
====================================================================
Every pipeline stage writes an artefact next to the point cloud, so how far a
site has been processed can be read without opening a viewer::

    Area_5_Site_11.ply
    Area_5_Site_11_ground.json            ground pick        (base/label/deviation)
    Area_5_Site_11_trench.json            trench pick        (deviation)
    Site_11_Instances/
      0_instance_0_type_7.ply             water instance     (tools/convert_...)
      20260615_151447/                    segment run        (segment_module)
        1_instance_0.ply ...
      labeled_20260706_115025/            label session      (label_module)
        1_instance_0_type_4.ply ...
        ler_matches.json                  LER matches        (label_module)
    Area_5_Site_11_LER_deviation_LAS/     deviation export   (deviation_module)

This module is the single source of truth for those conventions: which
``labeled_*`` folder wins when several exist, how a label filename maps back to
an instance, and which folders are leftovers. ``label_module``,
``data_loader`` and ``tools/pipeline_status.py`` all resolve through here so
they cannot drift apart.

Deliberately Open3D-free (and free of anything that pulls it in) so it stays
importable in a headless shell; instance point counts come from the PLY header
rather than from a real reader.
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# This is a library, but it is small enough to invite a direct "Run" from an
# editor, which puts core/ rather than the project root on the path.  Bootstrap
# the root so that still resolves; see the __main__ block at the bottom.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.config import PLY_BASE_DIR, TARGET_CLASS, ler_layers_for_type

# ─────────────────────────────────────────────────────────────────────────────
# NAMING CONVENTIONS
# ─────────────────────────────────────────────────────────────────────────────

LABELED_PREFIX = "labeled_"
MATCHES_FILENAME = "ler_matches.json"
# Picks the label viewer refused because the LER layer contradicted the
# instance's label, kept beside the matches so a refusal is recoverable.
CONFLICTS_FILENAME = "ler_match_conflicts.json"
DEVIATION_DIR_SUFFIX = "_LER_deviation_LAS"

# segment_module writes "<class>_instance_<cluster_id>.ply"; label_module writes
# "<class>_instance_<index>_type_<utility_type>.ply".  The two integers are NOT
# the same thing: see instance_order_matches_labels().
SEGMENT_FNAME_RE = re.compile(rf"^{TARGET_CLASS}_instance_(\d+)\.ply$")
LABELED_FNAME_RE = re.compile(rf"^{TARGET_CLASS}_instance_(\d+)_type_(\d+)\.ply$")

# Any class, used to spot the loose main-utility instance that
# tools/convert_main_utility_to_water_instance.py drops in the directory root.
ANY_LABELED_FNAME_RE = re.compile(r"^(\d+)_instance_(\d+)_type_(\d+)\.ply$")


def root_class_instances(instance_dir):
    """Instance PLYs sitting loose in the root of ``<base>_Instances/``.

    These come from tools/convert_main_utility_to_water_instance.py, which turns
    the semantic classes that are not clustered by segment_module (class 0 "Main
    Utility" and class 3 "Inactive Utility") into ready-made instances. Their
    class prefix is therefore never TARGET_CLASS, which is what separates them
    from anything a segment or label session writes.

    Several files per class are allowed (``0_instance_0_...``,
    ``0_instance_1_...``): the cluster id in the name is what identifies them, so
    a class blob split by hand into separate pipes yields separate instances,
    each able to carry its own LER match.
    """
    if not instance_dir or not Path(instance_dir).is_dir():
        return []
    return [
        f for f in sorted(Path(instance_dir).glob("*.ply"))
        if (m := ANY_LABELED_FNAME_RE.match(f.name)) and int(m.group(1)) != TARGET_CLASS
    ]

_STAMP_RE = re.compile(r"^\d{8}_\d{6}$")


def instance_base_name(ply_path):
    """
    Derive the base name used for a PLY's instance directory.

    Strips a redundant leading ``Area_N_`` prefix from the PLY stem, since the
    instance directory already lives inside the ``Water_Area_N`` folder (e.g.
    ``Area_5_Site_11`` -> ``Site_11``). PLY stems without that prefix are
    returned unchanged.
    """
    return re.sub(r"^Area_\d+_", "", Path(ply_path).stem)


def site_sidecar(ply_path, suffix):
    """Path of the cache file ``<stem>_<suffix>.json`` next to the PLY."""
    p = Path(ply_path)
    return p.parent / f"{p.stem}_{suffix}.json"


def instance_dir_for(ply_path):
    """Path of the permanent ``<base>_Instances/`` directory (may not exist)."""
    ply_path = Path(ply_path)
    return ply_path.parent / f"{instance_base_name(ply_path)}_Instances"


def deviation_dir_for(ply_path):
    """Path of the deviation LAS export directory (may not exist)."""
    ply_path = Path(ply_path)
    return ply_path.parent / f"{ply_path.stem}{DEVIATION_DIR_SUFFIX}"


# ─────────────────────────────────────────────────────────────────────────────
# PLY HEADER READING
# ─────────────────────────────────────────────────────────────────────────────

def ply_vertex_count(path):
    """
    Vertex count from a PLY header, without reading the body.

    Returns 0 for an unreadable or headerless file, which is also how a genuinely
    empty instance reads, and both are treated the same downstream.
    """
    try:
        with open(path, "rb") as f:
            while True:
                raw = f.readline()
                if not raw:
                    return 0
                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[-1])
                if line == "end_header":
                    return 0
    except (OSError, ValueError):
        return 0


def ordered_instance_files(files):
    """
    Put raw instance PLYs in the order label_module indexes them.

    label_module drops zero-point clouds and then sorts what is left by point
    count, largest first, so the index baked into a labelled filename is a
    *position in that order*, not the cluster id in the segment filename. The
    sort is stable, so ties keep the incoming (filename) order. Reproducing it
    here is the only way to map a saved label back to its instance.
    """
    counted = [(f, ply_vertex_count(f)) for f in files]
    counted = [(f, n) for f, n in counted if n > 0]
    counted.sort(key=lambda fn: fn[1], reverse=True)
    return [f for f, _ in counted]


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENT AND LABEL SESSION RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Session:
    """One timestamped run folder, segment or labelled."""
    path: Path
    stamp: str          # "20260706_115025", or "" for the legacy un-stamped dir
    ply_files: list = field(default_factory=list)

    @property
    def n_ply(self):
        return len(self.ply_files)

    @property
    def is_prunable(self):
        """True when the folder holds nothing at all, so deleting it loses
        nothing. A folder with no PLYs but some other file is left alone."""
        try:
            return not any(self.path.iterdir())
        except OSError:
            return False


def _sessions(instance_dir, predicate):
    """Run folders matching ``predicate``, newest first. Timestamps sort
    lexicographically, so the folder name is a valid sort key."""
    try:
        dirs = [d for d in instance_dir.iterdir() if d.is_dir() and predicate(d.name)]
    except OSError:
        return []
    out = [Session(path=d,
                   stamp=d.name[len(LABELED_PREFIX):] if d.name.startswith(LABELED_PREFIX) else d.name,
                   ply_files=sorted(d.glob("*.ply")))
           for d in dirs]
    out.sort(key=lambda s: s.path.name, reverse=True)
    return out


def segment_sessions(instance_dir):
    """All segment run folders, newest first. Empty ones are included."""
    if not instance_dir or not Path(instance_dir).is_dir():
        return []
    return _sessions(Path(instance_dir), lambda n: bool(_STAMP_RE.match(n)))


def labeled_sessions(instance_dir):
    """All label session folders, newest first, including the legacy un-stamped
    ``labeled/`` directory (ranked last, since it predates the convention)."""
    if not instance_dir or not Path(instance_dir).is_dir():
        return []
    out = _sessions(Path(instance_dir), lambda n: n.startswith(LABELED_PREFIX))
    legacy = Path(instance_dir) / "labeled"
    if legacy.is_dir():
        out.append(Session(path=legacy, stamp="", ply_files=sorted(legacy.glob("*.ply"))))
    return out


def resolve_segment_dir(instance_dir):
    """
    Authoritative segment run: the newest folder that actually holds instances.

    Returns ``(session_or_None, empty_sessions)``.
    """
    sessions = segment_sessions(instance_dir)
    live = next((s for s in sessions if s.n_ply), None)
    return live, [s for s in sessions if not s.n_ply]


def resolve_labeled_dir(instance_dir):
    """
    Decide which ``labeled_*`` folder counts when a site has several.

    The rule, applied everywhere: the newest folder holding at least one PLY
    wins. Empty folders never win (label_module used to create one on startup,
    so a site can carry several that record nothing). Older non-empty folders
    are earlier label passes and are kept, not deleted.

    Returns ``(authoritative_or_None, empty_sessions, superseded_sessions)``.
    """
    sessions = labeled_sessions(instance_dir)
    live = [s for s in sessions if s.n_ply]
    empty = [s for s in sessions if not s.n_ply]
    return (live[0] if live else None), empty, live[1:]


def authoritative_labeled_dir(instance_dir):
    """Path of the labelled folder that wins, or None. Convenience wrapper for
    callers that do not care about the leftovers."""
    live, _empty, _superseded = resolve_labeled_dir(instance_dir)
    return live.path if live else None


def read_labeled_indices(labeled_dir):
    """Map instance index -> utility_type id, parsed from the labelled filenames."""
    out = {}
    if not labeled_dir:
        return out
    for f in sorted(Path(labeled_dir).glob("*.ply")):
        m = LABELED_FNAME_RE.match(f.name)
        if m:
            out[int(m.group(1))] = int(m.group(2))
    return out


def _read_json_dict(path):
    """A JSON object from ``path``, or {} when absent or unreadable."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def read_ler_matches(labeled_dir):
    """Contents of ``ler_matches.json``, or {} when absent or unreadable."""
    if not labeled_dir:
        return {}
    return _read_json_dict(Path(labeled_dir) / MATCHES_FILENAME)


def read_match_conflicts(labeled_dir):
    """Contents of ``ler_match_conflicts.json``, or {} when absent."""
    if not labeled_dir:
        return {}
    return _read_json_dict(Path(labeled_dir) / CONFLICTS_FILENAME)


def match_disagrees_with_label(fname, layer):
    """True when a recorded match's LER layer contradicts the type in ``fname``.

    The filename carries the utility type the label viewer saved the instance
    under, so a match can be audited from disk without loading any geometry.
    Uses the same rule the viewers apply, so a Ledningstrace whose forsyningsart
    matches the type counts as agreeing. An unrecognised filename, or a type
    with no LER mapping, cannot disagree.
    """
    m = ANY_LABELED_FNAME_RE.match(str(fname))
    if not m or not layer:
        return False
    allowed = ler_layers_for_type(int(m.group(3)), {layer})
    return allowed is not None and layer not in allowed


# ─────────────────────────────────────────────────────────────────────────────
# PER-SITE STATUS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SiteStatus:
    """What has been done to one site, and what is wrong with it."""
    ply_path: Path
    area_name: str
    site_name: str

    has_ground: bool = False
    has_trench: bool = False

    instance_dir: Path = None
    segment: Session = None
    empty_segment_dirs: list = field(default_factory=list)
    n_instances: int = 0

    labeled: Session = None
    empty_labeled_dirs: list = field(default_factory=list)
    superseded_labeled_dirs: list = field(default_factory=list)
    labeled_indices: dict = field(default_factory=dict)   # index -> utility_type id

    n_matched: int = 0        # instances linked to a specific LER feature
    n_no_ler: int = 0         # instances confirmed to have no LER counterpart
    has_matches_file: bool = False
    # Matches whose LER layer contradicts the instance's own label, as
    # "<filename>: <label type> vs <layer>". Split by how they got there: a
    # stored match the label viewer now refuses (recorded before the check
    # existed, or deliberately overridden with Ctrl), and a refused pick logged
    # in ler_match_conflicts.json with no match written at all.
    conflicting_matches: list = field(default_factory=list)
    refused_matches: list = field(default_factory=list)

    water_instances: list = field(default_factory=list)
    deviation_dir: Path = None

    issues: list = field(default_factory=list)

    # ── derived ──────────────────────────────────────────────────────────────
    @property
    def n_labeled(self):
        """Labels that point at an instance the current segment run still has."""
        return len([i for i in self.labeled_indices if i < self.n_instances])

    @property
    def unlabeled_indices(self):
        return [i for i in range(self.n_instances) if i not in self.labeled_indices]

    @property
    def orphan_indices(self):
        """Labels whose instance index no longer exists, left by a re-segment."""
        return sorted(i for i in self.labeled_indices if i >= self.n_instances)

    @property
    def labels_are_stale(self):
        """True when the winning segment run is newer than the winning labels,
        so the labels were indexed against a different segmentation."""
        if not self.segment or not self.labeled or not self.labeled.stamp:
            return False
        return self.segment.stamp > self.labeled.stamp

    @property
    def is_segmented(self):
        return self.n_instances > 0

    @property
    def is_labeled(self):
        return self.is_segmented and not self.unlabeled_indices

    @property
    def is_matched(self):
        """Every label has been resolved against LER, either to a feature or to
        an explicit 'not in LER'."""
        return self.is_labeled and (self.n_matched + self.n_no_ler) >= self.n_instances

    @property
    def has_deviation(self):
        return self.deviation_dir is not None

    @property
    def stage(self):
        """Furthest completed stage, as a short word for tables and titles."""
        if self.has_deviation:
            return "deviation"
        if self.is_matched:
            return "matched"
        if self.is_labeled:
            return "labeled"
        if self.is_segmented:
            return "segmented"
        if self.has_ground:
            return "ground"
        return "untouched"

    @property
    def is_touched(self):
        return self.stage != "untouched"

    def label_summary(self):
        """'4/4 labelled, 3 matched' — the on-disk view of the live counter that
        label_module shows in its window title."""
        return format_label_summary(self.n_labeled, self.n_instances,
                                    self.n_matched, self.n_no_ler)


def format_label_summary(n_labeled, n_instances, n_matched=0, n_no_ler=0):
    """'4/4 labelled, 3 matched, 1 not in LER'.

    Kept as a free function so label_module can render the same line from its
    live in-memory counters, without re-reading the folder after every save.
    """
    if not n_instances:
        return "no instances"
    parts = [f"{n_labeled}/{n_instances} labelled"]
    if n_matched:
        parts.append(f"{n_matched} matched")
    if n_no_ler:
        parts.append(f"{n_no_ler} not in LER")
    return ", ".join(parts)


def _collect_issues(st):
    """Everything about this site that warrants a look, worst first."""
    issues = []
    if st.labels_are_stale:
        issues.append(
            f"labels ({st.labeled.stamp}) predate the segment run "
            f"({st.segment.stamp}); indices may point at different instances")
    if st.orphan_indices:
        issues.append(
            f"{len(st.orphan_indices)} label(s) beyond the current instance count: "
            f"index {', '.join(str(i) for i in st.orphan_indices)}")
    if st.is_segmented and st.unlabeled_indices:
        issues.append(
            f"{len(st.unlabeled_indices)} of {st.n_instances} instances unlabelled: "
            f"index {', '.join(str(i) for i in st.unlabeled_indices)}")
    if st.is_labeled and st.has_matches_file and not st.is_matched:
        unresolved = st.n_instances - st.n_matched - st.n_no_ler
        issues.append(f"{unresolved} labelled instance(s) with no LER match "
                      f"and not marked as absent from LER")
    if st.conflicting_matches:
        issues.append(
            f"{len(st.conflicting_matches)} LER match(es) contradicting their "
            f"label: {'; '.join(st.conflicting_matches)}")
    if st.refused_matches:
        issues.append(
            f"{len(st.refused_matches)} LER match(es) refused for disagreeing "
            f"with the label: {'; '.join(st.refused_matches)}")
    if st.empty_labeled_dirs:
        issues.append(f"{len(st.empty_labeled_dirs)} empty labeled_* folder(s)")
    if st.empty_segment_dirs:
        issues.append(f"{len(st.empty_segment_dirs)} empty segment folder(s)")
    if st.superseded_labeled_dirs:
        issues.append(f"{len(st.superseded_labeled_dirs)} superseded label session(s) "
                      f"kept alongside {st.labeled.path.name}")
    return issues


def site_status(ply_path):
    """Read every stage artefact for one site and return a :class:`SiteStatus`."""
    ply_path = Path(ply_path)
    st = SiteStatus(
        ply_path=ply_path,
        area_name=ply_path.parent.name,
        site_name=instance_base_name(ply_path),
    )

    st.has_ground = site_sidecar(ply_path, "ground").is_file()
    st.has_trench = site_sidecar(ply_path, "trench").is_file()

    dev_dir = deviation_dir_for(ply_path)
    if dev_dir.is_dir() and any(dev_dir.iterdir()):
        st.deviation_dir = dev_dir

    inst_dir = instance_dir_for(ply_path)
    if not inst_dir.is_dir():
        st.issues = _collect_issues(st)
        return st
    st.instance_dir = inst_dir

    st.water_instances = root_class_instances(inst_dir)

    st.segment, st.empty_segment_dirs = resolve_segment_dir(inst_dir)
    st.labeled, st.empty_labeled_dirs, st.superseded_labeled_dirs = \
        resolve_labeled_dir(inst_dir)

    # Instance count comes from the segment run. A site whose raw run was
    # deleted but whose labels survive still has a countable instance set.
    n_segmented = 0
    if st.segment:
        n_segmented = len(ordered_instance_files(st.segment.ply_files))
    elif st.labeled:
        n_segmented = len(ordered_instance_files(st.labeled.ply_files))
    # The root class instances are labellable and matchable too, so they count
    # here as label_module counts them; otherwise this table would report fewer
    # instances than the viewer's own "n/n labelled" counter.
    st.n_instances = n_segmented + len(st.water_instances)

    if st.labeled:
        st.labeled_indices = read_labeled_indices(st.labeled.path)
        st.has_matches_file = (st.labeled.path / MATCHES_FILENAME).is_file()

    # A root class instance carries its type in its own filename and is never
    # unlabelled. label_module appends them after the segmented instances, in
    # this same order, so these indices are the ones it uses.
    for _k, _f in enumerate(st.water_instances):
        _m = ANY_LABELED_FNAME_RE.match(_f.name)
        st.labeled_indices[n_segmented + _k] = int(_m.group(3))

    # Matches live next to the PLYs they describe, so the root class instances
    # keep their own file alongside the labelled session's.
    for _dir in ([st.labeled.path] if st.labeled else []) + [inst_dir]:
        for fname, entry in read_ler_matches(_dir).items():
            if not isinstance(entry, dict):
                continue
            if entry.get("no_ler"):
                st.n_no_ler += 1
            elif entry.get("gml_id"):
                st.n_matched += 1
                layer = entry.get("layer", "")
                if entry.get("conflict") or match_disagrees_with_label(fname, layer):
                    st.conflicting_matches.append(f"{fname}: {layer}")
        for fname, entry in read_match_conflicts(_dir).items():
            if isinstance(entry, dict) and not entry.get("overridden"):
                st.refused_matches.append(
                    f"{fname}: {entry.get('label', '?')} vs {entry.get('layer', '?')}")
    if st.water_instances and (inst_dir / MATCHES_FILENAME).is_file():
        st.has_matches_file = True

    st.issues = _collect_issues(st)
    return st


# ─────────────────────────────────────────────────────────────────────────────
# BATCH SCANNING
# ─────────────────────────────────────────────────────────────────────────────

def area_dirs(base_dir=None):
    """Area folders under Data/OpenTrench3D/ that contain at least one PLY."""
    base = Path(base_dir or PLY_BASE_DIR)
    if not base.is_dir():
        return []
    return sorted(d for d in base.iterdir() if d.is_dir() and any(d.glob("*.ply")))


def scan_area(area_dir):
    """Status of every site in one area folder, in filename order."""
    return [site_status(p) for p in sorted(Path(area_dir).glob("*.ply"))]


def scan_all(base_dir=None):
    """``{area_name: [SiteStatus, ...]}`` across every area."""
    return {d.name: scan_area(d) for d in area_dirs(base_dir)}


def prunable_dirs(statuses):
    """Empty run folders across the given statuses, as a flat list of paths.

    Only folders holding nothing at all are returned, so deleting them cannot
    discard a label or a match file.
    """
    out = []
    for st in statuses:
        for s in st.empty_labeled_dirs + st.empty_segment_dirs:
            if s.is_prunable:
                out.append(s.path)
    return out


if __name__ == "__main__":
    # Running this file gives a headline count and points at the real entry
    # point; the formatted tables, CSV export and pruning live in the tool.
    _scanned = scan_all()
    _flat = [st for sts in _scanned.values() for st in sts]
    print(f"{len(_flat)} sites in {len(_scanned)} areas: "
          f"{sum(1 for s in _flat if s.is_segmented)} segmented, "
          f"{sum(1 for s in _flat if s.is_labeled)} fully labelled, "
          f"{sum(1 for s in _flat if s.has_deviation)} with a deviation export, "
          f"{sum(1 for s in _flat if s.issues)} with issues")
    print("\nThis module is a library. For the per-site table, run:")
    print("  python tools/pipeline_status.py --area Water_Area_5 --todo")
