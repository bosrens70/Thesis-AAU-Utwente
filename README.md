# Utility Geometric Reconciliation: Point Cloud vs. LER Registry

Tools for visualising, segmenting, labelling, and comparing underground utility
infrastructure, combining the [OpenTrench3D](https://github.com/OpenTrench3D)
point-cloud dataset with Danish **LER** (Ledningsejerregistret) utility data.

The goal is **geometric reconciliation in full 3D**: comparing where a utility
actually is, as *measured* in an excavated point cloud, against where it is
*registered* in the LER utility registry, and quantifying the deviation between
the two. Depth (Z) is one component of that comparison, not the only axis.

Developed as part of an MSc thesis at the University of Twente.

## What is being compared

LER does not register the axis of a pipe. Horizontally it registers the
**centreline**, and vertically the **top** of the utility ("Vertikale
koordinater af geometrien angives for overkanten af ledningen"). Photogrammetry
has the mirror-image limitation: it reconstructs only the exposed upper arc of a
pipe, also never the axis.

Measuring the raw instance points against the registered line therefore compares
a surface against a line, and carries a lateral bias of roughly one pipe radius.
To avoid that, the measured cloud is first reduced to its **crown line** by
`core/crown.py`: the horizontal centreline recovered from the skeleton of the
plan-view footprint, carrying a crown height found by rolling a ball upwards
through the cloud. Crown line against registered line is then top against top,
on the datum the register actually uses, with no radius bias.

The viewer reports both. The **crown line** deviation is the headline metric.
The **per-point** deviation is kept alongside it because it is defined for every
instance, including those whose shape yields no crown line.

## Architecture

The codebase is organised into three layers. `core/` is the single source of
truth; nothing is duplicated across modules.

```
Thesis/
├── core/                      # Shared library, imported by every module and tool
│   │
│   │   Configuration and loading
│   ├── config.py              # Site resolution, class/layer/utility-type definitions,
│   │                          #   DLF display colours, depth and signature rule constants
│   ├── site_local.py          # Active site (gitignored, per machine; see Running)
│   ├── data_loader.py         # init_site(): PLY/GML loading, cropping, ground/trench picks
│   ├── crop.py                # Circular and rectangular crop primitives, CropRegion dispatch
│   ├── site_status.py         # How far a site has been processed, read from its artefacts
│   │
│   │   Geometry and measurement
│   ├── crown.py               # Measured crown line: plan-view skeleton + upward ball roll
│   ├── depth.py               # Fallback hierarchy for vertices with no usable registered Z
│   ├── geometry.py            # Stateless Open3D mesh primitives and spatial helpers
│   │
│   │   LER semantics
│   ├── ler_lines.py           # Grouping registry features into whole utility lines
│   ├── ler_matching.py        # Heuristic instance to LER line match suggestion
│   ├── ledningstrace.py       # Trace detection and forsyningsart colouring
│   ├── symbology.py           # LER "Signaturforklaring" line styles as numpy arrays
│   │
│   │   Rendering and GUI
│   ├── rendering.py           # Shared Open3D materials and scene lighting
│   ├── gui_helpers.py         # Legend swatches and rows, layer toggles, camera moves
│   ├── signature_render.py    # Open3D geometry for the line signatures
│   ├── signature_legend.py    # The Signaturforklaring as a collapsible panel section
│   ├── trace_render.py        # Trace corridor ribbon plus solid centreline
│   │
│   │   Export
│   └── ler_las_export.py      # Deviation clouds to georeferenced LAS for QGIS
│
├── modules/                   # Interactive Open3D GUI applications
│   ├── base_module.py         # Point cloud + LER overlays, indicative depth, click picking
│   ├── segment_module.py      # HDBSCAN/DBSCAN instance segmentation, live tuning controls
│   ├── label_module.py        # Utility-type labels, plus links from an instance to an LER line
│   ├── deviation_module.py    # * Geometric deviation: labelled instances vs. LER registry
│   ├── ERR_module.py          # Top-view plan of the Graveforesp: every site and utility in it
│   └── agent_module.py        # Natural-language queries over a site via a Claude AI agent
│
├── tools/                     # One-off command-line batch utilities
│   ├── ply_to_las.py          # Local-origin PLY to georeferenced LAS (UTM32 / EPSG:25832)
│   ├── convert_main_utility_to_water_instance.py  # Classes 0 and 3 to WaterLine instances
│   └── qml_distributor.py     # Copy a QGIS .qml style to every point cloud in target folders
│
├── Data/                      # Point clouds, GML packages, reference coords (git-ignored)
├── requirements.txt           # Pinned Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

`* deviation_module.py` is the core thesis deliverable; the other modules
prepare the data it consumes (segment, label, reconcile).

Several `core/` files are deliberately free of Open3D so they stay testable
without a display: `config.py`, `crop.py`, `depth.py`, `ledningstrace.py`,
`ler_lines.py`, `ler_las_export.py`, `ler_matching.py`, `site_status.py` and
`symbology.py`. Keep them that way when editing.

## Typical workflow

1. **Inspect** a site with `base_module.py`: the point cloud rendered with LER
   utility overlays, indicative depth estimation, and clickable feature
   inspection.
2. **Segment** the point cloud into instances with `segment_module.py`
   (HDBSCAN, with live `MIN_CLUSTER_SIZE` / `MIN_SAMPLES` tuning).
3. **Label** each instance with a utility type using `label_module.py`, which
   saves instances as PLY files carrying a `utility_type` attribute.

   An instance can also be linked to one specific registered utility. Left-click
   an LER line while the instance is active, or use "Suggest LER match" to have
   the most likely nearby line proposed automatically, ranked by proximity,
   direction, diameter and colour similarity (see `core/ler_matching.py`).

   A link covers the **whole line, not the clicked fragment**. The registry
   splits one physical utility into several features, so `core/ler_lines.py`
   groups the features that continue into each other and the match records every
   `gml_id` on the run. Lines that merely run alongside each other stay
   separate. Shift-click adds or removes a single feature where that grouping
   needs correcting.

   "Mark as NOT in LER" records that an instance has no registry counterpart at
   all. Matches are saved to `ler_matches.json` next to the labelled PLYs. A pick
   the viewer refused, because the LER layer contradicted the instance's label,
   is kept beside it in `ler_match_conflicts.json` so the refusal is recoverable.
4. **Reconcile** the labelled instances against the registry with
   `deviation_module.py`, which quantifies and visualises the deviation in XYZ,
   XY and Z, in both discrete accuracy-class and continuous-gradient colourings.
   Dutch KLIC/WIBON pass-fail colouring is available as a separate mode.

   An instance with a recorded LER link (step 3) is measured against only that
   one line, in both directions: its own deviation statistics and colouring, and
   the deviation clouds painted onto the registered geometry. An instance
   confirmed absent from LER is never measured at all, which avoids a false
   match to an unrelated nearby feature of the same type. Everything else falls
   back to being measured against every nearby feature whose layer matches its
   utility type.

   The discrete deviation clouds can be exported to georeferenced LAS
   (`core/ler_las_export.py`) and opened in QGIS as an overlay on the LER
   utilities. The continuous gradient modes do not map onto the five accuracy
   classes and are intentionally left out of the export.

## On-disk artefacts

Every stage writes its result next to the point cloud, so how far a site has
been processed can be read without opening a viewer. `core/site_status.py` is
the single source of truth for these conventions, including which `labeled_*`
folder wins when several exist.

```
Area_5_Site_11.ply
Area_5_Site_11_ground.json             ground pick        (base / label / deviation)
Area_5_Site_11_trench.json             trench pick        (deviation)
Site_11_Instances/
  0_instance_0_type_7.ply              water instance     (tools/convert_...)
  20260615_151447/                     segment run        (segment_module)
    1_instance_0.ply ...
  labeled_20260706_115025/             label session      (label_module)
    1_instance_0_type_4.ply ...
    ler_matches.json                   LER matches        (label_module)
    ler_match_conflicts.json           refused picks      (label_module)
Area_5_Site_11_LER_deviation_LAS/      deviation export   (deviation_module)
```

## Setup

```bash
pip install -r requirements.txt
```

Development uses a conda environment named `thesis`. On Windows this matters
beyond convenience: MKL's DLLs live in `<env>\Library\bin`, which only reaches
`PATH` through `conda activate`. Without activation, BLAS-backed numpy calls
(`@`, `np.dot`, `np.linalg.svd`, `np.linalg.eigh`, `sklearn.decomposition.PCA`)
terminate the interpreter with error `0xC06D007F` instead of raising. Always run
the modules from an activated environment.

Place the OpenTrench3D point clouds and LER utility packages under `Data/`
(see [Data model](#data-model)). Data paths are **project-root-relative** by
default, so no source edits are needed if `Data/` lives inside the project. To
keep the data elsewhere, point the `THESIS_DATA_DIR` environment variable at it:

```bash
# Windows (PowerShell)
$env:THESIS_DATA_DIR = "D:\thesis_data"
# macOS / Linux
export THESIS_DATA_DIR=/mnt/thesis_data
```

## Running

All modules and tools are run from the **project root** so that `core` is
importable:

```bash
python modules/base_module.py
python modules/deviation_module.py
python tools/ply_to_las.py --area 3
```

**Selecting a site.** The active site lives in `core/site_local.py`, which is
gitignored, so switching sites never dirties a tracked file. There is no
committed default: `core/config.py` reads both fields from it and raises a clear
error if the file or a field is missing, rather than silently loading a
different site. Create it once per machine with exactly these two fields:

```python
# core/site_local.py
SITE = "Water_Area_5/Area_5_Site_11.ply"   # path under Data/OpenTrench3D/
LEDNINGSPAKKE_DIR = "Ledningspakke_2803288_Area_4_and_5"   # folder under Data/ holding consolidated.gml
```

Every script then reads its configuration from `core/config.py`. Never put a
site path there.

**Crop region.** `CROP_MODE` in `core/config.py` defaults to `"rect"`: the cloud
is kept in full and utilities are selected and clipped to its XY bounding box,
expanded by `UTILITY_RECT_BUFFER` (2 m). Selection is XY-only, so a utility
passing through the footprint is kept whatever its depth. Set `CROP_MODE` to
`"circle"` for the legacy disc crop of radius `CROP_RADIUS` instead.

## Data model

| Input | Format | Notes |
|-------|--------|-------|
| Point clouds | PLY | OpenTrench3D dataset; attributes `x, y, z, r, g, b, class` |
| Utility registry | GML | Danish Ledningspakke / LER; one `consolidated.gml` per package |
| Reference coordinates | GeoJSON | Area origin points, UTM32 / ETRS89 (EPSG:25832) |
| Labelled instances | PLY | Per-instance clouds with a `utility_type` attribute |
| LER matches | JSON | Instance to utility-line links, written beside the labelled PLYs |

**Coordinates and units.** The horizontal CRS is **EPSG:25832** (ETRS89 / UTM
32N) and the vertical datum is **DVR90** (orthometric). Units are metres
throughout, with one exception: `vejledendeDybde` is carried in millimetres by
the GML. Viewer coordinates are *local*, that is UTM minus a per-area
translation, so any comparison mixing the two must translate first.

**OpenTrench3D semantic classes** (point-level): `Main Utility`,
`Other Utility`, `Trench`, `Inactive Utility`, `Misc`.

**Utility-type labels** (instance-level, assigned in `label_module`):
`PowerLine`, `DrainageLine`, `OilPipeLine`, `GasLine`, `ThermalLine`,
`Conduit`, `WaterLine`, `TelecomunicationLine`, `OtherLine`,
`LineUnknownServiceType`.

All class, layer and utility-type definitions and their DLF-recommended display
colours live in [`core/config.py`](core/config.py).

## Requirements

Python 3.10 or later. All dependencies are pinned in
[`requirements.txt`](requirements.txt):

- `open3d`: 3D rendering and GUI
- `geopandas` / `pyogrio` / `shapely`: reading and filtering GML utility layers
- `pyproj`: UTM32 / ETRS89 reprojection (`tools/ply_to_las.py`, `core/ler_las_export.py`)
- `numpy`
- `scipy`: cKDTree and ndimage (`core/crown.py`, `core/ler_lines.py`)
- `scikit-image`: skeletonize, the plan-view centreline in `core/crown.py`
- `scikit-learn`: HDBSCAN / DBSCAN segmentation, PCA in `core/ler_matching.py`
- `matplotlib`: trench footprint paths and colour conversion
- `plyfile`, `laspy`: PLY and LAS I/O

The `agent_module` talks to the Anthropic API directly over `urllib` (no SDK
package needed); it expects an API key in `API-KEY.env` at the project root
(git-ignored).

## Data and secrets

Data files (`*.ply`, `*.las`, `*.gml`, `*.gpkg`, and similar), the `Data/`
directory, API keys, interview material, and personal notes are excluded from
version control via `.gitignore`.

## Licence

MIT. See [LICENSE](LICENSE).
