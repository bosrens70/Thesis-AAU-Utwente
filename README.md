# Utility Depth Reconciliation: Point Cloud vs. LER Registry

Tools for visualising, segmenting, labelling, and comparing underground utility
infrastructure, combining the [OpenTrench3D](https://github.com/OpenTrench3D)
point-cloud dataset with Danish **LER** (Ledningsejerregistret) utility data.

The thesis goal is **depth reconciliation**: comparing the depth of utilities as
*measured* in the excavated point cloud against the depth/geometry *registered*
in the LER utility registry, and surfacing the deviation between the two.

Developed as part of an MSc thesis at the University of Twente.

## Architecture

The codebase is organised into three layers:

```
Thesis/
├── core/                 # Shared library, imported by every viewer and tool
│   ├── config.py             # Single source of truth: site paths, class labels, layer colours
│   ├── data_loader.py        # init_site() + PLY/GML loading, cropping, dataclasses
│   ├── geometry.py           # Stateless Open3D mesh primitives & spatial helpers
│   ├── gui_helpers.py        # Shared Open3D GUI widgets (legend swatches, toggles)
│   └── ledningstrace.py      # Ledningstrace forsyningsart detection & colouring
│
├── viewers/              # Interactive Open3D GUI applications
│   ├── base_viewer.py        # Point cloud + LER overlays, indicative depth, clickable picking
│   ├── segment_viewer.py     # HDBSCAN/DBSCAN instance segmentation with live tuning controls
│   ├── label_viewer.py       # Assign utility-type labels to instances; saves labelled PLYs
│   ├── deviation_viewer.py   # ★ Geometric deviation: labelled instances vs. LER registry
│   ├── graveforesp_viewer.py # Multi-site overview: all point clouds + all utilities, toggleable
│   └── agent_viewer.py       # Natural-language queries over a site via a Claude AI agent
│
├── tools/                # One-off command-line batch utilities
│   ├── ply_to_las.py         # Convert local-origin PLY → georeferenced LAS (UTM32 / EPSG:25832)
│   ├── convert_main_utility_to_water_instance.py  # Extract class-0 points as WaterLine instances
│   └── qml_distributor.py    # Copy a QGIS .qml style to every point cloud in target folders
│
├── Data/                 # Point clouds, GML utility packages, reference coords (git-ignored)
├── requirements.txt      # Pinned Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

`★ deviation_viewer.py` is the core thesis deliverable; the other viewers
prepare the data it consumes (segment → label → reconcile).

## Typical workflow

1. **Inspect** a site with `base_viewer.py`: point cloud rendered with LER
   utility overlays, indicative depth estimation, and clickable feature inspection.
2. **Segment** the point cloud into instances with `segment_viewer.py`
   (HDBSCAN, with live `MIN_CLUSTER_SIZE` / `MIN_SAMPLES` tuning).
3. **Label** each instance with a utility type using `label_viewer.py`
   (saves instances as PLY files carrying a `utility_type` attribute).
4. **Reconcile** the labelled instances against the LER registry with
   `deviation_viewer.py` to quantify and visualise depth/geometry deviation.

## Setup

```bash
pip install -r requirements.txt
```

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

All viewers and tools are run from the **project root** so that `core` is
importable:

```bash
python viewers/base_viewer.py
python viewers/deviation_viewer.py
python tools/ply_to_las.py --area 3
```

**Switching sites:** edit `PLY_FILE` (and, for a different Ledningspakke,
`GML_PATH` / `AREA_REF_GEOJSON`) in [`core/config.py`](core/config.py).
Every script reads its configuration from there.

## Data model

| Input | Format | Notes |
|-------|--------|-------|
| Point clouds | PLY | OpenTrench3D dataset; attributes `x, y, z, r, g, b, class` |
| Utility registry | GML | Danish Ledningspakke / LER 2.0 (per-owner GML files) |
| Reference coordinates | GeoJSON | Area origin points, UTM32 / ETRS89 (EPSG:25832) |
| Labelled instances | PLY | Per-instance clouds with a `utility_type` attribute |

**OpenTrench3D semantic classes** (point-level): `Main Utility`,
`Other Utility`, `Trench`, `Inactive Utility`, `Misc`.

**Utility-type labels** (instance-level, assigned in `label_viewer`):
`PowerLine`, `DrainageLine`, `OilPipeLine`, `GasLine`, `ThermalLine`,
`Conduit`, `WaterLine`, `TelecomunicationLine`, `OtherLine`,
`LineUnknownServiceType`.

All class/layer/utility-type definitions and their DLF-recommended display
colours live in [`core/config.py`](core/config.py).

## Requirements

Python 3.9+. All dependencies are pinned in
[`requirements.txt`](requirements.txt):

- `open3d`: 3D rendering and GUI
- `geopandas` / `pyogrio` / `shapely`: reading and filtering GML utility layers
- `pyproj`: UTM32 / ETRS89 reprojection (`tools/ply_to_las.py`)
- `numpy`
- `scikit-learn`: HDBSCAN / DBSCAN segmentation (`segment_viewer`)
- `plyfile`, `laspy`: PLY to LAS conversion (`tools/ply_to_las.py`)

The `agent_viewer` talks to the Anthropic API directly over HTTP (no SDK
package needed); it expects an API key in `API-KEY.env` at the project root
(git-ignored).

## Data & secrets

Data files (`*.ply`, `*.las`, `*.gml`, `*.gpkg`, …), the `Data/` directory, API
keys, and personal notes are excluded from version control via `.gitignore`.
