# OpenTrench3D — Utility Infrastructure Labeling Pipeline

Point cloud visualization and labeling tools for underground utility infrastructure, built on the [OpenTrench3D](https://github.com/OpenTrench3D) dataset and Danish LER utility data.

Developed as part of an MSc thesis at the University of Twente.

## Folder Structure

```
Thesis/
├── Data/                   # Point clouds, GML utility data, reference files (git-ignored)
├── Developing/
│   ├── 1. Base Viewer/     # 3D point cloud viewer with LER utility overlays
│   ├── 2. Segment Viewer/  # Instance segmentation (HDBSCAN-based)
│   ├── 3. Label Viewer/    # Instance labeling tool for assigning utility types
│   ├── 4. Linking Viewer/  # (in development)
│   ├── 5. Deviation Viewer/# (in development)
├── Testing/                # Earlier prototypes and experiments
└── .gitignore
```

## Key Scripts

| Script | Description |
|--------|-------------|
| `BASE1_visualise_single_pointcloud_with_utilities_depth_classlabels_clickable.py` | Base viewer — displays a cropped point cloud with LER utility overlays, depth estimation, class labels, and clickable feature inspection. |
| `LABEL1_visualise_single_pointcloud_with_instances.py` | Label viewer — shows segmented instances one at a time (largest first) for assigning one of 10 LER utility type labels. Saves labeled instances as PLY files with a `utility_type` attribute. |

## Requirements

- Python 3.9+
- Open3D
- GeoPandas
- NumPy

## Data

The pipeline expects:
- **Point clouds** in PLY format (OpenTrench3D dataset)
- **Utility data** in GML format (Danish Ledningspakke / LER 2.0)
- **Reference coordinates** in GeoJSON (area origin points in UTM32 / ETRS89)

Data files are excluded from version control via `.gitignore`.
