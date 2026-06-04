# -*- coding: utf-8 -*-
"""
Agent Viewer — Natural-language queries on a single site's utility data
========================================================================
Loads one point cloud + GML utilities filtered to its bounding box,
then lets you query the data via a Claude AI agent in the GUI.

Setup:  Place your API key in API-KEY.env in the project root.
Usage:  python viewers/agent_viewer.py
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
import geopandas as gpd
import numpy as np
import time
import os
import traceback
import urllib.request
import urllib.error
import json as _json

from core.config import (
    PLY_FILE, GML_PATH, AREA_REF_GEOJSON, CROP_RADIUS,
    CLASS_LABELS, DEFAULT_CLASS_COLOR,
    LINE_LAYERS, COMPONENT_LAYERS, COMP_TO_LINE,
    forsyningsart_color,
)
from core.data_loader import init_site, pick_ground_level
from core.geometry import segment_to_cylinder, segment_to_plane, linear_to_srgb
from core.ledningstrace import get_ledningstrace_display_info, get_storage_key, get_bredde_width

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load site data + ground picking
# ─────────────────────────────────────────────────────────────────────────────
_t0 = time.perf_counter()
site = init_site(load_instances=False)

area = site.area
TX, TY, TZ = area.TX, area.TY, area.TZ
pcd             = site.pc.pcd
pts             = site.pc.pts
original_colors = site.pc.original_colors
class_labels    = site.pc.class_labels
cloud_centroid  = site.pc.cloud_centroid
pc_min          = site.pc.pc_min
pc_max          = site.pc.pc_max

_ply_path = Path(PLY_FILE)
_crop_cx_local = site.pc.crop_center_local[0]
_crop_cy_local = site.pc.crop_center_local[1]
_crop_cx_utm   = site.pc.crop_center_utm[0]
_crop_cy_utm   = site.pc.crop_center_utm[1]

# Ground picking
GROUND_Z = pick_ground_level(site.pc)
_pick_method = site.pc.ground_z_method
print(f"  Ground level (UTM)   = {GROUND_Z + TZ:.3f} m")

# Local bounding box for utility filtering
_bbox_min_utm = pc_min + np.array([TX, TY, 0])
_bbox_max_utm = pc_max + np.array([TX, TY, 0])


def _in_bbox_utm(x, y):
    return (_bbox_min_utm[0] <= x <= _bbox_max_utm[0] and
            _bbox_min_utm[1] <= y <= _bbox_max_utm[1])


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Filter GML to site bbox + build local-coord utility meshes
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Filtering GML to site bbox ---")

from shapely.geometry import box as _box
_site_box = _box(_bbox_min_utm[0] - 2, _bbox_min_utm[1] - 2,
                 _bbox_max_utm[0] + 2, _bbox_max_utm[1] + 2)

gdfs = {}
if site.gml:
    for name, gdf in site.gml.line_gdfs.items():
        if gdf.empty:
            continue
        mask = gdf.geometry.notna() & gdf.intersects(_site_box)
        filtered = gdf[mask].copy()
        if not filtered.empty:
            gdfs[name] = filtered
    for name, gdf in site.gml.component_gdfs.items():
        if gdf.empty:
            continue
        mask = gdf.geometry.notna() & gdf.intersects(_site_box)
        filtered = gdf[mask].copy()
        if not filtered.empty:
            gdfs[name] = filtered

n_total = sum(len(g) for g in gdfs.values())
print(f"  {len(gdfs)} layers with features near site  ({n_total:,} features total)")
for name, gdf in gdfs.items():
    print(f"    {name:<35} {len(gdf):>4} features")

# Build utility meshes — only features whose geometry CROSSES the point cloud
# AABB are shown, and each such feature is drawn in full (no per-segment trimming),
# so the utilities that actually pass through the site appear as continuous lines.
_t_mesh = time.perf_counter()
_pc_aabb_box = _box(_bbox_min_utm[0], _bbox_min_utm[1],
                    _bbox_max_utm[0], _bbox_max_utm[1])


# Track Ledningstrace forsyningsart variants
_ledningstrace_variants = {}

_util_meshes = {}
for layer_name, gdf in gdfs.items():
    if layer_name not in LINE_LAYERS:
        continue
    cfg = LINE_LAYERS[layer_name]
    default_color = cfg["color"]
    radius = cfg["fallback_radius"]
    cyls = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        # Only draw features that cross the point cloud AABB (then drawn in full)
        if not geom.intersects(_pc_aabb_box):
            continue
        # Get Ledningstrace display info and width
        is_trace, display_fa, color = get_ledningstrace_display_info(layer_name, row, default_color)
        if is_trace and display_fa and display_fa not in _ledningstrace_variants:
            _ledningstrace_variants[display_fa] = color

        bredde_m = get_bredde_width(row)
        if is_trace and bredde_m is None:
            bredde_m = 0.25
        sub_geoms = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for sg in sub_geoms:
            coords = np.array(sg.coords, dtype=float)
            if coords.shape[1] == 2:
                coords = np.hstack([coords, np.zeros((len(coords), 1))])
            coords[:, 0] -= TX
            coords[:, 1] -= TY
            coords[:, 2] -= TZ
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i + 1]
                if bredde_m is not None:
                    mesh = segment_to_plane(p1, p2, bredde_m, color)
                else:
                    mesh = segment_to_cylinder(p1, p2, radius, color)
                if mesh is not None:
                    cyls.append(mesh)
    if cyls:
        merged = cyls[0]
        for c in cyls[1:]:
            merged += c
        merged.compute_vertex_normals()
        _util_meshes[layer_name] = merged

n_segs = sum(len(v.triangles) // 12 for v in _util_meshes.values()) if _util_meshes else 0
print(f"  Utility meshes: {len(_util_meshes)} layers, ~{n_segs} segments  "
      f"[{time.perf_counter() - _t_mesh:.2f}s]")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Build GML schema for the agent (from filtered data)
# ─────────────────────────────────────────────────────────────────────────────
def _build_schema_prompt():
    lines = [
        "You have access to GeoDataFrames in a dict called `gdfs`.",
        "These are spatially filtered to only features near this point cloud.",
        "Access a layer with: gdfs['LayerName']",
        "",
        f"Site: {_ply_path.stem}  |  Area: {area.area_name}",
        f"Points: {len(pts):,}  |  Ground Z: {GROUND_Z:.3f} m",
        f"Local bbox: X[{pc_min[0]:.1f}, {pc_max[0]:.1f}]  "
        f"Y[{pc_min[1]:.1f}, {pc_max[1]:.1f}]  "
        f"Z[{pc_min[2]:.1f}, {pc_max[2]:.1f}]",
        f"UTM offset: TX={TX:.3f}, TY={TY:.3f}, TZ={TZ:.3f}",
        "",
    ]
    for name, gdf in gdfs.items():
        n = len(gdf)
        cols = []
        for col in gdf.columns:
            if col == "geometry":
                geom_types = [t for t in gdf.geometry.geom_type.unique().tolist() if t is not None]
                cols.append(f"geometry ({', '.join(geom_types) if geom_types else 'mixed/null'})")
            else:
                dtype = str(gdf[col].dtype)
                non_null = gdf[col].dropna()
                if len(non_null) > 0:
                    samples = non_null.head(3).tolist()
                    sample_str = str(samples)
                    if len(sample_str) > 80:
                        sample_str = sample_str[:80] + "..."
                    cols.append(f"{col} ({dtype}, e.g. {sample_str})")
                else:
                    cols.append(f"{col} ({dtype}, all null)")
        lines.append(f"Layer: '{name}' — {n} features")
        for c in cols:
            lines.append(f"  - {c}")
        lines.append("")
    return "\n".join(lines)


SCHEMA_PROMPT = _build_schema_prompt()

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Claude API (raw urllib — no pip install needed)
# ─────────────────────────────────────────────────────────────────────────────
_api_key_cache = [None]


def _read_dotenv_key():
    for filename in [".env", "api-key.env", "API-KEY.env"]:
        env_file = _project_root / filename
        if not env_file.exists():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
            if line.startswith("sk-ant-"):
                return line
    return None


def _get_api_key():
    if _api_key_cache[0] is not None:
        return _api_key_cache[0]
    key = os.environ.get("ANTHROPIC_API_KEY") or _read_dotenv_key()
    if not key:
        print("[agent] No API key found.")
        for fn in [".env", "api-key.env", "API-KEY.env"]:
            p = _project_root / fn
            print(f"  - {p}: {'exists' if p.exists() else 'not found'}")
        return None
    _api_key_cache[0] = key
    return key


SYSTEM_PROMPT = f"""You are a GML utility data analyst embedded in an Open3D point cloud viewer.
You answer questions about Danish underground utility infrastructure (LER data)
for a specific point cloud site.  You remember the full conversation — you can
refer to previous answers and build on them.

{SCHEMA_PROMPT}

When the user asks a question, respond with a Python code block.
The code must assign the answer to a variable called `result` (a string).
You can use pandas, geopandas, numpy (as np), and shapely.
Keep code concise (1-20 lines).  Only use layers that exist in `gdfs`.

Example — data query:
```python
result = f"There are {{len(gdfs['Vandledning'])}} water pipe features near this site."
```

Example — highlight + answer:
```python
big = gdfs['Vandledning'][gdfs['Vandledning']['udvendigDiameter'] > 300]
highlight_features(big, color=[1, 1, 0])
result = f"Highlighted {{len(big)}} Vandledning with diameter > 300mm."
```

If it's a general knowledge question (no data query needed), answer directly in plain text.
Always be concise — the answer displays in a small GUI panel.

── ACTION FUNCTIONS (available in your code) ──────────────────────────

highlight_features(gdf, color=[1,1,0], radius=0.02)
    Highlight GeoDataFrame rows as bright cylinders/spheres in the 3D scene.
    Works with both line and point geometries.
    color: [R, G, B] 0-1.  Default yellow.
    radius: cylinder/sphere radius in metres.
    Automatically clears previous highlights.

clear_highlights()
    Remove all highlight meshes from the scene.

hide_layer(name)
    Hide a utility layer.  name must match a key in gdfs.

show_layer(name)
    Show a previously hidden utility layer.

show_all_layers()
    Show all utility layers.

pivot_to(x_local, y_local, z_local)
    Move the camera to look at a specific local coordinate.

get_visible_layers()
    Returns a list of currently visible layer names.

── CONTEXT ────────────────────────────────────────────────────────────
- This is a single site ({_ply_path.stem}) with only nearby utilities loaded.
- Danish LER data (Ledningsejerregistret).
- Coordinates: UTM32/ETRS89 (EPSG:25832). Local coords = UTM minus (TX, TY, TZ).
- Common columns: driftsstatus ("i drift"=active, "permanent ude af drift"=inactive),
  vejledendeDybde (indicative depth, mm), udvendigDiameter (external diameter, mm),
  ejerforhold (ownership), forsyningsart (utility sub-type for Ledningstrace).
- Z = -99 means no depth measurement registered.
- DLF = Danish utility colour standard.
- Available layers: {list(gdfs.keys())}
"""


def _query_agent(question: str) -> str:
    api_key = _get_api_key()
    if api_key is None:
        return "[ERROR] No API key found.\nCreate API-KEY.env with your key."

    # Build multi-turn messages from chat history (conversation memory)
    messages = []
    for role, text in _chat_history[:-1]:  # exclude the "Thinking..." placeholder
        if role == "user":
            messages.append({"role": "user", "content": text})
        else:
            messages.append({"role": "assistant", "content": text})
    # Add the current question
    messages.append({"role": "user", "content": question})

    # Ensure messages alternate user/assistant (API requirement)
    # Merge consecutive same-role messages
    merged = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(msg)
    # Must start with user
    if merged and merged[0]["role"] != "user":
        merged = merged[1:]

    payload = _json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": merged,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = _json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        return f"[API ERROR {e.code}] {error_body}"
    except Exception as e:
        return f"[API ERROR] {e}"

    reply = body["content"][0]["text"]

    if "```python" in reply:
        code = reply.split("```python")[1].split("```")[0].strip()
        return _execute_query(code)
    return reply


def _execute_query(code: str) -> str:
    from shapely.geometry import Point, LineString, Polygon

    _pending_gui_actions.clear()

    exec_globals = {
        "gdfs": gdfs,
        "np": np,
        "pd": __import__("pandas"),
        "gpd": gpd,
        "Point": Point,
        "LineString": LineString,
        "Polygon": Polygon,
        "GROUND_Z": GROUND_Z,
        "TX": TX, "TY": TY, "TZ": TZ,
        # Action functions
        "highlight_features": highlight_features,
        "clear_highlights": clear_highlights,
        "hide_layer": hide_layer,
        "show_layer": show_layer,
        "show_all_layers": show_all_layers,
        "pivot_to": pivot_to,
        "get_visible_layers": get_visible_layers,
    }

    try:
        exec(code, exec_globals)
        result = exec_globals.get("result", "[No result variable set by query]")

        # Flush any GUI actions on the main thread
        if _pending_gui_actions:
            actions = list(_pending_gui_actions)
            _pending_gui_actions.clear()
            def _flush():
                for action in actions:
                    action()
                window.post_redraw()
            gui.Application.instance.post_to_main_thread(window, _flush)

        return str(result)
    except Exception:
        tb = traceback.format_exc()
        return f"[QUERY ERROR]\n{code}\n\n{tb}"


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Chat history
# ─────────────────────────────────────────────────────────────────────────────
_chat_history = []


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Open3D GUI
# ─────────────────────────────────────────────────────────────────────────────
app = gui.Application.instance
app.initialize()

window = app.create_window(
    f"Agent Viewer  |  {_ply_path.stem}  |  press H for help",
    1600, 900,
)
em = window.theme.font_size

scene_widget = gui.SceneWidget()
scene_widget.scene = rendering.Open3DScene(window.renderer)
scene_widget.scene.set_background([0.10, 0.10, 0.10, 1.0])

# Sun light
scene_widget.scene.scene.set_sun_light([0.0, 0.0, -1.0], [1.0, 1.0, 1.0], 75000)
scene_widget.scene.scene.enable_sun_light(True)

# Add point cloud
mat_pt = rendering.MaterialRecord()
mat_pt.shader = "defaultUnlit"
mat_pt.point_size = 3.0
scene_widget.scene.add_geometry("point_cloud", pcd, mat_pt)

# Add utility meshes
mat_mesh = rendering.MaterialRecord()
mat_mesh.shader = "defaultLitTransparency"
mat_mesh.base_color = [1.0, 1.0, 1.0, 0.8]
for layer_name, mesh in _util_meshes.items():
    scene_widget.scene.add_geometry(f"util_{layer_name}", mesh, mat_mesh)

bounds = scene_widget.scene.bounding_box
scene_widget.setup_camera(60, bounds, cloud_centroid.tolist())

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Action functions (called by the agent via code execution)
# ─────────────────────────────────────────────────────────────────────────────
_highlight_geom_names = []
_layer_visible = {ln: True for ln in _util_meshes}
_pending_gui_actions = []  # list of callables to run on main thread


def highlight_features(gdf_subset, color=None, radius=0.02):
    """Highlight GeoDataFrame rows as bright meshes in the scene."""
    if color is None:
        color = [1.0, 1.0, 0.0]  # yellow

    clear_highlights()

    cyls = []
    spheres = []
    for _, row in gdf_subset.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        if geom.geom_type in ("Point", "PointZ"):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=radius * 5, resolution=10)
            pt = np.array([geom.x - TX, geom.y - TY,
                           (geom.z - TZ) if geom.has_z else GROUND_Z])
            s.translate(pt)
            s.paint_uniform_color(color)
            s.compute_vertex_normals()
            spheres.append(s)
        else:
            # Check for bledde and get Ledningstrace display info if applicable
            bredde_m = get_bredde_width(row)
            highlight_color = color  # use provided highlight color
            # Detect if this is Ledningstrace and get its forsyningsart color
            is_trace, display_fa, trace_color = get_ledningstrace_display_info(None, row, color)
            if bredde_m is not None and display_fa:
                highlight_color = trace_color
            sub_geoms = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
            for sg in sub_geoms:
                coords = np.array(sg.coords, dtype=float)
                if coords.shape[1] == 2:
                    coords = np.hstack([coords, np.zeros((len(coords), 1))])
                coords[:, 0] -= TX
                coords[:, 1] -= TY
                coords[:, 2] -= TZ
                for i in range(len(coords) - 1):
                    p1, p2 = coords[i], coords[i + 1]
                    if bredde_m is not None:
                        mesh = segment_to_plane(p1, p2, bredde_m, highlight_color)
                    else:
                        mesh = segment_to_cylinder(p1, p2, radius, highlight_color)
                    if mesh is not None:
                        mesh.compute_vertex_normals()
                        cyls.append(mesh)

    all_meshes = cyls + spheres
    if not all_meshes:
        return

    merged = all_meshes[0]
    for m in all_meshes[1:]:
        merged += m

    gn = f"_highlight_{len(_highlight_geom_names)}"
    _highlight_geom_names.append(gn)

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency"
    mat.base_color = [color[0], color[1], color[2], 1.0]

    def _add():
        scene_widget.scene.add_geometry(gn, merged, mat)
        window.post_redraw()
    _pending_gui_actions.append(_add)


def clear_highlights():
    """Remove all highlight meshes from the scene."""
    for gn in _highlight_geom_names:
        def _rm(_gn=gn):
            if scene_widget.scene.has_geometry(_gn):
                scene_widget.scene.remove_geometry(_gn)
        _pending_gui_actions.append(_rm)
    _highlight_geom_names.clear()


def hide_layer(name):
    """Hide a utility layer by name."""
    gn = f"util_{name}"
    _layer_visible[name] = False
    def _hide():
        if scene_widget.scene.has_geometry(gn):
            scene_widget.scene.show_geometry(gn, False)
        window.post_redraw()
    _pending_gui_actions.append(_hide)


def show_layer(name):
    """Show a utility layer by name."""
    gn = f"util_{name}"
    _layer_visible[name] = True
    def _show():
        if scene_widget.scene.has_geometry(gn):
            scene_widget.scene.show_geometry(gn, True)
        window.post_redraw()
    _pending_gui_actions.append(_show)


def show_all_layers():
    """Show all utility layers."""
    for name in _util_meshes:
        show_layer(name)


def pivot_to(x_local, y_local, z_local):
    """Move camera to look at a local coordinate."""
    target = np.array([float(x_local), float(y_local), float(z_local)])
    d = max(1.0, np.linalg.norm(pc_max - pc_min) * 0.6)
    eye = target + np.array([d, -d, d * 0.6])
    def _pivot():
        scene_widget.look_at(target.tolist(), eye.tolist(), [0.0, 0.0, 1.0])
    _pending_gui_actions.append(_pivot)


def zoom_to_point_cloud():
    """Frame the camera on the point cloud centroid, ignoring the far-below
    -99 (unregistered-depth) utilities that otherwise distort the view.
    Safe to call directly from a GUI callback (button or key)."""
    d = max(1.0, np.linalg.norm(pc_max - pc_min) * 0.6)
    eye = cloud_centroid + np.array([d, -d, d * 0.6])
    scene_widget.look_at(cloud_centroid.tolist(), eye.tolist(), [0.0, 0.0, 1.0])


def get_visible_layers():
    """Return list of currently visible layer names."""
    return [ln for ln, vis in _layer_visible.items() if vis]


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Right panel — site info + chat
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
PANEL_WIDTH = int(24 * em)
panel = gui.Vert(int(0.5 * em), gui.Margins(int(em), int(em), int(em), int(em)))

# Site info
panel.add_child(gui.Label(f"Site: {_ply_path.stem}"))
panel.add_child(gui.Label(f"Area: {area.area_name}  |  Points: {len(pts):,}"))
panel.add_child(gui.Label(f"Ground Z: {GROUND_Z:.3f} m ({_pick_method})"))
panel.add_child(gui.Label(f"Utilities near site: {n_total:,} features"))
panel.add_fixed(int(0.5 * em))

# Camera reset — reframe on the point cloud (utilities at -99 distort the view)
zoom_btn = gui.Button("Zoom to point cloud")
zoom_btn.set_on_clicked(zoom_to_point_cloud)
panel.add_child(zoom_btn)
panel.add_fixed(int(0.5 * em))

# Header
header = gui.Label("Ask the Agent")
header.text_color = gui.Color(0.85, 0.85, 0.20, 1.0)
panel.add_child(header)
panel.add_fixed(int(0.3 * em))

# Chat output (scrollable)
chat_scroll = gui.ScrollableVert(int(0.2 * em),
                                  gui.Margins(int(0.3 * em), 0, int(0.3 * em), 0))
panel.add_child(chat_scroll)

_MAX_CHAT_LABELS = 40
_chat_labels = []
for _ in range(_MAX_CHAT_LABELS):
    lbl = gui.Label("")
    lbl.visible = False
    chat_scroll.add_child(lbl)
    _chat_labels.append(lbl)


def _refresh_chat():
    display_lines = []
    for role, text in _chat_history:
        prefix = "You: " if role == "user" else "Agent: "
        for line in text.split("\n"):
            display_lines.append((role, prefix + line))
            prefix = "  "
    start = max(0, len(display_lines) - _MAX_CHAT_LABELS)
    visible = display_lines[start:]
    for i, lbl in enumerate(_chat_labels):
        if i < len(visible):
            role, text = visible[i]
            lbl.text = text
            lbl.visible = True
            lbl.text_color = (gui.Color(0.65, 0.75, 1.00, 1.0) if role == "user"
                              else gui.Color(0.90, 0.90, 0.90, 1.0))
        else:
            lbl.visible = False
    window.set_needs_layout()
    window.post_redraw()


# Example queries
panel.add_fixed(int(0.3 * em))
ex_lbl = gui.Label("Examples:")
ex_lbl.text_color = gui.Color(0.55, 0.55, 0.55, 1.0)
panel.add_child(ex_lbl)

for eq in [
    "How many water pipes are near this site?",
    "Highlight the water pipes",
    "Which utilities have diameter > 100mm?",
    "Hide everything except Vandledning",
    "Who owns the gas pipes here?",
]:
    _el = gui.Label(f"  • {eq}")
    _el.text_color = gui.Color(0.45, 0.45, 0.45, 1.0)
    panel.add_child(_el)

panel.add_fixed(int(0.5 * em))

# Input row
input_row = gui.Horiz(int(0.3 * em))
text_input = gui.TextEdit()
text_input.placeholder_text = "Ask about this site's utilities..."

send_btn = gui.Button("Ask")
send_btn.horizontal_padding_em = 1.0

_is_querying = [False]


def _on_send():
    if _is_querying[0]:
        return
    question = text_input.text_value.strip()
    if not question:
        return

    _is_querying[0] = True
    text_input.text_value = ""
    _chat_history.append(("user", question))
    _chat_history.append(("agent", "Thinking..."))
    _refresh_chat()

    import threading

    def _run():
        answer = _query_agent(question)
        _chat_history[-1] = ("agent", answer)

        def _update():
            _refresh_chat()
            _is_querying[0] = False
        gui.Application.instance.post_to_main_thread(window, _update)

    threading.Thread(target=_run, daemon=True).start()


send_btn.set_on_clicked(_on_send)
text_input.set_on_value_changed(lambda _: _on_send())

input_row.add_child(text_input)
input_row.add_fixed(int(0.3 * em))
input_row.add_child(send_btn)
panel.add_child(input_row)

panel.add_fixed(int(0.3 * em))

# Status
_status_lbl = gui.Label("")
_status_lbl.text_color = gui.Color(0.55, 0.55, 0.55, 1.0)
panel.add_child(_status_lbl)

if _get_api_key():
    _status_lbl.text = "Claude API key found."
    _status_lbl.text_color = gui.Color(0.3, 0.8, 0.3, 1.0)
else:
    _status_lbl.text = "No API key. Create API-KEY.env"
    _status_lbl.text_color = gui.Color(0.9, 0.3, 0.3, 1.0)

panel.add_stretch()

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Key callbacks
# ─────────────────────────────────────────────────────────────────────────────
HANDLED = gui.Widget.EventCallbackResult.HANDLED
IGNORED = gui.Widget.EventCallbackResult.IGNORED


def on_key(event):
    if event.type != gui.KeyEvent.DOWN:
        return IGNORED
    k = event.key
    if k in (ord('C'), ord('c')):
        zoom_to_point_cloud()
        return HANDLED
    if k in (ord('H'), ord('h')):
        print("\n-- Agent Viewer Shortcuts ---------------------------------------")
        print("  C              pivot to point cloud centroid")
        print("  H              show this help")
        print("  Type a question and press Ask or Enter")
        print("----------------------------------------------------------------\n")
        return HANDLED
    return IGNORED


scene_widget.set_on_key(on_key)

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Layout + run
# ─────────────────────────────────────────────────────────────────────────────
def on_layout(layout_ctx):
    r = window.content_rect
    scene_widget.frame = gui.Rect(r.x, r.y, r.width - PANEL_WIDTH, r.height)
    panel.frame = gui.Rect(r.x + r.width - PANEL_WIDTH, r.y, PANEL_WIDTH, r.height)


window.set_on_layout(on_layout)
window.add_child(scene_widget)
window.add_child(panel)

print(f"\nAgent Viewer ready in {time.perf_counter() - _t0:.1f}s")
print(f"Site: {_ply_path.stem}  |  {n_total} utilities near site")
print("Type a question in the panel and press Ask.\n")

app.run()
print("Viewer closed.")
