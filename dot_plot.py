import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stl import mesh


def load_stl(stl_path):
    """
    Loads an STL file and returns relevant data for Plotly and offsetting.
    - vehicle_mesh: The raw mesh-stl object.
    - plotly_data: Tuple of (x, y, z, i, j, k) for go.Mesh3d.
    """
    vehicle_mesh = mesh.Mesh.from_file(stl_path)
    # Flatten the 3D vectors into a long list of vertices
    vertices = vehicle_mesh.vectors.reshape(-1, 3)
    # Define triangle indices for Plotly (0,1,2), (3,4,5), etc.
    i = np.arange(0, len(vertices), 3)
    j = i + 1
    k = i + 2
    return vehicle_mesh, (vertices[:, 0], vertices[:, 1], vertices[:, 2], i, j, k)


def get_surface_offsets(vehicle_mesh, points, dist):
    """
    Calculates 3D offset vectors to lift points off the STL surface.
    Finds the nearest triangle normal to each point and scales it by 'dist'.
    """
    centroids = vehicle_mesh.centroids
    normals = vehicle_mesh.normals
    result = []
    for p in points:
        # Find nearest triangle centroid by squared Euclidean distance
        idx = np.argmin(np.sum((centroids - p) ** 2, axis=1))
        n = normals[idx]
        # Ensure normal is unit length
        mag = np.linalg.norm(n)
        if mag > 0:
            n = n / mag
        result.append(n * dist)
    return np.array(result)


def generate_dot_plot(config_path):
    """
    Main entry point for generating the 3D dot subplot comparison.
    Loads JSON, processes STL, applies offsets, and saves HTML with camera sync.
    """
    base_dir = os.path.dirname(config_path)

    # Load JSON and strip comments (supports # style)
    with open(config_path, "r") as f:
        content = f.read()
        content = re.sub(r"#.*", "", content)
        config = json.loads(content)

    # 1. Load STL Geometry
    stl_path = os.path.join(base_dir, config["stl_path"])
    vehicle_mesh, (mx, my, mz, mi, mj, mk) = load_stl(stl_path)

    # 2. Setup Subplots for comparison
    cases = config["cases"]
    n_cases = len(cases)
    settings = config["dot_settings"]
    opacity = settings.get("mesh_opacity", 0.3)
    offset_dist = settings.get("offset", 0.0)
    cam = config["camera"]

    # Create a layout with 1 row and N columns
    fig = make_subplots(
        rows=1,
        cols=n_cases,
        specs=[[{"type": "scene"}] * n_cases],
        subplot_titles=[case["name"] for case in cases],
    )

    # 3. Load Tap Positions
    pos_df = pd.read_csv(
        os.path.join(base_dir, config["tap_positions"]),
        header=None,
        names=["number", "x", "y", "z"],
    )

    # Apply surface offset to prevent dots from clipping through triangles
    if offset_dist != 0:
        offsets = get_surface_offsets(
            vehicle_mesh, pos_df[["x", "y", "z"]].values, offset_dist
        )
        pos_df["x"] += offsets[:, 0]
        pos_df["y"] += offsets[:, 1]
        pos_df["z"] += offsets[:, 2]

    # 4. Create Traces for each case subplot
    for idx, case in enumerate(cases):
        col = idx + 1
        case_df = pd.read_csv(os.path.join(base_dir, case["data_path"]))
        # Merge physical positions with Cp results via tap numbers
        df = pd.merge(pos_df, case_df, on="number")

        # Background Mesh trace (Geometry)
        mesh_trace = go.Mesh3d(
            x=mx,
            y=my,
            z=mz,
            i=mi,
            j=mj,
            k=mk,
            color="lightgrey",
            opacity=opacity,
            name="Geometry",
            hoverinfo="skip",
            showlegend=(idx == 0),
        )
        fig.add_trace(mesh_trace, row=1, col=col)

        # Dynamic Dot Sizing (Scale marker by Cp magnitude)
        if settings.get("scale_by_value", False):
            marker_size = df["cp"].abs() * settings.get("scale_factor", 10.0)
        else:
            marker_size = settings["size"]

        # Scatter3d trace (Pressure Tap Dots)
        trace = go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=df["cp"],
                colorscale=settings["colorscale"],
                cmin=settings["cmin"],
                cmax=settings["cmax"],
                reversescale=settings.get("reversescale", False),
                # Show colorbar only on the last plot to save space
                colorbar=dict(title="Cp", thickness=15, x=1.02)
                if idx == n_cases - 1
                else None,
                line=dict(
                    color=settings.get("line_color", "darkgrey"),
                    width=settings.get("line_width", 1.0),
                ),
            ),
            text=df["number"],
            name=f"Taps: {case['name']}",
            hovertemplate="<b>Tap %{text}</b><br>Cp: %{marker.color:.3f}<extra></extra>",
        )
        fig.add_trace(trace, row=1, col=col)

        # Apply specific camera view to this subplot's scene
        scene_name = f"scene{col if col > 1 else ''}"
        fig.update_layout(
            {
                scene_name: dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    aspectmode="data",
                    camera=dict(eye=cam["eye"], center=cam["center"], up=cam["up"]),
                )
            }
        )

    # 5. Global Figure Layout
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=60),
        title_text="3D Pressure Distribution Comparison",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # 6. Camera Sync JavaScript
    # This ensures rotating one plot rotates the other side-by-side
    div_id = "plotly-3d-sync"
    sync_script = f"""
        var gd = document.getElementById('{div_id}');
        gd.on('plotly_relayout', function(edata) {{
            var camera_keys = Object.keys(edata).filter(k => k.includes('camera'));
            if (camera_keys.length > 0 && !gd._syncing) {{
                gd._syncing = true;
                var cam = edata[camera_keys[0]];
                var update = {{}};
                for (var i = 1; i <= {n_cases}; i++) {{
                    var sceneName = 'scene' + (i > 1 ? i : '');
                    update[sceneName + '.camera'] = cam;
                }}
                Plotly.relayout(gd, update).then(() => {{ gd._syncing = false; }});
            }}
        }});
    """

    # 7. Write to HTML with CDN js and sync script
    output_path = os.path.join(base_dir, config["output_path"])
    html_content = fig.to_html(
        include_plotlyjs="cdn",
        post_script=sync_script,
        div_id=div_id,
        full_html=True,
    )
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Generated 3D plot with synced cameras: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 3D Pressure Tap Dot Plots from STL and CSV."
    )
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="demo_data/drivAer_dotplot/dotplot_config.json",
        help="Path to the dotplot configuration JSON file.",
    )
    args = parser.parse_args()

    if os.path.exists(args.config):
        generate_dot_plot(args.config)
    else:
        print(f"Error: Configuration file not found at {args.config}")
