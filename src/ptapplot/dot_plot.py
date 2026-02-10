import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import KDTree
from stl import mesh

try:
    from .utils import load_config_json, parse_taps_dataframe
except ImportError:
    from utils import load_config_json, parse_taps_dataframe


def load_stl(stl_path):
    """Loads an STL file and returns relevant data for Plotly and offsetting."""
    vehicle_mesh = mesh.Mesh.from_file(stl_path)
    # Extract vertices
    vertices = vehicle_mesh.vectors.reshape(-1, 3)
    # Plotly Mesh3d uses i, j, k to define triangles from vertices
    tri_i = np.arange(0, len(vertices), 3)
    tri_j = tri_i + 1
    tri_k = tri_i + 2
    return vehicle_mesh, (
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        tri_i,
        tri_j,
        tri_k,
    )


def get_surface_offsets(vehicle_mesh, points, dist):
    """
    Calculates offsets along the nearest triangle normals using KDTree for performance.
    """
    centroids = vehicle_mesh.centroids
    normals = vehicle_mesh.normals

    # Build KDTree for O(log N) lookup of nearest triangles
    tree = KDTree(centroids)
    _, indices = tree.query(points)

    selected_normals = normals[indices]
    # Normalize
    mags = np.linalg.norm(selected_normals, axis=1, keepdims=True)
    mags[mags == 0] = 1.0  # Avoid division by zero
    unit_normals = selected_normals / mags

    return unit_normals * dist


def generate_dot_plot(config_path):
    """
    Main entry point for generating the 3D dot subplot comparison.
    """
    config_path = Path(config_path)
    config = load_config_json(config_path)
    base_dir = config_path.parent

    # 1. Load STL Geometry
    stl_path = base_dir / config["stl_path"]
    vehicle_mesh, (mx, my, mz, mi, mj, mk) = load_stl(stl_path)

    # 2. Setup Subplots for comparison
    cases = config["cases"]
    n_cases = len(cases)
    settings = config["dot_settings"]
    opacity = settings.get("mesh_opacity", 0.3)
    mesh_color = settings.get("mesh_color", "lightgrey")
    offset_dist = settings.get("offset", 0.0)
    cam = config["camera"]

    fig = make_subplots(
        rows=1,
        cols=n_cases,
        specs=[[{"type": "scene"}] * n_cases],
        subplot_titles=[case["name"] for case in cases],
    )

    # 3. Load Tap Positions
    tap_source = config.get("tap_positions", config.get("taps"))
    pos_df = parse_taps_dataframe(tap_source, base_dir)

    # Apply offset to lift dots off the surface
    if offset_dist != 0:
        offsets = get_surface_offsets(
            vehicle_mesh, pos_df[["x", "y", "z"]].values, offset_dist
        )
        pos_df["x"] += offsets[:, 0]
        pos_df["y"] += offsets[:, 1]
        pos_df["z"] += offsets[:, 2]

    # 4. Create Traces for each subplot
    for idx, case in enumerate(cases):
        col = idx + 1
        case_df = pd.read_csv(base_dir / case["data_path"])
        df = pd.merge(pos_df, case_df, on="number")

        # Background mesh trace
        mesh_trace = go.Mesh3d(
            x=mx,
            y=my,
            z=mz,
            i=mi,
            j=mj,
            k=mk,
            color=mesh_color,
            opacity=opacity,
            name="Geometry",
            hoverinfo="skip",
            showlegend=(idx == 0),
        )
        fig.add_trace(mesh_trace, row=1, col=col)

        # Handle marker sizing
        if settings.get("scale_by_value", False):
            marker_size = df["cp"].abs() * settings.get("scale_factor", 10.0)
        else:
            marker_size = settings["size"]

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
                colorbar=dict(
                    title="Cp",
                    thickness=20,
                    x=1.02,
                    tickfont=dict(size=14),
                    titlefont=dict(size=18),
                )
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

        # Update scene settings
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

    # 5. Global Layout settings
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=60),
        title=dict(text="3D Pressure Distribution Comparison", font=dict(size=24)),
        font=dict(size=14),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=16),
            bgcolor="rgba(255,255,255,0.5)",
        ),
    )

    # 6. JavaScript Camera Sync
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

    output_path = base_dir / config["output_path"]
    html_content = fig.to_html(
        include_plotlyjs="cdn",
        post_script=sync_script,
        div_id=div_id,
        full_html=True,
    )
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Generated 3D plot with synced cameras: {output_path}")

    # Export PNG automatically
    png_path = output_path.with_suffix(".png")
    try:
        fig.write_image(png_path, width=1920, height=1080, scale=2)
        print(f"Saved: {png_path}")
    except Exception as e:
        print(f"PNG export failed (install kaleido: pip install kaleido): {e}")


def main():
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

    if Path(args.config).exists():
        generate_dot_plot(args.config)
    else:
        print(f"Error: Configuration file not found at {args.config}")


if __name__ == "__main__":
    main()
