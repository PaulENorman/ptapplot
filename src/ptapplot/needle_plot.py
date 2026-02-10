"""Needle Plot: 2D pressure tap visualization with individual Cp needles.

Unlike the Line Plot which connects Cp values with a continuous line, this plot
draws individual "needle" bars from the vehicle surface to each Cp value.
Needles extrude along surface normals with positive Cp pointing inward.
"""

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image

try:
    from .generate_json_normals import complete_json
    from .utils import get_image_bbox, load_config_json, parse_taps_dataframe
except ImportError:
    from generate_json_normals import complete_json
    from utils import get_image_bbox, load_config_json, parse_taps_dataframe


def render_needle_plot(json_path):
    """
    Renders a needle plot with individual Cp needles extruding along surface normals.
    Each tap gets its own needle from surface to Cp value (no connecting lines).
    """
    json_path = Path(json_path)
    config = load_config_json(json_path)
    base_dir = json_path.parent

    # Auto-generate normals if missing
    if "normals" not in config:
        print("Normals missing from config. Running preprocessor automatically...")
        complete_json(json_path)
        config = load_config_json(
            json_path.with_name(f"{json_path.stem}_complete.json")
        )

    img_path = base_dir / config["image_path"]
    img = Image.open(img_path)
    w, h = img.size
    pxmin, pymin, pxmax, pymax = get_image_bbox(img_path)
    pw, ph = pxmax - pxmin, pymax - pymin
    ext = config["extents"]
    fx, fy = pw / (ext["x_max"] - ext["x_min"]), ph / (ext["y_max"] - ext["y_min"])

    # Parse Taps
    taps_input = config.get("taps")
    df = parse_taps_dataframe(taps_input, base_dir)

    # Map geometric normals from config
    normals_map = config["normals"]
    df["nx"] = df["number"].apply(lambda n: normals_map[str(int(n))][0])
    df["ny"] = df["number"].apply(lambda n: normals_map[str(int(n))][1])

    # Map to pixel space
    df["xi"] = pxmin + (df["x"] - ext["x_min"]) * fx
    df["yi"] = (
        pymin + (1.0 - (df["y"] - ext["y_min"]) / (ext["y_max"] - ext["y_min"])) * ph
    )

    # Project normals into pixel space and normalize
    n_proj_x, n_proj_y = df["nx"] * fx, -df["ny"] * fy
    mag = np.sqrt(n_proj_x**2 + n_proj_y**2)
    df["nux"], df["nuy"] = n_proj_x / mag, n_proj_y / mag

    scale, ymin_ax, ymax_ax = (
        config["cp_scale"],
        config["y_range"][0],
        config["y_range"][1],
    )
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=img,
            xref="x",
            yref="y",
            x=0,
            y=h,
            sizex=w,
            sizey=h,
            sizing="stretch",
            opacity=1,
            layer="below",
        )
    )

    # --- Needle Axes: Lines along surface normals at each tap ---
    # Axis is centered at Cp=0 (surface), extending both directions
    # Positive Cp goes IN (opposite to normal), negative goes OUT (along normal)
    nx, ny, tx, ty = [], [], [], []
    tick_vals = np.linspace(ymin_ax, ymax_ax, config.get("num_ticks", 2))
    for i, r in df.iterrows():
        # Needle spine: from ymin_ax to ymax_ax, centered at surface (Cp=0)
        # Positive Cp = opposite normal direction (inward)
        # Negative Cp = along normal direction (outward)
        nx.extend(
            [
                r["xi"]
                - ymin_ax * scale * r["nux"],  # ymin_ax is negative, goes outward
                r["xi"]
                - ymax_ax * scale * r["nux"],  # ymax_ax is positive, goes inward
                None,
            ]
        )
        ny.extend(
            [
                h - (r["yi"] - ymin_ax * scale * r["nuy"]),
                h - (r["yi"] - ymax_ax * scale * r["nuy"]),
                None,
            ]
        )

        # Tick marks: placed at -v offset from tap (negative to invert)
        for v in tick_vals:
            px, py = r["nuy"], r["nux"]  # Perpendicular to needle
            tick_len = 5
            offset = -v * scale
            cx, cy = (
                r["xi"] + offset * r["nux"],
                h - (r["yi"] + offset * r["nuy"]),
            )
            tx.extend([cx - px * tick_len, cx + px * tick_len, None])
            ty.extend([cy - py * tick_len, cy + py * tick_len, None])

            # Axis labels (every 5th needle for clarity)
            if i % 5 == 0 or i == len(df) - 1:
                fig.add_annotation(
                    x=r["xi"] - v * scale * r["nux"],
                    y=h - (r["yi"] - v * scale * r["nuy"]),
                    text=f"{v:.1f}",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    xanchor="right" if r["nux"] < 0 else "left",
                )

    fig.add_trace(
        go.Scatter(
            x=nx,
            y=ny,
            mode="lines",
            line=dict(color="lightgrey", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=tx,
            y=ty,
            mode="lines",
            line=dict(color="grey", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # No gridlines for needle plot (unlike line plot)

    # Base surface line (Cp=0)
    fig.add_trace(
        go.Scatter(
            x=df["xi"],
            y=h - df["yi"],
            mode="lines",
            line=dict(color="lightgrey", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # --- Multi-Series Plotting ---
    reserved_cols = {"number", "x", "y", "z", "nx", "ny", "xi", "yi", "nux", "nuy"}
    cp_cols = [c for c in df.columns if c not in reserved_cols]
    series_names = config.get("series_names", cp_cols)
    series_prefs = config.get("series_preferences", [])
    series_colors = config.get(
        "series_colors", ["red", "blue", "green", "purple", "orange"]
    )
    n_series = len(cp_cols)
    series_offset = config.get(
        "series_offset", 8
    )  # Perpendicular offset between series

    for i, col_name in enumerate(cp_cols):
        name = series_names[i] if i < len(series_names) else col_name
        cp_vals = df[col_name].values

        pref = {
            "line_color": series_colors[i % len(series_colors)],
            "line_width": 2,
            "line_dash": "solid",
            "show_markers": True,
            "marker_size": 4,
            "marker_symbol": "circle",
        }
        if i < len(series_prefs):
            pref.update(series_prefs[i])

        # Each needle is a line from surface (Cp=0) to the Cp value along the normal
        # Positive Cp = opposite normal (inward), Negative Cp = along normal (outward)
        # Apply perpendicular offset so series don't overlap
        x_needles, y_needles, cp_hover, tap_hover = [], [], [], []
        for v_idx in range(len(df)):
            r = df.iloc[v_idx]

            # Perpendicular offset to separate series
            # Perp vector: (nuy, nux) in plot space
            perp_x, perp_y = r["nuy"], r["nux"]
            # Center the series around the tap, offset by series index
            offset_amount = (i - (n_series - 1) / 2.0) * series_offset
            base_x = r["xi"] + offset_amount * perp_x
            base_y = h - r["yi"] - offset_amount * perp_y

            # Cp offset along normal direction (negative = inward for positive Cp)
            cp_offset = -cp_vals[v_idx] * scale
            xi_cp = base_x + cp_offset * r["nux"]
            yi_cp = base_y - cp_offset * r["nuy"]

            # Draw line from surface to Cp value along normal
            x_needles.extend([base_x, xi_cp, None])
            y_needles.extend([base_y, yi_cp, None])
            cp_hover.extend([cp_vals[v_idx], cp_vals[v_idx], None])
            tap_hover.extend([r["number"], r["number"], None])

        fig.add_trace(
            go.Scatter(
                x=x_needles,
                y=y_needles,
                mode="lines",
                name=name,
                line=dict(
                    color=pref["line_color"],
                    width=pref["line_width"],
                    dash=pref["line_dash"],
                ),
                text=cp_hover,
                customdata=tap_hover,
                hovertemplate=(
                    f"<b>{name}</b><br>Tap: %{{customdata}}<br>"
                    "Cp: %{text:.3f}<extra></extra>"
                ),
            )
        )

    # Final Dots at Surface
    fig.add_trace(
        go.Scatter(
            x=df["xi"],
            y=h - df["yi"],
            mode="markers",
            marker=dict(size=4, color="black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Auto-Range and Update Layout
    all_px = nx + tx + [pxmin, pxmax]
    all_py = ny + ty + [h - pymin, h - pymax]
    v_px = [p for p in all_px if p is not None]
    v_py = [p for p in all_py if p is not None]
    pad = 50
    fig.update_xaxes(
        range=[min(v_px) - pad, max(v_px) + pad],
        showgrid=False,
        zeroline=False,
        visible=False,
    )
    fig.update_yaxes(
        range=[min(v_py) - pad, max(v_py) + pad],
        showgrid=False,
        zeroline=False,
        visible=False,
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=60),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=16),
            bgcolor="rgba(255,255,255,0.8)",
        ),
        title=dict(text=f"Needle Plot: {json_path.name}", font=dict(size=24)),
        font=dict(size=14),  # Global font size
    )

    out_file = base_dir / config.get("output_path", "needleplot_output.html")
    fig.write_html(out_file, include_plotlyjs="cdn")
    print(f"Saved: {out_file}")

    # Export PNG automatically
    png_file = out_file.with_suffix(".png")
    try:
        fig.write_image(png_file, width=1920, height=1080, scale=2)
        print(f"Saved: {png_file}")
    except Exception as e:
        print(f"PNG export failed (install kaleido: pip install kaleido): {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Render a 2D needle plot with Cp bars along normals."
    )
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="demo_data/drivAer_needleplot/needleplot_config.json",
        help="Path to the source JSON configuration file.",
    )
    args = parser.parse_args()

    if Path(args.config).exists():
        render_needle_plot(args.config)
    else:
        print(f"Error: Configuration file not found at {args.config}")


if __name__ == "__main__":
    main()
