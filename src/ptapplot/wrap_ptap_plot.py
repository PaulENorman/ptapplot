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


def render_plot(json_path):
    """
    Main rendering function. Preserves the 'Original Look' with needle ticks and labels
    while integrating with the modernized project structure.
    """
    json_path = Path(json_path)
    config = load_config_json(json_path)
    base_dir = json_path.parent

    # Auto-generate normals if missing
    if "normals" not in config:
        print("Normals missing from config. Running preprocessor automatically...")
        complete_json(json_path)
        # Reload to ensure we have the absolute state
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

    # Map geometric normals and CP data
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

    # --- Local Cp Axes: Lines along surface normals at each tap ---
    # The axis starts at the tap location (representing ymin_ax) and extends
    # OUTWARD ONLY (along the normal) to represent ymax_ax.
    # The range of the axis is (ymax_ax - ymin_ax).
    nx, ny, tx, ty = [], [], [], []
    axis_range = ymax_ax - ymin_ax
    tick_vals = np.linspace(ymin_ax, ymax_ax, config.get("num_ticks", 2))
    for i, r in df.iterrows():
        # Needle spine: starts at tap (offset=0), extends to axis_range * scale
        nx.extend(
            [
                r["xi"],
                r["xi"] + axis_range * scale * r["nux"],
                None,
            ]
        )
        ny.extend(
            [
                h - r["yi"],
                h - (r["yi"] + axis_range * scale * r["nuy"]),
                None,
            ]
        )

        # Tick marks: placed at (v - ymin_ax) offset from tap
        for v in tick_vals:
            px, py = r["nuy"], r["nux"]
            tick_len = 5
            offset = (v - ymin_ax) * scale
            cx, cy = (
                r["xi"] + offset * r["nux"],
                h - (r["yi"] + offset * r["nuy"]),
            )
            tx.extend([cx - px * tick_len, cx + px * tick_len, None])
            ty.extend([cy - py * tick_len, cy + py * tick_len, None])

            # Axis labels (every 5th needle for clarity)
            if i % 5 == 0 or i == len(df) - 1:
                # Calculate rotation angle (perpendicular to axis, +90 degrees)
                angle = np.degrees(np.arctan2(-r["nuy"], r["nux"])) + 90

                # Normalize angle to [-180, 180]
                while angle > 180:
                    angle -= 360
                while angle <= -180:
                    angle += 360

                # Keep text upright
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180

                # Perpendicular offset (shift text away from spine)
                # Perp vector in Plotly space is (nuy, nux)
                shift = 10
                spine_x = r["xi"] + offset * r["nux"]
                spine_y = h - (r["yi"] + offset * r["nuy"])

                fig.add_annotation(
                    x=spine_x + r["nuy"] * shift,
                    y=spine_y + r["nux"] * shift,
                    text=f"{v:.1f}",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    textangle=angle,
                    xanchor="center",
                    yanchor="middle",
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

    # --- Horizontal Gridlines: Connect ticks across needles ---
    # Determine the "central" value (midpoint of the axis range, like x=0 on a normal plot)
    # Respect line_breaks defined in the JSON
    breaks = [set(b) for b in config.get("line_breaks", [])]
    central_val = (ymin_ax + ymax_ax) / 2.0
    for v in tick_vals:
        gx, gy = [], []
        for idx in range(len(df)):
            r = df.iloc[idx]
            offset = (v - ymin_ax) * scale
            gx.append(r["xi"] + offset * r["nux"])
            gy.append(h - (r["yi"] + offset * r["nuy"]))

            # Insert None to break the line at line_breaks
            if idx < len(df) - 1:
                t1, t2 = int(df.iloc[idx]["number"]), int(df.iloc[idx + 1]["number"])
                if {t1, t2} in breaks:
                    gx.append(None)
                    gy.append(None)

        # Darker grey for central value, light grey dotted for others
        if abs(v - central_val) < 1e-6:
            line_style = dict(color="darkgrey", width=1, dash="dot")
        else:
            line_style = dict(color="lightgrey", width=0.5, dash="dot")

        fig.add_trace(
            go.Scatter(
                x=gx,
                y=gy,
                mode="lines",
                line=line_style,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Base surface line (CP=0)
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
    breaks = [set(b) for b in config.get("line_breaks", [])]

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

        # Displacement calculation:
        # The axis starts at the tap (ymin_ax) and extends outward to ymax_ax.
        # A Cp value of ymin_ax plots at the tap location (offset=0).
        # A Cp value of ymax_ax plots at the end of the axis (offset=axis_range*scale).
        # We map Cp to offset: offset = (cp - ymin_ax) * scale
        cp_offsets = (cp_vals - ymin_ax) * scale
        xp = df["xi"] + cp_offsets * df["nux"]
        yp = h - (df["yi"] + cp_offsets * df["nuy"])

        # Handle line breaks for multi-segment cars
        x_plot, y_plot, cp_plot, tap_plot = [], [], [], []
        for v_idx in range(len(df)):
            x_plot.append(xp.iloc[v_idx])
            y_plot.append(yp.iloc[v_idx])
            cp_plot.append(cp_vals[v_idx])
            tap_plot.append(df["number"].iloc[v_idx])

            if v_idx < len(df) - 1:
                t1, t2 = (
                    int(df["number"].iloc[v_idx]),
                    int(df["number"].iloc[v_idx + 1]),
                )
                if {t1, t2} in breaks:
                    for arr in [x_plot, y_plot, cp_plot, tap_plot]:
                        arr.append(None)

        fig.add_trace(
            go.Scatter(
                x=x_plot,
                y=y_plot,
                mode="lines+markers" if pref["show_markers"] else "lines",
                name=name,
                line=dict(
                    color=pref["line_color"],
                    width=pref["line_width"],
                    dash=pref["line_dash"],
                ),
                marker=dict(size=pref["marker_size"], symbol=pref["marker_symbol"]),
                text=cp_plot,
                customdata=tap_plot,
                hovertemplate=f"<b>{name}</b><br>Tap: %{{customdata}}<br>Cp: %{{text:.3f}}<extra></extra>",
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
        title=dict(text=f"Line Plot: {json_path.name}", font=dict(size=24)),
        font=dict(size=14),  # Global font size
    )

    out_file = base_dir / config.get("output_path", "plot_output.html")
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
        description="Render a 2D pressure tap line plot with local Cp axes following surface normals."
    )
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="demo_data/drivAer_lineplot/drivAer_top.json",
        help="Path to the source JSON configuration file.",
    )
    args = parser.parse_args()

    if Path(args.config).exists():
        render_plot(args.config)
    else:
        print(f"Error: Configuration file not found at {args.config}")


if __name__ == "__main__":
    main()
