import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image

try:
    from .generate_json_normals import complete_json
    from .utils import load_config_json, parse_taps_dataframe
except ImportError:
    from generate_json_normals import complete_json
    from utils import load_config_json, parse_taps_dataframe


def get_image_bbox(image_path):
    """
    Detects the bounding box of the non-white/non-transparent content in an image.
    This is used to map physical coordinates to the actual car boundaries.
    """
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)
    # Check for non-white and non-transparent pixels
    is_content = (
        (data[:, :, 0] < 250) | (data[:, :, 1] < 250) | (data[:, :, 2] < 250)
    ) & (data[:, :, 3] > 10)

    rows, cols = np.any(is_content, axis=1), np.any(is_content, axis=0)
    if not np.any(rows):
        return 0, 0, img.width, img.height
    return (
        np.where(cols)[0][0],
        np.where(rows)[0][0],
        np.where(cols)[0][-1],
        np.where(rows)[0][-1],
    )


def render_plot(json_path):
    """
    Main rendering function for 2D needle plots.
    """
    json_path = Path(json_path)
    config = load_config_json(json_path)
    base_dir = json_path.parent

    # Auto-generate normals if missing
    if "normals" not in config:
        print("Normals missing from config. Running preprocessor automatically...")
        config = complete_json(json_path)
        # Reload the augmented config
        json_path = json_path.with_name(f"{json_path.stem}_complete.json")
        config = load_config_json(json_path)

    img_path = base_dir / config["image_path"]
    img = Image.open(img_path)
    w, h = img.size
    pxmin, pymin, pxmax, pymax = get_image_bbox(img_path)
    pw, ph = pxmax - pxmin, pymax - pymin
    ext = config["extents"]
    fx, fy = pw / (ext["x_max"] - ext["x_min"]), ph / (ext["y_max"] - ext["y_min"])

    # Parse taps
    taps_input = config["taps"]
    df = parse_taps_dataframe(taps_input, base_dir)

    normals = config["normals"]
    df["n"] = df["number"].astype(int) if "number" in df else np.arange(1, len(df) + 1)

    # Map normals (dictionary keyed by tap number or list)
    if isinstance(normals, dict):
        df["nx"] = [normals[str(int(n))][0] for n in df["n"]]
        df["ny"] = [normals[str(int(n))][1] for n in df["n"]]
    else:
        df["nx"] = [n[0] for n in normals]
        df["ny"] = [n[1] for n in normals]

    # Extract Cp columns
    reserved = {"number", "x", "y", "z", "normals", "n", "nx", "ny"}
    cp_cols = [c for c in df.columns if c.lower() not in reserved]
    cp_data = [df[c].tolist() for c in cp_cols]

    # Map to pixel space
    df["xi"] = pxmin + (df["x"] - ext["x_min"]) * fx
    df["yi"] = (
        pymin + (1.0 - (df["y"] - ext["y_min"]) / (ext["y_max"] - ext["y_min"])) * ph
    )

    n_proj_x, n_proj_y = df["nx"] * fx, -df["ny"] * fy
    mag = np.sqrt(n_proj_x**2 + n_proj_y**2)
    n_proj_x, n_proj_y = n_proj_x / mag, n_proj_y / mag

    # Plotly Figure
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

    colors = config.get("series_colors", ["red", "blue", "green"])
    names = config.get("series_names")
    if not names:
        names = cp_cols
    prefs = config.get("series_preferences", [])

    for s_idx, cp_vals in enumerate(cp_data):
        pref = prefs[s_idx] if s_idx < len(prefs) else {}
        scale = config.get("cp_scale", 100.0)

        # Build vectorized segments
        # Points are [xi, yi] to [xi + nx*cp*scale, yi + ny*cp*scale]
        x_ends = df["xi"] + n_proj_x * np.array(cp_vals) * scale
        y_ends = df["yi"] + n_proj_y * np.array(cp_vals) * scale

        # For Plotly lines, we need [x1, x2, None, x3, x4, None...]
        px = np.empty(len(df) * 3)
        px[0::3], px[1::3], px[2::3] = df["xi"], x_ends, None
        py = np.empty(len(df) * 3)
        py[0::3], py[1::3], py[2::3] = df["yi"], y_ends, None

        fig.add_trace(
            go.Scatter(
                x=px,
                y=py,
                mode="lines" + ("+markers" if pref.get("show_markers") else ""),
                name=names[s_idx],
                line=dict(
                    color=pref.get("line_color", colors[s_idx % len(colors)]),
                    width=pref.get("line_width", 2),
                    dash=pref.get("line_dash", "solid"),
                ),
                marker=dict(symbol=pref.get("marker_symbol", "circle"), size=6),
                hoverinfo="skip",
            )
        )

    # Local Axes (Zero-line for Cp)
    fig.add_trace(
        go.Scatter(
            x=df["xi"],
            y=df["yi"],
            mode="lines",
            name="Surface",
            line=dict(color="black", width=1, dash="dot"),
            hoverinfo="skip",
        )
    )

    # Tooltips
    fig.add_trace(
        go.Scatter(
            x=df["xi"],
            y=df["yi"],
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)", size=10),
            name="Taps",
            text=[f"Tap {int(n)}" for n in df["n"]],
            customdata=df[cp_cols],
            hovertemplate="<b>%{text}</b><br>"
            + "<br>".join(
                [f"{c}: %{{customdata[{i}]:.3f}}" for i, c in enumerate(cp_cols)]
            )
            + "<extra></extra>",
        )
    )

    fig.update_xaxes(showgrid=False, zeroline=False, visible=False, range=[0, w])
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        visible=False,
        range=[0, h],
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        title=f"Pressure Distribution: {json_path.name}",
    )

    out_file = base_dir / config.get("output_path", "plot_output.html")
    fig.write_html(out_file, include_plotlyjs="cdn")
    print(f"Saved: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Render a 2D pressure tap needle plot."
    )
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="demo_data/drivAer_lineplot/drivAer_top_complete.json",
        help="Path to the source JSON configuration file.",
    )
    args = parser.parse_args()

    if Path(args.config).exists():
        render_plot(args.config)
    else:
        print(f"Error: Configuration file not found at {args.config}")


if __name__ == "__main__":
    main()
