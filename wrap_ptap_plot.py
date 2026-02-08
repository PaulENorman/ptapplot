import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image


def get_image_bbox(image_path):
    """
    Detects the bounding box of the non-white/non-transparent content in an image.
    This is used to map physical coordinates to the actual car boundaries
    rather than the full image canvas.
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
    Main rendering function. Reads the augmented JSON configuration,
    maps physical coordinates to image pixel space, and generates
    an interactive Plotly figure with local pressure axes (needles).
    """
    import re

    with open(json_path, "r") as f:
        # Remove comments before parsing JSON (supports # style comments)
        content = f.read()
        content = re.sub(r"#.*", "", content)
        config = json.loads(content)
    img = Image.open(os.path.join(os.path.dirname(json_path), config["image_path"]))
    w, h = img.size
    pxmin, pymin, pxmax, pymax = get_image_bbox(
        os.path.join(os.path.dirname(json_path), config["image_path"])
    )
    pw, ph = pxmax - pxmin, pymax - pymin
    ext = config["extents"]
    fx, fy = pw / (ext["x_max"] - ext["x_min"]), ph / (ext["y_max"] - ext["y_min"])

    taps = config["taps"]
    df = pd.DataFrame(
        {
            "n": taps["number"],
            "x": taps["x"],
            "y": taps["y"],
            "nx": [n[0] for n in taps["normals"]],
            "ny": [n[1] for n in taps["normals"]],
        }
    )
    df["xi"] = pxmin + (df["x"] - ext["x_min"]) * fx
    df["yi"] = (
        pymin + (1.0 - (df["y"] - ext["y_min"]) / (ext["y_max"] - ext["y_min"])) * ph
    )

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

    # Needs, Ticks, Cp=0
    nx, ny, tx, ty, tt = [], [], [], [], []
    tick_vals = np.linspace(ymin_ax, ymax_ax, config.get("num_ticks", 2))
    for i, r in df.iterrows():
        nx.extend(
            [
                r["xi"] + ymin_ax * scale * r["nux"],
                r["xi"] + ymax_ax * scale * r["nux"],
                None,
            ]
        )
        ny.extend(
            [
                h - (r["yi"] + ymin_ax * scale * r["nuy"]),
                h - (r["yi"] + ymax_ax * scale * r["nuy"]),
                None,
            ]
        )
        # Add tick lines and labels
        for v in tick_vals:
            # Perpendicular vector in (x, h-y) space
            # Needle vector is (nux, -nuy), so perp is (nuy, nux)
            px, py = r["nuy"], r["nux"]
            tick_len = 5
            cx, cy = (
                r["xi"] + v * scale * r["nux"],
                h - (r["yi"] + v * scale * r["nuy"]),
            )
            tx.extend([cx - px * tick_len, cx + px * tick_len, None])
            ty.extend([cy - py * tick_len, cy + py * tick_len, None])
            if i % 5 == 0 or i == len(df) - 1:  # Label every 5th needle
                fig.add_annotation(
                    x=r["xi"] + v * scale * r["nux"],
                    y=h - (r["yi"] + v * scale * r["nuy"]),
                    text=f"{v:.1f}",
                    showarrow=False,
                    font=dict(size=8, color="grey"),
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

    series_names = config.get("series_names", [])
    series_prefs = config.get("series_preferences", [])
    breaks = [set(b) for b in config.get("line_breaks", [])]
    for i, cp in enumerate(taps["Cp"]):
        name = series_names[i] if i < len(series_names) else f"Case {i + 1}"

        # Default preferences
        pref = {
            "line_color": config["series_colors"][i % len(config["series_colors"])],
            "line_width": 2,
            "line_dash": "solid",
            "show_markers": True,
            "marker_size": 4,
            "marker_symbol": "circle",
        }
        # Override with user preferences if available
        if i < len(series_prefs):
            pref.update(series_prefs[i])

        xp, yp = (
            df["xi"] + np.array(cp) * scale * df["nux"],
            h - (df["yi"] + np.array(cp) * scale * df["nuy"]),
        )
        x_plot, y_plot, cp_plot, tap_plot = [], [], [], []
        for j in range(len(df)):
            x_plot.append(xp[j])
            y_plot.append(yp[j])
            cp_plot.append(cp[j])
            tap_plot.append(df["n"].iloc[j])
            if (
                j < len(df) - 1
                and {int(df["n"].iloc[j]), int(df["n"].iloc[j + 1])} in breaks
            ):
                for a in [x_plot, y_plot, cp_plot, tap_plot]:
                    a.append(None)

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

    fig.add_trace(
        go.Scatter(
            x=df["xi"],
            y=h - df["yi"],
            mode="markers",
            marker=dict(size=6, color="black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Determine plot bounds to prevent clipping and remove excess white space
    # We want to see the car (pxmin, pymin to pxmax, pymax) and the needles
    all_px = nx + tx + [pxmin, pxmax]
    all_py = ny + ty + [h - pymin, h - pymax]
    valid_px = [p for p in all_px if p is not None]
    valid_py = [p for p in all_py if p is not None]

    x_min, x_max = min(valid_px), max(valid_px)
    y_min, y_max = min(valid_py), max(valid_py)
    pad = 50

    fig.update_xaxes(range=[x_min - pad, x_max + pad], showgrid=False, zeroline=False)
    fig.update_yaxes(
        range=[y_min - pad, y_max + pad],
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
    )
    out_html = os.path.join(os.path.dirname(json_path), "drivAer_multi_series.html")
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"Saved: {out_html}")


if __name__ == "__main__":
    import sys

    render_plot(
        sys.argv[1]
        if len(sys.argv) > 1
        else "demo_data/drivAer/drivAer_top_complete.json"
    )
