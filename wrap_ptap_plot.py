import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image


def get_image_bbox(image_path):
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)
    is_not_white = ~(
        (data[:, :, 0] > 250) & (data[:, :, 1] > 250) & (data[:, :, 2] > 250)
    )
    rows, cols = np.any(is_not_white, axis=1), np.any(is_not_white, axis=0)
    if not np.any(rows):
        return 0, 0, img.width, img.height
    return (
        np.where(cols)[0][0],
        np.where(rows)[0][0],
        np.where(cols)[0][-1],
        np.where(rows)[0][-1],
    )


def render_plot(json_path):
    with open(json_path, "r") as f:
        config = json.load(f)
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
    breaks = [set(b) for b in config.get("line_breaks", [])]
    for i, cp in enumerate(taps["Cp"][:2]):
        name = series_names[i] if i < len(series_names) else f"Case {i + 1}"
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
                mode="lines+markers",
                name=name,
                line=dict(color=config["series_colors"][i], width=2),
                marker=dict(size=4),
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
    fig.update_xaxes(showgrid=False, zeroline=False).update_yaxes(
        showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
    )
    fig.write_html("drivAer_multi_series.html", include_plotlyjs="cdn")
    print("Saved: drivAer_multi_series.html")


if __name__ == "__main__":
    import sys

    render_plot(
        sys.argv[1]
        if len(sys.argv) > 1
        else "demo_data/drivAer/drivAer_top_complete.json"
    )
