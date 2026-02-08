import io
import json
import os
import re

import numpy as np
import pandas as pd


def complete_json(json_path, sort_dir="x"):
    """
    Augments a pressure tap configuration JSON with geometric normals and
    flattens table data for the renderer.

    Args:
        json_path (str): Path to the source JSON file containing metadata and tap data.
        sort_dir (str): Dataframe column name to sort taps by (default 'x').
    """
    with open(json_path, "r") as f:
        # Remove comments before parsing JSON (supports # style comments)
        content = f.read()
        content = re.sub(r"#.*", "", content)
        config = json.loads(content)

    base_dir = os.path.dirname(json_path)
    taps_input = config.get("taps", config.get("points_source"))

    if isinstance(taps_input, str):
        if taps_input.endswith(".csv") and os.path.exists(
            os.path.join(base_dir, taps_input)
        ):
            df = pd.read_csv(os.path.join(base_dir, taps_input))
        else:
            df = pd.read_csv(io.StringIO(taps_input))
    elif isinstance(taps_input, list):
        if len(taps_input) > 0 and isinstance(taps_input[0], str):
            df = pd.read_csv(io.StringIO("\n".join(taps_input)))
        else:
            df = pd.DataFrame(taps_input)
    else:
        raise ValueError("Invalid taps configuration.")

    df = df.sort_values(by=sort_dir).reset_index(drop=True)

    # Calculate geometric normals based on adjacent tap targets
    flip = config.get("normals_flip", False)
    normals = []
    for i in range(len(df)):
        # Select neighbors based on position in the sequence
        p1, p2 = (
            (df.iloc[i], df.iloc[i + 1])
            if i == 0
            else (df.iloc[i - 1], df.iloc[i])
            if i == len(df) - 1
            else (df.iloc[i - 1], df.iloc[i + 1])
        )
        # Vector along the surface (dx, dy)
        dx, dy = p2["x"] - p1["x"], p2["y"] - p1["y"]
        # Rotate 90 degrees outward to get the normal
        nx, ny = (dy, -dx) if not flip else (-dy, dx)
        norm = np.sqrt(nx**2 + ny**2)
        normals.append(
            [float(nx / norm), float(ny / norm), 0.0] if norm > 0 else [0.0, 1.0, 0.0]
        )

    # Relaxation Algorithm: Smooth and uncross normals to improve plot legibility
    if config.get("relax_normals", False):
        iters = config.get("relax_iterations", 10)
        alpha = config.get("relax_factor", 0.1)
        breaks = [set(b) for b in config.get("line_breaks", [])]

        for _ in range(iters):
            new_normals = [n[:] for n in normals]
            for i in range(1, len(normals) - 1):
                # Don't relax across line breaks
                n_id = df.iloc[i]["number"]
                n_prev_id = df.iloc[i - 1]["number"]
                n_next_id = df.iloc[i + 1]["number"]

                if {int(n_id), int(n_prev_id)} in breaks or {
                    int(n_id),
                    int(n_next_id),
                } in breaks:
                    continue

                # Smooth with neighbors
                avg_nx = (normals[i - 1][0] + normals[i + 1][0]) / 2.0
                avg_ny = (normals[i - 1][1] + normals[i + 1][1]) / 2.0

                new_normals[i][0] = normals[i][0] * (1 - alpha) + avg_nx * alpha
                new_normals[i][1] = normals[i][1] * (1 - alpha) + avg_ny * alpha

                # Explicit Uncrossing: If adjacent normals are converging too sharply, make them more parallel
                if config.get("relax_uncross", False):
                    # Position at max Cp range
                    max_y = config.get("y_range", [0, 1.2])[1]
                    scale = config.get("cp_scale", 100)
                    # Check convergence with previous
                    p_curr = np.array(
                        [
                            df.iloc[i]["x"] + new_normals[i][0] * max_y * scale,
                            df.iloc[i]["y"] + new_normals[i][1] * max_y * scale,
                        ]
                    )
                    p_prev = np.array(
                        [
                            df.iloc[i - 1]["x"] + normals[i - 1][0] * max_y * scale,
                            df.iloc[i - 1]["y"] + normals[i - 1][1] * max_y * scale,
                        ]
                    )
                    d_base = np.sqrt(
                        (df.iloc[i]["x"] - df.iloc[i - 1]["x"]) ** 2
                        + (df.iloc[i]["y"] - df.iloc[i - 1]["y"]) ** 2
                    )
                    d_tip = np.sqrt(np.sum((p_curr - p_prev) ** 2))
                    if d_tip < d_base * 0.5:
                        new_normals[i][0] = (
                            new_normals[i][0] + normals[i - 1][0]
                        ) / 2.0
                        new_normals[i][1] = (
                            new_normals[i][1] + normals[i - 1][1]
                        ) / 2.0

                # Re-normalize
                mag = np.sqrt(new_normals[i][0] ** 2 + new_normals[i][1] ** 2)
                if mag > 0:
                    new_normals[i][0] /= mag
                    new_normals[i][1] /= mag
            normals = new_normals

    reserved = {"number", "x", "y", "z", "normals"}
    cp_cols = [c for c in df.columns if c.lower() not in reserved]

    # Convert normals list to a dictionary keyed by tap number
    normals_map = {}
    for idx, row in df.iterrows():
        normals_map[str(int(row["number"]))] = normals[idx]

    config["normals"] = normals_map

    # If the input was a list of strings (CSV), let's keep it that way
    if (
        isinstance(taps_input, list)
        and len(taps_input) > 0
        and isinstance(taps_input[0], str)
    ):
        config["taps"] = df.to_csv(index=False).strip().split("\n")
    elif isinstance(taps_input, list):
        config["taps"] = df.to_dict("records")
    elif isinstance(taps_input, str):
        config["taps"] = df.to_csv(index=False).strip()

    defaults = {
        "cp_scale": 100.0,
        "y_range": [0.0, 1.2],
        "num_ticks": 4,
        "series_colors": ["red", "blue", "green", "purple"],
        "series_names": cp_cols or ["Series 1"],
    }
    for k, v in defaults.items():
        config[k] = config.get(k, v)

    if "points_source" in config:
        del config["points_source"]

    out = json_path.replace(".json", "_complete.json")
    with open(out, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Generated: {out}")


if __name__ == "__main__":
    import sys

    complete_json(
        sys.argv[1] if len(sys.argv) > 1 else "demo_data/drivAer/drivAer_top.json"
    )
