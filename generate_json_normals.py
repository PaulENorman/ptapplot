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
        # Remove comments before parsing JSON
        content = f.read()
        content = re.sub(r"//.*", "", content)
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
        nx, ny = dy, -dx
        norm = np.sqrt(nx**2 + ny**2)
        normals.append(
            [float(nx / norm), float(ny / norm), 0.0] if norm > 0 else [0.0, 1.0, 0.0]
        )

    reserved = {"number", "x", "y", "z", "normals"}
    cp_cols = [c for c in df.columns if c.lower() not in reserved]

    config["taps"] = {
        "number": df["number"].astype(int).tolist()
        if "number" in df
        else list(range(1, len(df) + 1)),
        "x": df["x"].tolist(),
        "y": df["y"].tolist(),
        "z": df["z"].tolist() if "z" in df else [0.0] * len(df),
        "normals": normals,
        "Cp": [df[c].tolist() for c in cp_cols] if cp_cols else [[0.0] * len(df)],
    }

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
