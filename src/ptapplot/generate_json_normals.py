import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .utils import load_config_json, parse_taps_dataframe
except ImportError:
    from utils import load_config_json, parse_taps_dataframe


def complete_json(json_path, sort_dir=None):
    """
    Augments a pressure tap configuration JSON with geometric normals and
    flattens table data for the renderer.
    """
    json_path = Path(json_path)
    config = load_config_json(json_path)
    base_dir = json_path.parent

    # Support 'taps', 'points_source' (2D workflow) or 'tap_positions' (3D workflow)
    taps_source = config.get(
        "taps", config.get("points_source", config.get("tap_positions"))
    )

    if not taps_source:
        raise ValueError(
            f"No tap data found in {json_path}. Expected 'taps', 'points_source', or 'tap_positions'."
        )

    df = parse_taps_dataframe(taps_source, base_dir)
    if sort_dir and sort_dir in df.columns:
        print(f"Sorting taps by {sort_dir}...")
        df = df.sort_values(by=sort_dir).reset_index(drop=True)

    # Calculate geometric normals based on adjacent tap targets
    flip = config.get("normals_flip", False)
    ext = config.get("extents", {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1})
    center_x = (ext["x_min"] + ext["x_max"]) / 2.0
    center_y = (ext["y_min"] + ext["y_max"]) / 2.0

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
        # Rotate 90 degrees to get a candidate normal
        nx, ny = (dy, -dx) if not flip else (-dy, dx)

        # Heuristic: Ensure the normal points AWAY from the center of the bounding box
        # This ensures the 'Cp axis' extrudes outward from the car body.
        vec_from_center = [df.iloc[i]["x"] - center_x, df.iloc[i]["y"] - center_y]
        if (nx * vec_from_center[0] + ny * vec_from_center[1]) < 0:
            nx, ny = -nx, -ny

        norm = np.sqrt(nx**2 + ny**2)
        normals.append(
            [float(nx / norm), float(ny / norm), 0.0] if norm > 0 else [0.0, 1.0, 0.0]
        )

    # Relaxation Algorithm: Smooth and uncross normals to improve plot legibility
    if config.get("relax_normals", False):
        iters = config.get("relax_iterations", 10)
        alpha = config.get("relax_factor", 0.1)
        breaks = [set(b) for b in config.get("line_breaks", [])]
        fix_endpoints = config.get("relax_fix_endpoints", False)

        for _ in range(iters):
            new_normals = [n[:] for n in normals]
            for i in range(len(normals)):
                # If fix_endpoints is True, don't smooth the first or last tap
                if fix_endpoints and (i == 0 or i == len(normals) - 1):
                    continue

                if 0 < i < len(normals) - 1:
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
                        max_y = config.get("y_range", [0, 1.2])[1]
                        scale = config.get("cp_scale", 100)
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

    # Convert normals to a dictionary keyed by tap number
    normals_map = {
        str(int(df.iloc[idx]["number"])): normals[idx] for idx in range(len(df))
    }
    config["normals"] = normals_map

    # Update the 'taps' key with the sorted/processed data in its original format
    taps_input = config.get("taps", config.get("points_source"))
    if isinstance(taps_input, list):
        if len(taps_input) > 0 and isinstance(taps_input[0], str):
            config["taps"] = df.to_csv(index=False).strip().split("\n")
        else:
            config["taps"] = df.to_dict("records")
    else:
        config["taps"] = df.to_csv(index=False).strip()

    defaults = {
        "cp_scale": 100.0,
        "y_range": [0.0, 1.2],
        "num_ticks": 4,
        "series_colors": ["red", "blue", "green", "purple"],
        "series_names": cp_cols or ["Series 1"],
    }
    for k, v in defaults.items():
        if k not in config or (k == "series_names" and not config[k]):
            config[k] = v

    if "points_source" in config:
        del config["points_source"]

    out_path = json_path.with_name(f"{json_path.stem}_complete.json")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Generated: {out_path}")
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Augment pressure tap JSON with surface normals."
    )
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="demo_data/drivAer_lineplot/drivAer_top.json",
        help="Path to the source JSON configuration file.",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default=None,
        help="Coordinate column to sort taps by (default: None - preserves original order).",
    )
    args = parser.parse_args()

    if Path(args.config).exists():
        complete_json(args.config, sort_dir=args.sort)
    else:
        print(f"Error: Configuration file not found at {args.config}")


if __name__ == "__main__":
    main()
