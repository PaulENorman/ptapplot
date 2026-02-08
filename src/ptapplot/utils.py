import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def load_config_json(json_path):
    """Loads a JSON configuration file, stripping # style comments."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {json_path}")

    with open(path, "r") as f:
        content = f.read()
        # Remove # style comments
        content = re.sub(r"#.*", "", content)
        return json.loads(content)


def get_image_bbox(image_path):
    """
    Detects the bounding box of the non-white/non-transparent content in an image.
    Used to map physical coordinates to actual vehicle boundaries in the image.

    Returns (x_min, y_min, x_max, y_max) in pixel coordinates.
    """
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)
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


def parse_taps_dataframe(taps_input, base_dir):
    """
    Consistently parses tap data from various formats into a pandas DataFrame.
    Supports:
    - Path to a CSV file (relative to base_dir)
    - Raw CSV string
    - List of CSV-formatted strings
    - List of dictionaries
    """
    base_dir = Path(base_dir)

    if isinstance(taps_input, str):
        # Check if it's a path to a CSV
        potential_path = base_dir / taps_input
        if taps_input.endswith(".csv") and potential_path.exists():
            # Check for header
            header_sample = pd.read_csv(potential_path, nrows=1, header=None)
            try:
                # If all columns in first row are numeric, assume no header
                header_sample.iloc[0].astype(float)
                return pd.read_csv(
                    potential_path, header=None, names=["number", "x", "y", "z"]
                )
            except (ValueError, TypeError):
                return pd.read_csv(potential_path)
        else:
            # Assume it's a raw CSV string
            return pd.read_csv(io.StringIO(taps_input))

    elif isinstance(taps_input, list):
        if len(taps_input) > 0 and isinstance(taps_input[0], str):
            # List of CSV lines
            return pd.read_csv(io.StringIO("\n".join(taps_input)))
        else:
            # List of records (dicts)
            return pd.DataFrame(taps_input)

    raise ValueError("Invalid taps configuration format.")
