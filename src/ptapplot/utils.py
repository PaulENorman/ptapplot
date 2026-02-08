import io
import json
import re
from pathlib import Path

import pandas as pd


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
