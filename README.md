# ptapplot
A python library for making pressure tap plots

*Click the image to view an interactive Plotly*

[![DrivAer Pressure Distribution](demo_data/drivAer/drivAer_multi_series.png)](https://raw.githack.com/PaulENorman/ptapplot/main/demo_data/drivAer/drivAer_multi_series.html)


## Usage Flow
1. **Source Configuration**: Define your vehicle image, physical extents, and CSV tap data in a `.json` file.
2. **Preprocessing**: Run `generate_json_normals.py` to calculate surface normals and prepare the data.
3. **Rendering**: Run `wrap_ptap_plot.py` to generate the interactive HTML Plotly visualization.

```mermaid
graph LR
    A[config.json] --> B(generate_json_normals.py)
    B --> C[config_complete.json]
    C --> D(wrap_ptap_plot.py)
    D --> E[plot_output.html]
```

### Installation
```bash
pip install -r requirements.txt
```

### Quick Start
```bash
# 1. Prepare configuration (Calculate surface normals)
python generate_json_normals.py demo_data/drivAer/drivAer_top.json

# 2. Render plot
python wrap_ptap_plot.py demo_data/drivAer/drivAer_top_complete.json
```

## Advanced Features

### JSON Configuration Options
The source `.json` supports Python-style `#` comments and the following keys:

- `image_path`: Path to the background PNG/JPG.
- `taps`: Array of CSV-formatted strings (header: `number,x,y,z,Case1,Case2...`).
- `extents`: Physical bounding box of the vehicle (`x_min`, `x_max`, etc.).
- `line_breaks`: List of tap-number pairs `[ID1, ID2]` to disconnect the plot line between.
- `normals_flip`: (bool) Reverses the direction of all calculated surface normals (useful if plots are appearing "inside" the vehicle).
- `cp_scale`: Scaling factor for the local $C_p$ axes (pixels per unit $C_p$).
- `series_preferences`: Array of style objects (keys: `line_color`, `line_width`, `line_dash`, `show_markers`, `marker_symbol`).

### Normal Relaxation
To improve plot legibility in regions with complex geometry, enable relaxation in the JSON:

- `relax_normals`: (bool) Enables the relaxation pipeline.
- `relax_iterations`: (int) Number of smoothing passes (e.g., 50).
- `relax_factor`: (float) Strength of the smoothing blend (0.0 to 1.0).
- `relax_uncross`: (bool) Prevents local axes from converging/crossing within the plot range.
- `relax_fix_endpoints`: (bool) Keeps segment start/end normals fixed to their true geometric orientation.

## Developer Features
- **Auto-Formatting**: The project is configured for **Ruff** (formatting and import sorting) with VS Code settings provided in `.vscode/`.
