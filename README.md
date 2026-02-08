# ptapplot
A python library for making pressure tap plots

*Click the images to view interactive Plotly visualizations*

| 2D Line Plot | 2D Needle Plot | 3D Dot Plot |
| :---: | :---: | :---: |
| [![DrivAer 2D](demo_data/drivAer_lineplot/drivAer_multi_series.png)](https://raw.githack.com/PaulENorman/ptapplot/main/demo_data/drivAer_lineplot/drivAer_multi_series.html) | [![DrivAer Needle](demo_data/drivAer_needleplot/drivAer_needleplot.png)](https://raw.githack.com/PaulENorman/ptapplot/main/demo_data/drivAer_needleplot/drivAer_needleplot.html) | [![DrivAer 3D](demo_data/drivAer_dotplot/drivAer_dotplot.png)](https://raw.githack.com/PaulENorman/ptapplot/main/demo_data/drivAer_dotplot/drivAer_dotplot.html) |

## Installation

```bash
# Clone the repository
git clone https://github.com/PaulENorman/ptapplot.git
cd ptapplot

# Install in editable mode
pip install -e .
```

## Usage Flow

### 2D Line Plots
1. **Source Configuration**: Define your vehicle image, physical extents, and CSV tap data in a `.json` file.
2. **Preprocessing**: (Optional) Calculate surface normals.
3. **Rendering**: Generate the interactive HTML Plotly visualization.

```bash
# 1. Prepare (Optional if running plot directly)
ptap-2d-prep demo_data/drivAer_lineplot/drivAer_top.json

# 2. Render plot
ptap-2d-plot demo_data/drivAer_lineplot/drivAer_top_complete.json
```

*Note: `ptap-2d-plot` will automatically trigger the prep step if normals are missing from the configuration.*

### 3D Dot Plots (Geometry Overlay)
1. **Source Configuration**: Define paths to your STL geometry, tap positions, and results CSVs in a `.json` file.
2. **Rendering**: Generate a 3D Plotly scene with side-by-side comparison and synced cameras.

```bash
ptap-3d-plot demo_data/drivAer_dotplot/dotplot_config.json
```

### 2D Needle Plots
1. **Source Configuration**: Same as Line Plots (normals auto-generated if missing).
2. **Rendering**: Generates a plot with individual $C_p$ bars at each tap location.

```bash
ptap-needle-plot demo_data/drivAer_needleplot/needleplot_config.json
```

*Needle plots show individual bars instead of a connected line. Positive $C_p$ (pressure) points inward toward the body, negative $C_p$ (suction) points outward. Multiple series are offset perpendicular to the normal for clarity.*

## Advanced Features

### JSON Configuration Options
Both workflows support Python-style `#` comments in their `.json` configuration files.

#### Common & 2D Keys:
- `image_path`: Path to the background PNG/JPG.
- `taps`: Array of CSV strings, path to `.csv`, or list of records.
- `extents`: Physical bounding box of the vehicle (`x_min`, `x_max`, etc.).
- `line_breaks`: List of tap-number pairs `[ID1, ID2]` to disconnect the plot line.
- `normals_flip`: (bool) Reverse the direction of calculated surface normals.
- `cp_scale`: Scaling factor for the local $C_p$ axes.
- `series_preferences`: Array of style objects (line color, width, etc.).

#### 2D Normal Relaxation:
Enable these in your JSON for complex geometry:
- `relax_normals`: (bool) Enable the smoothing pipeline.
- `relax_iterations`: (int) Number of smoothing passes.
- `relax_factor`: (float) Smoothing blend strength (0.0 to 1.0).
- `relax_uncross`: (bool) Prevent local axes from crossing.
- `relax_fix_endpoints`: (bool) Keep endpoints fixed to true geometry.

#### 3D Dot Plot Keys:
- `stl_path`: Path to the vehicle surface mesh.
- `mesh_opacity`: Transparency of the vehicle geometry (0.0 to 1.0).
- `scale_by_value`: (bool) Size dots based on $C_p$ magnitude.
- `scale_factor`: Multiplier for the magnitude scaling.
- `offset`: (float) Lift dots off the STL surface to prevent clipping.
- `camera`: Initial eye, center, and up vectors for the 3D scene.

## Developer Features
- **Auto-Formatting**: The project is configured for **Ruff** (formatting and import sorting) with VS Code settings provided in `.vscode/`.
