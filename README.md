# ptapplot
A python library for making pressure tap plots

[![DrivAer Pressure Distribution](demo_data/drivAer/drivAer_multi_series.png)](https://raw.githack.com/PaulENorman/ptapplot/main/demo_data/drivAer/drivAer_multi_series.html)

*Click the image above to view the interactive Plotly plot.*

## Usage Flow
1. **Source Configuration**: Define your vehicle image, physical extents, and CSV tap data in a `.json` file.
2. **Preprocessing**: Run `generate_json_normals.py` to calculate surface normals and prepare the data.
3. **Rendering**: Run `wrap_ptap_plot.py` to generate the interactive HTML Plotly visualization.

```mermaid
graph LR
    A[drivAer_top.json] --> B(generate_json_normals.py)
    B --> C[drivAer_top_complete.json]
    C --> D(wrap_ptap_plot.py)
    D --> E[drivAer_multi_series.html]
```

### Installation
```bash
pip install -r requirements.txt
```

### Quick Start
```bash
# Calculate normals
python generate_json_normals.py demo_data/drivAer/drivAer_top.json

# Render plot
python wrap_ptap_plot.py demo_data/drivAer/drivAer_top_complete.json
```
