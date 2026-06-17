# ICON particle-advection tutorial

Advecting Lagrangian particles in unstructured **ICON** ocean output with
[Parcels](https://github.com/OceanParcels/Parcels) v4, using the cell-view linear
velocity reconstruction from the companion
[FESOM notebook](../fesom_reconstruction/fesom_velocity_reconstruction.ipynb).

ICON stores horizontal velocities at triangle centres (face-registered data).
Parcels' default face interpolator is piecewise constant; `tutorial_icon.ipynb`
replaces it with `LinearFaceRecon`, which fits a linear velocity field inside
each triangle by least squares to its edge neighbours, then advects particles
through a real `levy01` (r2b8) eddy-simulation snapshot.

## Contents

| File | Description |
| --- | --- |
| `tutorial_icon.ipynb` | The notebook: open → convert to UGRID → build `FieldSet` → attach reconstruction → advect → plot. |
| `icon_levy_grid.nc` | ICON grid description, trimmed to the variables uxarray's ICON reader needs (cell/edge/vertex coordinates and cell connectivities). |
| `pixi.toml` / `pixi.lock` | Reproducible environment (Parcels v4 from `main`, uxarray, the scientific stack, JupyterLab). |

`icon_levy_grid.nc` keeps only the 11 grid variables uxarray reads — `vlon`,
`vlat`, `elon`, `elat`, `clon`, `clat`, `vertex_of_cell`, `edge_of_cell`,
`neighbor_cell_index`, `adjacent_cell_of_edge`, `edge_vertices` — down from the
~85 variables in the original full ICON grid file.

## Data file (not in git)

The velocity snapshot `levy01_r2b8_P1D_3d_20500101T000000Z_10days.nc` (288 MB:
10-day daily-mean `u`, `v`, `w`) exceeds GitHub's 100 MB limit, so it is **not
committed**. Obtain it separately and drop it in this directory before running
the notebook.

## Running it

```bash
pixi run lab        # launch JupyterLab, then open tutorial_icon.ipynb
# or register a kernel for an existing Jupyter:
pixi run kernel
```

The notebook reads the grid and data `.nc` files by relative path, so run it
from this directory. Executing the advection cell writes `output-icon.parquet`.
