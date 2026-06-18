"""Probe the actual fieldset/grid structure so a numba prototype matches it."""

from pathlib import Path

import numpy as np
import parcels
import xarray as xr

DATA_DIR = "/work/bk1450/b381575/elphe-hackathon_data"

ds = xr.open_zarr(Path(DATA_DIR) / "cmems_uovo_2001.zarr")
fields = {"U": ds["uo"], "V": ds["vo"]}
ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields=fields)
ds_fset = ds_fset.fillna(0.0).isel(depth=slice(0, 2))
fieldset = parcels.FieldSet.from_sgrid_conventions(ds_fset)

U = fieldset.UV.U
grid = U.grid
print("=== field ===")
print("U.data type:", type(U.data))
print("U.data dims:", U.data.dims)
print("U.data shape:", U.data.shape)
print("U.data dtype:", U.data.dtype)
print("=== grid ===")
print("mesh:", grid._mesh)
print("lon ndim/shape:", np.asarray(grid.lon).ndim, np.asarray(grid.lon).shape)
print("lat ndim/shape:", np.asarray(grid.lat).ndim, np.asarray(grid.lat).shape)
lon = np.asarray(grid.lon)
lat = np.asarray(grid.lat)
if lon.ndim == 1:
    print("lon monotonic increasing:", bool(np.all(np.diff(lon) > 0)))
    print(
        "lon spacing uniform? std/mean:",
        float(np.std(np.diff(lon))),
        float(np.mean(np.diff(lon))),
    )
if lat.ndim == 1:
    print("lat monotonic increasing:", bool(np.all(np.diff(lat) > 0)))
print("depth:", np.asarray(ds.depth.values))

# windowed array internals
wfs = fieldset.to_windowed_arrays(max_levels=2)
Uw = wfs.UV.U
print("=== windowed U.data ===")
print("type:", type(Uw.data))
da = Uw.data
print("has _cache:", hasattr(da, "_cache"))
print("slab_bytes:", getattr(da, "_slab_bytes", None))
print(
    "full field nbytes (per time level):",
    int(np.prod(U.data.shape[1:])) * U.data.dtype.itemsize,
)
