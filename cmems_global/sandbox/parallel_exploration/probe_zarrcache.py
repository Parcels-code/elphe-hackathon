"""Probe the open_raw_zarr + CacheStore setup (PR #2668) to learn how to pull
per-time-level numpy slabs for the JIT kernel without eager full reads."""

import time as _t
from pathlib import Path

import numpy as np
import parcels
import zarr
from zarr.experimental.cache_store import CacheStore

DATA_DIR = "/work/bk1450/b381575/elphe-hackathon_data"

store = CacheStore(
    store=zarr.storage.LocalStore(Path(DATA_DIR) / "cmems_uovo_2001.zarr"),
    cache_store=zarr.storage.MemoryStore(),
    max_size=2**30,
)
ds = parcels.open_raw_zarr(store)
fields = {"U": ds["uo"], "V": ds["vo"]}
ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields=fields)
fieldset = parcels.FieldSet.from_sgrid_conventions(ds_fset)

Uda = fieldset.UV.U.data
print("U.data type:", type(Uda))
print("U.data dims:", Uda.dims)
print("U.data shape:", Uda.shape)
print("U.data.data type:", type(Uda.data))
print("U.data.variable._data type:", type(Uda.variable._data))

grid = fieldset.UV.U.grid
print("grid.lon ndim/shape:", np.asarray(grid.lon).ndim, np.asarray(grid.lon).shape)
print("mesh:", grid._mesh)

# try extracting one time level, top 2 depths, as numpy
t0 = _t.perf_counter()
slab = np.asarray(Uda.isel(time=0).values)[:2]
print(
    "isel(time=0)[:2] shape:",
    slab.shape,
    "dtype:",
    slab.dtype,
    f"  {(_t.perf_counter() - t0):.2f}s",
)

# second read should hit the cache
t0 = _t.perf_counter()
slab2 = np.asarray(Uda.isel(time=1).values)[:2]
print("isel(time=1)[:2] shape:", slab2.shape, f"  {(_t.perf_counter() - t0):.2f}s")

# direct zarr-array indexing (only needed chunks)?
raw = ds["uo"]
print("ds['uo'].data type:", type(raw.data))
try:
    za = raw.data
    t0 = _t.perf_counter()
    s = np.asarray(za[0, 0:2, :, :])
    print("za[0,0:2] shape:", s.shape, f"  {(_t.perf_counter() - t0):.2f}s")
except Exception as e:  # noqa: BLE001
    print("direct zarr index failed:", repr(e))
