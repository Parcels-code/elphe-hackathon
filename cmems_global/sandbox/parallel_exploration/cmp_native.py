"""Time native parcels pset.execute for one IO backend, for the 02[b-e] compare.

IO=windowed  -> 02b path: xr.open_zarr + to_windowed_arrays (run on pr2671 env)
IO=zarrcache -> 02c path: open_raw_zarr + CacheStore        (run on pr2668 env)

Same particles/runtime/dt as the JIT notebooks. Prints one RESULT line.
"""

import os
import time
from pathlib import Path

import numpy as np
import parcels
import xarray as xr

DATA_DIR = "/work/bk1450/b381575/elphe-hackathon_data"
IO = os.environ["IO"]
N = int(os.environ.get("N", "1000000"))

if IO == "windowed":
    ds = xr.open_zarr(Path(DATA_DIR) / "cmems_uovo_2001.zarr")
    fields = {"U": ds["uo"], "V": ds["vo"]}
    ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields=fields)
    ds_fset = ds_fset.fillna(0.0).isel(depth=slice(0, 2))
    fieldset = parcels.FieldSet.from_sgrid_conventions(ds_fset)
    fs = fieldset.to_windowed_arrays(max_levels=2)
elif IO == "zarrcache":
    import zarr
    from zarr.experimental.cache_store import CacheStore

    store = CacheStore(
        store=zarr.storage.LocalStore(Path(DATA_DIR) / "cmems_uovo_2001.zarr"),
        cache_store=zarr.storage.MemoryStore(),
        max_size=2**30,
    )
    ds = parcels.open_raw_zarr(store)
    fields = {"U": ds["uo"], "V": ds["vo"]}
    ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields=fields)
    fieldset = parcels.FieldSet.from_sgrid_conventions(ds_fset)
    fs = fieldset
else:
    raise SystemExit(f"unknown IO={IO}")

rng = np.random.default_rng(0)
lon = rng.uniform(-80, 20, size=N)
lat = rng.uniform(-35, 40, size=N)
z = np.full(N, ds.depth.values[0])
t0 = ds.time.values[0]
ptime = np.full(N, t0)

pset = parcels.ParticleSet(
    fieldset=fs, pclass=parcels.Particle, time=ptime, z=z, lat=lat, lon=lon
)
out = parcels.ParticleFile(
    f"/tmp/cmp_native_{IO}.parquet", outputdt=np.timedelta64(6, "h"), mode="w"
)

t = time.perf_counter()
pset.execute(
    [parcels.kernels.AdvectionRK4],
    runtime=np.timedelta64(9, "D"),
    dt=np.timedelta64(2, "h"),
    output_file=out,
)
wall = time.perf_counter() - t
print(f"RESULT native {IO} N={N} wall={wall:.1f}s", flush=True)
