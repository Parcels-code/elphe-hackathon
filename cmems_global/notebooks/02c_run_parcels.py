# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     formats: py:percent,md,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: 'Pixi: cmems_global (pr2668-open-raw-zarr)'
#     language: python
#     name: cmems_global-pr2668-open-raw-zarr
# ---

# %% [markdown]
# # Run Parcels — raw zarr + cache (PR #2668), 1000 particles
#
# Advect 1000 surface particles using the raw-zarr loader from parcels PR
# [#2668](https://github.com/parcels-code/Parcels/pull/2668): the CMEMS store is
# read via `parcels.open_raw_zarr` behind a zarr `CacheStore` (in-memory chunk
# cache), bypassing dask. This mirrors the `zarr-with-cache` mode of the
# `raw_zarr_testing/raw_zarr_profiling.py` experiment. Kernel:
# `Pixi: cmems_global (pr2668-open-raw-zarr)`.
#
# `open_raw_zarr` reads the store raw (no CF-decoding), so it relies on
# `01_retrieve_data` having written the fields as NaN-filled `float32` (packed
# int16 would come through unscaled). The variables stay backed by bare
# `zarr.Array`s and are read lazily — chunk-by-chunk through the `CacheStore`
# during interpolation. We deliberately do NOT slice depth here: any xarray
# indexing (e.g. `.isel`) on a raw `zarr.Array` triggers an eager full read,
# which would defeat the cache; depth handling is left to the cached store.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import parcels
import zarr
from zarr.experimental.cache_store import CacheStore

# %% tags=["parameters"]
data_dir = "/work/bk1450/b381575/elphe-hackathon_data"

# %%
print(parcels.__version__)

# %%
store = CacheStore(
    store=zarr.storage.LocalStore(Path(data_dir) / "cmems_uovo_2001.zarr"),
    cache_store=zarr.storage.MemoryStore(),
    max_size=2**30,
)
ds_fields = parcels.open_raw_zarr(store)
ds_fields

# %%
fields = {"U": ds_fields["uo"], "V": ds_fields["vo"]}
ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields=fields)
fieldset = parcels.FieldSet.from_sgrid_conventions(ds_fset)
print(fieldset)

# %%
n_particles = 1_000

lon = np.random.uniform(-80, 20, size=(n_particles,))
lat = np.random.uniform(-35, 40, size=(n_particles,))
z = np.full_like(lon, ds_fields.depth.values[0])  # surface
time = np.array(
    [ds_fields.time.values[0] for _ in range(n_particles)]
)  # initial time of the input data

pset = parcels.ParticleSet(
    fieldset=fieldset,
    pclass=parcels.Particle,
    time=time,
    z=z,
    lat=lat,
    lon=lon,
)
print(pset)

# %%
kernels = [parcels.kernels.AdvectionRK4]

# %%
output_file = parcels.ParticleFile(
    "02c_trajectories.parquet", outputdt=np.timedelta64(6, "h"), mode="w"
)

# %%
pset.execute(
    kernels,
    runtime=np.timedelta64(9, "D"),
    dt=np.timedelta64(2, "h"),
    output_file=output_file,
)

# %%
df = parcels.read_particlefile("02c_trajectories.parquet")
df

# %%
fig, ax = plt.subplots(figsize=(12, 9))
_df = (
    df.to_pandas()
    .sort_values("particle_id")
    .set_index("particle_id")
    .loc[range(0, n_particles, 1)]
)
scatter = ax.scatter(
    _df["lon"], _df["lat"], c=_df["time"], s=1, alpha=0.5, cmap="viridis_r"
)
ax.set_xlabel("Longitude [deg E]")
ax.set_ylabel("Latitude [deg N]")
fig.colorbar(scatter, ax=ax, label="time")
plt.show()
