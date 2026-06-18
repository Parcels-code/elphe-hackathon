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
#     display_name: 'Pixi: cmems_global (main)'
#     language: python
#     name: cmems_global-main
# ---

# %% [markdown]
# # Run Parcels — `main`, 1000 particles
#
# Advect 1000 surface particles using the `main` build of parcels with a plain
# `FieldSet` (no windowed arrays). Kernel: `Pixi: cmems_global (main)`.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import parcels
import xarray as xr

# %% tags=["parameters"]
data_dir = "/work/bk1450/b381575/elphe-hackathon_data"

# %%
print(parcels.__version__)

# %%
ds_fields = xr.open_zarr(Path(data_dir) / "cmems_uovo_2001.zarr")
ds_fields

# %%
fields = {"U": ds_fields["uo"], "V": ds_fields["vo"]}
ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields=fields)
ds_fset = ds_fset.fillna(0.0)
ds_fset = ds_fset.isel(depth=slice(0, 2))
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
    "02a_trajectories.parquet", outputdt=np.timedelta64(6, "h"), mode="w"
)

# %%
pset.execute(
    kernels,
    runtime=np.timedelta64(9, "D"),
    dt=np.timedelta64(2, "h"),
    output_file=output_file,
)

# %%
df = parcels.read_particlefile("02a_trajectories.parquet")
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
