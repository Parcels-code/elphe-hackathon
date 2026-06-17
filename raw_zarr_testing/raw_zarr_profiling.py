import parcels
import argparse

import xarray as xr
import numpy as np


def run_simulation(ds):
    fieldset = parcels.FieldSet.from_sgrid_conventions(ds, mesh="spherical")

    pfile = parcels.ParticleFile(
        "profiling.parquet",
        outputdt=np.timedelta64(2, "h"),
        mode="w",
    )

    N = 10_000
    X, Y = np.meshgrid(
        np.linspace(-80, -60, int(np.sqrt(N))), np.linspace(-10, 10, int(np.sqrt(N)))
    )
    pset = parcels.ParticleSet(fieldset=fieldset, lon=X, lat=Y, z=10 * np.ones_like(X))

    pset.execute(
        kernels=parcels.kernels.AdvectionRK4,
        runtime=np.timedelta64(6, "D"),
        dt=np.timedelta64(1, "h"),
        output_file=pfile,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load-mode",
        choices=["zarr", "numpy", "dask", "zarr-with-cache"],
        default="zarr",
        help="How to open physics.zarr for the simulation.",
    )
    args = parser.parse_args()

    if args.load_mode == "zarr":
        ds = parcels.open_raw_zarr("physics.zarr")
    elif args.load_mode == "numpy":
        ds = xr.open_zarr("physics.zarr")
        ds.load()
    elif args.load_mode == "dask":
        ds = xr.open_zarr("physics.zarr")
    elif args.load_mode == "zarr-with-cache":
        import zarr
        from zarr.experimental.cache_store import CacheStore

        source_store = zarr.storage.LocalStore("physics.zarr")
        cache_store = zarr.storage.MemoryStore()
        store = CacheStore(store=source_store, cache_store=cache_store, max_size=2**30)
        ds = parcels.open_raw_zarr(store)

    run_simulation(ds)
