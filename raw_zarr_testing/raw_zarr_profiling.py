import parcels
import argparse

import xarray as xr
import numpy as np

try:
    import zarr
    from zarr.experimental.cache_store import CacheStore
except ImportError:
    zarr = None
    CacheStore = None


def run_simulation(load_mode: str):

    parcels_version = 4

    if load_mode == "zarr":
        ds = parcels.open_raw_zarr("physics.zarr")
    elif load_mode == "numpy":
        ds = xr.open_zarr("physics.zarr")
        ds.load()
    elif load_mode == "dask" or load_mode == "windowed-arrays":
        ds = xr.open_zarr("physics.zarr")
    elif load_mode == "zarr-with-cache":
        if zarr is None or CacheStore is None:
            raise ImportError("zarr or CacheStore is not available")
        source_store = zarr.storage.LocalStore("physics.zarr")
        cache_store = zarr.storage.MemoryStore()
        store = CacheStore(store=source_store, cache_store=cache_store, max_size=2**30)
        ds = parcels.open_raw_zarr(store)
    elif load_mode == "parcels-v3":
        ds = xr.open_dataset("physics.nc")
        parcels_version = 3

    N = 10_000
    X, Y = np.meshgrid(
        np.linspace(-80, -60, int(np.sqrt(N))), np.linspace(-10, 10, int(np.sqrt(N)))
    )

    if parcels_version == 4:
        fieldset = parcels.FieldSet.from_sgrid_conventions(ds, mesh="spherical")
        if load_mode == "windowed-arrays":
            fieldset.to_windowed_arrays()
        pset = parcels.ParticleSet(
            fieldset=fieldset, lon=X, lat=Y, z=10 * np.ones_like(X)
        )
        pfile = parcels.ParticleFile(
            "profiling.parquet",
            outputdt=np.timedelta64(2, "h"),
            mode="w",
        )
        kernel = parcels.kernels.AdvectionRK4
    elif parcels_version == 3:
        dimensions = {
            "U": {"lat": "lat", "lon": "lon", "depth": "depth", "time": "time"},
            "V": {"lat": "lat", "lon": "lon", "depth": "depth", "time": "time"},
        }
        variables = {"U": "U", "V": "V"}
        fieldset = parcels.FieldSet.from_xarray_dataset(
            ds, dimensions=dimensions, variables=variables, mesh="spherical"
        )

        pset = parcels.ParticleSet(
            fieldset=fieldset, lon=X, lat=Y, depth=10 * np.ones_like(X)
        )
        pfile = parcels.ParticleFile(
            "profiling.zarr",
            pset,
            outputdt=np.timedelta64(2, "h"),
        )
        kernel = parcels.AdvectionRK4

    pset.execute(
        kernel,
        runtime=np.timedelta64(6, "D"),
        dt=np.timedelta64(1, "h"),
        output_file=pfile,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load-mode",
        choices=[
            "zarr",
            "numpy",
            "dask",
            "zarr-with-cache",
            "windowed-arrays",
            "parcels-v3",
        ],
        default="zarr",
        help="How to open physics.zarr for the simulation.",
    )
    args = parser.parse_args()

    run_simulation(args.load_mode)
