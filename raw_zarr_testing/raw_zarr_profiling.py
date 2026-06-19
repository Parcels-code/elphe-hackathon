import argparse
import cProfile
from pathlib import Path
import pstats

import xarray as xr
import numpy as np

import parcels

try:
    import zarr
    from zarr.experimental.cache_store import CacheStore
except ImportError:
    zarr = None
    CacheStore = None


def profile_execution_time(load_mode: str, compression_mode: str):
    report_name = f"profiles/time/{load_mode}_{compression_mode}.prof"
    Path(report_name).parent.mkdir(parents=True, exist_ok=True)
    prof = cProfile.Profile()
    prof.enable()
    run_simulation(load_mode, compression_mode)
    prof.disable()
    stats = pstats.Stats(prof)
    stats.print_stats(r"run_simulation|particleset\.py:\d+\(execute\)")
    stats.dump_stats(report_name)


def run_simulation(load_mode: str, compression_mode: str) -> None:
    filename = f"physics_{compression_mode}.zarr"
    parcels_version = 3 if load_mode == "parcels-v3" else 4

    N = 10_000
    X, Y = np.meshgrid(
        np.linspace(-80, -60, int(np.sqrt(N))), np.linspace(-10, 10, int(np.sqrt(N)))
    )

    if parcels_version == 4:
        if load_mode == "zarr":
            ds = parcels.open_raw_zarr(filename)
        elif load_mode == "numpy":
            ds = xr.open_zarr(filename)
            ds.load()
        elif load_mode == "dask" or load_mode == "windowed-arrays":
            ds = xr.open_zarr(filename)
        elif load_mode == "zarr-with-cache":
            if zarr is None or CacheStore is None:
                raise ImportError("zarr or CacheStore is not available")
            source_store = zarr.storage.LocalStore(filename)
            cache_store = zarr.storage.MemoryStore()
            store = CacheStore(
                store=source_store, cache_store=cache_store, max_size=2**32
            )
            ds = parcels.open_raw_zarr(store)

        fieldset = parcels.FieldSet.from_sgrid_conventions(ds, mesh="spherical")
        if load_mode == "windowed-arrays":
            fieldset.to_windowed_arrays()
        pset = parcels.ParticleSet(
            fieldset=fieldset, lon=X, lat=Y, z=10 * np.ones_like(X)
        )
        pfile = parcels.ParticleFile(
            "output_profiling.parquet",
            outputdt=np.timedelta64(2, "h"),
            mode="w",
        )
        kernel = parcels.kernels.AdvectionRK4
    elif parcels_version == 3:
        files = filename.replace(".zarr", ".nc")
        dimensions = {
            "U": {"lat": "lat", "lon": "lon", "depth": "depth", "time": "time"},
            "V": {"lat": "lat", "lon": "lon", "depth": "depth", "time": "time"},
        }
        variables = {"U": "U", "V": "V"}
        fieldset = parcels.FieldSet.from_netcdf(
            files, dimensions=dimensions, variables=variables, mesh="spherical"
        )

        pset = parcels.ParticleSet(
            fieldset=fieldset, lon=X, lat=Y, depth=10 * np.ones_like(X)
        )
        pfile = parcels.ParticleFile(
            "output_profiling.zarr",
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
    parser.add_argument(
        "--no-compression",
        action="store_true",
        default=False,
        help="Whether to load files without compression.",
    )
    args = parser.parse_args()

    compression_mode = "uncompressed" if args.no_compression else "compressed"
    profile_execution_time(args.load_mode, compression_mode=compression_mode)
