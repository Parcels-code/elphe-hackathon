"""Microbenchmark: does splitting particles across threads / processes speed up
the parcels kernel execution?

GIL is ENABLED in this CPython 3.14 build, so the question is whether the hot
path (vectorised xarray .isel + numpy.stack + dask zarr load) releases the GIL
enough for threads to help, vs. needing separate processes.

Strategy: split N particles into K chunks, advect each chunk in an independent
ParticleSet, and compare wall time of:
  - serial   : K chunks one after another (sanity baseline ~= single pset)
  - threads  : K chunks in K threads (shared GIL)
  - processes: K chunks in K processes (true parallelism, each rebuilds fieldset)

Each worker builds its OWN fieldset (own windowed-array cache) to avoid the
thread-safety question for now and isolate the parallelism signal.

    pixi run -e pr2671-windowed-array python scripts/parallel_exploration/bench_parallel.py
Env knobs: N_PARTICLES, RUNTIME_DAYS, N_CHUNKS.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

DATA_DIR = "/work/bk1450/b381575/elphe-hackathon_data"
N_PARTICLES = int(os.environ.get("N_PARTICLES", "8000"))
RUNTIME_DAYS = int(os.environ.get("RUNTIME_DAYS", "2"))
N_CHUNKS = int(os.environ.get("N_CHUNKS", "4"))


def _make_inits(n, seed):
    import xarray as xr

    ds = xr.open_zarr(Path(DATA_DIR) / "cmems_uovo_2001.zarr")
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-80, 20, size=(n,))
    lat = rng.uniform(-35, 40, size=(n,))
    z = np.full_like(lon, ds.depth.values[0])
    t0 = ds.time.values[0]
    return lon, lat, z, np.array([t0 for _ in range(n)])


def _advect_chunk(args):
    """Build a fieldset + pset for one chunk and execute. Returns wall time."""
    chunk_id, n, seed = args
    import parcels
    import xarray as xr

    ds = xr.open_zarr(Path(DATA_DIR) / "cmems_uovo_2001.zarr")
    fields = {"U": ds["uo"], "V": ds["vo"]}
    ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields=fields)
    ds_fset = ds_fset.fillna(0.0).isel(depth=slice(0, 2))
    fieldset = parcels.FieldSet.from_sgrid_conventions(ds_fset)

    lon, lat, z, t = _make_inits(n, seed)
    pset = parcels.ParticleSet(
        fieldset=fieldset.to_windowed_arrays(max_levels=2),
        pclass=parcels.Particle,
        time=t,
        z=z,
        lat=lat,
        lon=lon,
    )
    out = parcels.ParticleFile(
        f"/tmp/bench_chunk_{chunk_id}.parquet",
        outputdt=np.timedelta64(6, "h"),
        mode="w",
    )
    t0 = time.perf_counter()
    pset.execute(
        [parcels.kernels.AdvectionRK4],
        runtime=np.timedelta64(RUNTIME_DAYS, "D"),
        dt=np.timedelta64(2, "h"),
        output_file=out,
    )
    return time.perf_counter() - t0


def main():
    per_chunk = N_PARTICLES // N_CHUNKS
    jobs = [(i, per_chunk, 100 + i) for i in range(N_CHUNKS)]
    print(
        f"N={N_PARTICLES} chunks={N_CHUNKS} ({per_chunk}/chunk) runtime={RUNTIME_DAYS}d"
    )

    # serial
    t = time.perf_counter()
    serial_inner = [_advect_chunk(j) for j in jobs]
    serial_wall = time.perf_counter() - t
    print(
        f"serial   : wall={serial_wall:6.1f}s  inner={[f'{x:.1f}' for x in serial_inner]}"
    )

    # threads
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N_CHUNKS) as ex:
        thr_inner = list(ex.map(_advect_chunk, jobs))
    thr_wall = time.perf_counter() - t
    print(
        f"threads  : wall={thr_wall:6.1f}s  inner={[f'{x:.1f}' for x in thr_inner]}"
        f"  speedup_vs_serial={serial_wall / thr_wall:.2f}x"
    )

    # processes
    t = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_CHUNKS) as ex:
        proc_inner = list(ex.map(_advect_chunk, jobs))
    proc_wall = time.perf_counter() - t
    print(
        f"processes: wall={proc_wall:6.1f}s  inner={[f'{x:.1f}' for x in proc_inner]}"
        f"  speedup_vs_serial={serial_wall / proc_wall:.2f}x"
    )


if __name__ == "__main__":
    main()
