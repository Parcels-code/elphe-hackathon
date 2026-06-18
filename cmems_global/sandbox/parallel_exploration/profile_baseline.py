"""Profile a downscaled single-threaded parcels run to locate the hot path.

Run with the pr2671-windowed-array pixi env, e.g.:
    pixi run -e pr2671-windowed-array python scripts/parallel_exploration/profile_baseline.py

Downscaled vs 02b (100k particles / 9 d) so it finishes fast while keeping the
same code path: windowed-array fieldset + AdvectionRK4.
"""

import cProfile
import io
import pstats
import time
from pathlib import Path

import numpy as np
import parcels
import xarray as xr

DATA_DIR = "/work/bk1450/b381575/elphe-hackathon_data"
N_PARTICLES = int(__import__("os").environ.get("N_PARTICLES", "5000"))
RUNTIME_DAYS = int(__import__("os").environ.get("RUNTIME_DAYS", "2"))


def build():
    ds_fields = xr.open_zarr(Path(DATA_DIR) / "cmems_uovo_2001.zarr")
    fields = {"U": ds_fields["uo"], "V": ds_fields["vo"]}
    ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields=fields)
    ds_fset = ds_fset.fillna(0.0).isel(depth=slice(0, 2))
    fieldset = parcels.FieldSet.from_sgrid_conventions(ds_fset)

    rng = np.random.default_rng(0)
    lon = rng.uniform(-80, 20, size=(N_PARTICLES,))
    lat = rng.uniform(-35, 40, size=(N_PARTICLES,))
    z = np.full_like(lon, ds_fields.depth.values[0])
    t0 = ds_fields.time.values[0]
    time_arr = np.array([t0 for _ in range(N_PARTICLES)])

    pset = parcels.ParticleSet(
        fieldset=fieldset.to_windowed_arrays(max_levels=2),
        pclass=parcels.Particle,
        time=time_arr,
        z=z,
        lat=lat,
        lon=lon,
    )
    return pset


def run(pset):
    out = parcels.ParticleFile(
        "/tmp/profile_traj.parquet", outputdt=np.timedelta64(6, "h"), mode="w"
    )
    pset.execute(
        [parcels.kernels.AdvectionRK4],
        runtime=np.timedelta64(RUNTIME_DAYS, "D"),
        dt=np.timedelta64(2, "h"),
        output_file=out,
    )


def main():
    print(f"parcels {parcels.__version__}  N={N_PARTICLES}  runtime={RUNTIME_DAYS}d")
    t = time.perf_counter()
    pset = build()
    print(f"build (incl. first lazy touch): {time.perf_counter() - t:.2f}s")

    # warm + time a clean execute
    t = time.perf_counter()
    run(pset)
    print(f"execute wall: {time.perf_counter() - t:.2f}s")

    # profile a fresh pset's execute
    pset = build()
    pr = cProfile.Profile()
    pr.enable()
    run(pset)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(35)
    print("\n===== cumulative top 35 =====")
    print(s.getvalue())

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(35)
    print("\n===== tottime top 35 =====")
    print(s.getvalue())


if __name__ == "__main__":
    main()
