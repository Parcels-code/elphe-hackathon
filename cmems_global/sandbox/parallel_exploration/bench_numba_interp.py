"""Does numba njit(parallel) interpolation beat the parcels XLinear path at 1e6
particles, and does it scale with threads?

Scenario mirrors 02b: regular A-grid (1D lon/lat/depth), spherical mesh, 2 time
levels resident, AdvectionRK4-style trilinear U/V interpolation. We compare:

  - parcels  : the real vectorised XLinear path (incl. windowed-array np.stack)
  - numba    : an njit(parallel=True) per-particle trilinear interp over the
               cached numpy slabs, swept over thread counts.

Correctness is checked with np.allclose against the parcels result.

    pixi run -e pr2671-windowed-array python sandbox/parallel_exploration/bench_numba_interp.py
Env knobs: N_PARTICLES (default 1_000_000).
"""

import os
import time
from pathlib import Path

import numba
import numpy as np
import parcels
import xarray as xr
from numba import njit, prange

DATA_DIR = "/work/bk1450/b381575/elphe-hackathon_data"
N = int(os.environ.get("N_PARTICLES", "1000000"))
DEG2M = 1852.0 * 60.0


# --------------------------------------------------------------------------- #
# numba kernel
# --------------------------------------------------------------------------- #
@njit(inline="always")
def _locate(arr, x):
    """Binary search on monotonic-increasing arr -> (i, frac) like parcels."""
    n = arr.shape[0]
    if n < 2:
        return 0, 0.0
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if arr[mid] <= x:
            lo = mid
        else:
            hi = mid
    i = lo
    frac = (x - arr[i]) / (arr[i + 1] - arr[i])
    if frac < 0.0:
        frac = 0.0
    elif frac > 1.0:
        frac = 1.0
    return i, frac


@njit(parallel=True, fastmath=True, cache=True)
def interp_uv(U2, V2, lon, lat, depth, plon, plat, pz, tau, out_u, out_v):
    """Trilinear + time-linear interp of U,V for each particle (prange)."""
    npart = plon.shape[0]
    for p in prange(npart):
        x = plon[p]
        y = plat[p]
        z = pz[p]
        t = tau[p]

        xi, xsi = _locate(lon, x)
        yi, eta = _locate(lat, y)
        zi, zeta = _locate(depth, z)

        w000 = (1 - zeta) * (1 - eta) * (1 - xsi)
        w001 = (1 - zeta) * (1 - eta) * xsi
        w010 = (1 - zeta) * eta * (1 - xsi)
        w011 = (1 - zeta) * eta * xsi
        w100 = zeta * (1 - eta) * (1 - xsi)
        w101 = zeta * (1 - eta) * xsi
        w110 = zeta * eta * (1 - xsi)
        w111 = zeta * eta * xsi

        u_acc = 0.0
        v_acc = 0.0
        for ti in range(2):
            wt = (1.0 - t) if ti == 0 else t
            u = (
                w000 * U2[ti, zi, yi, xi]
                + w001 * U2[ti, zi, yi, xi + 1]
                + w010 * U2[ti, zi, yi + 1, xi]
                + w011 * U2[ti, zi, yi + 1, xi + 1]
                + w100 * U2[ti, zi + 1, yi, xi]
                + w101 * U2[ti, zi + 1, yi, xi + 1]
                + w110 * U2[ti, zi + 1, yi + 1, xi]
                + w111 * U2[ti, zi + 1, yi + 1, xi + 1]
            )
            v = (
                w000 * V2[ti, zi, yi, xi]
                + w001 * V2[ti, zi, yi, xi + 1]
                + w010 * V2[ti, zi, yi + 1, xi]
                + w011 * V2[ti, zi, yi + 1, xi + 1]
                + w100 * V2[ti, zi + 1, yi, xi]
                + w101 * V2[ti, zi + 1, yi, xi + 1]
                + w110 * V2[ti, zi + 1, yi + 1, xi]
                + w111 * V2[ti, zi + 1, yi + 1, xi + 1]
            )
            u_acc += wt * u
            v_acc += wt * v

        out_u[p] = u_acc / (DEG2M * np.cos(np.deg2rad(y)))
        out_v[p] = v_acc / DEG2M


def main():
    print(f"numba {numba.__version__}  N={N}  NUMBA threads avail={numba.config.NUMBA_NUM_THREADS}")

    ds = xr.open_zarr(Path(DATA_DIR) / "cmems_uovo_2001.zarr")
    fields = {"U": ds["uo"], "V": ds["vo"]}
    ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields=fields)
    ds_fset = ds_fset.fillna(0.0).isel(depth=slice(0, 2))
    fieldset = parcels.FieldSet.from_sgrid_conventions(ds_fset)
    wfs = fieldset.to_windowed_arrays(max_levels=2)

    lon_g = np.asarray(fieldset.UV.U.grid.lon, dtype=np.float64)
    lat_g = np.asarray(fieldset.UV.U.grid.lat, dtype=np.float64)
    depth_g = np.asarray(ds.depth.values[:2], dtype=np.float64)

    # particles: surface, between time level 0 and 1 (tau in (0,1)) -> lenT=2
    rng = np.random.default_rng(0)
    plon = rng.uniform(-80, 20, size=N)
    plat = rng.uniform(-35, 40, size=N)
    pz = np.full(N, depth_g[0])
    t0 = ds.time.values[0]
    t_half = t0 + (ds.time.values[1] - t0) / 2
    ptime = np.full(N, t_half)

    # ---- parcels baseline -------------------------------------------------- #
    pset = parcels.ParticleSet(
        fieldset=wfs, pclass=parcels.Particle, time=ptime, z=pz, lat=plat, lon=plon
    )
    view = pset[np.ones(N, dtype=bool)]
    # warm (triggers windowed load + first eval)
    u0, v0 = fieldset.UV[view]
    reps = 3
    t = time.perf_counter()
    for _ in range(reps):
        u_par, v_par = fieldset.UV[view]
    par_t = (time.perf_counter() - t) / reps
    print(f"\nparcels XLinear (vectorised): {par_t*1000:8.1f} ms/eval  "
          f"({N/par_t/1e6:.2f} M part/s)")

    # ---- extract cached numpy slabs for numba ------------------------------ #
    Uw = wfs.UV.U.data
    Vw = wfs.UV.V.data
    levels = sorted(Uw._cache)
    print("resident time levels:", levels)
    U2 = np.ascontiguousarray(np.stack([Uw._cache[l] for l in levels[:2]]), dtype=np.float32)
    V2 = np.ascontiguousarray(np.stack([Vw._cache[l] for l in levels[:2]]), dtype=np.float32)
    print("U2 shape:", U2.shape)

    # tau for the chosen sample time within the resident window (timedelta ratio)
    tvals = ds.time.values
    tau = ((ptime - tvals[0]) / (tvals[1] - tvals[0])).astype(np.float64)
    print("tau sample:", float(tau[0]))

    out_u = np.empty(N)
    out_v = np.empty(N)

    # compile (1 thread) + correctness check
    numba.set_num_threads(1)
    interp_uv(U2, V2, lon_g, lat_g, depth_g, plon, plat, pz, tau, out_u, out_v)
    ok_u = np.allclose(out_u, np.asarray(u_par), rtol=1e-3, atol=1e-6)
    ok_v = np.allclose(out_v, np.asarray(v_par), rtol=1e-3, atol=1e-6)
    maxdiff = float(np.nanmax(np.abs(out_u - np.asarray(u_par))))
    print(f"correctness vs parcels: u={ok_u} v={ok_v}  max|du|={maxdiff:.2e}")

    # ---- numba thread sweep ----------------------------------------------- #
    print("\nnumba interp_uv thread sweep:")
    avail = numba.config.NUMBA_NUM_THREADS
    threadcounts = [t for t in (1, 2, 4, 8, 16, 28, avail) if t <= avail]
    threadcounts = sorted(set(threadcounts))
    base = None
    for nt in threadcounts:
        numba.set_num_threads(nt)
        # warm at this thread count
        interp_uv(U2, V2, lon_g, lat_g, depth_g, plon, plat, pz, tau, out_u, out_v)
        t = time.perf_counter()
        for _ in range(reps):
            interp_uv(U2, V2, lon_g, lat_g, depth_g, plon, plat, pz, tau, out_u, out_v)
        nt_t = (time.perf_counter() - t) / reps
        if base is None:
            base = nt_t
        print(f"  {nt:3d} thr: {nt_t*1000:8.2f} ms/eval  "
              f"({N/nt_t/1e6:7.2f} M part/s)  "
              f"scaling={base/nt_t:5.2f}x  vs_parcels={par_t/nt_t:6.1f}x")


if __name__ == "__main__":
    main()
