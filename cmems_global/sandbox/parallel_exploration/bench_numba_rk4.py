"""Fused AdvectionRK4 step in numba njit(parallel) over 1e6 particles.

Extends the interpolation prototype to the WHOLE kernel: one RK4 step does 4
trilinear U/V interps at displaced positions + the RK4 combine, all inside a
single prange loop (no Python-level per-substage array temporaries). This is the
"JIT the kernel eval" case. Compares throughput + thread scaling against a
parcels lower-bound proxy (4x its single vectorised interp).

    pixi run -e pr2671-windowed-array python sandbox/parallel_exploration/bench_numba_rk4.py
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


@njit(inline="always")
def _locate(arr, x):
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
    frac = (x - arr[lo]) / (arr[lo + 1] - arr[lo])
    if frac < 0.0:
        frac = 0.0
    elif frac > 1.0:
        frac = 1.0
    return lo, frac


@njit(inline="always")
def _uv(U2, V2, lon, lat, depth, x, y, z, tau):
    """Return (u, v) in degrees/sec at one point (trilinear + time)."""
    xi, xsi = _locate(lon, x)
    yi, eta = _locate(lat, y)
    zi, zeta = _locate(depth, z)
    u_acc = 0.0
    v_acc = 0.0
    for ti in range(2):
        wt = (1.0 - tau) if ti == 0 else tau
        for dz in range(2):
            wz = (1 - zeta) if dz == 0 else zeta
            for dy in range(2):
                wy = (1 - eta) if dy == 0 else eta
                for dx in range(2):
                    wx = (1 - xsi) if dx == 0 else xsi
                    w = wt * wz * wy * wx
                    u_acc += w * U2[ti, zi + dz, yi + dy, xi + dx]
                    v_acc += w * V2[ti, zi + dz, yi + dy, xi + dx]
    u_acc = u_acc / (DEG2M * np.cos(np.deg2rad(y)))
    v_acc = v_acc / DEG2M
    return u_acc, v_acc


@njit(parallel=True, fastmath=True, cache=True)
def rk4_step(U2, V2, lon, lat, depth, plon, plat, pz, tau0, dtau, dt, out_dlon, out_dlat):
    npart = plon.shape[0]
    for p in prange(npart):
        x = plon[p]
        y = plat[p]
        z = pz[p]
        u1, v1 = _uv(U2, V2, lon, lat, depth, x, y, z, tau0)
        x1, y1 = x + u1 * 0.5 * dt, y + v1 * 0.5 * dt
        u2, v2 = _uv(U2, V2, lon, lat, depth, x1, y1, z, tau0 + 0.5 * dtau)
        x2, y2 = x + u2 * 0.5 * dt, y + v2 * 0.5 * dt
        u3, v3 = _uv(U2, V2, lon, lat, depth, x2, y2, z, tau0 + 0.5 * dtau)
        x3, y3 = x + u3 * dt, y + v3 * dt
        u4, v4 = _uv(U2, V2, lon, lat, depth, x3, y3, z, tau0 + dtau)
        out_dlon[p] = (u1 + 2 * u2 + 2 * u3 + u4) / 6.0 * dt
        out_dlat[p] = (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt


def main():
    print(f"numba {numba.__version__}  N={N}  threads avail={numba.config.NUMBA_NUM_THREADS}")
    ds = xr.open_zarr(Path(DATA_DIR) / "cmems_uovo_2001.zarr")
    fields = {"U": ds["uo"], "V": ds["vo"]}
    ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields=fields)
    ds_fset = ds_fset.fillna(0.0).isel(depth=slice(0, 2))
    fieldset = parcels.FieldSet.from_sgrid_conventions(ds_fset)
    wfs = fieldset.to_windowed_arrays(max_levels=2)

    lon_g = np.asarray(fieldset.UV.U.grid.lon, dtype=np.float64)
    lat_g = np.asarray(fieldset.UV.U.grid.lat, dtype=np.float64)
    depth_g = np.asarray(ds.depth.values[:2], dtype=np.float64)

    rng = np.random.default_rng(0)
    plon = rng.uniform(-80, 20, size=N)
    plat = rng.uniform(-35, 40, size=N)
    pz = np.full(N, depth_g[0])
    t0 = ds.time.values[0]
    t_half = t0 + (ds.time.values[1] - t0) / 2
    ptime = np.full(N, t_half)

    # parcels lower-bound proxy: 4x one vectorised interp
    pset = parcels.ParticleSet(
        fieldset=wfs, pclass=parcels.Particle, time=ptime, z=pz, lat=plat, lon=plon
    )
    view = pset[np.ones(N, dtype=bool)]
    fieldset.UV[view]  # warm
    t = time.perf_counter()
    for _ in range(3):
        fieldset.UV[view]
    par_interp = (time.perf_counter() - t) / 3
    par_rk4_proxy = 4 * par_interp
    print(f"parcels 1 interp={par_interp*1000:.0f} ms -> RK4 proxy (4x)={par_rk4_proxy*1000:.0f} ms")

    Uw, Vw = wfs.UV.U.data, wfs.UV.V.data
    levels = sorted(Uw._cache)[:2]
    U2 = np.ascontiguousarray(np.stack([Uw._cache[l] for l in levels]), dtype=np.float32)
    V2 = np.ascontiguousarray(np.stack([Vw._cache[l] for l in levels]), dtype=np.float32)

    dt = float(np.timedelta64(2, "h") / np.timedelta64(1, "s"))
    win = float((ds.time.values[1] - t0) / np.timedelta64(1, "s"))
    tau0 = 0.5
    dtau = dt / win

    out_dlon = np.empty(N)
    out_dlat = np.empty(N)

    print("\nfused RK4 step thread sweep:")
    avail = numba.config.NUMBA_NUM_THREADS
    tcs = sorted({t for t in (1, 2, 4, 8, 16, 28, avail) if t <= avail})
    base = None
    for nt in tcs:
        numba.set_num_threads(nt)
        rk4_step(U2, V2, lon_g, lat_g, depth_g, plon, plat, pz, tau0, dtau, dt, out_dlon, out_dlat)
        t = time.perf_counter()
        for _ in range(3):
            rk4_step(U2, V2, lon_g, lat_g, depth_g, plon, plat, pz, tau0, dtau, dt, out_dlon, out_dlat)
        st = (time.perf_counter() - t) / 3
        if base is None:
            base = st
        print(f"  {nt:3d} thr: {st*1000:8.2f} ms/step  ({N/st/1e6:7.2f} M part/s)  "
              f"scaling={base/st:5.2f}x  vs_parcels_proxy={par_rk4_proxy/st:6.1f}x")

    # sanity: displacement magnitude reasonable (deg over a 2h step)
    print(f"\nmean |dlon|={np.mean(np.abs(out_dlon)):.4e} deg, "
          f"max |dlon|={np.max(np.abs(out_dlon)):.4e} deg")


if __name__ == "__main__":
    main()
