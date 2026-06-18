# cmems_global

CMEMS global ocean experiments driven by [OceanParcels](https://github.com/parcels-code/Parcels).

## Layout

```
cmems_global/
├── pixi.toml / pixi.lock     # one pixi env per parcels git rev (see below)
├── scripts/                  # env setup (run from the cmems_global/ dir)
│   ├── configure_pixi.sh
│   └── register_kernels.sh
├── sandbox/                  # throwaway exploration / benchmarks (not prod)
│   └── parallel_exploration/ # numba-JIT kernel + parallelism experiments
└── notebooks/                # jupytext-paired .py / .md / .ipynb
    ├── 01_retrieve_data.*
    ├── 01a_zarr_v2_copy.*
    ├── 02a_run_parcels.*
    ├── 02b_run_parcels.*
    ├── 02c_run_parcels.*
    ├── 02d_run_parcels.*
    ├── 02e_run_parcels.*
    └── 02f_run_parcels.*
```

- `notebooks/01_retrieve_data` — pull CMEMS global `uo`/`vo` via
  `copernicusmarine`, fill land NaNs with 0, and store as unpacked `float32`
  zarr (so the raw-zarr reader in `02c` sees real velocities, not packed int16).
  Writes `cmems_uovo_2001.zarr` in **zarr format 3** (the modern stack default).
- `notebooks/01a_zarr_v2_copy` — write a **zarr format 2** copy
  (`cmems_uovo_2001_zarr2.zarr`) of the field store. Needed because `02f`'s
  parcels-v3 env is pinned to `zarr < 3`, which cannot read the zarr-v3 original.
  Runs on the `main` env (zarr 3 reads the v3 source and writes a v2 copy); the
  original is left untouched, so `02a`–`02e` are unaffected.
- `notebooks/02a_run_parcels` — advect 1000 particles on the `main` build (plain
  `FieldSet`).
- `notebooks/02b_run_parcels` — advect 100k particles on the windowed-array build
  (PR #2671), via `fieldset.to_windowed_arrays(...)`.
- `notebooks/02c_run_parcels` — advect 1000 particles on the raw-zarr build
  (PR #2668), loading the store with `parcels.open_raw_zarr` behind a zarr
  `CacheStore` (in-memory, dask-free). Mirrors the `zarr-with-cache` mode of
  `raw_zarr_testing/raw_zarr_profiling.py` on the `raw_zarr_profiling` branch.
- `notebooks/02d_run_parcels` — advect **1M** particles on the windowed-array
  build (PR #2671), but replace the single-threaded parcels kernel with a
  `numba.njit(parallel=True)` fused `AdvectionRK4` over all cores. The windowed
  fieldset is used only as the IO layer (load each time-level slab once per
  window); the JIT kernel does the index search + trilinear interp + RK4 combine.
  Driver-level only (no parcels patch); specific to this regular A-grid.
- `notebooks/02e_run_parcels` — same JIT kernel as `02d`, but with the `02c`
  zarr-`CacheStore` IO layer (PR #2668) instead of windowed arrays — so the two
  IO strategies can be compared under the identical fast kernel.
- `notebooks/02f_run_parcels` — advect **1M** particles using **native parcels
  v3 JIT** (`parcels.JITParticle` + `parcels.AdvectionRK4`, v3's own JIT-compiled
  C kernel), on a `FieldSet.from_xarray_dataset` whose fields are **eager-loaded
  once** up front (top 2 depth levels, all times) so there is no IO during the
  run. The idiomatic-v3 reference point for the series (no custom kernel, no
  driver loop). Reads the **zarr-v2** copy from `01a` (its env pins `zarr < 3`).
  Output is buffered in an in-memory `zarr.MemoryStore` during the run and dumped
  to the on-disk `.zarr` in a single `zarr.copy_store` (Lustre-friendly: written
  once, not streamed).

The notebooks are jupytext-paired (`.py` / `.md` / `.ipynb`); the `.py`
(py:percent) is the source of truth — see [Notebooks](#notebooks-jupytext) below.

## Pixi environments — one per parcels git rev

This workspace keeps a single shared conda stack (Python, xarray, dask,
copernicusmarine, jupyterlab, …) and layers several pinned **parcels** builds on
top of it as separate pixi environments. This lets us compare parcels revisions
side by side from one directory, each as its own JupyterHub kernel.

| pixi env                | parcels rev | SHA (resolved 2026-06-18) |
| ----------------------- | ----------- | ------------------------- |
| `main`                  | `parcels-code/Parcels` `main`            | `481decc` |
| `pr2671-windowed-array` | PR [#2671](https://github.com/parcels-code/Parcels/pull/2671) "Issue 2656 windowed array" head | `8136bf5` |
| `pr2668-open-raw-zarr`  | PR [#2668](https://github.com/parcels-code/Parcels/pull/2668) "Add `open_raw_zarr` helper" head | `97c3324` |
| `v3`                    | conda-forge `parcels` v3 release (`>=3.1,<4`) — native v3 JIT reference for `02f`. **Self-contained** (`no-default-feature`): pins `zarr<3` + Python 3.12, the combo parcels 3.1 targets | `3.1.0` (installed) |

The shared deps live in pixi's implicit `default` feature, which is merged into
every environment automatically (see `pixi.toml`). The bare `default`
environment carries no parcels and gets no kernel (`register_kernels.sh` skips
it); `pixi install --all` does still build it, but it is otherwise unused.

## Setup on DKRZ Levante

Run from the `cmems_global/` directory, in order (each step is safe to re-run):

```bash
bash scripts/configure_pixi.sh     # 1. global pixi config: envs on $HOME, cache on /scratch
pixi install --all                 # 2. install every named env (uses the /scratch cache)
bash scripts/register_kernels.sh   # 3. register one JupyterHub kernel per env
```

- `scripts/configure_pixi.sh` parks per-project envs on `$HOME` (VAST) and points
  the default package cache at `/scratch` (DKRZ purges it; `$HOME` quota stays
  clean). Global and idempotent.
- `pixi install --all` installs every environment in `pixi.toml` into its
  detached `$HOME` prefix. No wrapper script is needed once the cache lives on
  `/scratch` — the tarballs land there and get purged on DKRZ's schedule, while
  the installed envs on `$HOME` are self-contained and survive a cache purge.
- `scripts/register_kernels.sh` writes one `kernel.json` per environment, each
  launching `pixi run --environment <env>` so pixi activation applies; `PATH` is
  pinned in the spec because the DKRZ spawner does not source `~/.bashrc`. It
  resolves the workspace root from its own location, so `$PWD` doesn't matter.

In JupyterHub, the kernels appear as:

- `Pixi: cmems_global (main)`
- `Pixi: cmems_global (pr2671-windowed-array)`
- `Pixi: cmems_global (pr2668-open-raw-zarr)`

Pick the kernel matching the parcels rev you want to run a notebook against.

To run a notebook headless against a specific env:

```bash
pixi run --environment pr2671-windowed-array python -m ipykernel_launcher ...
# or drop into a shell:
pixi shell --environment main
```

## Notebooks (jupytext)

Each notebook exists in three jupytext-paired forms; **edit the `.py`** (it is
the source of truth) and re-sync — never hand-edit the `.md` or `.ipynb`:

- `<nb>.py` — py:percent source of truth (plain Python; lint/run it directly).
- `<nb>.md` — markdown rendering for readable diffs (generated).
- `<nb>.ipynb` — Jupyter/JupyterHub execution form (generated). Execution outputs
  are kept (the repo's `nbstripout` pre-commit hook was removed).

Each notebook pins its kernel in the `.py` frontmatter: `01`/`02a` use
`cmems_global-main`, `02b`/`02d` use `cmems_global-pr2671-windowed-array`,
`02c`/`02e` use `cmems_global-pr2668-open-raw-zarr`, and `02f` uses
`cmems_global-v3` (the conda-forge `parcels` v3 release `3.1.0`). The `02d`/`02e`
JIT notebooks reach into parcels *private* internals (windowed-array cache; the
raw zarr handle), so each pins the exact parcels commit it was verified against
in a note near the top. `02f` uses only the **public** parcels v3 API
(`FieldSet.from_xarray_dataset`, `JITParticle`, `AdvectionRK4`, `ParticleFile`),
so it pins the conda-forge release version rather than a git commit.

The `02*` notebooks read the input store through a papermill `parameters`-tagged
cell — `data_dir = "/work/bk1450/b381575/elphe-hackathon_data"` (absolute) — so
the path can be overridden per run without editing the body.

```bash
# after editing a .py, propagate to .md and .ipynb (no execution):
pixi run -e main jupytext --sync notebooks/02a_run_parcels.py

# run a quick headless sanity check (writes its own outputs, e.g. 02a_trajectories.parquet):
MPLBACKEND=Agg pixi run -e main python notebooks/02a_run_parcels.py
```

`jupytext` is part of the shared conda deps, so it is available in every env.
Sanity-check each notebook on its own env, e.g.
`pixi run -e pr2671-windowed-array python notebooks/02b_run_parcels.py` or
`pixi run -e pr2668-open-raw-zarr python notebooks/02c_run_parcels.py`.

## Adding or bumping a parcels rev

1. Resolve the rev to a full SHA (reproducible even after a branch moves):

   ```bash
   git ls-remote https://github.com/parcels-code/Parcels.git refs/heads/main
   git ls-remote https://github.com/parcels-code/Parcels.git refs/pull/<N>/head
   ```

2. In `pixi.toml`, add/update a `[feature.<name>.pypi-dependencies]` block
   pinning `parcels` to that SHA, and add a matching entry under
   `[environments]`.
3. Re-run `pixi install --all` then `bash scripts/register_kernels.sh` to
   re-solve the lock, install the env, and register its kernel.

## Background / references

Why envs live on `$HOME` and the cache on `/scratch`:

- DKRZ recommends `$HOME` (VAST) for conda-style envs and discourages `/work`
  (Lustre): <https://docs.dkrz.de/doc/levante/code-development/python.html>,
  <https://docs.dkrz.de/doc/levante/file-systems.html>
- `/scratch` has a 14-day purge, so an ephemeral per-run cache avoids stale
  mtime/atime issues and `$HOME` quota use:
  <https://docs.dkrz.de/doc/levante/containers/singularity.html>
- JupyterHub kernels on DKRZ:
  <https://docs.dkrz.de/doc/software&services/jupyterhub/kernels.html>
- Reference parcels DKRZ setup:
  <https://github.com/geomar-od-lagrange/2025_dkrz_setup>
- pixi config knobs (`detached-environments`, `cache.root`):
  <https://pixi.sh/latest/reference/pixi_configuration/>
