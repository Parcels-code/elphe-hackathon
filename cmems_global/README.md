# cmems_global

CMEMS global ocean experiments driven by [OceanParcels](https://github.com/parcels-code/Parcels).

## Layout

```
cmems_global/
├── pixi.toml / pixi.lock     # one pixi env per parcels git rev (see below)
├── scripts/                  # env setup (run from the cmems_global/ dir)
│   ├── configure_pixi.sh
│   └── register_kernels.sh
└── notebooks/                # jupytext-paired .py / .md / .ipynb
    ├── 01_retrieve_data.*
    ├── 02a_run_parcels.*
    ├── 02b_run_parcels.*
    └── 02c_run_parcels.*
```

- `notebooks/01_retrieve_data` — pull CMEMS global `uo`/`vo` via
  `copernicusmarine`, fill land NaNs with 0, and store as unpacked `float32`
  zarr (so the raw-zarr reader in `02c` sees real velocities, not packed int16).
- `notebooks/02a_run_parcels` — advect 1000 particles on the `main` build (plain
  `FieldSet`).
- `notebooks/02b_run_parcels` — advect 100k particles on the windowed-array build
  (PR #2671), via `fieldset.to_windowed_arrays(...)`.
- `notebooks/02c_run_parcels` — advect 1000 particles on the raw-zarr build
  (PR #2668), loading the store with `parcels.open_raw_zarr` behind a zarr
  `CacheStore` (in-memory, dask-free). Mirrors the `zarr-with-cache` mode of
  `raw_zarr_testing/raw_zarr_profiling.py` on the `raw_zarr_profiling` branch.

The notebooks are jupytext-paired (`.py` / `.md` / `.ipynb`); the `.py`
(py:percent) is the source of truth — see [Notebooks](#notebooks-jupytext) below.

## Pixi environments — one per parcels git rev

This workspace keeps a single shared conda stack (Python, xarray, dask,
copernicusmarine, jupyterlab, …) and layers several pinned **parcels** builds on
top of it as separate pixi environments. This lets us compare parcels revisions
side by side from one directory, each as its own JupyterHub kernel.

| pixi env                | parcels rev                                                                                     | SHA (resolved 2026-06-18) |
| ----------------------- | ----------------------------------------------------------------------------------------------- | ------------------------- |
| `main`                  | `parcels-code/Parcels` `main`                                                                   | `481decc`                 |
| `pr2671-windowed-array` | PR [#2671](https://github.com/parcels-code/Parcels/pull/2671) "Issue 2656 windowed array" head  | `8136bf5`                 |
| `pr2668-open-raw-zarr`  | PR [#2668](https://github.com/parcels-code/Parcels/pull/2668) "Add `open_raw_zarr` helper" head | `97c3324`                 |

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
`cmems_global-main`, `02b` uses `cmems_global-pr2671-windowed-array`, and `02c`
uses `cmems_global-pr2668-open-raw-zarr`.

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
