---
jupyter:
  jupytext:
    cell_metadata_filter: tags,-all
    formats: py:percent,md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: 'Pixi: cmems_global (main)'
    language: python
    name: cmems_global-main
---

# Convert the field store to zarr format 2

`01_retrieve_data` writes `cmems_uovo_2001.zarr` in **zarr format 3** (the
default of the modern zarr 3 stack shared by `02a`–`02e`). The parcels **v3**
environment used by `02f` is pinned to **zarr < 3** (because parcels 3.1.0's
`ParticleFile` uses the zarr 2 API), and **zarr 2 cannot read a zarr-v3 store**.

This notebook writes a **zarr format 2** copy of the store so the v3 env can
read it. It runs on the `main` env (zarr 3): zarr 3 can read the v3 source and
write a v2 copy; zarr 2 (the v3 env) can then read that copy. The original v3
store is left untouched, so `02a`–`02e` are unaffected. Kernel:
`Pixi: cmems_global (main)`.

```python
from pathlib import Path

import xarray as xr
```

```python tags=["parameters"]
data_dir = "/work/bk1450/b381575/elphe-hackathon_data"
```

```python
src = Path(data_dir) / "cmems_uovo_2001.zarr"
dst = Path(data_dir) / "cmems_uovo_2001_zarr2.zarr"
ds = xr.open_zarr(src)
ds
```

```python
# `zarr_format=2` makes xarray write the older format that zarr 2 can read.
# `drop_encoding()` strips the source's zarr-v3 codec/serializer encoding, which
# is not expressible in zarr format 2 (otherwise to_zarr raises on `serializer`).
ds.drop_encoding().to_zarr(dst, mode="w", zarr_format=2)
print("wrote zarr v2 copy:", dst)
```

```python
# Sanity check: re-open the copy and confirm it carries the same fields/shape.
check = xr.open_zarr(dst)
print(check)
```
