---
jupyter:
  jupytext:
    cell_metadata_filter: -all
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

# Retrieve CMEMS global fields

Pull daily global `uo`/`vo` (2001-01-01..2001-01-10) from CMEMS via
`copernicusmarine` and write them to a local zarr store.

Land NaNs are filled with 0 and the fields are stored as plain `float32`
(`drop_encoding` removes the source int16 packing) so that `02c`, which reads
the store raw via `parcels.open_raw_zarr` (no CF-decoding), sees real
velocities rather than packed integers.

```python
from pathlib import Path

import copernicusmarine
```

```python
output_path = "/work/bk1450/b381575/elphe-hackathon_data"
```

```python
ds = copernicusmarine.open_dataset(dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m")
ds
```

```python
ds = ds[["uo", "vo"]].sel(time=slice("2001-01-01", "2001-01-10"))
ds
```

```python
ds = ds.fillna(0.0)
ds["uo"] = ds["uo"].astype("float32")
ds["vo"] = ds["vo"].astype("float32")
ds.drop_encoding().to_zarr(Path(output_path) / "cmems_uovo_2001.zarr/", mode="w")
```
