import argparse

import parcels
import copernicusmarine

import xarray as xr

ModelId = str
UsedFields = tuple[str, ...]
Grid = list[tuple[ModelId, UsedFields]]

DATASET_IDs_BY_GRID: list[tuple[str, Grid]] = [
    (
        "physics",
        [
            ("cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m", ("uo", "vo")),
        ],
    ),
]

DATASET_IDs: list[str] = [list(ids) for ids in DATASET_IDs_BY_GRID]


def download_data(no_compression: bool, **copernicus_kwargs) -> dict[str, xr.Dataset]:
    copernicus_kwargs = (
        dict(
            minimum_longitude=-100,
            maximum_longitude=16,
            minimum_latitude=-25,
            maximum_latitude=45,
            start_datetime="2024-07-01",
            end_datetime="2024-07-07",
            minimum_depth=0.5,
            maximum_depth=15,
        )
        | copernicus_kwargs
    )

    copernicusmarine.login()

    datasets = {}
    for name, grid_datasets in DATASET_IDs_BY_GRID:
        datasets_list = [
            copernicusmarine.open_dataset(id_, **copernicus_kwargs)[list(used_vars)]
            for id_, used_vars in grid_datasets
        ]
        # Should this processing go to copernicusmarine_to_sgrid?
        for i in range(len(datasets_list)):
            if "depth" not in datasets_list[i].dims:
                datasets_list[i] = datasets_list[i].expand_dims(dim={"depth": [0]})
        ds = xr.merge(datasets_list)

        # Should this processing go to copernicusmarine_to_sgrid?
        ds = ds.rename({"uo": "U", "vo": "V"})
        ds["U"] = ds["U"].fillna(0)
        ds["V"] = ds["V"].fillna(0)

        datasets[name] = parcels.convert.copernicusmarine_to_sgrid(
            fields={name: da for name, da in ds.data_vars.items()}
        )

    for name, ds in datasets.items():
        if no_compression:
            encoding = {
                v: {"compressors": None} for v in list(ds.data_vars) + list(ds.coords)
            }
            ds.drop_encoding().to_zarr(
                f"{name}_uncompressed.zarr", mode="w", encoding=encoding
            )
        else:
            ds.drop_encoding().to_zarr(f"{name}_compressed.zarr", mode="w")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-compression",
        action="store_true",
        default=False,
        help="Whether to disable compression when saving the zarr files.",
    )
    args = parser.parse_args()
    download_data(no_compression=args.no_compression)
