import parcels
import copernicusmarine
import argparse

import xarray as xr
import numpy as np

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

def download_data(**copernicus_kwargs) -> dict[str, xr.Dataset]:
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
        ds.drop_encoding().to_zarr(f"{name}.zarr", mode="w")

def run_simulation(ds):
    fieldset = parcels.FieldSet.from_sgrid_conventions(ds, mesh="spherical")

    pfile = parcels.ParticleFile(
        "Sargassum_Simulation.parquet",
        outputdt=np.timedelta64(2, "h"),
        mode="w",
    )

    N = 10_000
    X, Y = np.meshgrid(np.linspace(-80, -60, int(np.sqrt(N))), np.linspace(-10, 10, int(np.sqrt(N))))
    pset = parcels.ParticleSet(fieldset=fieldset, lon=X, lat=Y, z=10*np.ones_like(X))

    pset.execute(
        kernels=parcels.kernels.AdvectionRK4,
        runtime=np.timedelta64(6, "D"),
        dt=np.timedelta64(1, "h"),
        output_file=pfile,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load-mode",
        choices=["zarr", "numpy", "dask"],
        default="zarr",
        help="How to open physics.zarr for the simulation.",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Whether to download the data from Copernicus Marine and save it as zarr files.",
    )
    args = parser.parse_args()

    if args.download_data:
        download_data()

    if args.load_mode == "zarr":
        ds = parcels.open_raw_zarr("physics.zarr")
    elif args.load_mode == "numpy":
        ds = xr.open_zarr("physics.zarr")
        ds.load()
    elif args.load_mode == "dask":
        ds = xr.open_zarr("physics.zarr")

    run_simulation(ds)
