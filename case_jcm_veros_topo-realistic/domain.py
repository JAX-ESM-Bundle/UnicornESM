from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.collections import PolyCollection

from EarthSystemGrids.base import StructuredQuadMesh
from EarthSystemGrids.domain_master import DomainMaster, make_mask_regridder
from EarthSystemGrids.RotatedGaussianLatLon import RotatedGaussianLatLon
from EarthSystemGrids import GaussianLatLon

import xarray as xr

def get_domain_master(
    grid_dir,
    truncation_number:int = 31,
    ocean_resolution:float = 1.5,
    mask_threshold:float = 0.3
):
    
    grid_dir = Path(grid_dir)
    
    atm_grid = StructuredQuadMesh.from_SCRIP_file(grid_dir / f"JCM_T{truncation_number:d}.SCRIP.nc")
    sfc_grid = StructuredQuadMesh.from_SCRIP_file(grid_dir / "RotatedGaussianLatLon.SCRIP.nc")

    sfc_lat = np.asarray(sfc_grid.face_lat)
    sfc_lon = np.asarray(sfc_grid.face_lon)
    sfc_land_fraction = jnp.asarray(xr.open_dataset(grid_dir / "landsea_mask_fraction_RotatedGaussianLatLon.nc")["lsm"].isel(valid_time=0).to_numpy())

    atm_landsea_ds = xr.open_dataset(grid_dir / f"landsea_mask_fraction_JCM_T{truncation_number:d}.nc")
    atm_land_fraction = jnp.asarray(atm_landsea_ds["lsm"].isel(valid_time=0).to_numpy())
    atm_topography = jnp.asarray(atm_landsea_ds["topography"].isel(valid_time=0).to_numpy())

    dm = DomainMaster()
    dm.register_domain(
        name = "atm",
        grid = atm_grid,
        mask = jnp.ones(atm_grid.shape),
        land_fraction = atm_land_fraction,
        topography = atm_topography,
    )

    dm.register_domain(
        name = "exchange",
        grid = sfc_grid,
        is_exchange_grid = True,
        mask = jnp.zeros(sfc_grid.shape),
    )


    dm.register_domain(
        name = "ocn",
        grid = sfc_grid,
        land_fraction = sfc_land_fraction,
        mask = sfc_land_fraction < mask_threshold,
    )

    dm.register_domain(
        name = "lnd",
        grid = sfc_grid,
        land_fraction = sfc_land_fraction,
        mask = sfc_land_fraction >= mask_threshold,
    )

    dm.register_transformation(source="atm", target="exchange", method="bilinear", weight_file=grid_dir / f"weight_algo-bilinear_JCM_T{truncation_number:d}_to_RotatedGaussianLatLon.nc")
    dm.register_transformation(source="atm", target="exchange", method="conserve", weight_file=grid_dir / f"weight_algo-conserve_JCM_T{truncation_number:d}_to_RotatedGaussianLatLon.nc")

    dm.register_transformation(source="exchange", target="atm", method="bilinear", weight_file=grid_dir / f"weight_algo-bilinear_RotatedGaussianLatLon_to_JCM_T{truncation_number:d}.nc")
    dm.register_transformation(source="exchange", target="atm", method="conserve", weight_file=grid_dir / f"weight_algo-conserve_RotatedGaussianLatLon_to_JCM_T{truncation_number:d}.nc")

    dm.register_transformation(source="exchange", target="ocn", method="identity", regridder=make_mask_regridder(lambda: dm.domains["ocn"].mask))
    dm.register_transformation(source="exchange", target="lnd", method="identity", regridder=make_mask_regridder(lambda: dm.domains["lnd"].mask))

    # Reverse direction: put an "ocn"/"lnd"-domain field onto the shared
    # "exchange" grid, masked to that domain's own cells with the rest
    # filled 0 (not the module default NaN) -- so an "ocn" contribution and
    # a "lnd" contribution can be summed directly into one composite
    # exchange-grid field, since the two masks partition it exactly.
    dm.register_transformation(source="ocn", target="exchange", method="identity", regridder=make_mask_regridder(lambda: dm.domains["ocn"].mask, fill_value=0.0))
    dm.register_transformation(source="lnd", target="exchange", method="identity", regridder=make_mask_regridder(lambda: dm.domains["lnd"].mask, fill_value=0.0))

    problems = dm.validate(raise_error=False)
    if len(problems) == 0:
        print("There is no problem detected in DomainMaster!")
    else:
        print("There are problems detected in DomainMaster:")
        for i, problem in enumerate(problems):
            print(f" - Problem {i+1:d} : {str(problem):s}")

    return dm

