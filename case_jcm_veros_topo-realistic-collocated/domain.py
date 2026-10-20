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
):
    
    grid_dir = Path(grid_dir)
    
    atm_grid = StructuredQuadMesh.from_SCRIP_file(grid_dir / f"JCM_T{truncation_number:d}.SCRIP.nc")
    sfc_grid = StructuredQuadMesh.from_SCRIP_file(grid_dir / "RotatedGaussianLatLon.SCRIP.nc")

    # Topography files written by EarthSystemGrids' preprocessing_workflow
    # (04_clean_surface.py / 05_regrid_to_atm.py), all (nj, ni) = (lat, lon).
    # The surface `landsea_mask` (1 = land) is that workflow's cleaned mask
    # (isolated ocean bodies filled), so it is used as is rather than
    # re-thresholding `land_fraction` here: the "ocn"/"lnd" domain split,
    # Veros' kbot (veros_case_setup.GridInfo) and fakelnd's active cells all
    # read this one field and therefore cannot disagree about the coastline.
    sfc_topo_ds = xr.open_dataset(grid_dir / "sfc_topo_RotatedGaussianLatLon.nc")
    sfc_land_fraction = jnp.asarray(sfc_topo_ds["land_fraction"].to_numpy())
    sfc_is_land = jnp.asarray(sfc_topo_ds["landsea_mask"].to_numpy() == 1)

    atm_topo_ds = xr.open_dataset(grid_dir / f"atm_topo_JCM_T{truncation_number:d}.nc")
    atm_land_fraction = jnp.asarray(atm_topo_ds["land_fraction"].to_numpy())
    atm_topography = jnp.asarray(atm_topo_ds["topography"].to_numpy())

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
        mask = ~sfc_is_land,
    )

    dm.register_domain(
        name = "lnd",
        grid = sfc_grid,
        land_fraction = sfc_land_fraction,
        mask = sfc_is_land,
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

