# Construction of the coupled JCM + Veros + fake-land system used by the
# realistic-topography experiment, packaged as a function so that the forward
# run (main_forward.py) and other scripts (e.g. a jax.grad sensitivity
# experiment) build exactly the same model.

import glob
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData

import xarray as xr

from jem import Coupler
# Importing `VerosComponent` is what points Veros at the JAX backend (and
# turns on x64) -- it has to come before anything that imports
# `veros.core.operators`, which binds its array backend at import time.
from jem.components import (
    SlabGrid,
    SlabLandModel,
    SlabLandParameters,
    SlabSeaiceModel,
    JCMComponent,
    VerosComponent,
)
from jem.components.veros_component import GHOST_CELLS, SEAWATER_HEAT_CAPACITY
from veros_case_setup import generateVerosSetup
from veros.core.operators import update, at
from domain import get_domain_master

def is_pytree_all_finite(tree):
    return jax.tree_util.tree_reduce(
        lambda acc, x: acc & jnp.all(jnp.isfinite(x)),
        tree,
        jnp.array(True),
    )

def report_first_nonfinite(name, x, lon=None, lat=None):
    """If `x` has any non-finite value, debug-print the index of its
    first occurrence (and lon/lat, if `x` is a 2D (lon, lat) field).

    `jax.debug.print` cannot format a tuple-of-tracers (e.g. the result
    of `jnp.unravel_index`) directly, so each index component is passed
    as its own scalar kwarg. The format string is built in Python from
    `x.ndim`, which is static under tracing.
    """
    x = jnp.asarray(x)
    finite = jnp.isfinite(x)
    any_bad = jnp.any(~finite)
    flat_idx = jnp.argmin(finite.ravel().astype(jnp.int32))
    idx = jnp.unravel_index(flat_idx, x.shape)
    val = x.ravel()[flat_idx]

    idx_fmt = ", ".join(f"{{i{d}}}" for d in range(x.ndim))
    idx_kwargs = {f"i{d}": idx[d] for d in range(x.ndim)}

    def report(_):
        fmt = name + f": first non-finite at idx=({idx_fmt}), value={{val:.6e}}"
        kwargs = dict(idx_kwargs, val=val)
        if lon is not None and lat is not None and x.ndim == 2:
            fmt += ", lon={lo:.2f}, lat={la:.2f}"
            # `lon`/`lat` are plain numpy arrays (since `jnp.pi` is a
            # Python float, `numpy_array * 180 / jnp.pi` stays numpy).
            # Indexing a numpy array with a JAX tracer raises
            # TracerArrayConversionError, so convert to jnp first.
            kwargs["lo"] = jnp.asarray(lon)[idx[0]]
            kwargs["la"] = jnp.asarray(lat)[idx[1]]
        jax.debug.print(fmt, **kwargs)

    jax.lax.cond(any_bad, report, lambda _: None, None)



def _freeze_season(raw_atm_model, freeze_season_at_day):
    """Pin `atm_model`'s seasonal (day-of-year) forcing to a fixed day, while
    leaving the diurnal (time-of-day) cycle running.

    `jcm.model.Model._date_from_sim_time` turns elapsed `sim_time` into a
    `DateData`; its "days" component sets the season (`tyear`/`orbital_phase`
    in `jcm.forcing._solar_from_date`, and any climatological `TimeSeries`
    forcing sliced by `ForcingData.select`), while its "seconds-within-day"
    component sets the diurnal phase (`synodic_phase`). Replacing only the
    days component -- with the real `sim_time % 86400` still feeding the
    seconds component -- freezes the season (the sun stops advancing past
    `freeze_season_at_day`) without freezing day/night.

    This monkey-patches the *instance*, not the `jcm.model.Model` class:
    Python attribute lookup checks the instance's `__dict__` before the
    class's, so `atm_model._date_from_sim_time` (called internally as
    `self._date_from_sim_time(...)`) resolves to this wrapper without
    touching the `jcm` package itself.
    """
    original_date_from_sim_time = atm_model._date_from_sim_time
    frozen_days_seconds = freeze_season_at_day * 86400.0

    def frozen_date_from_sim_time(sim_time):
        seconds_of_day = sim_time % 86400.0
        return original_date_from_sim_time(frozen_days_seconds + seconds_of_day)

    atm_model._date_from_sim_time = frozen_date_from_sim_time


def _geometric_stretched_ddz(n_layers: int, surface_thickness: float, total_depth: float, tol: float = 1e-9):
    """Vertical layer thicknesses for a constant-ratio ("geometric")
    stretched z-grid: dz[k] = surface_thickness * ratio**k for
    k = 0, ..., n_layers-1, with `ratio` chosen so the thicknesses sum to
    exactly `total_depth`. This concentrates resolution near the surface
    (thermocline/mixed layer, where gradients and diurnal forcing are
    sharpest) and coarsens smoothly with depth, with no explicit
    bathymetry-dependent cap -- appropriate here since this setup's
    GridInfo still uses a flat-bottom  (uniform total_depth
    everywhere; see GridInfo's own TODO about real bathymetry in
    veros_case_setup.py).

    `ratio` has no closed form for a given (n_layers, surface_thickness,
    total_depth), so it's solved by bisection on the geometric series sum
    sum_{k=0}^{n-1} dz0*r**k == dz0*(r**n - 1)/(r - 1) == total_depth.
    """
    def series_sum(ratio: float) -> float:
        if abs(ratio - 1.0) < 1e-12:
            return surface_thickness * n_layers
        return surface_thickness * (ratio ** n_layers - 1.0) / (ratio - 1.0)

    lo, hi = 1.0 + 1e-9, 2.0
    while series_sum(hi) < total_depth:
        hi *= 1.5
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if series_sum(mid) < total_depth:
            lo = mid
        else:
            hi = mid
    ratio = (lo + hi) / 2.0

    return [surface_thickness * ratio ** k for k in range(n_layers)]


def _shortwave_penetration_divergence(
    zw,
    dzt,
    rpart_shortwave: float = 0.58,
    efold1_shortwave: float = 0.35,
    efold2_shortwave: float = 23.0,
):
    """Vertical divergence of the solar penetration profile, `divpen` (1/m).

    Ported verbatim from `veros/setups/global_1deg/global_1deg.py`
    (`set_initial_conditions`). Solar radiation is not absorbed entirely in
    the 's top layer: it decays with depth as a two-band exponential
    (Paulson & Simpson 1977 / the SPEEDY-era "Jerlov water type I" fit),

        pen(z) = rpart * exp(z / efold1) + (1 - rpart) * exp(z / efold2)

    with `z <= 0` (Veros' `zw` is the *upper* interface of each T-cell, so
    `zw[-1] == 0` is the surface and `zw[0]` the  floor). `rpart` is
    the red/near-IR fraction, killed off within `efold1` (0.35 m) of the
    surface; the remainder is the blue-green band, reaching `efold2` (23 m).

    `pen` is deliberately set to 0 -- not 1 -- at the surface. The full
    shortwave flux is *already* being applied to the top layer as part of
    the net surface heat flux (`forc_temp_surface`), so what this profile
    contributes is a pure *redistribution*: `sum(divpen * dzt) == 0` by
    construction (telescoping sum, ending at `pen[-1] == 0`), taking the
    penetrating fraction back out of the top layer and depositing it in the
    layers below. That means callers do not have to (and must not) also
    subtract the shortwave part out of the surface heat flux themselves.

    Below the sea floor the deposited heat is dropped rather than
    redistributed, since `tempsalt_sources` multiplies `temp_source` by
    `maskT` -- the same behaviour as global_1deg. Harmless for this setup's
    flat-bottom (5200 m) , where `pen` is ~1e-10 at the deepest
    interface anyway.
    """
    zw = jnp.asarray(zw)
    dzt = jnp.asarray(dzt)

    pen = (
        rpart_shortwave * jnp.exp(zw / efold1_shortwave)
        + (1.0 - rpart_shortwave) * jnp.exp(zw / efold2_shortwave)
    )
    pen = pen.at[-1].set(0.0)

    divpen = jnp.empty_like(pen)
    divpen = divpen.at[1:].set((pen[1:] - pen[:-1]) / dzt[1:])
    divpen = divpen.at[0].set(pen[0] / dzt[0])
    return divpen


def build_model(
    *,
    truncation_number,
    start_datetime,
    coupling_timestep,
    veros_dt_mom=3600.0,
    veros_dt_tracer=3600.0,
    jcm_dt=1800.0,
    number_of_ocean_layers=None,
    debug_mode=False,
    freeze_season_at_day=None,
    grid_folder: str,
    ocean_surface_thickness: float = 10.0,
    enable_shortwave_penetration: bool = True,
    shortwave_rpart: float = 0.58,
    shortwave_efold1: float = 0.35,
    shortwave_efold2: float = 23.0,
    landsea_threshold: float = 0.25,
):
    """Build the coupled JCM + Veros + SlabLand system.

    Returns the assembled `Coupler` (which owns the workflow and the clock)
    together with a `config` dict of supporting objects, so that callers can
    build exactly the same model. The clock is `jax_datetime`'s proleptic
    Gregorian calendar -- neither JCM nor the `Coupler` offers another.

    `freeze_season_at_day`: if not `None`, the number of days after
    `start_datetime` at which to freeze the atmosphere's seasonal (solar and
    climatological-forcing) cycle -- e.g. `0` pins it at `start_datetime`
    itself. The diurnal cycle keeps advancing normally. See `_freeze_season`.

    `enable_shortwave_penetration`: let the absorbed solar flux be deposited
    over a finite depth range instead of entirely in the 's top layer,
    using the two-band exponential profile of
    `_shortwave_penetration_divergence` (`shortwave_rpart` /
    `shortwave_efold1` / `shortwave_efold2` are its parameters). Without it,
    100% of the daytime insolation lands in a single layer, so the implied
    top-layer heating rate scales as 1/`_surface_thickness` -- which is
    why thinning the surface layer below ~30 m used to blow the coupled model
    up within a couple of days. With penetration on, roughly the top
    `shortwave_efold2` metres share the flux regardless of how the grid is
    cut, so the surface layer can be made thin without the heating rate
    running away with it.
    """

    grid_folder = Path(grid_folder)
    print(f"grid_folder = {str(grid_folder):s}")

    coords = get_speedy_coords(spectral_truncation=truncation_number)

    # Real topography, on JCM's own native (unrotated) grid -- not the
    # rotated  grid's mask file. JCM's Coriolis parameter is
    # `2*omega*sin(lat)` computed directly from its own native latitude
    # (dinosaur.primitive_equations.PrimitiveEquations.coriolis_parameter,
    # hard-coded, no override hook), so terrain must be geographically
    # consistent with *that* latitude, not the 's rotated one. The
    # grid folder ships this file already in JCM's canonical layout (dims
    # (lon, lat), vars lsm/orog); it holds the same land fraction and
    # topography as the "atm" domain of `dm`.

    ds_terrain = xr.load_dataset(grid_folder / f"topo_JCM_T{truncation_number:d}.nc")
    terrain = TerrainData.from_coords(
        coords=coords,
        # The file is (nj, ni) = (lat, lon); JCM wants (lon, lat).
        orography=jnp.asarray(ds_terrain["topography"].transpose("ni", "nj").values),
        fmask=jnp.asarray(ds_terrain["land_fraction"].transpose("ni", "nj").values),
    )
    lon_deg = coords.horizontal.longitudes * 180 / jnp.pi
    lat_deg = coords.horizontal.latitudes * 180 / jnp.pi

    terrain.fmask = jnp.where(terrain.fmask <= landsea_threshold, 0.0, jnp.where(terrain.fmask >= landsea_threshold, 1.0, terrain.fmask))
    
    # Create JCM
    raw_atm_model = jcm.model.Model(
        coords=coords,
        start_time=start_datetime,
        terrain=terrain,
        time_step=jcm_dt/60.0,
    )
    if freeze_season_at_day is not None:
        _freeze_season(raw_atm_model, freeze_season_at_day)
    
    atm_component = JCMComponent(raw_atm_model)
    atm_D2_nodal_shape = raw_atm_model.coords.nodal_shape[1:]
    
    lon_deg = raw_atm_model.coords.horizontal.longitudes * 180 / jnp.pi
    lat_deg = raw_atm_model.coords.horizontal.latitudes * 180 / jnp.pi

    slab_grid = SlabGrid.from_coords(
        raw_atm_model.coords.horizontal,
        raw_atm_model.terrain.fmask,
    )

    # Seaice Model
    seaice_component = SlabSeaiceModel(slab_grid)
 
    # Land Model
    lnd_component = SlabLandModel(
        slab_grid,
        SlabLandParameters(
            temperature_relaxation_timescale=86400.0 * 10,
            drainage_timescale=jnp.asarray([86400.0 * 180.0,]),
        ),
        land_surface_temperature_from_clim_file = False,
        snow_depth_from_clim_file = False,
    )

    # Create Veros
    # First need to remove `output_veros.*.nc` files, otherwise veros complains.
    for f in glob.glob("output_veros.*.nc"):
        print(f"Deleting: {f}")
        os.remove(f)
   
    ocean_grid = "JCM_T31"#RotatedGaussianLatLon" 
    # `generateVerosSetup` needs raw file access (native_lat/native_lon axis
    # metadata that isn't part of a `Domain`'s in-memory land_fraction/mask
    # arrays), so it's handed the  grid/mask files directly rather than
    # going through `dm` -- these are the same files `dm` itself loaded the
    # "ocn"/"exchange" domains' grid and land-sea mask from, in
    # `get_domain_master`.
    ocn_model = generateVerosSetup(
        rotated_grid=False,
        scrip_grid_file = grid_folder / f"{ocean_grid:s}.SCRIP.nc",
        landsea_mask_file = grid_folder / f"topo_{ocean_grid:s}.nc",
        dt_mom=veros_dt_mom,
        dt_tracer=veros_dt_tracer,
        # 30-layer geometric-stretch grid, same 5200 m total depth as the
        # previous hand-tuned 15-layer profile. `_surface_thickness`
        # is exposed as its own parameter -- a 10 m surface layer (5x
        # smaller heat capacity for the same heat flux) used to trigger an
        # immediate atm humidity explosion (~day 2) regardless of 
        # timestep (tested 3600/1800/900s, all failed identically). The
        # suspected cause was that the *entire* absorbed solar flux was
        # being dumped into that one top layer, making the daytime heating
        # rate scale as 1/`_surface_thickness`; see
        # `enable_shortwave_penetration` and
        # `_shortwave_penetration_divergence`, which spread it over the
        # top ~`shortwave_efold2` metres instead.
        ddz=_geometric_stretched_ddz(n_layers=30, surface_thickness=ocean_surface_thickness, total_depth=5200.0)[:number_of_ocean_layers],
    )()
    ocn_model.setup()
    ocn_component = VerosComponent(ocn_model)

    # Solar penetration profile (see `_shortwave_penetration_divergence`).
    # `zw`/`dzt` are fixed by the grid, so this is a plain (nz,) constant
    # computed once here and closed over by `interaction` below. Disabled ->
    # all-zeros, i.e. the old behaviour where the whole shortwave flux is
    # absorbed in the top layer via the net surface heat flux alone.
    if enable_shortwave_penetration:
        divpen_shortwave = _shortwave_penetration_divergence(
            ocn_model.state.variables.zw,
            ocn_model.state.variables.dzt,
            rpart_shortwave=shortwave_rpart,
            efold1_shortwave=shortwave_efold1,
            efold2_shortwave=shortwave_efold2,
        )
        print(
            "Shortwave penetration enabled: "
            f"rpart={shortwave_rpart:g}, efold1={shortwave_efold1:g} m, "
            f"efold2={shortwave_efold2:g} m"
        )
        print(f"  divpen_shortwave (surface -> bottom) = {np.asarray(divpen_shortwave)[::-1]}")
    else:
        divpen_shortwave = jnp.zeros_like(jnp.asarray(ocn_model.state.variables.dzt))
        print("Shortwave penetration disabled: all solar absorbed in the top  layer.")

    # Reference heat capacity / density used to turn W/m^2 into K m/s -- the
    # same pair `jem/components/veros_component.py` uses for
    # `forc_temp_surface`, pulled out of the traced `interaction` below.
    ocn_cp_0 = SEAWATER_HEAT_CAPACITY
    ocn_rho_0 = float(ocn_model.state.settings.rho_0)

    if not ocn_model.state.settings.enable_tempsalt_sources:
        raise RuntimeError(
            "Veros setting `enable_tempsalt_sources` must be True for the "
            "penetrating-shortwave `temp_source` term to be allocated and "
            "applied (see veros_case_setup.py)."
        )
    

    # Creating Flux and Scalar Exchange between Components
    #
    # Here we demonstrate the flexibility of JEM: you do not have to use
    # the `BasicMapper` that JEM provides. You can define your own mapping
    # function. In this example, we simply define a function `interaction`
    # as below.

    # An exchanger takes every component's carry plus the coupler's clock
    # and returns the carries to use for this coupled step. Carries are
    # never mutated: each write below builds a new forcing/state struct with
    # `.replace(...)`, cast to the dtype of the field it replaces -- Veros
    # switches JAX to double precision when it is imported, so which of the
    # atmosphere's fields are float32 depends on what was allocated before
    # that, and `jax.lax.scan` refuses a carry whose dtypes change.
    def interaction(components, time):
        del time  # this coupling does not depend on the date
        atm = components["atm"]
        ocn = components["ocn"]
        lnd = components["lnd"]
        seaice = components["seaice"]

        # ===== compute wind stress begin =====
        # Tien-Yiao's ad-hoc way to compute wind stress.
        # This conveniently demonstrates how the flux computation can be
        # its own function or module.
        drag_coefficient = 1e-3  # dimensionless
        air_density = 1.22  # kg / m^3
        # `dm.transform_vector` rotates (u0, v0) from JCM's own grid-local
        # frame to geographic east/north (a no-op here, since JCM's grid is
        # unrotated), regrids each geographic component atm -> exchange,
        # then rotates into the  grid's local (rotated) frame using
        # its per-cell cos_angle/sin_angle.
        wind_x, wind_y = (
            atm["derived"].u0,
            atm["derived"].v0,
        )
        # `sqrt` has an infinite derivative at 0, so AD through `sqrt(x)`
        # blows up to NaN as x -> 0, even though the primal value (0) stays
        # finite. Floor the *squared* speed -- sqrt's argument -- at
        # min_speed_magnitude**2 so sqrt and its derivative stay bounded;
        # this naturally caps the resulting wind speed from below at
        # min_speed_magnitude. Mirrors the analogous fix in
        # `jem.components.Veros.make_jem_compatible` for `forc_tke_surface`.
        min_speed_magnitude = 1e-3
        wind_speed_squared = wind_x**2 + wind_y**2
        wind_velocity = jnp.sqrt(jnp.maximum(wind_speed_squared, min_speed_magnitude ** 2))
        surface_taux = drag_coefficient * air_density * wind_velocity * wind_x
        surface_tauy = drag_coefficient * air_density * wind_velocity * wind_y
        # ===== compute wind stress end =====

        ocean_heat_flux = atm["derived"].total_heat_flux
        freshwater_flux = atm["derived"].total_freshwater_flux

        # ===== penetrating shortwave radiation begin =====
        # JCM's sea-surface net heat flux is
        #   hfluxn[:, :, 1] = rsds*(1 - alb_s) + rlds - (rlus + shf + alhc*evap)
        # (jcm/physics/surface/speedy_surface_flux.py), which is what
        # `total_heat_flux` above carries (negated, and blended with the
        # land tile's by JCM's own `fmask` in coastal cells). So the absorbed
        # solar part of the flux the  is already receiving is
        # `rsds * (1 - alb_s)`, downward-positive, recomputed here from the
        # same two diagnostics JCM itself used.
        #
        # `_shortwave_penetration_divergence` is a *redistribution* profile
        # (it integrates to zero over the column), so `heat_flux` is left
        # exactly as it is: the top layer still receives the full solar flux
        # through `forc_temp_surface`, and `temp_source` then moves the
        # penetrating fraction out of it and down into the layers below.
        # Mirrors `set_forcing_kernel` in
        # `veros/setups/global_1deg/global_1deg.py`.
        solar_flux = atm["derived"].physics["_shortwave_rad"].rsds * (1.0 - atm["derived"].physics["_mod_radcon"].alb_s)

        # W/m^2 * 1/m / (J/kg/K) / (kg/m^3) = K/s. `tempsalt_sources`
        # multiplies by `maskT` itself, so no mask is needed here.
        vs_ocn_forcing = ocn["state"].variables
        interior = slice(GHOST_CELLS, -GHOST_CELLS)
        with vs_ocn_forcing.unlock():
            vs_ocn_forcing.temp_source = update(
                vs_ocn_forcing.temp_source,
                at[interior, interior, :],
                solar_flux[:, :, None]
                * divpen_shortwave[None, None, :]
                / ocn_cp_0
                / ocn_rho_0,
            )
        # ===== penetrating shortwave radiation end =====
            
        sea_surface_temperature = ocn["derived"].sea_surface_temperature

        atmgrid_sea_surface_temperature = jnp.clip(
            sea_surface_temperature,
            250.0,
            310.0, 
        )
        atmgrid_stl_am = lnd["state"].land_surface_temperature
        

        # Exchange information
        atm = dict(atm, forcing=atm["forcing"].replace(
            sea_surface_temperature = atmgrid_sea_surface_temperature,
            sice_am=seaice["state"].ice_fraction,
            stl_am=atmgrid_stl_am,
            snowc_am=lnd["state"].snow_cover,
            soilw_am=lnd["state"].soil_water_content[:, :, 0],
        ))

        seaice = dict(seaice, forcing=seaice["forcing"].replace(
            sea_surface_temperature=sea_surface_temperature,
        ))

        lnd = dict(lnd, forcing=lnd["forcing"].replace(
            total_heat_flux=atm["derived"].total_heat_flux,
            precipitation=atm["derived"].precipitation,
        ))

        ocn = dict(ocn, forcing=ocn["forcing"].replace(
            surface_taux=surface_taux,
            surface_tauy=surface_tauy,
            heat_flux=atm["derived"].total_heat_flux,
            freshwater_flux=atm["derived"].total_freshwater_flux,
        ))
 
        updated_components = dict(components, atm=atm, ocn=ocn, lnd=lnd, seaice=seaice)


        if debug_mode:

            print("Notice: debug_mode is on.")

            # Debug: report wind speed (atm),  velocity, and stratification (ocn) extremes
            vs_ocn = ocn["state"].variables
            jax.debug.print(
                "max|wind_u| (atm) = {wu:.6e}, max|wind_v| (atm) = {wv:.6e}, "
                "max|u| (ocn) = {ou:.6e}, max|v| (ocn) = {ov:.6e}, min(Nsqr) (ocn) = {nsqr:.6e}",
                wu=jnp.max(jnp.abs(wind_x)),
                wv=jnp.max(jnp.abs(wind_y)),
                ou=jnp.max(jnp.abs(vs_ocn.u[..., vs_ocn.tau])),
                ov=jnp.max(jnp.abs(vs_ocn.v[..., vs_ocn.tau])),
                nsqr=jnp.min(vs_ocn.Nsqr[..., vs_ocn.tau]),
            )

            # Debug model explosion
            def breakpoint_if_nonfinite(x):
                all_finite = is_pytree_all_finite(x)
                def true_fn(x):
                    pass
                def false_fn(x):
                    jax.debug.print("Model exploded. Activate breakpoint to see what happened.")
                    report_first_nonfinite("wind_x", wind_x, lon_deg, lat_deg)
                    report_first_nonfinite("wind_y", wind_y, lon_deg, lat_deg)
                    report_first_nonfinite("ocn.forcing.heat_flux", ocn["forcing"].heat_flux, lon_deg, lat_deg)
                    report_first_nonfinite("atm.forcing.stl_am", atm["forcing"].stl_am, lon_deg, lat_deg)
                    # The crash was first seen in the  state (u, v, Nsqr
                    # all NaN while the atmosphere was still finite), so check
                    # the 's prognostic fields too (3D: x, y, depth).
                    report_first_nonfinite("ocn.state.temp", vs_ocn.temp[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.salt", vs_ocn.salt[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.u", vs_ocn.u[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.v", vs_ocn.v[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.Nsqr", vs_ocn.Nsqr[..., vs_ocn.tau])
                    # jax.debug.breakpoint()  # disabled: no TTY in SLURM batch jobs -> JaxRuntimeError
                jax.lax.cond(all_finite, true_fn, false_fn, x)

            breakpoint_if_nonfinite(updated_components) 


        return updated_components

    # One coupled step: exchange first, then every component integrates the
    # same exchanged state over `coupling_timestep`.
    model = Coupler(
        components=dict(
            atm=atm_component,
            seaice=seaice_component,
            lnd=lnd_component,
            ocn=ocn_component,

        ),
        exchangers=dict(mapper=interaction),
        coupling_timestep=coupling_timestep,
        start_date=start_datetime,
        workflow=["mapper", "atm", "ocn", "lnd", "seaice"],
    )

    config = dict(
        coords=coords,
    )

    return model, config
