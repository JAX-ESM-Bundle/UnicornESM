# Construction of the coupled JCM + Veros + SlabOceanModel system used by
# the double-drake experiment, packaged as a function so that the forward
# run (main.py) and other scripts (e.g. a jax.grad sensitivity experiment)
# build exactly the same model.

import glob
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import numpy as np
import xarray as xr

import jcm
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.terrain import TerrainData
from jem.tool_scripts.generate_jcm_forcing_and_topography_files import generate_jcm_forcing_and_topography_files

from jem.components import JCM, Veros, SlabOceanModel
from jem.components.slab.grid import generate_slab_grid_from_scrip
from jem.base.coupler import Coupler

from modify_jcm_terrain import modify_jcm_terrain
from veros_case_setup import generateVerosSetup

from veros.core.operators import update, at

from domain import get_domain_master

one_second = jdt.to_timedelta(1, "second")

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



def _freeze_season(atm_model, freeze_season_at_day):
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
    GridInfo still uses a flat-bottom ocean (uniform total_depth
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
    the ocean's top layer: it decays with depth as a two-band exponential
    (Paulson & Simpson 1977 / the SPEEDY-era "Jerlov water type I" fit),

        pen(z) = rpart * exp(z / efold1) + (1 - rpart) * exp(z / efold2)

    with `z <= 0` (Veros' `zw` is the *upper* interface of each T-cell, so
    `zw[-1] == 0` is the surface and `zw[0]` the ocean floor). `rpart` is
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
    flat-bottom (5200 m) ocean, where `pen` is ~1e-10 at the deepest
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


def _build_jcm_terrain_file(atm_grid, land_fraction, topography, output_file):
    """Build a JCM-canonical terrain file (dims (lon, lat), vars lsm/orog --
    see jcm.terrain.TerrainData.from_file) directly from `dm`'s in-memory
    "atm" domain -- `atm_grid` is `dm.domains["atm"].grid` (a
    StructuredQuadMesh, (nj, ni) = (lat, lon) logical shape) and
    `land_fraction`/`topography` are that domain's flat (nface,) arrays,
    row-major over the same (lat, lon) shape -- rather than re-reading the
    ERA5-derived landsea_mask_fraction_JCM_T{n}.nc file from disk a second
    time. Deliberately *not* using modify_jcm_terrain here: that only
    supports idealized synthetic geographies (aquaplanet, double_drake,
    ...), not real ERA5 topography.
    """
    nj, ni = atm_grid.shape  # (lat, lon)
    lat_deg = np.asarray(atm_grid.face_lat).reshape(nj, ni)[:, 0] * 180.0 / np.pi
    lon_deg = np.asarray(atm_grid.face_lon).reshape(nj, ni)[0, :] * 180.0 / np.pi
    lsm = np.asarray(land_fraction).reshape(nj, ni)
    orog = np.asarray(topography).reshape(nj, ni)

    out_ds = xr.Dataset(
        data_vars=dict(
            lsm=(("lon", "lat"), lsm.T),
            orog=(("lon", "lat"), orog.T),
        ),
        coords=dict(
            lon=("lon", lon_deg),
            lat=("lat", lat_deg),
        ),
        attrs=dict(grid_type="gaussian"),
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(output_file)
    return output_file


def build_model(
    *,
    truncation_number,
    start_datetime,
    coupling_timestep,
    calendar,
    veros_dt_mom=3600.0,
    veros_dt_tracer=3600.0,
    jcm_dt=1800.0,
    number_of_ocean_layers=None,
    debug_mode=False,
    freeze_season_at_day=None,
    grid_folder: str,
    landsea_mask_threshold:float = 0.01,
    ocean_surface_thickness: float = 10.0,
    enable_shortwave_penetration: bool = True,
    shortwave_rpart: float = 0.58,
    shortwave_efold1: float = 0.35,
    shortwave_efold2: float = 23.0,
):
    """Build the coupled JCM + Veros + SlabOceanModel system.

    Returns the assembled `Coupler` together with the underlying component
    models and supporting objects (e.g. `ocn_model`, needed by Veros'
    checkpoint loader), so that callers can build exactly the same model.

    `freeze_season_at_day`: if not `None`, the number of days after
    `start_datetime` at which to freeze the atmosphere's seasonal (solar and
    climatological-forcing) cycle -- e.g. `0` pins it at `start_datetime`
    itself. The diurnal cycle keeps advancing normally. See `_freeze_season`.

    `enable_shortwave_penetration`: let the absorbed solar flux be deposited
    over a finite depth range instead of entirely in the ocean's top layer,
    using the two-band exponential profile of
    `_shortwave_penetration_divergence` (`shortwave_rpart` /
    `shortwave_efold1` / `shortwave_efold2` are its parameters). Without it,
    100% of the daytime insolation lands in a single layer, so the implied
    top-layer heating rate scales as 1/`ocean_surface_thickness` -- which is
    why thinning the surface layer below ~30 m used to blow the coupled model
    up within a couple of days. With penetration on, roughly the top
    `shortwave_efold2` metres share the flux regardless of how the grid is
    cut, so the surface layer can be made thin without the heating rate
    running away with it.
    """

    grid_folder = Path(grid_folder)
    print(f"grid_folder = {str(grid_folder):s}")

    dm = get_domain_master(
        grid_folder,
        truncation_number = truncation_number,
        mask_threshold = landsea_mask_threshold,
    )
    coords = get_speedy_coords(spectral_truncation=truncation_number)

    # Real (ERA5-derived) topography, on JCM's own native (unrotated) grid --
    # not the rotated ocean grid's mask file. JCM's Coriolis parameter is
    # `2*omega*sin(lat)` computed directly from its own native latitude
    # (dinosaur.primitive_equations.PrimitiveEquations.coriolis_parameter,
    # hard-coded, no override hook), so terrain must be geographically
    # consistent with *that* latitude, not the ocean's rotated one.
    jcm_terrain_file = _build_jcm_terrain_file(
        dm.domains["atm"].grid,
        dm.domains["atm"].land_fraction,
        dm.domains["atm"].topography,
        grid_folder / f"terrain_JCM_T{truncation_number:d}.nc",
    )
    terrain = TerrainData.from_file(jcm_terrain_file, coords=coords)
    
    # Create JCM
    atm_model = jcm.model.Model(
        coords=coords,
        start_date=start_datetime,
        terrain=terrain,
        time_step=jcm_dt/60.0,
        calendar=calendar,
    )
    JCM.make_jem_compatible(atm_model, coupling_timestep=coupling_timestep)
    if freeze_season_at_day is not None:
        _freeze_season(atm_model, freeze_season_at_day)
    atm_D2_nodal_shape = atm_model.coords.nodal_shape[1:]
    
    # Longitude/latitude (in degrees) of the atmosphere's nodal grid, used by
    # `report_first_nonfinite` in `interaction` to locate non-finite values.
    lon_deg = atm_model.coords.horizontal.longitudes * 180 / jnp.pi
    lat_deg = atm_model.coords.horizontal.latitudes * 180 / jnp.pi
    
    # Create Veros
    # First need to remove `output_veros.*.nc` files, otherwise veros complains.
    for f in glob.glob("output_veros.*.nc"):
        print(f"Deleting: {f}")
        os.remove(f)
    
    # `generateVerosSetup` needs raw file access (native_lat/native_lon axis
    # metadata that isn't part of a `Domain`'s in-memory land_fraction/mask
    # arrays), so it's handed the ocean grid/mask files directly rather than
    # going through `dm` -- these are the same files `dm` itself loaded the
    # "ocn"/"exchange" domains' grid from, in `get_domain_master`.
    ocn_model = generateVerosSetup(
        scrip_grid_file = grid_folder / "RotatedGaussianLatLon.SCRIP.nc",
        landsea_mask_file = grid_folder / "landsea_mask_fraction_RotatedGaussianLatLon.nc",
        landsea_mask_threshold = landsea_mask_threshold,
        dt_mom=veros_dt_mom,
        dt_tracer=veros_dt_tracer,
        # 30-layer geometric-stretch grid, same 5200 m total depth as the
        # previous hand-tuned 15-layer profile. `ocean_surface_thickness`
        # is exposed as its own parameter -- a 10 m surface layer (5x
        # smaller heat capacity for the same heat flux) used to trigger an
        # immediate atm humidity explosion (~day 2) regardless of ocean
        # timestep (tested 3600/1800/900s, all failed identically). The
        # suspected cause was that the *entire* absorbed solar flux was
        # being dumped into that one top layer, making the daytime heating
        # rate scale as 1/`ocean_surface_thickness`; see
        # `enable_shortwave_penetration` and
        # `_shortwave_penetration_divergence`, which spread it over the
        # top ~`shortwave_efold2` metres instead.
        ddz=_geometric_stretched_ddz(n_layers=30, surface_thickness=ocean_surface_thickness, total_depth=5200.0)[:number_of_ocean_layers],
    )()
    ocn_model.setup()
    Veros.make_jem_compatible(ocn_model, coupling_timestep=coupling_timestep)

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
        print("Shortwave penetration disabled: all solar absorbed in the top ocean layer.")

    # Reference heat capacity / density used to turn W/m^2 into K m/s -- the
    # same pair `jem/components/veros_component.py` uses for
    # `forc_temp_surface`, pulled out of the traced `interaction` below.
    ocn_cp_0 = 3991.86795711963
    ocn_rho_0 = float(ocn_model.state.settings.rho_0)

    if not ocn_model.state.settings.enable_tempsalt_sources:
        raise RuntimeError(
            "Veros setting `enable_tempsalt_sources` must be True for the "
            "penetrating-shortwave `temp_source` term to be allocated and "
            "applied (see veros_case_setup.py)."
        )
    
    # Create Slab Ocean model, standing in for a land model -- on the same
    # rotated ocean grid as "ocn"/"exchange"/"lnd" (not JCM's own atm
    # grid), so its forcing/state flow through dm's atm<->exchange
    # transformations exactly like `ocn`'s do. `dm`'s "lnd" domain
    # land_fraction is a flat (nface,) array, row-major over the mesh's
    # (lat, lon) shape, rather than JEM's (lon, lat) -- reshape and
    # transpose before handing it to SlabGrid.
    #
    # `SlabOceanModel` is a genuine *ocean* model being repurposed here as
    # a stand-in land model: internally it only actively simulates cells
    # where `grid.binary_mask == mask_value` (0, "ocean" -- see
    # `jem/components/slab/slab_ocean_model/slab_ocean_model.py`'s
    # `ocn_idx`/`nonocn_idx`); every other cell ("land", binary_mask != 0)
    # gets hard-reset to a constant `default_land_surface_temperature`
    # (288.15 K) on init *and every step* (`slab_ocean_model.py:218-220,309`)
    # -- it never actually evolves there. We want fakelnd to actively
    # simulate the true *land* cells (that's the whole point of it standing
    # in for a land model), so its own fractional_mask must be *inverted*
    # relative to the real land fraction: true land needs a LOW value here
    # (so SlabOceanModel treats it as its own "ocean", i.e. active), and
    # true ocean needs a HIGH value (so it gets frozen/ignored there,
    # since Veros already owns that domain). `threshold` is inverted to
    # match, so the binary_mask flips over at the same real-world
    # land_fraction cutoff as everywhere else (`landsea_mask_threshold`,
    # also used for Veros's own kbot classification and dm's "ocn"/"lnd"
    # domain split in `get_domain_master`) -- keeping all land/ocean
    # classifications consistent with each other is what fixed the earlier
    # coastal-contamination bug; passing fakelnd its fractional_mask
    # un-inverted was a second, separate instance of the same class of
    # mistake (a cell counted as "ocean" by whichever model owns it having
    # no real data from that model), just showing up as a permanently
    # frozen land temperature instead of a regridding artifact.
    lnd_grid_shape = dm.domains["lnd"].grid.shape  # (nj, ni) = (lat, lon)
    lnd_fractional_mask = jnp.asarray(dm.domains["lnd"].land_fraction).reshape(lnd_grid_shape).T  # (lat, lon) -> (lon, lat)

    fakelnd_grid = generate_slab_grid_from_scrip(
        str(grid_folder / "RotatedGaussianLatLon.SCRIP.nc"),
        fractional_mask=1.0 - lnd_fractional_mask,
        threshold=1.0 - landsea_mask_threshold,
    )
    fakelnd_model = SlabOceanModel(
        grid=fakelnd_grid,
        start_datetime=start_datetime,
        timestep=coupling_timestep / one_second,
        forcing_method=None,
        calendar=calendar,
    )

    # Creating Flux and Scalar Exchange between Components
    #
    # Here we demonstrate the flexibility of JEM: you do not have to use
    # the `BasicMapper` that JEM provides. You can define your own mapping
    # function. In this example, we simply define a function `interaction`
    # as below.


    # Note: Remember to return the `coupled_carry` at the end.
    def interaction(coupled_carry):
        atm = coupled_carry["atm"]
        ocn = coupled_carry["ocn"]
        fakelnd = coupled_carry["fakelnd"]



        # ===== compute wind stress begin =====
        # Tien-Yiao's ad-hoc way to compute wind stress.
        # This conveniently demonstrates how the flux computation can be
        # its own function or module.
        drag_coefficient = 1e-3  # dimensionless
        air_density = 1.22  # kg / m^3
        # `dm.transform_vector` rotates (u0, v0) from JCM's own grid-local
        # frame to geographic east/north (a no-op here, since JCM's grid is
        # unrotated), regrids each geographic component atm -> exchange,
        # then rotates into the ocean grid's local (rotated) frame using
        # its per-cell cos_angle/sin_angle.
        wind_x, wind_y = dm.transform_vector(
            "atm", "exchange", "conserve",
            atm["derived"].physics["_surface_flux"].u0,
            atm["derived"].physics["_surface_flux"].v0,
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

        heat_flux = dm.transform_scalar("atm", "exchange", "conserve", atm["derived"].total_heat_flux)
        freshwater_flux = dm.transform_scalar("atm", "exchange", "conserve", atm["derived"].total_freshwater_flux)

        # ===== simple "swamp" sea ice mask begin =====
        # Ported from `veros/setups/global_1deg/global_1deg.py`: once the
        # ocean surface reaches the freezing point, sea ice insulates it
        # from further heat/freshwater exchange with the atmosphere. Flux
        # that would warm the ocean (melt) is always let through. `heat_flux`
        # is upward-positive (see `jem.components.Veros`), so a flux that
        # warms the ocean is negative.
        freezing_point_K = 273.15 - 1.8
        not_frozen = ocn["derived"].sea_surface_temperature > freezing_point_K
        would_warm = heat_flux < 0
        ice_free = jnp.logical_or(not_frozen, would_warm)
        heat_flux = heat_flux * ice_free
        freshwater_flux = freshwater_flux * ice_free
        # ===== simple "swamp" sea ice mask end =====

        # ===== penetrating shortwave radiation begin =====
        # JCM's sea-surface net heat flux is
        #   hfluxn[:, :, 1] = rsds*(1 - alb_s) + rlds - (rlus + shf + alhc*evap)
        # (jcm/physics/surface/speedy_surface_flux.py), and `total_heat_flux`
        # above is exactly -hfluxn[:, :, 1] (JCM's `hfluxn` only has a land
        # and a sea component, so the `[:, :, 2]` index used by
        # `jem/components/jcm_component.py` clamps to the sea one). So the
        # absorbed solar part of the flux the ocean is already receiving is
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
        solar_flux = dm.transform_scalar(
            "atm", "exchange", "conserve",
            atm["derived"].physics["_shortwave_rad"].rsds
            * (1.0 - atm["derived"].physics["_mod_radcon"].alb_s),
        )
        # Sea ice cut `heat_flux` off above, which means the solar flux never
        # entered the ocean at all there -- so there is nothing to
        # redistribute either. Same masking as global_1deg's `ice[..., None]`.
        solar_flux = solar_flux * ice_free

        # W/m^2 * 1/m / (J/kg/K) / (kg/m^3) = K/s. `tempsalt_sources`
        # multiplies by `maskT` itself, so no mask is needed here.
        vs_ocn_forcing = ocn["state"].variables
        with vs_ocn_forcing.unlock():
            vs_ocn_forcing.temp_source = update(
                vs_ocn_forcing.temp_source,
                at[2:-2, 2:-2, :],
                solar_flux[:, :, None]
                * divpen_shortwave[None, None, :]
                / ocn_cp_0
                / ocn_rho_0,
            )
        # ===== penetrating shortwave radiation end =====

        # Mapping
        ocn["forcing"].surface_taux = surface_taux
        ocn["forcing"].surface_tauy = surface_tauy
        ocn["forcing"].heat_flux = heat_flux
        ocn["forcing"].freshwater_flux = freshwater_flux
        ocn["forcing"].wind_x = wind_x
        ocn["forcing"].wind_y = wind_y
        fakelnd["forcing"].total_heat_flux = heat_flux
        fakelnd["state"].sea_surface_temperature = jnp.clip(
            fakelnd["state"].sea_surface_temperature,
            200.0, #273.15 - 50.0,
            273.15 + 30.0,
        )

        # JCM does its own land/sea blend internally, using its own
        # continuous fmask at its native resolution:
        #   tsfc = sea_surface_temperature + fmask*(stl_am - sea_surface_temperature)
        # (jcm/physics/surface/speedy_surface_flux.py). It needs
        # `sea_surface_temperature`/`stl_am` as two separate, unblended
        # fields to do that blend correctly -- pre-merging them into one
        # composite here (as an earlier version of this code did) makes
        # them identical, which silently zeroes out JCM's own fmask term
        # (fmask*(x-x) = 0) and replaces its fine-resolution blend with
        # whatever crude blending happened during our own coarser
        # exchange-grid compositing instead. So each is regridded
        # separately here, straight off "exchange" (the same grid `ocn`/
        # `fakelnd` are already on, so no domain-specific masking is
        # needed -- masking either to zero here, rather than letting
        # JCM's own fmask blend handle the coastline, would inject a
        # literal 0 Kelvin discontinuity at every coastal cell).
        atm["forcing"].sea_surface_temperature = jnp.clip(
            dm.transform_scalar("exchange", "atm", "bilinear", ocn["derived"].sea_surface_temperature),
            250.0,
            310.0, 
        )
        atm["forcing"].stl_am = dm.transform_scalar("exchange", "atm", "bilinear", fakelnd["state"].sea_surface_temperature)


        if debug_mode:

            print("Notice: debug_mode is on.")

            # Debug: report wind speed (atm), ocean velocity, and stratification (ocn) extremes
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
                    # The crash was first seen in the ocean state (u, v, Nsqr
                    # all NaN while the atmosphere was still finite), so check
                    # the ocean's prognostic fields too (3D: x, y, depth).
                    report_first_nonfinite("ocn.state.temp", vs_ocn.temp[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.salt", vs_ocn.salt[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.u", vs_ocn.u[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.v", vs_ocn.v[..., vs_ocn.tau])
                    report_first_nonfinite("ocn.state.Nsqr", vs_ocn.Nsqr[..., vs_ocn.tau])
                    # jax.debug.breakpoint()  # disabled: no TTY in SLURM batch jobs -> JaxRuntimeError
                jax.lax.cond(all_finite, true_fn, false_fn, x)

            breakpoint_if_nonfinite(coupled_carry) 


        return coupled_carry

    model = Coupler(
        components=dict(
            atm=atm_model,
            ocn=ocn_model,
            fakelnd=fakelnd_model,
        ),
        mappers=dict(mapper=interaction),
    )

    config = dict(
        #terrain=terrain,
        #modified_jcm_terrain_file=modified_jcm_terrain_file,
        coords=coords,
        workflow=["mapper", "atm", "ocn", "fakelnd"],
    )

    return model, config
