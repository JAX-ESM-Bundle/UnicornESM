# The "fake land" of this case: JEM's SlabOceanModel standing in for a land
# model, with the two sea-water assumptions SlabOceanModel makes removed.

import jax.numpy as jnp

from jem import constants
from jem.components import SlabOceanModel
from jem.components.slab.base import MASKED_SURFACE_TEMPERATURE
from jem.components.slab.slab_ocean_model.slab_ocean_model import OceanDerived
from jem.utils.idealized_distribution import positive_cosine_cubic_latitude_squared

# Idealized initial surface temperature: `base + range * profile(latitude)`,
# i.e. 27 degC at the equator falling to 0 degC poleward of 60 degrees.
INITIAL_TEMPERATURE_BASE = 273.15
INITIAL_TEMPERATURE_RANGE = 27.0


class FakeLandModel(SlabOceanModel):
    """A slab of `mixed_layer_depth` metres of water, used as a land surface.

    `SlabOceanModel` is a model of *sea water*: it floors its temperature at
    the seawater freezing point (271.35 K) both in `initialize` and at the end
    of every `step`, diverting the deficit into `ice_frazil_melt_energy` for a
    sea-ice model to pick up. A land surface has no such floor -- continental
    interiors go far below freezing in winter -- and this case couples no
    sea-ice model to take the diverted energy, so with the floor in place that
    energy would simply leave the system. This subclass therefore integrates
    the same slab energy equation without it, and starts from the wider
    idealized profile above instead of `SlabOceanParameters.initial_sst`'s.

    Only `forcing_method == "none"` is supported: the relaxation and Q-flux
    terms of the parent's `step` are not reproduced here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.params.forcing_method != "none":
            raise ValueError(
                "FakeLandModel only supports forcing_method='none'; got "
                f"{self.params.forcing_method!r}."
            )

    def initialize(self, params=None):
        carry = super().initialize(params)
        active = self._ocean_cells(carry["params"])
        temperature = INITIAL_TEMPERATURE_BASE + INITIAL_TEMPERATURE_RANGE * (
            positive_cosine_cubic_latitude_squared(self.grid.latitude_radian)
        )
        temperature = jnp.where(active, temperature, MASKED_SURFACE_TEMPERATURE)
        return dict(
            carry,
            state=carry["state"].replace(sea_surface_temperature=temperature),
        )

    def step(self, carry, time):
        params = carry["params"]
        state = carry["state"]
        forcing = carry["forcing"]

        active = self._ocean_cells(params)
        mixed_layer_depth = self._mixed_layer_depth(params)
        heat_capacity = (
            constants.ocean_density
            * constants.ocean_specific_heat_capacity
            * mixed_layer_depth
        )
        # `total_heat_flux` is upward-positive, hence the negation.
        temperature = state.sea_surface_temperature + time.dt / heat_capacity * (
            -forcing.total_heat_flux
        )
        temperature = jnp.where(active, temperature, MASKED_SURFACE_TEMPERATURE)

        new_state = state.replace(sea_surface_temperature=temperature)
        new_derived = OceanDerived.zeros(
            self.grid.shape,
            mixed_layer_depth=mixed_layer_depth,
            effective_total_heat_flux=forcing.total_heat_flux,
        )
        new_carry = {
            "params": params,
            "state": new_state,
            "forcing": forcing,
            "derived": new_derived,
        }
        diagnostics = {
            "state": new_state,
            "forcing": forcing,
            "derived": new_derived,
        }
        return new_carry, diagnostics
