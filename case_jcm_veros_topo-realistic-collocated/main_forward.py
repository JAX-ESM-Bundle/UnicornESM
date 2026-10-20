# Couple JCM and Veros using JAX-ESM (JEM).

import sys
from pathlib import Path

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)  # only cache if compile took >1s

#jax.config.update("jax_enable_x64", False) 
import jcm

import jax_datetime as jdt

import jem

from model_setup import build_model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--total-simulation-days", type=int, help="Total time of simulation in days", default=10)
parser.add_argument("--simulation-interval-days", type=int, help="Simulation interval in days", default=5)
parser.add_argument("--simulation-name", type=str, help="Simulation name for output", default="default")
parser.add_argument("--truncation-number", type=int, help="Truncation number", default=31)
parser.add_argument("--jcm-timestep-min", type=int, help="JCM timestep in minutes", default=30)
parser.add_argument("--veros-timestep-min", type=int, help="Veros timestep in minutes", default=30)
parser.add_argument("--do-not-average-time", action="store_true", help="Do not average time dimension for each interval.")
parser.add_argument("--debug-mode", action="store_true", help="Turn on debug mode. Detect NaN and enter breakpoint.")
parser.add_argument("--grid-folder", type=str, help="Grid folder containing grid, land-sea mask, and regrid weightings.", required=True)
parser.add_argument("--coupling-timestep-hours", type=float, help="Interval between atm<->ocn coupling exchanges, in hours.", default=24.0)
parser.add_argument("--checkpoint-interval", type=int, help="Save a checkpoint every this many simulation intervals.", default=1)
parser.add_argument("--ocean-surface-thickness", type=float, help="Thickness (m) of the ocean's top vertical layer, in the geometric-stretched ddz grid built by model_setup._geometric_stretched_ddz.", default=30.0)
parser.add_argument("--no-shortwave-penetration", action="store_true", help="Absorb the whole solar flux in the ocean's top layer instead of spreading it over depth with the two-band exponential profile of model_setup._shortwave_penetration_divergence. Mainly a control knob for stability experiments.")
args = parser.parse_args()

print(f"jcm library is located at: {jcm.__file__}")
print(f"jem library is located at: {jem.__file__}")
import dinosaur
print(f"dinosaur library is located at: {dinosaur.__file__}")

# Check available devices
print(f"Available devices: {jax.devices()}")
print(f"Number of devices: {len(jax.devices())}")

# Configurations
truncation_number = args.truncation_number
start_datetime = jdt.to_datetime("2000-01-01")
coupling_timestep = jdt.to_timedelta(int(args.coupling_timestep_hours * 3600), "second")

output_dir = (Path(f"output_T{truncation_number}") / args.simulation_name).resolve()
output_dir.mkdir(exist_ok=True, parents=True)

# Build the coupled JCM + Veros + fake-land system. Packaged as a function
# in `model_setup.py` so that other scripts (e.g. a jax.grad sensitivity
# experiment) can build exactly the same model.
model, config = build_model(
    truncation_number=truncation_number,
    start_datetime=start_datetime,
    coupling_timestep=coupling_timestep,
    debug_mode=args.debug_mode,
    jcm_dt = args.jcm_timestep_min * 60.0,
    veros_dt_mom=args.veros_timestep_min * 60,
    veros_dt_tracer=args.veros_timestep_min * 60,
    grid_folder = args.grid_folder,
    ocean_surface_thickness = args.ocean_surface_thickness,
    enable_shortwave_penetration = not args.no_shortwave_penetration,
)

print(f"Model info: {model!r}")

atm_model = model.components["atm"].model
transport_scheme = type(atm_model.dycore._primitive).__name__
print(f"jcm dycore transport scheme: {transport_scheme}")

# Run Coupled Model
#
# `jem.run_chunked` owns the loop: it integrates one simulation interval
# ("chunk") at a time, writes `<component>-<first coupled step>.nc` for each,
# runs JCM's blow-up diagnostics on the chunk, and keeps a single restart
# directory at `output_dir / "checkpoint"`, rewritten at each checkpoint
# interval. Re-running this same command resumes from that checkpoint. A
# chunk that fails the health check is still written (for debugging) but not
# checkpointed, and the run stops there.
result = jem.run_chunked(
    model,
    total_time=f"{args.total_simulation_days:d} days",
    chunk=f"{args.simulation_interval_days:d} days",
    output_dir=output_dir,
    output_averages=not args.do_not_average_time,
    checkpoint_interval=f"{args.checkpoint_interval * args.simulation_interval_days:d} days",
)

if not result.completed:
    print(f"Error: the health check stopped the run at coupled step {result.steps_completed:d}.")
    print(f"Last report: {result.reports[-1] if result.reports else '(none)'}")
    print("Exit program.")
    sys.exit(1)

print(f"Program ends.")
