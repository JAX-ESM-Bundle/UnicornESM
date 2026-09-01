# Ocean instability: thin surface layer — status notes (2026-09-01)

Branch: `dev/hosing-experiments-double-drake`. Case dir: `case_jcm_veros_topo-realistic/`.

Companion to `old/case_jcm_veros_topo-realistic/CFL_INVESTIGATION_NOTES.md`
(2026-08-21), which investigated a *different trigger* (open ocean adjacent to
the rotated grid's non-cyclic y-wall) and reached a root cause that this note
builds on directly. **Read that file first** — the ruled-out table there is
still valid and is not repeated here.

## TL;DR

- `--ocean-surface-thickness` below ~30 m blows the coupled model up on
  simulated **day 1–2**, regardless of ocean timestep.
- The failure is a **momentum runaway in `u`**, not a thermodynamic one. The
  `enable_nan_checks` report naming `integrate_eke` is a **red herring**:
  `eke ~ u²` and `K_diss_h ~ u⁴`, so EKE is simply the first field to overflow
  float64.
- Proximate cause is arithmetic: `du/dt = τ/(ρ·Δz)`. A 10 m top layer
  accelerates **3× faster** than 30 m for the same wind stress, so the
  wind-driven surface jet reaches ~1.15 m/s in five hours instead of
  ~0.2–0.3 m/s. Every CFL-type limit in the model then scales with that
  velocity.
- Adding penetrating shortwave radiation (this session's change) **does not
  fix it** — 10 m explodes identically with it on and off.
- **Unresolved regression:** 30 m *with* penetration exploded at batch 12
  (~2.4 yr), vs batch 575 (~115 yr) for the previous 30 m run without it.
  See "Open items".

## Configuration

| | |
|---|---|
| Grid | `grid_res-3deg`, rotated-pole Gaussian, 120×60 ocean cells |
| Vertical | 30-layer geometric stretch, 5200 m total, flat bottom |
| Ocean Δt | `dt_mom = dt_tracer = 3600 s` (unless stated) |
| Atmos Δt | 1800 s, JCM T31 |
| Coupling | 24 h, workflow `mapper, atm, ocn, fakelnd` |
| Ocean init | cold start, near-unstratified; salinity uniform 35 psu (`deltaS_psu = 0`) |
| EOS | `eq_of_state_type = 1` → **linear** (`veros/core/density/linear_eq.py`) |

Note: `CFL_INVESTIGATION_NOTES.md` labels `eq_of_state_type=1` as "Vallis 2008
nonlinear". That is a mislabel — type 1 dispatches to `linear_eq_of_state_rho`.
Does not affect that note's conclusions, but matters if you recompute `N²`.

## Symptom

`Error: batch=0, attempt=1/1: model exploded (non-finite humidity)`.

Ocean NaNs first (output day 1), atmosphere follows (day 2). The forcing handed
to the ocean at the failing step is **finite and physically reasonable** —
heat flux −362…+278 W/m², |τ| ≤ 0.7 N/m² — so this is not bad coupled forcing.

## Evidence

### 1. The runaway is in `u`, and EKE is a passenger

`settings.enable_nan_checks = True` (already implemented in this Veros fork,
`veros/debug_tools.py`) reports per sub-step. First non-finite is
`integrate_eke` → `eke`, `deke`, **with Inf as well as NaN**. But `u` is already
astronomically large by then:

| itt | max\|u\| (m/s) | max eke |
|---|---|---|
| 26 | 0.371 | 1.2e-09 |
| 30 | 1.153 | 6.5e-03 |
| 34 | 2.701 | 1.9e+01 |
| 35 | **171.6** | 3.3e+01 |
| 37 | 9.68e+02 | 5.4e+03 |
| 39 | 8.74e+05 | 1.7e+07 |
| 41 | 1.28e+18 | 7.6e+28 |
| 43 | 5.23e+66 | 1.2e+124 |
| 45 | 1.01e+262 | 1.1e+308 → Inf |

From itt=35 on, `u` **squares every step**:
`u_{n+1} ≈ (Δt/Δx)·u_n²`. Check: `Δt/Δx = 3600/333 km = 0.0108`, and
`0.0108·(8.74e5)² = 8.2e9` vs `9.4e9` observed. Classic nonlinear advective
runaway — this is the *terminal* stage, not the trigger.

This independently reproduces the 2026-08-21 finding that `enable_eke=False`
does not fix anything: EKE was never the driver.

### 2. Momentum tendency breakdown (10 m)

Instrumented via `after_timestep` in `veros_case_setup.py` (see "Reproducing").

| itt | max\|u\| | du_cor | du_adv | du_mix | p_hydro | psi | max rho |
|---|---|---|---|---|---|---|---|
| 26 | 0.371 | 3.2e-05 | 1.0e-06 | 1.4e-06 | 23.42 | 0.072 | 1.891 |
| 30 | 1.153 | 9.0e-05 | 8.8e-06 | 5.7e-05 | 23.45 | 0.157 | 1.891 |
| 34 | 2.701 | 3.3e-04 | 8.0e-05 | **4.8e-02** | 23.45 | 0.246 | 1.891 |
| 35 | 171.6 | 1.6e-03 | 4.2e-02 | 1.9e-02 | 23.45 | 0.266 | 1.891 |
| 37 | 968 | 8.4e-03 | 1.50 | 0.14 | 24.80 | 60.8 | 5.06 |

`du_mix` (vertical friction/mixing) grows **3.4e4×** between itt 26 and 34,
while `du_cor` grows 10× and `du_adv` 80×. This matches the 2026-08-21
momentum trace, which also found Coriolis and `du_mix` dominant with advection
negligible.

### 3. TKE feedback is the amplifier

| | 10 m | 30 m |
|---|---|---|
| surface \|u\| | rises to 1.15+ m/s, no saturation | saturates 0.20–0.30 m/s |
| `tke` | 2e-6 → 0.93 → 45 | steady ~1.6e-3 |
| `kappaM` | 2e-4 (floor) → 2.3 → 11 → **100 (cap)** | steady ~0.3 |

Confirmed by ablation: with `enable_tke = False` at 10 m, `u` **turns over and
decays** (1.24 → 1.19 → 1.11 → 1.01) instead of running away at itt≈33, and the
model survives ~25 days instead of 1.5.

### 4. CFL numbers

⚠️ **Correction to earlier in-session analysis:** `vs.dxt` is already converted
to metres at `veros/core/numerics.py:66`. An initial diagnostic multiplied by
`settings.degtom` a second time, understating horizontal CFL by 111,195×.
Corrected values:

| | itt 31 | itt 34 (last clean) | itt 35 (blowup) |
|---|---|---|---|
| 10 m horizontal `uΔt/Δx` | 0.076 | 0.125 | **3.55** |
| 10 m vertical `wΔt/Δz` | 0.074 | 0.264 | **3.22** |
| 30 m horizontal | 0.017 | 0.017 | 0.017 |
| 30 m vertical | 0.020 | 0.015 | 0.015 |

Both cross 1 in the same step; vertical is closer at the last clean step.

### 5. Timestep insensitivity — rules out a simple CFL limit

| Veros Δt | 10 m result |
|---|---|
| 3600 s | explodes, output day 1 |
| 1800 s | explodes, output day 1 |
| 900 s | explodes, output day 1 |
| 300 s | **untested** (killed by a tool timeout) |

All fail at the same *simulated* time, not the same step count. A pure CFL or
explicit-Coriolis `f·Δt` limit would have been cured by this. The growth rate is
set by the physics, not by Δt. Same conclusion as 2026-08-21 (900 s vs 3600 s).

### 6. Failure locations

| Run | grid lat | true lat | local Δx |
|---|---|---|---|
| 10 m, default | −58.5 … −76.5 | −64 … −68 | 78–174 km |
| 10 m, TKE off | +82.5 | +77.1 | 43.5 km |

Always high \|f\|, and on this rotated grid also where meridians converge.

## Mechanism

1. Wind stress switches on at the first coupling exchange. A 10 m top layer
   gives `τ/(ρΔz)·Δt = 0.7/(1024·10)·3600 = 0.246 m/s per step` — matching the
   observed 0.37 → 0.65 → 0.89 → 1.05 increments exactly.
2. That jet sits on a near-unstratified cold-start column ⇒ large shear over
   10 m ⇒ TKE production explodes ⇒ `kappaM` saturates at its 100 m²/s cap ⇒
   `du_mix` runs away.
3. `w` rises with the enhanced mixing; vertical CFL crosses 1.
4. Nonlinear horizontal advection takes over as `u_{n+1} ∝ u_n²` → overflow.
5. `eke ~ u²` overflows first and gets the blame in the NaN report.

At 30 m every step of this chain is 3–9× weaker (`u` 3× smaller, shear
`∂u/∂z ~ u/Δz` 9× smaller, TKE production ~ shear² ~80× smaller) and the loop
never closes.

## Relationship to the 2026-08-21 root cause

That investigation's decisive experiment (job 1863389: zeroing the hydrostatic
pressure gradient → 280 days vs 2 days, ~140× improvement) identified the
**hydrostatic pressure gradient acting on an under-resolved Rossby radius** as
the dominant driver, with `L_rossby/dx ≈ 0.78` at the culprit cell.

**That conclusion is not contradicted by anything here.** An earlier statement
in this session — that the PGF was exonerated because `p_hydro`, `psi` and `rho`
stay flat through the onset — was **methodologically weak and should not be
relied on**: those diagnostics are *global maxima*, dominated by the deep
column everywhere in the domain, and are insensitive to the local near-surface
gradient at the failing cell. A term's global magnitude is not evidence about
its causal role. The ablation experiment is much stronger evidence than the
magnitude trace, and it points at the PGF.

New measurement, cold-start initial condition, this 3° grid (linear EOS,
WKB `C_rossby = ∫√max(N²,0) dz / π`, `L_rossby = C_rossby/|f|`):

- `C_rossby = 2.093 m/s` (uniform — the cold start is horizontally uniform)
- `L_rossby/dx`: **min 0.071, median 0.118**, max 31.2
- **96.5 % of ocean cells have `L_rossby/dx < 1`**
- at the 10 m failure cell (lon 22.5, lat −61.5): Δx 159 km, `L_rossby` 15.9 km,
  **ratio 0.100**
- at the TKE-off failure cell (lon 58.5, lat 82.5): Δx 43.5 km, `L_rossby`
  14.7 km, **ratio 0.338**

**Refinement of the 2026-08-21 picture:** on this 3° grid the resolvability
criterion is violated across essentially the whole ocean, not just near the
pole — yet 30 m ran 115 simulated years. So `L_rossby/dx < 1` is a *necessary
background condition, not a sufficient one*. What decides whether the model
survives is the **amplitude of the forcing exciting the unresolved mode**. The
surface layer thickness sets that amplitude directly, via `τ/(ρΔz)`. This is
why thinning the top layer, which changes no resolution parameter at all, moves
the model from stable-for-a-century to dead-in-a-day.

## Ruled out this session

| Hypothesis | Test | Result |
|---|---|---|
| Penetrating shortwave absorption would fix it | 10 m with and without `--no-shortwave-penetration` | **No effect** — identical failure at batch 0 |
| Bad coupled forcing (NaN/spike from regridding) | Inspected `heat_flux`, `τ`, freshwater at the failing step | All finite and reasonable |
| Simple CFL / step-size limit | Δt 3600 / 1800 / 900 s | All fail at same simulated time |
| EKE closure is the driver | Magnitude trace; `eke ~ u²` overflow ordering | Passenger, not driver (matches 2026-08-21 `enable_eke=False`) |
| TKE closure is the sole driver | `enable_tke = False` at 10 m | Delays failure 1.5 d → 25 d, does not cure. **Amplifier, not root cause** |

## Open items

1. **Suspected regression — most likely GPU non-determinism, not the
   penetration change.** Three data points on 30 m:

   | Job | Config | Result |
   |---|---|---|
   | `sbatch.64.1960787.out` | 30 m, penetration **off** | batch 575 ≈ 115 yr |
   | 1979449 | 30 m, penetration **on** | **exploded batch 12 ≈ 2.4 yr** |
   | 1990208 | 30 m, penetration **on** (identical to 1979449) | **batch 18+ clean** |

   Two runs of the *identical* config differing by >50% in survival is exactly
   the behaviour `CFL_INVESTIGATION_NOTES.md` documented: **GPU runs on this
   cluster are not deterministic** (`XLA_FLAGS=--xla_gpu_deterministic_ops=true`
   is commented out in `submit.slurm`), with >100× swings observed on identical
   configs. So the batch-12 death is probably noise, not a deterministic
   regression from penetrating shortwave. **Not yet conclusive** — needs either
   1990208 to run well past batch 575, or repeat trials with deterministic ops
   enabled. Set that XLA flag before attempting any further single-run A/B on
   this cluster.

   (A separate penetration-off control was attempted but died on a filesystem
   race against the concurrently-running production job — no result. Run
   controls with `--simulation-name` pointing outside `output_T31/`, or not at
   all while a production job is live.)
2. **Δt = 300 s at 10 m** — the one untested point in the timestep sweep.
3. **Does a thin layer work at all?** If a 10 m surface layer is wanted (for
   diurnal cycle / air-sea coupling fidelity), the options are the same three
   from 2026-08-21 — refine the grid, add a scale-aware eddy parameterization
   keyed off `L_rossby/dx`, or accept the resolution limit — plus one specific
   to this trigger: **cap the wind-driven acceleration**, e.g. distribute τ
   over a fixed mixed-layer depth rather than over `dzt[-1]`, so `du/dt` stops
   scaling as `1/Δz`. Not implemented, not tested.
4. `kappaM` hitting its 100 m²/s cap during the runaway is worth a look —
   whether the cap itself is participating, or merely recording saturation.

## Reproducing the diagnostics

All instrumentation below was **reverted** after use; `veros_case_setup.py`
differs from git HEAD only by `enable_tempsalt_sources = True`.

Per-sub-step NaN localisation (already in the Veros fork, off by default):

```python
# veros_case_setup.py, in set_parameter
settings.enable_nan_checks = True   # prints stage + variable on first non-finite
```

Per-timestep field tracing — add to the (empty) `after_timestep` hook in
`veros_case_setup.py`; `jax.debug.print` works inside the jitted scan:

```python
import jax as _jax, jax.numpy as _jnp
_jax.debug.print(
    "DIAG itt={itt} maxu={u:.3e} du_cor={c:.3e} du_adv={a:.3e} du_mix={m:.3e} "
    "p_hydro={p:.3e} psi={s:.3e} tke={t:.3e} kappaM={k:.3e}",
    itt=vs.itt, u=_jnp.max(_jnp.abs(vs.u[..., vs.tau])),
    c=_jnp.max(_jnp.abs(vs.du_cor)), a=_jnp.max(_jnp.abs(vs.du_adv)),
    m=_jnp.max(_jnp.abs(vs.du_mix)), p=_jnp.max(_jnp.abs(vs.p_hydro)),
    s=_jnp.max(_jnp.abs(vs.psi[..., vs.tau])),
    t=_jnp.max(vs.tke[..., vs.tau]), k=_jnp.max(vs.kappaM),
)
```

Beware: wrap reads in `_jnp.where(_jnp.isfinite(x), x, 0.0)` once NaNs appear,
or every max reads `nan`. And **`vs.dxt`/`dyt` are already in metres** — do not
multiply by `degtom`.

Fast reproduction (~2 min each on the login-node A40, no SLURM needed):

```bash
python3 main_forward.py --total-simulation-days 2 --simulation-interval-days 2 \
  --checkpoint-interval 1000 --truncation-number 31 --simulation-name diag \
  --jcm-timestep-min 30 --veros-timestep-min 60 --max-rerun-attempts 0 \
  --grid-folder grid_res-3deg --ocean-surface-thickness 10 \
  --explode-log explode.diag.log
```

`--no-shortwave-penetration` (added in `main_forward.py` this session) toggles
the solar penetration profile for A/B tests.

## Key file locations

- `model_setup.py` — `_shortwave_penetration_divergence()`, `build_model()`;
  `interaction` mapper writes `temp_source`.
- `veros_case_setup.py` — `GridInfo`, `generateVerosSetup()`, `after_timestep`
  hook (empty; the instrumentation point).
- `main_forward.py` — entry point, `--ocean-surface-thickness`,
  `--no-shortwave-penetration`.
- `veros-jittable/veros/core/momentum.py` — `tend_coriolisf` (Coriolis is
  **fully explicit at `tau`**), advection, friction.
- `veros-jittable/veros/core/external/solve_pressure.py` — `prepare_forcing`
  builds `p_hydro` and adds the PGF to `du`/`dv`; AB2(+eps) time extrapolation.
- `veros-jittable/veros/core/eke.py` — EKE closure, computes `L_rossby` every step.
- `veros-jittable/veros/debug_tools.py` — `detect_nan_in_state`.
