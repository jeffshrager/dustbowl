# CLAUDE.md — Great Plains CA project notes

## Environment
- Use `conda activate test` (at `/Users/jeffshrager/opt/anaconda3/envs/test`)
- Python interpreter: `python` (not `python3`) inside that env
- numpy 2.2.6, matplotlib 3.10.0

## Running the simulation
```bash
source /Users/jeffshrager/opt/anaconda3/etc/profile.d/conda.sh && conda activate test
python plains_ca.py --steps 200
python plains_ca.py --steps 200 --animate
```

## Key files
- `plains_ca.py` — single-file simulation + visualization
- `results/` — all output lands here, auto-created, timestamped `YYYYMMDDhhmm_<kind>.<ext>`
- `README.md` — user-facing documentation
- `CLAUDE.md` — this file
- `slides/great_plains_ca.tex` — Beamer slide deck (12 slides); compiles with pdflatex or Overleaf
- `slides/gen_images.py` — runs simulation, saves figs to `results/` with timestamps, copies to `slides/fig_*.png`
- `slides/fig_*.png` — current slide figures (copies from results/; regenerate with gen_images.py)

## Architecture notes
- All parameters in `PARAMS` dict at top of file
- `GreatPlainsCA` class holds all state as numpy arrays
- Flux is fully vectorized (no Python loops over cells)
- `_results_path(kind, ext)` generates timestamped output paths
- Visualization uses a 2×3 GridSpec; `ax_map` is a named dict (not a flat array)
  - Keys: `resources`, `wildlife`, `humans`, `infra`, `history`, `history_r`
  - `history_r` is a pre-created twinx — do NOT call `twinx()` inside the update loop

## Results traceability
- Every CLI run dumps `results/YYYYMMDDhhmm_params.json` (full PARAMS dict) before running
- Slide figure runs dump `results/YYYYMMDDhhmm_slides_params.json`
- Slide PNGs in results/ are named `YYYYMMDDhhmm_slides_<kind>.png`; copied to `slides/fig_<kind>.png`
- To regenerate slide figures: `python slides/gen_images.py` from project root

## Code comments status
- Block comments added to all sections (imports, PARAMS, terrain, init, resource
  dynamics, _flux, population updates, simulation control, visualization, CLI)
- TODO: finer-grained inline comments still needed inside the `_flux` method body
  (agreed with user to do this in a future session)

## Known model behavior
- Default params cause bison boom/crash in first ~100 steps (wildlife outgrazes resources)
- Tuning levers: lower `wl_consumption`, raise `res_regen_rate`, or lower `wl_carrying_cap`

## Conventions
- Row 0 = North, Row 59 = South; Col 0 = West, Col 79 = East
- `mask` array: 1.0 = valid plains cell, 0.0 = mountain (impassable)
- Always multiply state arrays by `mask` after updating to keep mountains clean
