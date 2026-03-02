# Great Plains Population Flux — Cellular Automaton Model

Simulates the movement of human (settler) and wildlife (bison) populations
across the Great Plains of the US using a 2-D cellular automaton.

## Setup

```bash
conda activate test
```

numpy 2.2 and matplotlib 3.10 are required (both present in the `test` env).

## Running

```bash
python plains_ca.py                        # 200 steps, show final plot + save snapshot
python plains_ca.py --steps 500            # longer run
python plains_ca.py --steps 200 --animate  # also save animated GIF
python plains_ca.py --seed 7               # different random seed
python plains_ca.py --steps 200 --record-every 10  # coarser animation frames
```

All output goes to `./results/YYYYMMDDhhmm_<kind>.<ext>` automatically:

| File | When produced |
|---|---|
| `…_params.json` | Every run — full parameter dict so results are traceable to their inputs |
| `…_snapshot.png` | Every run |
| `…_animation.gif` | When `--animate` is given |
| `…_slides_params.json` | When `slides/gen_images.py` is run |
| `…_slides_<kind>.png` | When `slides/gen_images.py` is run |

## Grid

- **60 rows × 80 cols** — row 0 = North, row 59 = South; col 0 = West, col 79 = East
- Each cell represents roughly one spatial unit of the Great Plains

### Terrain layers (static)

| Terrain | Location | Effect |
|---|---|---|
| Rocky Mountains | Western ~12 cols, jagged boundary | Impassable barrier; zero resources |
| Rivers | 4 E-W bands (Platte, Republican, Arkansas, Red-ish) | Higher resource cap (1.4 vs 1.0) |
| Woodland edge | Eastern ~7 cols | Lower resource cap (0.75) |
| Grassland | Everything else | Standard resource cap (1.0) |

## Model

### State (updated every step)

| Layer | Description |
|---|---|
| `resources` | Grass/water scalar per cell; drives population movement |
| `wildlife` | Bison population density |
| `humans` | Settler population density |
| `infra` | Infrastructure (towns/trails); grows with human presence |

### Flux mechanics

Each step, every cell emits a fraction of its population to its 4 cardinal
neighbors. The split uses a **softmax over neighbor attractiveness**, so
populations drift up resource (and infrastructure) gradients. An additional
**pressure term** accelerates outflow when local density exceeds carrying
capacity. Mountains are excluded via a valid-neighbor mask.

### Update order per step

1. Resource logistic regrowth
2. Stochastic climate event (drought or fire — Gaussian damage footprint)
3. Wildlife flux + logistic birth/death + grazing (depletes resources)
4. Human flux + logistic birth/death + consumption (depletes resources)
5. Infrastructure growth/decay

### Flux drivers

| Driver | Wildlife | Humans |
|---|---|---|
| Resource gradient | ✓ | ✓ |
| Carrying-capacity pressure | ✓ | ✓ |
| Climate events (drought/fire) | indirect | indirect |
| Infrastructure attractors | — | ✓ |

## Visualization

The output figure is a **2×3 panel**:

```
[ Resources ] [ Wildlife ] [ Humans      ]
[ Infra     ] [ Population & Resource History (spans 2 cols) ]
```

The history panel shows wildlife total (orange), human total (red), and mean
resources (green dashed, right axis) up to the current step. Mountains are
rendered in grey across all spatial panels.

## Parameters

All tunable values live in the `PARAMS` dict at the top of `plains_ca.py`.
Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `wl_carrying_cap` | 150 | Max bison per cell before pressure kicks in |
| `hu_carrying_cap` | 30 | Max humans per cell |
| `res_regen_rate` | 0.08 | Logistic resource recovery rate per step |
| `wl_consumption` | 0.07 | Resources consumed per unit wildlife per step |
| `hu_consumption` | 0.10 | Resources consumed per unit human per step |
| `drought_prob` | 0.025 | Per-step probability of a drought event |
| `fire_prob` | 0.012 | Per-step probability of a fire event |
| `wl_drift_strength` | 2.5 | Softmax sharpness for wildlife gradient following |
| `hu_drift_strength` | 3.5 | Softmax sharpness for human gradient following |

## Known dynamics

With default parameters, bison grow exponentially in the first ~100 steps and
crash the resource base before self-limiting. Tuning `wl_consumption` down or
`res_regen_rate` up produces a stable boom/bust cycle instead.
