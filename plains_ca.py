#!/usr/bin/env python
"""
Great Plains Population Flux  —  Cellular Automaton Model
==========================================================
Simulates the movement of human (settler) and wildlife (bison) populations
across the Great Plains over discrete time steps.

Flux drivers:
  1. Resource gradients   – populations drift toward grass/water-rich cells
  2. Carrying-capacity pressure – overcrowded cells shed population
  3. Climate events       – stochastic drought / fire reduces local resources
  4. Social attractors    – towns / rivers pull humans disproportionately

Grid convention
  Row 0 = North  →  Row ROWS-1 = South
  Col 0 = West   →  Col COLS-1 = East

Run:
  conda activate test
  python plains_ca.py [--steps N] [--animate]

Outputs go to ./results/YYYYMMDDhhmm_<kind>.<ext> automatically.
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
# Standard library:
#   argparse    — command-line argument parsing
#   os          — filesystem operations (creating the results/ directory)
#   datetime    — generating the YYYYMMDDhhmm timestamp in output filenames
#
# Third-party (requires the 'test' conda env):
#   numpy       — all grid state is stored as 2-D float arrays; every update
#                 rule is written as vectorized array arithmetic (no Python
#                 loops over cells) for performance
#   matplotlib  — figure layout, colormaps, and animation export
#     colormaps — accessed directly to avoid the deprecated plt.cm.get_cmap()
#     FuncAnimation — drives the frame-by-frame GIF export

import argparse
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation

# ─── Results directory helpers ────────────────────────────────────────────────
# All output files are written to a single results/ subdirectory so they don't
# clutter the source tree.  Every filename is prefixed with a YYYYMMDDhhmm
# timestamp so successive runs never overwrite each other.  The 'kind' argument
# is a short descriptive label (e.g. "snapshot", "animation") and 'ext' is the
# file extension ("png", "gif").  The directory is created on first use.

def _results_path(kind: str, ext: str, results_dir: str = "results") -> str:
    """Return a timestamped path inside results_dir, creating it if needed.

    Format: <results_dir>/YYYYMMDDhhmm_<kind>.<ext>
    Example: results/202602281435_snapshot.png
    """
    os.makedirs(results_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d%H%M")
    return os.path.join(results_dir, f"{stamp}_{kind}.{ext}")


# ─── Default Parameters ───────────────────────────────────────────────────────
# All tunable values live here in one place.  Passing a partial dict to
# GreatPlainsCA() overrides individual entries; everything else falls back to
# these defaults.  Parameters are grouped by concern:
#
#   Grid        — overall dimensions.  Rows run N-S (row 0 = northernmost),
#                 cols run W-E (col 0 = westernmost).  The grid represents
#                 the Great Plains as a rectangle ~60 units tall by 80 wide.
#
#   Terrain     — the Rocky Mountains form a vertical barrier along the western
#                 edge: every cell in the leftmost ~12 columns is impassable.
#                 The column count varies row-by-row (±2) to give the boundary
#                 a jagged, realistic silhouette — so "mountain_cols" is an
#                 average width, not a hard wall.  Rivers run horizontally
#                 (W-E), so each river is specified by a row index (its N-S
#                 position) plus a half-width in rows; "river_rows: [10,22,35,48]"
#                 places four rivers at those latitudes, loosely corresponding
#                 to the Platte, Republican, Arkansas, and Red rivers.  The
#                 woodland edge is a fixed band of columns on the eastern side.
#
#   Resources   — each cell holds a scalar in [0, res_cap] representing
#                 combined grass/water abundance.  The cap varies by terrain
#                 (rivers support more, woodland less).  Each step resources
#                 recover via logistic growth toward the cap at res_regen_rate.
#
#   Climate     — independent probabilities, severities, and spatial radii for
#                 drought and fire events each step.
#
#   Wildlife /
#   Humans      — each cell also holds a population count (continuous, not
#                 integer agents).  "carrying_cap" is the population level above
#                 which a cell starts actively pushing residents outward — it is
#                 not a hard ceiling, just the threshold where overcrowding
#                 pressure begins.  Below it, only the gentler base_flux drives
#                 movement.  Humans additionally weight existing infrastructure
#                 alongside resources when deciding which neighbor to move to.
#
#   Infrastructure — how quickly towns grow under human presence and decay
#                    without it.
#
# The four _PLAINS/_MOUNTAIN/_RIVER/_WOODLAND integer codes below are used as
# cell labels in the terrain array; keeping them module-level constants (rather
# than, say, an Enum) keeps the numpy comparisons simple.

PARAMS: dict = {
    # Grid
    "rows": 60,
    "cols": 80,

    # Terrain
    "mountain_cols": 12,          # western cols that form the Rocky Mountain barrier
    "woodland_cols": 7,           # eastern cols — woodland / prairie edge
    "river_rows": [10, 22, 35, 48],   # N→S positions: Platte, Republican, Arkansas, Red-ish
    "river_width": 1,             # half-width in rows

    # Resources
    "res_regen_rate": 0.08,       # logistic growth rate per step
    "res_cap_plains":  1.00,
    "res_cap_river":   1.40,      # rivers support more
    "res_cap_woodland": 0.75,
    "res_init_noise":  0.35,      # random depletion at startup

    # Climate events (per step probabilities)
    "drought_prob":     0.025,
    "drought_severity": 0.55,
    "drought_radius":   9,

    "fire_prob":        0.012,
    "fire_severity":    0.70,
    "fire_radius":      5,

    # Wildlife (bison)
    "wl_carrying_cap":   150.0,
    "wl_base_flux":       0.12,   # fraction that "wants" to move each step
    "wl_pressure_rate":   0.40,   # extra outflow per unit above carrying cap
    "wl_drift_strength":  2.5,    # softmax sharpness toward resources
    "wl_resource_weight": 1.0,
    "wl_consumption":     0.07,   # resources consumed per unit wildlife
    "wl_birth_rate":      0.05,
    "wl_death_rate":      0.02,
    "wl_init_total":      3000.0,

    # Humans (settlers)
    "hu_carrying_cap":   30.0,
    "hu_base_flux":       0.05,
    "hu_pressure_rate":   0.30,
    "hu_drift_strength":  3.5,
    "hu_resource_weight": 0.55,
    "hu_infra_weight":    0.45,
    "hu_consumption":     0.10,
    "hu_birth_rate":      0.03,
    "hu_death_rate":      0.015,
    "hu_init_total":      200.0,

    # Infrastructure
    "infra_growth_rate": 0.012,
    "infra_decay_rate":  0.003,
    "infra_max":         1.0,

    "seed": 42,
}

# Terrain type codes
_PLAINS   = 0
_MOUNTAIN = 1
_RIVER    = 2
_WOODLAND = 3


# ─── Cellular Automaton ───────────────────────────────────────────────────────

class GreatPlainsCA:
    # GreatPlainsCA is the simulation's single stateful object.  Construction
    # proceeds in two phases:
    #
    #   1. Static geography — terrain, mask, and resource caps are built once
    #      and never change.  The mask is a float array (1.0 = passable,
    #      0.0 = mountain) derived directly from the terrain; multiplying any
    #      state array by the mask at the end of an update guarantees that
    #      mountains stay permanently empty.
    #
    #   2. Dynamic state — resources, wildlife, humans, and infrastructure are
    #      initialised to starting conditions and will evolve each step.
    #
    # self.t counts elapsed steps.  self.history accumulates snapshots for the
    # time-series panel and animation export.

    def __init__(self, params: dict | None = None):
        self.p = {**PARAMS, **(params or {})}
        self.rng = np.random.default_rng(self.p["seed"])
        self.t = 0

        self.terrain      = self._build_terrain()
        self.mask         = (self.terrain != _MOUNTAIN).astype(float)
        self.resource_cap = self._build_resource_cap()

        self.resources = self._init_resources()
        self.wildlife  = self._init_wildlife()
        self.humans    = self._init_humans()
        self.infra     = self._init_infra()

        self.history: list[dict] = []

    # ── Terrain ──────────────────────────────────────────────────────────────
    # _build_terrain fills a (rows, cols) integer array with terrain type codes.
    # Placement priority is mountains first, then woodland, then rivers — this
    # matters because rivers are laid down last and the inner loop explicitly
    # skips mountain cells, so the Rocky Mountain barrier is never overwritten.
    # The jagged mountain boundary is produced by adding a small random offset
    # (±2 cols) independently for each row.
    #
    # _build_resource_cap translates terrain codes into per-cell resource ceilings
    # using np.select (a vectorized if/elif/else over arrays).  Mountains get 0
    # (the mask will zero them out anyway); rivers get the highest cap since they
    # provide permanent water; woodland is slightly below the plains default.
    # Multiplying by self.mask at the end is a belt-and-suspenders guarantee that
    # mountain cells are always zero regardless of the select logic.

    def _build_terrain(self) -> np.ndarray:
        p = self.p
        rows, cols = p["rows"], p["cols"]
        t = np.full((rows, cols), _PLAINS, dtype=np.int8)

        # Mountains – slightly jagged western boundary
        for r in range(rows):
            edge = p["mountain_cols"] + int(self.rng.integers(-2, 3))
            edge = max(3, edge)
            t[r, :edge] = _MOUNTAIN

        # Woodland edge (east) — only in non-mountain cells
        wd = p["woodland_cols"]
        t[:, -wd:] = np.where(t[:, -wd:] == _MOUNTAIN, _MOUNTAIN, _WOODLAND)

        # Rivers — run E-W through plains (don't enter mountains)
        for rr in p["river_rows"]:
            hw = p["river_width"]
            for r in range(max(0, rr - hw), min(rows, rr + hw + 1)):
                for c in range(cols):
                    if t[r, c] != _MOUNTAIN:
                        t[r, c] = _RIVER

        return t

    def _build_resource_cap(self) -> np.ndarray:
        p = self.p
        cap = np.select(
            [self.terrain == _MOUNTAIN,
             self.terrain == _RIVER,
             self.terrain == _WOODLAND],
            [0.0,
             p["res_cap_river"],
             p["res_cap_woodland"]],
            default=p["res_cap_plains"],
        )
        return cap * self.mask

    # ── Initialization ───────────────────────────────────────────────────────
    # Each method sets up one dynamic state layer at t=0.
    #
    # _init_resources  — starts every cell near its terrain cap, then subtracts
    #                    a uniform random amount (up to res_init_noise) to produce
    #                    spatial variability in initial grass/water abundance.
    #
    # _init_wildlife   — places bison as a 2-D Gaussian blob centred slightly east
    #                    of the grid midpoint (reflecting the historical core of the
    #                    Great Plains herd).  The blob is normalised so the total
    #                    across all valid cells equals wl_init_total.
    #
    # _init_humans     — distributes settlers evenly across the eastern woodland
    #                    columns only, representing a population that has just
    #                    reached the edge of the plains from the east and has not
    #                    yet begun moving westward.
    #
    # _init_infra      — seeds a small non-zero infrastructure value along rivers
    #                    and the woodland edge, representing pre-existing trails,
    #                    fords, and trading posts before the simulation begins.

    def _init_resources(self) -> np.ndarray:
        r = self.resource_cap.copy()
        noise = self.rng.uniform(0, self.p["res_init_noise"], r.shape)
        return np.clip(r - noise, 0.0, self.resource_cap) * self.mask

    def _init_wildlife(self) -> np.ndarray:
        p = self.p
        rows, cols = p["rows"], p["cols"]
        # Bison concentrated in central plains (historical distribution)
        ri, ci = np.mgrid[0:rows, 0:cols]
        wl = np.exp(-((ri - rows * 0.5) / 14) ** 2
                    - ((ci - cols * 0.55) / 22) ** 2)
        wl *= self.mask
        total = wl.sum()
        if total > 0:
            wl *= p["wl_init_total"] / total
        return wl

    def _init_humans(self) -> np.ndarray:
        p = self.p
        rows, cols = p["rows"], p["cols"]
        # Settlers arrive from the east
        hu = np.zeros((rows, cols))
        wd = p["woodland_cols"]
        eastern = (self.terrain[:, -wd:] != _MOUNTAIN)
        count = eastern.sum()
        if count > 0:
            hu[:, -wd:][eastern] = p["hu_init_total"] / count
        return hu * self.mask

    def _init_infra(self) -> np.ndarray:
        # Small seed infrastructure along rivers and eastern edge
        river_seed   = (self.terrain == _RIVER).astype(float) * 0.10
        woodland_seed = (self.terrain == _WOODLAND).astype(float) * 0.15
        return (river_seed + woodland_seed) * self.mask

    # ── Update Rules ─────────────────────────────────────────────────────────
    # ── Resource dynamics ────────────────────────────────────────────────────
    # _regenerate_resources applies logistic growth to every cell each step:
    #   growth = rate * r * (1 - r/cap)
    # This produces fast recovery when resources are low and slows as they
    # approach the terrain cap — matching how grasslands actually recover.
    # The np.where guard prevents division by zero on mountain cells (cap=0).
    #
    # _climate_event independently rolls the dice for drought and fire each step.
    # If an event fires, a random valid (non-mountain) cell is chosen as the
    # epicentre and a Gaussian damage footprint is computed over the whole grid.
    # Resources are multiplied by (1 - damage), so the epicentre takes the full
    # hit (damage ≈ severity) and impact fades smoothly with distance.  Fire has
    # a smaller radius but higher severity than drought; both can occur in the
    # same step.

    def _regenerate_resources(self) -> None:
        r = self.resources
        cap = self.resource_cap
        growth = self.p["res_regen_rate"] * r * (1.0 - r / np.where(cap > 0, cap, 1.0))
        self.resources = np.clip(r + growth, 0.0, cap) * self.mask

    def _climate_event(self) -> None:
        p = self.p
        rows, cols = p["rows"], p["cols"]
        events = [
            (p["drought_prob"], p["drought_severity"], p["drought_radius"]),
            (p["fire_prob"],    p["fire_severity"],    p["fire_radius"]),
        ]
        ri, ci = np.mgrid[0:rows, 0:cols]
        valid_cells = np.argwhere(self.mask > 0)

        for prob, severity, radius in events:
            if self.rng.random() < prob and len(valid_cells):
                er, ec = valid_cells[self.rng.integers(len(valid_cells))]
                dist_sq = ((ri - er) / radius) ** 2 + ((ci - ec) / radius) ** 2
                damage = severity * np.exp(-2.0 * dist_sq)
                self.resources = np.maximum(0.0, self.resources * (1.0 - damage))

    def _flux(
        self,
        pop: np.ndarray,
        attractiveness: np.ndarray,
        base_flux: float,
        pressure_rate: float,
        carrying_cap: float,
        drift_strength: float,
    ) -> np.ndarray:
        """
        Compute net population flux across the 2-D grid (4-connected).

        Each cell emits a fraction of its population to valid neighbors,
        weighted by an exponential (softmax) over neighbor attractiveness.
        Overcrowded cells emit extra population proportional to the excess.

        Returns delta array (same shape as pop).
        """
        mask = self.mask

        # Attractiveness of each cell, zeroed on mountains
        att = attractiveness * mask

        # Neighbor attractiveness arrays (boundary stays 0 → unattractive)
        att_N = np.zeros_like(att); att_N[1:,  :]  = att[:-1, :]   # cell to north
        att_S = np.zeros_like(att); att_S[:-1, :]  = att[1:,  :]   # cell to south
        att_E = np.zeros_like(att); att_E[:,  :-1] = att[:,  1:]    # cell to east
        att_W = np.zeros_like(att); att_W[:,   1:] = att[:, :-1]   # cell to west

        # Valid neighbor masks (1 where destination is a non-mountain in-bounds cell)
        valid_N = np.zeros_like(mask); valid_N[1:,  :]  = mask[:-1, :]
        valid_S = np.zeros_like(mask); valid_S[:-1, :]  = mask[1:,  :]
        valid_E = np.zeros_like(mask); valid_E[:,  :-1] = mask[:,  1:]
        valid_W = np.zeros_like(mask); valid_W[:,   1:] = mask[:, :-1]

        # Softmax weights (exponential drift) — zeroed for invalid neighbors
        w_N = np.exp(drift_strength * att_N) * valid_N
        w_S = np.exp(drift_strength * att_S) * valid_S
        w_E = np.exp(drift_strength * att_E) * valid_E
        w_W = np.exp(drift_strength * att_W) * valid_W

        w_out = w_N + w_S + w_E + w_W                         # total outward weight
        has_exit = w_out > 1e-10                               # cells with ≥1 valid neighbor

        # Normalised fractions (safe divide)
        denom = np.where(has_exit, w_out, 1.0)
        f_N = w_N / denom
        f_S = w_S / denom
        f_E = w_E / denom
        f_W = w_W / denom

        # Total outflow: base flux + overcrowding pressure
        pressure = np.maximum(0.0, pop - carrying_cap)
        total_out = np.minimum(pop, pop * base_flux + pressure * pressure_rate)
        total_out *= mask * has_exit.astype(float)

        out_N = total_out * f_N   # leaving (i,j) → (i-1,j)
        out_S = total_out * f_S   # leaving (i,j) → (i+1,j)
        out_E = total_out * f_E   # leaving (i,j) → (i,j+1)
        out_W = total_out * f_W   # leaving (i,j) → (i,j-1)

        # Inflow from each direction
        in_S = np.zeros_like(pop); in_S[:-1, :]  = out_N[1:,  :]   # from south going north
        in_N = np.zeros_like(pop); in_N[1:,  :]  = out_S[:-1, :]   # from north going south
        in_W = np.zeros_like(pop); in_W[:,   1:] = out_E[:,  :-1]  # from west going east
        in_E = np.zeros_like(pop); in_E[:,  :-1] = out_W[:,   1:]  # from east going west

        inflow  = in_N + in_S + in_E + in_W
        outflow = out_N + out_S + out_E + out_W

        return (inflow - outflow) * mask

    def _update_wildlife(self) -> None:
        p = self.p
        wl = self.wildlife

        attractiveness = self.resources * p["wl_resource_weight"]
        delta = self._flux(wl, attractiveness,
                           p["wl_base_flux"], p["wl_pressure_rate"],
                           p["wl_carrying_cap"], p["wl_drift_strength"])

        # Logistic birth/death
        birth = p["wl_birth_rate"] * wl * (1.0 - wl / (p["wl_carrying_cap"] + 1e-10))
        death = p["wl_death_rate"] * wl

        # Grazing reduces resources
        grazing = np.minimum(self.resources, wl * p["wl_consumption"])
        self.resources = np.maximum(0.0, self.resources - grazing)

        self.wildlife = np.maximum(0.0, wl + delta + birth - death) * self.mask

    def _update_humans(self) -> None:
        p = self.p
        hu = self.humans

        attractiveness = (p["hu_resource_weight"] * self.resources
                          + p["hu_infra_weight"]   * self.infra)
        delta = self._flux(hu, attractiveness,
                           p["hu_base_flux"], p["hu_pressure_rate"],
                           p["hu_carrying_cap"], p["hu_drift_strength"])

        birth = p["hu_birth_rate"] * hu * (1.0 - hu / (p["hu_carrying_cap"] + 1e-10))
        death = p["hu_death_rate"] * hu

        consumed = np.minimum(self.resources, hu * p["hu_consumption"])
        self.resources = np.maximum(0.0, self.resources - consumed)

        self.humans = np.maximum(0.0, hu + delta + birth - death) * self.mask

    def _update_infra(self) -> None:
        p = self.p
        hu_fraction = self.humans / (p["hu_carrying_cap"] + 1e-10)
        growth = p["infra_growth_rate"] * hu_fraction
        decay  = p["infra_decay_rate"]  * self.infra * (1.0 - hu_fraction)
        self.infra = np.clip(self.infra + growth - decay, 0.0, p["infra_max"]) * self.mask

    def step(self) -> None:
        """Advance simulation by one time step."""
        self._regenerate_resources()
        self._climate_event()
        self._update_wildlife()
        self._update_humans()
        self._update_infra()
        self.t += 1

    def get_state(self) -> dict:
        return dict(
            t         = self.t,
            resources = self.resources.copy(),
            wildlife  = self.wildlife.copy(),
            humans    = self.humans.copy(),
            infra     = self.infra.copy(),
        )

    def run(self, n_steps: int, record_every: int = 5) -> list[dict]:
        """Run for n_steps, recording a snapshot every record_every steps."""
        self.history = [self.get_state()]
        for _ in range(n_steps):
            self.step()
            if self.t % record_every == 0:
                self.history.append(self.get_state())
        return self.history


# ─── Visualization ────────────────────────────────────────────────────────────

_MOUNTAIN_GREY = np.array([0.40, 0.40, 0.40])   # RGB for imshow overlay


def _to_rgba(data: np.ndarray, cmap_name: str,
             vmin: float, vmax: float, mask: np.ndarray) -> np.ndarray:
    """Convert a 2-D array to an RGBA image with mountains greyed out."""
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps.get_cmap(cmap_name)
    rgba = cmap(norm(data))                        # (rows, cols, 4)
    rgba[mask == 0, :3] = _MOUNTAIN_GREY
    rgba[mask == 0,  3] = 1.0
    return rgba


def _make_axes(fig: plt.Figure) -> dict:
    """Create the standard 2×3 GridSpec axis layout and return a named dict."""
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)
    ax_h = fig.add_subplot(gs[1, 1:])   # history panel spans last 2 cols
    return {
        "resources": fig.add_subplot(gs[0, 0]),
        "wildlife":  fig.add_subplot(gs[0, 1]),
        "humans":    fig.add_subplot(gs[0, 2]),
        "infra":     fig.add_subplot(gs[1, 0]),
        "history":   ax_h,
        "history_r": ax_h.twinx(),      # right y-axis for resources
    }


def plot_state(model: "GreatPlainsCA",
               state: dict | None = None,
               fig: plt.Figure | None = None,
               ax_map: dict | None = None,
               save_path: str | None = None):
    """
    Render the 2×3 panel figure:
      top row  — Resources | Wildlife | Humans
      bot row  — Infrastructure | History time-series (spans 2 cols)
    """
    if state is None:
        state = model.get_state()

    p    = model.p
    mask = model.mask

    if fig is None or ax_map is None:
        fig    = plt.figure(figsize=(18, 10))
        ax_map = _make_axes(fig)

    # ── Four spatial grid panels ──────────────────────────────────────────────
    grid_panels = [
        ("resources", "Resources",          state["resources"], "YlGn",   0, p["res_cap_river"]),
        ("wildlife",  "Wildlife  (bison)",  state["wildlife"],  "YlOrBr", 0, p["wl_carrying_cap"]),
        ("humans",    "Humans  (settlers)", state["humans"],    "Reds",   0, p["hu_carrying_cap"]),
        ("infra",     "Infrastructure",     state["infra"],     "Purples",0, p["infra_max"]),
    ]

    for key, title, data, cmap_name, vmin, vmax in grid_panels:
        ax = ax_map[key]
        ax.clear()
        rgba = _to_rgba(data, cmap_name, vmin, vmax, mask)
        ax.imshow(rgba, origin="upper", aspect="auto", interpolation="nearest")

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm   = plt.cm.ScalarMappable(cmap=colormaps.get_cmap(cmap_name), norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)

        total = data[mask > 0].sum()
        ax.set_title(f"{title}  (Σ={total:.0f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("West → East", fontsize=8)
        ax.set_ylabel("North → South", fontsize=8)
        ax.tick_params(labelsize=7)

    # ── History time-series panel ─────────────────────────────────────────────
    ax_h = ax_map["history"]
    ax_r = ax_map["history_r"]
    ax_h.clear()
    ax_r.clear()

    history = [s for s in model.history if s["t"] <= state["t"]]

    if len(history) > 1:
        steps     = [s["t"]                          for s in history]
        wl_totals = [s["wildlife"][mask > 0].sum()   for s in history]
        hu_totals = [s["humans"][mask > 0].sum()     for s in history]
        res_means = [s["resources"][mask > 0].mean() for s in history]

        ax_h.plot(steps, wl_totals, color="#b5651d", lw=1.8, label="Wildlife")
        ax_h.plot(steps, hu_totals, color="#c0392b", lw=1.8, label="Humans")
        ax_h.set_ylabel("Population total", fontsize=9)
        ax_h.legend(loc="upper left", fontsize=8)

        ax_r.plot(steps, res_means, color="#27ae60", lw=1.5, ls="--", label="Mean resources")
        ax_r.set_ylim(0, p["res_cap_river"])
        ax_r.set_ylabel("Mean resources", fontsize=9, color="#27ae60")
        ax_r.tick_params(axis="y", colors="#27ae60", labelsize=8)
        ax_r.legend(loc="upper right", fontsize=8)

        ax_h.axvline(state["t"], color="grey", lw=0.8, ls=":")

    ax_h.set_xlabel("Step", fontsize=9)
    ax_h.set_title("Population & Resource History", fontsize=10, fontweight="bold")
    ax_h.tick_params(labelsize=8)

    fig.suptitle(f"Great Plains CA  |  step {state['t']}", fontsize=13, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")

    return fig, ax_map


def animate_history(model: "GreatPlainsCA",
                    interval: int = 200,
                    save_path: str | None = None):
    """Animate recorded history with FuncAnimation."""
    history = model.history
    if not history:
        print("No history recorded — call model.run() first.")
        return None

    fig    = plt.figure(figsize=(18, 10))
    ax_map = _make_axes(fig)

    def _update(frame: int):
        plot_state(model, state=history[frame], fig=fig, ax_map=ax_map)
        return list(ax_map.values())

    anim = FuncAnimation(fig, _update, frames=len(history),
                         interval=interval, blit=False)

    if save_path:
        anim.save(save_path, writer="pillow", fps=5)
        print(f"Animation saved → {save_path}")
    else:
        plt.show()

    return anim


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Great Plains Population Flux CA")
    parser.add_argument("--steps",        type=int, default=200,
                        help="Number of simulation steps  (default 200)")
    parser.add_argument("--record-every", type=int, default=5,
                        help="Snapshot interval for animation  (default 5)")
    parser.add_argument("--animate",      action="store_true",
                        help="Save an animated GIF of recorded history")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    print("Initialising Great Plains CA …")
    model = GreatPlainsCA({"seed": args.seed})

    print(f"Running {args.steps} steps …")
    model.run(args.steps, record_every=args.record_every)

    print(f"\nDone.  Final state at step {model.t}:")
    print(f"  Wildlife total : {model.wildlife[model.mask > 0].sum():.1f}")
    print(f"  Human total    : {model.humans[model.mask  > 0].sum():.1f}")

    # Always save a timestamped static snapshot
    snap_path = _results_path("snapshot", "png")
    fig, ax_map = plot_state(model)
    fig.savefig(snap_path, dpi=100, bbox_inches="tight")
    print(f"  Snapshot  → {snap_path}")

    if args.animate:
        plt.close(fig)
        anim_path = _results_path("animation", "gif")
        animate_history(model, save_path=anim_path)
    else:
        plt.show()
