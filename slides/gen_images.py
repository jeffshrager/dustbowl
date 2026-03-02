#!/usr/bin/env python
"""
Generate figures for the Great Plains CA slide deck.

Figures are saved to results/ with full timestamps (so the run is traceable),
then copied into slides/ under fixed names that the .tex file references.

Run from the project root:
    conda activate test
    python slides/gen_images.py

Produces in results/:
    YYYYMMDDhhmm_params.json
    YYYYMMDDhhmm_slides_terrain.png
    YYYYMMDDhhmm_slides_t0.png
    YYYYMMDDhhmm_slides_t200.png
    YYYYMMDDhhmm_slides_history.png

Copies into slides/ as:
    fig_terrain.png
    fig_t0.png
    fig_t200.png
    fig_history.png
"""

import json
import os
import shutil
import sys

# Allow importing plains_ca from the parent directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colormaps

from plains_ca import GreatPlainsCA, _results_path, \
                      _PLAINS, _MOUNTAIN, _RIVER, _WOODLAND

SLIDES_DIR = os.path.dirname(os.path.abspath(__file__))


def save_and_copy(fig, slide_name, results_label):
    """Save to results/ with timestamp, then copy to slides/ with fixed name."""
    results_path = _results_path(f"slides_{results_label}", "png")
    fig.savefig(results_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    slides_path = os.path.join(SLIDES_DIR, slide_name)
    shutil.copy(results_path, slides_path)
    print(f'  {results_path}  →  {slides_path}')


# ── Terrain ───────────────────────────────────────────────────────────────────

def make_terrain(model):
    t = model.terrain
    rows, cols = t.shape

    COLORS = {
        _PLAINS:   [0.93, 0.87, 0.62],
        _MOUNTAIN: [0.45, 0.45, 0.45],
        _RIVER:    [0.27, 0.55, 0.80],
        _WOODLAND: [0.42, 0.62, 0.35],
    }

    img = np.zeros((rows, cols, 3))
    for code, color in COLORS.items():
        img[t == code] = color

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(img, origin='upper', aspect='auto', interpolation='nearest')
    ax.set_title('Terrain', fontsize=13, fontweight='bold')
    ax.set_xlabel('West  →  East', fontsize=9)
    ax.set_ylabel('North  →  South', fontsize=9)

    patches = [
        mpatches.Patch(color=COLORS[_MOUNTAIN], label='Rocky Mountains (impassable)'),
        mpatches.Patch(color=COLORS[_RIVER],    label='Rivers: Platte, Republican, Arkansas, Red'),
        mpatches.Patch(color=COLORS[_WOODLAND], label='Woodland edge'),
        mpatches.Patch(color=COLORS[_PLAINS],   label='Grassland plains'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=8, framealpha=0.9)
    save_and_copy(fig, 'fig_terrain.png', 'terrain')


# ── State panels (2×2) ────────────────────────────────────────────────────────

def make_state(model, state, slide_name, results_label, title):
    p    = model.p
    mask = model.mask

    panels = [
        ('Resources',          state['resources'], 'YlGn',   0, p['res_cap_river']),
        ('Wildlife  (bison)',  state['wildlife'],  'YlOrBr', 0, p['wl_carrying_cap']),
        ('Humans  (settlers)', state['humans'],    'Reds',   0, p['hu_carrying_cap']),
        ('Infrastructure',     state['infra'],     'Purples',0, p['infra_max']),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.subplots_adjust(hspace=0.38, wspace=0.32)
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for ax, (label, data, cmap_name, vmin, vmax) in zip(axes.flat, panels):
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = colormaps.get_cmap(cmap_name)
        rgba = cmap(norm(data))
        rgba[mask == 0] = [0.45, 0.45, 0.45, 1.0]
        ax.imshow(rgba, origin='upper', aspect='auto', interpolation='nearest')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        total = data[mask > 0].sum()
        ax.set_title(f'{label}  (\u03a3={total:.0f})', fontsize=9, fontweight='bold')
        ax.set_xlabel('W \u2192 E', fontsize=7)
        ax.set_ylabel('N \u2192 S', fontsize=7)
        ax.tick_params(labelsize=6)

    save_and_copy(fig, slide_name, results_label)


# ── History time series ───────────────────────────────────────────────────────

def make_history(model):
    p       = model.p
    history = model.history
    m_mask  = model.mask

    steps     = [s['t']                            for s in history]
    wl_totals = [s['wildlife'][m_mask > 0].sum()   for s in history]
    hu_totals = [s['humans'][m_mask > 0].sum()     for s in history]
    res_means = [s['resources'][m_mask > 0].mean() for s in history]

    fig, ax_h = plt.subplots(figsize=(9, 4))
    ax_h.plot(steps, wl_totals, color='#b5651d', lw=2.0, label='Wildlife (bison)')
    ax_h.plot(steps, hu_totals, color='#c0392b', lw=2.0, label='Humans (settlers)')
    ax_h.set_ylabel('Population total', fontsize=10)
    ax_h.set_xlabel('Step', fontsize=10)
    ax_h.legend(loc='upper left', fontsize=9)
    ax_h.set_title('Population & Resource History  (200 steps)',
                   fontsize=12, fontweight='bold')

    ax_r = ax_h.twinx()
    ax_r.plot(steps, res_means, color='#27ae60', lw=1.8, ls='--',
              label='Mean resources')
    ax_r.set_ylim(0, p['res_cap_river'])
    ax_r.set_ylabel('Mean resources', fontsize=10, color='#27ae60')
    ax_r.tick_params(axis='y', colors='#27ae60')
    ax_r.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    save_and_copy(fig, 'fig_history.png', 'history')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Running simulation (200 steps)...')
    model    = GreatPlainsCA()
    state_t0 = model.get_state()
    model.run(200, record_every=5)
    state_t200 = model.get_state()

    # Dump parameters so this slide run is traceable
    params_path = _results_path("slides_params", "json")
    with open(params_path, "w") as f:
        json.dump(model.p, f, indent=2)
    print(f'  params    → {params_path}')

    print('Generating figures...')
    make_terrain(model)
    make_state(model, state_t0,   'fig_t0.png',   't0',   'Initial State  (t = 0)')
    make_state(model, state_t200, 'fig_t200.png', 't200', 'After 200 Steps  (t = 200)')
    make_history(model)
    print('Done.')
