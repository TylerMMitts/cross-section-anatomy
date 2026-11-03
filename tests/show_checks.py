"""Pictures of check_numbers.py's synthetic sections: each case whole + zoomed on the edit, outlines coloured by what
the measuring code counts them as (cortex / epidermis / not counted / air space / excluded by a sector), excluded
arcs as grey wedges, vessels red, stele shaded. Title = the case's checks, FAIL in red.

  .venv/bin/python tests/show_checks.py <out_dir of check_numbers.py>    -> <out_dir>/checks_sheet.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P          # noqa: E402
from check_numbers import at, EDGES   # noqa: E402

COL = {"cortex": (1.0, 0.85, 0.1), "epidermis": (0.6, 0.6, 0.6), "unassigned": (0.85, 0.2, 0.85),
       "air_space": (0.1, 0.9, 0.95), "cortex_excluded": (0.45, 0.4, 0.2), "epidermis_excluded": (0.3, 0.3, 0.3),
       "stele": (0.35, 0.55, 1.0)}
WEDGE = (0.3, 0.3, 0.32)
# where to zoom: (radius, degrees, half-width px)
ZOOM = {"00_base": ((EDGES[0] + EDGES[-1]) / 2, 45, 200),
        "01_delete_cell": (EDGES[4], 115, 110), "02_add_missed_cell": (EDGES[4], 135, 110),
        "03_merge_two_cells": (EDGES[4], 135, 110), "04_split_cell": (EDGES[5], 225, 110),
        "05_exclude_quarter_sector": (300, 225, 330), "06_old_dropped_quarter": (300, 225, 330),
        "06b_edge_cells_by_centre": (300, 13, 200), "06c_arc_across_zero": (300, 0, 200),
        "06d_overlapping_arcs": (300, 45, 300), "06e_narrow_kept_sector": (300, 20, 250),
        "06f_drag_then_restore": (300, 175, 330),
        "07_remove_vessel": (105, 135, 90), "08_add_vessel": (105, 45, 90), "09_stele_takes_layer1": (160, 45, 110),
        "10_root_without_epidermis": (455, 45, 90), "11_tag_aerenchyma": (EDGES[4], 30, 140),
        "12_lateral_root_outside": (500, 75, 110), "13_lateral_root_inside_outline": (500, 75, 110)}


def picture(npz):
    cells, region, mx, _ = P.load_state(npz)
    centre = P.centre_of(region, P.load_centre(npz))
    arcs = P.load_sectors(npz)
    props = P.exclude_sectors(P.derive(cells, region, mx, 0, centre)[0], arcs)
    rgb = np.full((*cells.shape, 3), 0.08)
    rgb[region == 2] = (0.13, 0.17, 0.3)
    rgb[region == 1] = (0.13, 0.13, 0.13)
    yy, xx = np.nonzero(region == 1)
    w = P.in_arcs(P.angle_of(yy, xx, centre), arcs)
    rgb[yy[w], xx[w]] = WEDGE                                                       # excluded arcs
    lut = np.zeros((int(cells.max()) + 1, 3))
    for t, c in COL.items():
        lut[props.label[props.tissue == t].values] = c
    b = find_boundaries(cells, mode="inner") & (cells > 0)
    rgb[b] = lut[cells[b]]
    rgb[mx > 0] = (0.9, 0.15, 0.15)
    rb = find_boundaries(region >= 1, mode="inner")
    rgb[rb] = (1, 1, 1)                                                              # root outline
    return rgb, P.kept_deg(arcs), int((props.tissue == "cortex").sum())


def legend(fig):
    handles = [plt.Line2D([], [], color=c, lw=3, label=t.replace("_", " ")) for t, c in COL.items()] + [
        plt.Line2D([], [], color=WEDGE, lw=8, label="excluded sector"),
        plt.Line2D([], [], color=(0.9, 0.15, 0.15), lw=8, label="metaxylem"),
        plt.Line2D([], [], color="0.5", lw=1, label="root outline (white)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False)


def main(out: Path):
    """One png per case in <out>/pictures/ (whole section | zoom on the edit, checks underneath)."""
    checks = pd.read_excel(out / "checks.xlsx")
    pics = out / "pictures"
    pics.mkdir(exist_ok=True)
    for case in [c for c in ZOOM if (out / c / "anatomy/masks/S.npz").exists()]:
        rgb, kept, ncx = picture(out / case / "anatomy/masks/S.npz")
        fig, row = plt.subplots(1, 2, figsize=(13, 7.6), facecolor="white")
        row[0].imshow(rgb)
        r, deg, h = ZOOM[case]
        y, x = at(r, deg)
        row[0].add_patch(plt.Rectangle((x - h, y - h), 2 * h, 2 * h, fill=False, ec="white", lw=1.2, ls="--"))
        row[1].imshow(rgb[max(y - h, 0):y + h, max(x - h, 0):x + h], interpolation="nearest")
        for a in row:
            a.set_xticks([]); a.set_yticks([])
        c = checks[(checks.case == ("base" if case == "00_base" else case)) & ~checks.check.str.startswith("status")]
        if case == "00_base":
            c = c[c.check.isin(["n_cortex_cells", "n_cortical_layers", "mx_n", "root_area_px", "stele_area_px",
                                "cortex_width_px", "mean_cortex_cell_area_px"])]
        fail = (c.ok == "FAIL").any()
        fig.suptitle(f"{case.replace('_', ' ')}   ·   {'FAIL' if fail else 'PASS'}", fontsize=15, x=0.02, ha="left",
                     color="#c0262d" if fail else "#1d6b34", fontweight="bold")
        row[0].set_title(f"as measured: {ncx} cortical cells · {kept:.0f}° kept", fontsize=11, loc="left")
        row[1].set_title("zoom (dashed box)", fontsize=11, loc="left")
        note = c.note.dropna().iloc[0] if c.note.notna().any() else ""
        lines = [f"{'✗' if k.ok == 'FAIL' else '✓'}  {k.check}:  expected {k.expected},  got {k.got}" for k in c.itertuples()]
        fig.text(0.02, 0.135, (note + "\n" if note else "") + "\n".join(lines), fontsize=10.5, va="top", family="monospace")
        legend(fig)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.2 + 0.022 * max(0, len(lines) - 3), wspace=0.04)
        fig.savefig(pics / f"{case}.png", dpi=100)
        plt.close(fig)
    print(pics)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
