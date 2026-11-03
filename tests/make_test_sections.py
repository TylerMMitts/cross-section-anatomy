"""Ten synthetic root cross-sections with a known answer, to test Rooh end to end: the auto pass (does it find every
cell?) and a person's edits (aerenchyma, torn / cracked sectors, lateral root, debris) against the truth.

  .venv/bin/python tests/make_test_sections.py [out_dir]      (default tests/output/sections, git-ignored)

<out_dir>/raw/T01..T10.tif         the pictures (bright walls, dark lumens, like a DAPI section), 0.5 µm/px
<out_dir>/truth/masks/T??.npz      the true masks in Rooh's format (cells, region, mx, excluded sectors, aerenchyma labels)
<out_dir>/truth/truth.xlsx         per image: what a correct review should give, and what to do to get there
<out_dir>/truth/pictures/T??.png   the picture beside its truth (tissues coloured)

Geometry: concentric cortex layers whose walls wave a little; each layer is cut into cells by tilted, jittered
radial walls (outer layers have more cells, as in a root). A ray from the stele crosses one cell per layer, so
n_cortical_layers = layers wherever the cortex is whole. Stele = small Voronoi cells + metaxylem vessels (thick walls).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import io, segmentation

N = 1100
UM = 0.5
yy, xx = np.mgrid[:N, :N].astype(float)

# name: layers, cortex widths (first -> last as a ramp), stele r, vessels, aerenchyma lacunae, torn, extras
CASES = {
    "T01": dict(layers=8, stele=140, mx=6, note="clean section"),
    "T02": dict(layers=10, stele=160, mx=7, w0=18, w1=32, note="clean, more layers"),
    "T03": dict(layers=6, stele=120, mx=4, w0=30, w1=50, note="clean, few large layers"),
    "T04": dict(layers=8, stele=140, mx=6, aer=[(3, 6, 20, 26), (3, 6, 100, 24), (4, 7, 160, 22), (3, 6, 250, 26),
                                                (4, 6, 330, 20)],
                note="aerenchyma: 5 lacunae in mid cortex"),
    "T05": dict(layers=8, stele=140, mx=6, aer=[(2, 7, a, 22) for a in (10, 50, 95, 140, 185, 230, 275, 320)],
                note="heavy aerenchyma: 8 lacunae"),
    "T06": dict(layers=8, stele=140, mx=6, torn=(3, 4), note="torn: outer half of the cortex gone in Q3"),
    "T07": dict(layers=8, stele=140, mx=6, crack=(55, 4), note="crack: a wedge of cortex missing in Q1"),
    "T08": dict(layers=8, stele=140, mx=6, lateral=60, note="lateral root breaking out at 60° (Q1)"),
    "T09": dict(layers=8, stele=140, mx=5, aer=[(3, 6, 30, 24), (3, 6, 300, 24), (4, 7, 200, 22)], torn=(2, 3),
                note="aerenchyma + torn Q2"),
    "T10": dict(layers=8, stele=140, mx=6, debris=True, epi_gap=(200, 40),
                note="debris outside the root + epidermis missing over 40° in Q3"),
}


def quadrant(deg):
    return (np.floor(np.mod(deg, 360) / 90) + 1).astype(int)


def make(name, p, rng):
    c0 = (N / 2 + rng.uniform(-15, 15), N / 2 + rng.uniform(-15, 15))
    dy, dx = yy - c0[0], xx - c0[1]
    R = np.hypot(dy, dx)
    TH = np.mod(np.degrees(np.arctan2(-dy, dx)), 360)
    L, rs = p["layers"], p["stele"]
    w = np.linspace(p.get("w0", 22), p.get("w1", 40), L)
    w = np.r_[w[: L // 2 + 1], w[: L // 2 + 1][::-1][1:]][:L] if "w1" not in p else w   # rise then fall (default)
    w = np.interp(np.arange(L), np.arange(len(w)), w) if len(w) == L else np.resize(w, L)
    if "w1" not in p:
        w = 20 + 26 * np.sin(np.pi * (np.arange(L) + 0.5) / L)                            # small - big - small
    edges = np.r_[rs, rs + np.cumsum(w)]
    r_epi = edges[-1] + 16
    # wavy ring boundaries
    wav = [e + 2.5 * np.sin(np.radians(TH) * rng.integers(3, 7) + rng.uniform(0, 6)) for e in edges]
    cells = np.zeros((N, N), np.int32)
    lab = 1
    truth_layer = {}
    for k in range(L):
        ring = (R >= wav[k]) & (R < wav[k + 1])
        n = max(12, int(round(2 * np.pi * (edges[k] + w[k] / 2) / (w[k] * 1.15))))   # cells about as wide as tall
        cuts = np.sort(np.mod(np.arange(n) * 360 / n + rng.uniform(-0.25, 0.25, n) * 360 / n + rng.uniform(0, 360), 360))
        # one wall per cut, waving a little along the radius: no slivers
        th_eff = np.mod(TH + np.degrees(2.0 * np.sin(R / 11 + rng.uniform(0, 6)) / np.maximum(R, 1)), 360)
        sec = np.searchsorted(cuts, th_eff) % n
        cells[ring] = lab + sec[ring]
        for s in range(n):
            truth_layer[lab + s] = k + 1
        lab += n
    # epidermis
    ring = (R >= wav[-1]) & (R < r_epi)
    ne = int(2 * np.pi * r_epi / 14)
    cells[ring] = lab + (TH[ring] * ne / 360).astype(int)
    epi_labels = set(range(lab, lab + ne))
    lab += ne
    # stele: Voronoi
    pts = []
    while len(pts) < int(np.pi * rs ** 2 / 180):
        q = rng.uniform(-rs, rs, 2)
        if np.hypot(*q) < rs - 4:
            pts.append(q)
    pts = np.array(pts) + c0
    inn = R < wav[0]
    _, idx = cKDTree(pts).query(np.c_[yy[inn], xx[inn]])
    cells[inn] = lab + idx
    lab += len(pts)
    # vessels
    mx = np.zeros((N, N), np.int32)
    a0 = rng.uniform(0, 360)
    vr = []
    for i in range(p["mx"]):
        a = np.radians(a0 + i * 360 / p["mx"] + rng.uniform(-8, 8))
        cy, cx = c0[0] - 0.62 * rs * np.sin(a), c0[1] + 0.62 * rs * np.cos(a)
        r = rng.uniform(24, 32)
        vr.append(r)
        mx[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = i + 1
    cells[mx > 0] = 0
    region = np.where(R < r_epi, 1, 0).astype(np.uint8)
    region[R < wav[0]] = 2
    base = cells.copy()
    gone = np.zeros((N, N), bool)                                    # tissue missing from the section
    aer = np.zeros((N, N), np.int32)                                 # aerenchyma lacunae (truth objects)
    for i, (k0, k1, a, span) in enumerate(p.get("aer", [])):
        d = np.abs(np.mod(TH - a + 180, 360) - 180)
        m = (R >= edges[k0 - 1] + 3) & (R < edges[k1] - 3) & (d < span / 2 * (1 + 0.15 * np.sin(R / 9)))
        aer[m] = i + 1
    if "torn" in p:                                                  # (quarter, from layer): outer cortex gone
        q, k = p["torn"]
        m = (quadrant(TH) == q) & (R >= edges[k - 1] + 6 * np.sin(np.radians(TH) * 7)) & \
            (np.abs(np.mod(TH - (q - 1) * 90 - 45 + 180, 360) - 180) < 38)
        gone |= m
    if "crack" in p:                                                 # wedge from the epidermis down to layer k
        a, k = p["crack"]
        d = np.mod(TH - a + 180, 360) - 180
        gone |= (R >= edges[k - 1]) & (np.abs(d) < 3 + 12 * (R - edges[k - 1]) / (r_epi - edges[k - 1]))
    if "epi_gap" in p:
        a, span = p["epi_gap"]
        gone |= (R >= wav[-1] - 1) & (np.abs(np.mod(TH - a + 180, 360) - 180) < span / 2)
    cells[gone] = 0
    region[gone] = 0
    # lateral root: a cone of small cells from the cortex out past the epidermis
    lr = np.zeros((N, N), bool)
    if "lateral" in p:
        a = np.radians(p["lateral"])
        ux, uy = np.cos(a), -np.sin(a)
        t = dx * ux + dy * uy
        s = np.abs(-dx * uy + dy * ux)
        lr = (R > rs + 3) & (t > 0) & (t < r_epi + 110) & (s < 42 - 0.05 * (t - rs))
        ptsl = np.c_[yy[lr], xx[lr]][rng.choice(lr.sum(), int(lr.sum() / 160), replace=False)]
        _, idx = cKDTree(ptsl).query(np.c_[yy[lr], xx[lr]])
        cells[lr] = lab + idx
        lab += len(ptsl)
        region[lr & (R < r_epi)] = 1                                  # inside the root it is root; outside it isn't
    if p.get("debris"):
        for _ in range(4):
            a = np.radians(rng.uniform(0, 360)); rr = r_epi + rng.uniform(40, 120)
            cy, cx = c0[0] - rr * np.sin(a), c0[1] + rr * np.cos(a)
            cells[(yy - cy) ** 2 + ((xx - cx) * 1.4) ** 2 <= rng.uniform(12, 22) ** 2] = lab
            lab += 1
    # a cell that loses > 40% to a lacuna, the lateral root or a tear goes into it whole (lacunae form from whole
    # cells); the rest of a cut cell stays a cell
    before = np.bincount(base.ravel(), minlength=base.max() + 1)
    for mask, kind in ((aer > 0, "aer"), (lr, "lr"), (gone, "gone")):
        if not mask.any():
            continue
        lost = np.bincount(base[mask].ravel(), minlength=base.max() + 1)
        whole = np.nonzero((lost > 0.4 * np.maximum(before, 1)) & (np.arange(len(lost)) > 0))[0]
        whole = [l for l in whole if l in truth_layer or l in epi_labels]
        grow = np.isin(base, whole) & ~mask
        if kind == "aer":
            near = ndi.distance_transform_edt(aer == 0, return_indices=True)[1]
            aer[grow] = aer[near[0][grow], near[1][grow]]
        elif kind == "lr":
            lr |= grow
            cells[grow] = lab
            lab += 1
            region[grow & (R < r_epi)] = 1
        else:
            gone |= grow
            region[grow] = 0
        cells[grow] = 0 if kind != "lr" else cells[grow]
    cells[aer > 0] = 0
    cells[gone] = 0
    cells, fw, _ = segmentation.relabel_sequential(cells)
    truth_layer = {int(fw[k]): v for k, v in truth_layer.items() if k < len(fw) and fw[k]}
    epi_labels = {int(fw[k]) for k in epi_labels if k < len(fw) and fw[k]}
    return dict(cells=cells, region=region, mx=mx, aer=aer, gone=gone, lr=lr, c0=c0, edges=edges, r_epi=r_epi,
                layer=truth_layer, epi=epi_labels, TH=TH, R=R, vr=vr)


def render(t, rng):
    cells, mx, aer = t["cells"], t["mx"], t["aer"]
    obj = cells + (mx > 0) * (cells.max() + 1 + mx) + (aer > 0) * (cells.max() + 100 + aer)
    walls = segmentation.find_boundaries(obj, mode="thick")
    walls = ndi.binary_dilation(walls, iterations=1) & (obj > 0)
    g = np.where(obj > 0, 0.13, 0.03)
    g = g + 0.04 * ndi.gaussian_filter(rng.standard_normal(g.shape), 6) * (obj > 0)          # cytoplasm texture
    g[walls] = 0.75 + 0.15 * rng.standard_normal(walls.sum()).clip(-2, 2)
    vw = ndi.binary_dilation(mx > 0, iterations=3) & ~(mx > 0)                                # thick vessel walls
    g[vw] = 0.95
    g[mx > 0] = 0.05
    g[aer > 0] = 0.04
    g = ndi.gaussian_filter(g, 1.1)
    g *= 0.85 + 0.25 * (xx / N)                                                               # uneven light
    g += rng.normal(0, 0.025, g.shape)
    return (np.clip(g, 0, 1) * 65535).astype(np.uint16)


def truth_arcs(p) -> list:
    """The arcs (start, end, reason) a careful reviewer excludes: the damaged tissue, all radii (sectors are wedges)."""
    arcs = []
    if "torn" in p:                                   # torn within +-38 deg of the quarter's middle
        mid = (p["torn"][0] - 1) * 90 + 45
        arcs.append((mid - 40, mid + 40, "torn"))
    if "crack" in p:                                  # wedge up to +-15 deg at the epidermis
        arcs.append((p["crack"][0] - 17, p["crack"][0] + 17, "torn"))
    return [(a % 360, b % 360, r) for a, b, r in arcs]


def truth_numbers(name, p, t):
    """What a correct review gives: aerenchyma tagged, torn/cracked sectors excluded, debris and lateral root out."""
    cells = t["cells"]
    lab = np.arange(cells.max() + 1)
    cent = np.array(ndi.center_of_mass(np.ones_like(cells), cells, lab[1:]))
    deg = np.mod(np.degrees(np.arctan2(-(cent[:, 0] - t["c0"][0]), cent[:, 1] - t["c0"][1])), 360)
    arcs = truth_arcs(p)
    out = np.zeros(len(deg), bool)
    for a, b, _ in arcs:                              # a cell is out when its centre is in an arc
        out |= ((deg >= a) & (deg < b)) if a <= b else ((deg >= a) | (deg < b))
    out_of = dict(zip(lab[1:], out))
    cortex = [l for l, k in t["layer"].items() if (cells == l).any() and not out_of[l]]
    area = np.bincount(cells.ravel())
    cortex_area = ((t["region"] == 1) & ~t["lr"]).sum() + (t["aer"] > 0).sum()
    row = {"image": f"{name}.tif", "what": p["note"], "um_per_px": UM,
           "sectors_to_exclude": "; ".join(f"{a:.0f}-{b:.0f} {r}" for a, b, r in arcs),
           "aerenchyma_to_tag": int(t["aer"].max()),
           "n_cortex_cells": len(cortex),
           "mean_cortex_cell_area_um2": float(np.mean([area[l] for l in cortex])) * UM ** 2,
           "n_cortical_layers": float(p["layers"]),
           "mx_n": p["mx"], "mx_mean_area_um2": float(np.mean([(t["mx"] == i).sum() for i in range(1, p["mx"] + 1)])) * UM ** 2,
           "stele_area_um2": float((t["region"] == 2).sum()) * UM ** 2,
           "aerenchyma_n": int(t["aer"].max()),
           "aerenchyma_area_um2": float((t["aer"] > 0).sum()) * UM ** 2,
           "aerenchyma_pct_cortex": 100 * float((t["aer"] > 0).sum()) / max(cortex_area, 1),
           "lateral_root": "yes" if "lateral" in p else "", "debris": "yes" if p.get("debris") else ""}
    return row, arcs


def picture(img, t, f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = np.zeros((N, N, 3))
    cells = t["cells"]
    lut = np.zeros((cells.max() + 1, 3))
    for l, k in t["layer"].items():
        lut[l] = (1, 0.85, 0.1)
    for l in t["epi"]:
        lut[l] = (0.6, 0.6, 0.6)
    b = segmentation.find_boundaries(cells, mode="inner") & (cells > 0)
    rgb[b] = lut[cells[b]]
    rgb[(t["region"] == 2) & ~b] = (0.13, 0.17, 0.3)
    stele_b = b & (t["region"] == 2)
    rgb[stele_b] = (0.35, 0.55, 1.0)
    rgb[t["mx"] > 0] = (0.9, 0.15, 0.15)
    rgb[t["aer"] > 0] = (0.1, 0.9, 0.95)
    rgb[t["lr"] & b] = (0.85, 0.2, 0.85)
    rgb[b & (t["region"] == 0) & ~t["lr"]] = (0.85, 0.2, 0.85)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6.2))
    ax[0].imshow(img, cmap="gray")
    ax[1].imshow(rgb)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    ax[0].set_title("picture (what Rooh sees)", loc="left")
    ax[1].set_title("truth: cortex yellow, epidermis grey, stele blue, vessels red, aerenchyma cyan, not root magenta",
                    loc="left", fontsize=9)
    fig.suptitle(f"{f.stem}: {t['note']}", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(f, dpi=80)
    plt.close(fig)


def main(out: Path):
    for d in ("raw", "truth/masks", "truth/pictures"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (name, p) in enumerate(CASES.items()):
        rng = np.random.default_rng(100 + i)
        t = make(name, p, rng)
        t["note"] = p["note"]
        img = render(t, rng)
        io.imsave(out / "raw" / f"{name}.tif", img, check_contrast=False)
        row, arcs = truth_numbers(name, p, t)
        rows.append(row)
        np.savez_compressed(out / "truth/masks" / f"{name}.npz", cells=t["cells"], region=t["region"], mx=t["mx"],
                            aerenchyma=t["aer"], status=np.array("truth"), image_file=np.array(f"{name}.tif"),
                            excl_start=np.array([a for a, _, _ in arcs], float),
                            excl_end=np.array([b for _, b, _ in arcs], float),
                            excl_reason=np.array([r for _, _, r in arcs], dtype="U40"), centre=np.array(t["c0"]),
                            layer=np.array(list(t["layer"].items()), np.int32))
        picture(img, t, out / "truth/pictures" / f"{name}.png")
        print(name, p["note"], "->", row["n_cortex_cells"], "cortex cells")
    tab = pd.DataFrame(rows)
    readme = pd.DataFrame({"column": tab.columns, "what it is": [
        "picture", "what is special about it", "scale", "arcs a careful reviewer excludes (degrees ccw from the right)",
        "aerenchyma lacunae to mark", "cortical cells that should count (aerenchyma, excluded sectors out)",
        "their mean area", "cortical layers (rays from the stele out)", "metaxylem vessels", "mean vessel area", "stele area",
        "aerenchyma lacunae", "their total area", "aerenchyma as % of cortex area", "a lateral root is in the picture",
        "debris outside the root"]})
    with pd.ExcelWriter(out / "truth/truth.xlsx") as w:
        tab.to_excel(w, sheet_name="truth", index=False)
        readme.to_excel(w, sheet_name="README", index=False)
    print(out / "truth/truth.xlsx")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "output/sections")
