"""Masks -> tables. Run any time; it reads whatever is saved (auto or reviewed).

  python summarise.py <anatomy_dir> [--image-figures]     (also one cortex pattern figure per image)

Uses <anatomy_dir>/samples.xlsx (one row per image; made by run_batch.py, or here if missing). Fill it in:
  plant      the replicate; images of the same plant are averaged
  group_*    columns from the folder names; rename the headers (species, genotype ...)
             or add columns (treatment ...). Every column except image_file, plant,
             um_per_px and include is treated as a grouping column.
  um_per_px  microns per pixel, from the microscope/camera: always given by the user (scale bars are not read).
             Fill it any time after measuring: the scale is used only here, never for outlining.
  include    "no" drops the image
Writes <anatomy_dir>/../results/anatomy_summary_YYYY-MM-DD.xlsx
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import measure

import pattern_plots as PP
import pipeline as P

FIXED = ["image_file", "plant", "um_per_px", "include"]
N_BINS = 10   # radial bins across the cortex: 1 = next to the stele ... 10 = next to the epidermis; epidermis its own point
# what every column means: written into each workbook's README sheet. Columns ending _px are the same
# measure in pixels (kept at the end of each sheet, for images without a scale). Areas run to mid-wall.
COLUMNS = {
    "image_file": "image path inside the image folder (the tag that links every row to its picture)",
    "plant": "plant (biological replicate), from the folder name; images of one plant share it",
    "group_1": "first folder level (e.g. species); rename in anatomy/samples.xlsx",
    "group_2": "second folder level (e.g. genotype); rename in anatomy/samples.xlsx",
    "tech_rep": "image number within its plant (technical replicate)",
    "status": "auto = computer outlines only; reviewed = checked by hand (Save + next)",
    "um_per_px": "scale given in samples.xlsx (micrometres per pixel); empty = no µm columns",
    "n_images": "images averaged into this plant row",
    "n_reviewed": "of those, images checked by hand",
    "angle_kept_deg": "degrees of the section measured: 360 minus the excluded sectors (damaged tissue taken out by hand)",
    "excluded_sectors": "the excluded arcs, 'start-end reason' in degrees counter-clockwise from the image's right",
    "low_coverage": f"TRUE when fewer than {P.MIN_KEPT_DEG} degrees are kept: cortex numbers rest on few cells",
    "label": "cell number inside the image's mask",
    "tissue": "cortex | stele | epidermis | air_space | metaxylem | unassigned; '..._excluded' = centre in an excluded "
              "sector (kept in this table, out of every cortex number)",
    "sector": "kept or excluded (by the cell's centre)",
    "ring": "adjacency steps from the stele (a gap check; runs high)",
    "x": "cell centre, pixels from the left", "y": "cell centre, pixels from the top",
    "angle_rad": "direction of the cell centre from the centre, counter-clockwise from the right (radians)",
    "angle_deg": "the same in degrees",
    "r_um": "distance of the cell centre from the centre (µm)",
    "r_norm": "radial position of the cell centre: 0 = stele edge, 1 = root edge, along its own direction",
    "radial_bin": f"r_norm in {N_BINS} equal bins: 1 = next to the stele",
    "dist_from_stele_um": "distance from the stele edge to the cell centre (µm)",
    "area_um2": "cell area (µm², to mid-wall)", "major_um": "cell length, long axis (µm)",
    "minor_um": "cell width, short axis (µm)",
    "root_area_um2": "whole cross-section area as outlined (µm²); with excluded sectors use root_area_est_um2",
    "stele_area_um2": "stele area, whole section (µm²; never sector dependent)",
    "cortex_area_kept_um2": "root outside the stele (cortex + epidermis) inside the kept angles (µm²)",
    "cortex_area_est_um2": "the same scaled to the whole circle (x 360 / angle_kept_deg): an ESTIMATE that assumes "
                           "the excluded arcs looked like the kept ones",
    "root_area_est_um2": "stele area + cortex_area_est_um2 (estimate)",
    "stele_share_est": "stele area / root_area_est (estimate; whole-section stele over the scaled root)",
    "cortex_width_um": "mean root radius minus stele radius over the kept degrees (µm)",
    "n_cortical_layers": "cortical cells crossed by a ray from the stele out, median over rays every 5 degrees in the "
                         "kept angles (_min / _max: the range: one side can carry an extra layer)",
    "n_cortex_cells": "cortical cells counted in the kept angles", "n_cortex_cells_est": "scaled to 360 degrees (estimate)",
    "mean_cortex_cell_area_um2": "mean cortical cell area, kept cells (µm²)",
    "median_cortex_cell_area_um2": "median cortical cell area, kept cells (µm²): the reference for rel",
    "n_air_spaces": "small spaces between kept cortical cells", "n_air_spaces_est": "scaled to 360 degrees (estimate)",
    "air_space_total_area_um2": "their total area (µm²)",
    "mx_n": "metaxylem vessels (whole section)", "mx_mean_area_um2": "mean metaxylem vessel area (µm²)",
    "mx_total_area_um2": "total metaxylem area (µm²)",
    "bin": f"1..{N_BINS} = r_norm bins of the cortex (1 next to the stele); epidermis = its own point",
    "x_position": "bin centre on the 0-1 axis (epidermis drawn just past 1)",
    "n_cells": "cells in this bin (kept angles); few_cells = TRUE below " + str(P.MIN_CELLS_PER_BIN),
    "few_cells": f"TRUE when the bin has fewer than {P.MIN_CELLS_PER_BIN} cortical cells: read it with care",
    "median_area_um2": "median cell area in this bin (µm²)",
    "rel": "median area in the bin / the image's median cortical cell area (shape without overall size)",
    "epidermis_rel": "epidermis median cell area / the image's median cortical cell area",
    "min_cells_in_bin": "fewest cortical cells in any cortex bin of this image",
    "area_unit": "unit of the absolute sizes in this row: um2, or px when the image has no scale",
    "peak_height_rel": "largest value on the fitted curve; _rel = relative size (shape), _abs = in area_unit, "
                       "_rel_epi = relative with the epidermis as the last point (same for the four columns below)",
    "peak_pos_rel": "where the peak is on the 0-1 axis (stele side 0, epidermis side 1)",
    "rise_slope_rel": "(peak - first value) / distance: how fast cells grow from the stele to the peak",
    "decay_slope_rel": "slope just past the peak (over 20% of the range): how fast cells shrink outward",
    "outer_rise_rel": "last value - lowest value after the peak: an upturn at the epidermis",
    "axis": "rel (the curve of relative size)", "fitted": "the curve's value at x",
}
FEATURE_SETS = ("_rel", "_abs", "_rel_epi")


def make_samples(files: list[Path]) -> pd.DataFrame:
    rows = []
    for f in files:
        rel = Path(str(np.load(f)["image_file"]))
        parts = list(rel.parent.parts)
        plant = (parts[-1] if len(parts) >= 3 else rel.stem).lower()   # "P1" and "p1" are one plant
        groups = parts[:-1] if len(parts) >= 3 else parts
        rows.append({"image_file": str(rel), "plant": plant, "um_per_px": None, "include": "yes",
                     **{f"group_{k + 1}": g for k, g in enumerate(groups)}})
    return pd.DataFrame(rows)


def write_samples(anatomy_dir: Path) -> Path:
    """samples.xlsx for the user to fill (never overwritten once it exists)."""
    sfile = anatomy_dir / "samples.xlsx"
    if not sfile.exists():
        make_samples(sorted((anatomy_dir / "masks").glob("*.npz"))).to_excel(sfile, index=False)
        print(f"\nWrote {sfile}\nFill in um_per_px (microns per pixel) for every image, rename group_1, group_2 ... "
              "(e.g. species, genotype), check plant, then click Make results.")
    return sfile


def px_last(df):
    """Pixel columns at the end: the µm columns are the ones to read."""
    px = [c for c in df.columns if c.endswith("_px") and c != "um_per_px"]
    return df[[c for c in df.columns if c not in px] + px]


def describe(col):
    if col in COLUMNS:
        return COLUMNS[col]
    if col.endswith("_px"):
        base = col[:-3]
        for unit in ("_um2", "_um"):
            if base + unit in COLUMNS:
                return COLUMNS[base + unit].split("(µm")[0].strip() + " (pixels)"
        return "pixels"
    for sfx in FEATURE_SETS:                           # profile features: same meaning on another scale
        if col.endswith(sfx) and col[:-len(sfx)] + "_rel" in COLUMNS:
            return COLUMNS[col[:-len(sfx)] + "_rel"].split(";")[0] + f" ({sfx[1:]})"
    return ""


def to_um(d: dict, s: float) -> dict:
    """Pixel traits -> µm: *area*_px -> *_um2, *width_px -> *_um (NaN without a scale)."""
    out = {}
    for k, v in d.items():
        if k.endswith("_px") and "area" in k:
            out[k[:-3] + "_um2"] = v * s * s
        elif k.endswith("width_px"):
            out[k[:-3] + "_um"] = v * s
    return out


def main(anatomy_dir: Path, image_figures: bool = False):
    """The project's tables (xlsx) and group figures. image_figures: also one cortex pattern figure per image
    (slow for big sets; the editors draw it live for the image on screen).
    Cortex numbers come from the kept angles only (excluded sectors out, cells by their centre); stele and metaxylem
    are whole-section. Unit of analysis: the plant. Per image first (medians per bin), then plant means; cells are
    never pooled across images."""
    settings = json.loads((anatomy_dir / "settings.json").read_text())
    files = sorted((anatomy_dir / "masks").glob("*.npz"))
    sfile = write_samples(anatomy_dir)
    samples = pd.read_excel(sfile, dtype={"image_file": str, "plant": str})
    samples["include"] = samples["include"].fillna("yes").astype(str).str.lower()
    groups = [c for c in samples.columns if c not in FIXED]
    samples = samples.set_index("image_file")
    tech_rep = samples.groupby(["plant", *groups], dropna=False).cumcount() + 1   # image number within a plant

    angle = float(settings.get("quadrant_angle_deg", 0))       # only to read old dropped quarters as sectors
    fig_dir = anatomy_dir.parent / "results" / "figures" / "cortex_pattern"
    if image_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)
    img_rows, cell_rows, prof_rows, all_rows, feat_rows, curve_rows = [], [], [], [], [], []
    grid = np.round(np.linspace(0, 1, 21), 2)
    for f in files:
        z = np.load(f)
        rel = str(z["image_file"])
        if rel not in samples.index or samples.at[rel, "include"] == "no":
            continue
        cells, region, mx, status = P.load_state(f)
        centre = P.centre_of(region, P.load_centre(f))
        props, mxp, tr = P.derive(cells, region, mx, angle, centre)
        arcs = P.load_sectors(f, angle)
        props = P.exclude_sectors(props, arcs)
        st = P.sector_traits(cells, region, props, centre, arcs)
        s = samples.at[rel, "um_per_px"]
        s = float(s) if pd.notna(s) else np.nan
        tech = int(tech_rep[rel])
        meta = {"image_file": rel, "plant": samples.at[rel, "plant"], **{g: samples.at[rel, g] for g in groups}}
        whole = {k: tr[k] for k in ("root_area_px", "stele_area_px", "mx_n", "mx_mean_area_px", "mx_total_area_px")}

        prof = P.sector_profile(props, N_BINS)
        um = not np.isnan(s)
        prof["median_area_um2"] = prof.median_area_px * s * s
        cx = prof[prof.bin != "epidermis"]
        epi = prof[prof.bin == "epidermis"].iloc[0]
        img_rows.append({**meta, "tech_rep": tech, "status": status, "um_per_px": s, **whole, **st,
                         "epidermis_rel": epi.rel, "min_cells_in_bin": int(cx.n_cells.min()),
                         **to_um({**whole, **st}, s)})
        prof_rows.append(prof.rename(columns={"x": "x_position"}).assign(**meta, tech_rep=tech, status=status))

        ycol = "median_area_um2" if um else "median_area_px"
        feat = {**meta, "tech_rep": tech, "status": status, "area_unit": "um2" if um else "px",
                "angle_kept_deg": st["angle_kept_deg"], "low_coverage": st["low_coverage"],
                "n_cortex_cells": st["n_cortex_cells"], "min_cells_in_bin": int(cx.n_cells.min()),
                "epidermis_rel": epi.rel}
        w = cx.n_cells.where(cx.n_cells > 0)
        with_epi = prof.dropna(subset=["rel"])
        for sfx, (x, y, ww) in {"_rel": (cx.x, cx.rel, w), "_abs": (cx.x, cx[ycol], w),
                                "_rel_epi": (with_epi.x, with_epi.rel, with_epi.n_cells.where(with_epi.n_cells > 0))}.items():
            feat.update({k + sfx: v for k, v in PP.profile_features(x, y, ww).items()})
        feat_rows.append(feat)
        sp = PP.spline(cx.x, cx.rel, w, at=grid)
        curve_rows += [{**meta, "tech_rep": tech, "axis": "rel", "x": xv, "fitted": yv}
                       for xv, yv in zip(grid, sp[1] if sp is not None else [np.nan] * len(grid))]
        if image_figures:
            PP.image_figure(prof.assign(y=prof[ycol]), "Cell area (µm²)" if um else "Cell area (px)",
                            f"{rel}  ·  {status}  ·  {st['angle_kept_deg']:.0f}° kept",
                            fig_dir / (Path(rel).with_suffix("").as_posix().replace("/", "__") + ".png"))

        mxt = pd.DataFrame(measure.regionprops_table(mx, properties=("label", "area", "centroid", "axis_major_length",
                                                                     "axis_minor_length"))) if mx.any() else pd.DataFrame()
        if len(mxt):
            mxt = mxt.rename(columns={"centroid-0": "y", "centroid-1": "x", "area": "area_px",
                                      "axis_major_length": "major_px", "axis_minor_length": "minor_px"})
            mxt = mxt.join(P.radial_position(mxt, region, centre))
            mxt["tissue"] = "metaxylem"
        allc = pd.concat([props, mxt], ignore_index=True)
        allc["sector"] = np.where(allc.tissue.astype(str).str.endswith("_excluded"), "excluded", "kept")
        allc["ring"] = allc.layer if "layer" in allc else np.nan
        allc = allc.assign(area_um2=allc.area_px * s * s, major_um=allc.major_px * s, minor_um=allc.minor_px * s,
                           r_um=allc.r_px * s, dist_from_stele_um=allc.dist_from_stele_px * s,
                           angle_deg=np.degrees(allc.angle_rad))
        all_rows.append(allc.assign(**meta, tech_rep=tech, status=status)[
            [*meta, "tech_rep", "status", "label", "tissue", "sector", "ring", "x", "y",
             "angle_rad", "angle_deg", "r_px", "r_um", "r_norm", "dist_from_stele_um",
             "area_px", "area_um2", "major_um", "minor_um"]])

        c = props[props.tissue == "cortex"].copy()
        c["area_um2"] = c.area_px * s * s
        c["major_um"], c["minor_um"] = c.major_px * s, c.minor_px * s
        c["dist_from_stele_um"] = c.dist_from_stele_px * s
        c["radial_bin"] = (np.clip(c.r_norm, 0, 0.999) * N_BINS).astype(int) + 1
        cell_rows.append(c.assign(**meta, status=status)[[*meta, "status", "r_norm", "radial_bin", "dist_from_stele_um",
                                                          "label", "area_px", "area_um2", "major_um", "minor_um"]])

    keys = ["plant", *groups]
    first = lambda df, cols: df[[*cols, *[c for c in df.columns if c not in cols]]]
    per_image = pd.DataFrame(img_rows)
    per_cell = pd.concat(cell_rows, ignore_index=True) if cell_rows else pd.DataFrame()
    per_image_profile = first(pd.concat(prof_rows, ignore_index=True), ["image_file", *keys, "tech_rep", "status", "bin"]) \
        if prof_rows else pd.DataFrame()
    per_plant_profile = (per_image_profile.groupby([*keys, "bin", "x_position"], dropna=False)
                         .agg(n_images=("image_file", "size"), median_area_um2=("median_area_um2", "mean"),
                              rel=("rel", "mean"), n_cells=("n_cells", "sum")).reset_index()
                         .sort_values([*keys, "x_position"])) if len(per_image_profile) else pd.DataFrame()
    num = [c for c in per_image.columns if c.endswith(("_px", "_um2", "_um", "_deg", "_est", "_rel"))
           or c.startswith(("n_", "mx_", "min_"))]
    per_plant = (per_image.groupby(keys, dropna=False)
                 .agg(n_images=("image_file", "size"),
                      n_reviewed=("status", lambda s: int((s == "reviewed").sum())),
                      n_low_coverage=("low_coverage", "sum"),
                      **{c: (c, "mean") for c in num}).reset_index())
    all_cells = pd.concat(all_rows, ignore_index=True)
    profile_features = pd.DataFrame(feat_rows)
    spline_curves = pd.DataFrame(curve_rows)
    out_dir = anatomy_dir.parent / "results"
    out_dir.mkdir(exist_ok=True)
    today = dt.date.today().isoformat()
    files_out = {
        f"cortex_pattern_{today}.xlsx": {
            "per_image_profile": (per_image_profile, f"one row per image x bin: every cortical cell in the kept angles "
                                  f"(centre rule), r_norm in {N_BINS} bins (1 = next to the stele), median area per bin, "
                                  "then the epidermis as its own row. rel = bin median / the image's median cortical "
                                  f"cell area. few_cells = fewer than {P.MIN_CELLS_PER_BIN} cells in the bin"),
            "profile_features": (profile_features, "one row per image: smoothing spline (GCV) through the bin medians "
                                 "(weights = cells per bin); features read off the curve. _rel = relative size (the "
                                 "shape), _abs = in area_unit, _rel_epi = relative with the epidermis as the last point. "
                                 "peak_height, peak_pos; rise_slope = (peak - first) / distance; decay_slope = local, "
                                 "over 20% of the range past the peak; outer_rise = last value - minimum after the peak. "
                                 f"NaN = fewer than {PP.MIN_PTS} bins with cells or feature undefined (peak at an end). "
                                 "Check low_coverage and min_cells_in_bin before using a row"),
            "spline_curves": (spline_curves, "the fitted relative-size curves on 0, 0.05 ... 1 (long format; NaN outside "
                              "the data): the input for fPCA"),
            "per_plant_profile": (per_plant_profile, "plant x bin: mean of the image medians (display; analyse "
                                  "per_image_profile with plant as the unit)")},
        f"all_cells_{today}.xlsx": {
            "all_cells": (all_cells, "every segmented object: tissue = cortex | stele | epidermis | air_space | "
                          "unassigned | metaxylem, '..._excluded' when its centre is in an excluded sector. angle_rad "
                          "counter-clockwise from the image's right, around the stele centre. r_norm 0 = stele edge, "
                          "1 = root edge along the object's direction. ring = adjacency hops from the stele (gap check, runs high)")},
        f"anatomy_summary_{today}.xlsx": {
            "per_plant": (per_plant, "plant means of per_image; n_reviewed = images checked by hand; "
                          "n_low_coverage = images with too few degrees kept"),
            "per_image": (per_image, "per image. Stele, metaxylem, root_area: whole section. Cortex traits: kept angles "
                          "only; _est = scaled to 360 degrees (estimate). angle_kept_deg and excluded_sectors say what "
                          "was taken out and why"),
            "per_cell": (per_cell, "kept cortical cells (centre rule), one per row")},
    }
    for fname, tabs in files_out.items():
        tabs = {name: (px_last(df), note) for name, (df, note) in tabs.items()}
        readme = pd.DataFrame([(k, "", n) for k, (_, n) in tabs.items()], columns=["sheet", "column", "what it is"])
        cols = pd.DataFrame([(name, c, describe(c)) for name, (df, _) in tabs.items() for c in df.columns],
                            columns=["sheet", "column", "what it is"])
        with pd.ExcelWriter(out_dir / fname) as w:
            pd.concat([readme, cols], ignore_index=True).to_excel(w, sheet_name="README", index=False)
            for name, (df, _) in tabs.items():
                df.to_excel(w, sheet_name=name, index=False)
    if len(per_image_profile) and groups:
        sp, gt = groups[0], groups[-1]            # e.g. species, genotype
        (out_dir / "figures").mkdir(exist_ok=True)
        for spv, d in per_image_profile.groupby(sp):
            um = d.median_area_um2.notna().any()   # µm² if any image has a scale; images without one are left out
            drop = d.loc[d.median_area_um2.isna(), "image_file"].nunique() if um else 0
            PP.group_figure(d[d.median_area_um2.notna()] if um else d, gt, spv,
                            "median_area_um2" if um else "median_area_px",
                            "Cortical cell area (µm²)" if um else "Cortical cell area (px)",
                            out_dir / "figures" / f"cortex_pattern_{spv}_{today}.png",
                            f"  {drop} image(s) without a scale left out." if drop else "")
    no_scale = per_image.loc[per_image.um_per_px.isna(), "image_file"].tolist()
    if no_scale:
        print(f"\nWARNING: {len(no_scale)} image(s) have no scale (um_per_px empty); their sizes "
              f"stay in pixels and they are left out of the um figures. Type um_per_px in {sfile.name} and run again:")
        print("  " + "\n  ".join(no_scale[:10]) + ("\n  ..." if len(no_scale) > 10 else ""))
    low = per_image.loc[per_image.low_coverage, "image_file"].tolist()
    if low:
        print(f"\nNOTE: {len(low)} image(s) keep fewer than {P.MIN_KEPT_DEG} degrees: low_coverage = TRUE.")
    print(f"{len(per_image)} images ({(per_image.status == 'reviewed').sum()} reviewed), "
          f"{per_plant.shape[0]} plants -> {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).resolve(), image_figures="--image-figures" in sys.argv)
