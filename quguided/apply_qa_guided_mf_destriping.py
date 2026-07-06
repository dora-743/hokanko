from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_DIR = Path(
    r"D:\research\code\outputs_detected_slopes_orthogonal_thin_eachiter_then_broad_median"
)
DEFAULT_TAKAKU_DIR = Path(r"D:\research\code\takaku_wavelet_mf_destriping\outputs")
DEFAULT_TAKAKU_SCRIPT = Path(
    r"D:\research\code\takaku_wavelet_mf_destriping\apply_takaku_wavelet_mf_destriping.py"
)
DEFAULT_OUTPUT_DIR = Path(r"D:\research\code\qa_guided_mf_destriping\outputs")
DEFAULT_METADATA_TXT = Path(
    "E:/\u30e1\u30bf\u30f3/2025_HISUI_72_The Permian Basin-"
    "\u8ad6\u6587\u7167\u5408\u7528/"
    "HSHL1G_N320W1032_20221030160051_20231127193053/"
    "HSHL1G_N320W1032_20221030160051_20231127193053.txt"
)

# Slopes are row/column slopes in the 200 x 200 MF image.
# AT_META_SLOPE is the mean of the left/right observation-footprint edges.
# CT_MIRROR_SLOPE is the positive mirror of the top/bottom observation-footprint edges.
AT_META_SLOPE = 0.9771963630247031
CT_MIRROR_SLOPE = 1.2667203308533619
DETECTED_THIN_SLOPE = 0.9792723507257799
DETECTED_BROAD_SLOPE = 1.257172298918948

RECOMMENDED_NAME = "qa_guided_existing_thin_then_ct_dwt"


def load_takaku_module(path: Path):
    spec = importlib.util.spec_from_file_location("takaku_destripe_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_hisui_metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def latlon_to_utm_easting_northing(lat_deg: float, lon_deg: float, zone: int) -> tuple[float, float]:
    """Convert WGS84 lat/lon to UTM easting/northing.

    This small forward transform keeps the script dependency-free; it is accurate
    enough for estimating local image-axis slopes over this 20 m L1G grid.
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    k0 = 0.9996
    e2 = f * (2.0 - f)
    ep2 = e2 / (1.0 - e2)

    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)

    n = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    t = math.tan(lat) ** 2
    c = ep2 * math.cos(lat) ** 2
    aa = math.cos(lat) * (lon - lon0)
    m = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * lat)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * lat)
        - (35 * e2**3 / 3072) * math.sin(6 * lat)
    )
    easting = k0 * n * (
        aa
        + (1 - t + c) * aa**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * ep2) * aa**5 / 120
    ) + 500000.0
    northing = k0 * (
        m
        + n
        * math.tan(lat)
        * (
            aa**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * aa**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * ep2) * aa**6 / 720
        )
    )
    if lat_deg < 0:
        northing += 10000000.0
    return easting, northing


def metadata_footprint_slopes(path: Path) -> tuple[float, float, list[dict[str, float | str]]]:
    meta = parse_hisui_metadata(path)
    if not meta:
        return AT_META_SLOPE, CT_MIRROR_SLOPE, []

    zone = int(float(meta["UTMZone"]))
    grid_m = float(meta["GridCellSizeMeter"])
    line_offset = float(meta["LineProjectionOffsetMeter"])
    sample_offset = float(meta["SampleProjectionOffsetMeter"])

    corners: dict[str, tuple[float, float]] = {}
    for key in ["UpperLeft", "UpperRight", "LowerLeft", "LowerRight"]:
        lat = float(meta[f"Observation{key}LatitudeDegree"])
        lon = float(meta[f"Observation{key}LongitudeDegree"])
        easting, northing = latlon_to_utm_easting_northing(lat, lon, zone)
        row = (line_offset - northing) / grid_m
        col = (easting - sample_offset) / grid_m
        label = key.replace("Upper", "U").replace("Lower", "L").replace("Left", "L").replace("Right", "R")
        corners[label] = (row, col)

    edge_defs = [
        ("left_edge_UL_to_LL_inferred_AT", "UL", "LL", "AT"),
        ("right_edge_UR_to_LR_inferred_AT", "UR", "LR", "AT"),
        ("top_edge_UL_to_UR_inferred_CT", "UL", "UR", "CT"),
        ("bottom_edge_LL_to_LR_inferred_CT", "LL", "LR", "CT"),
    ]
    rows: list[dict[str, float | str]] = []
    for edge, a, b, axis in edge_defs:
        row0, col0 = corners[a]
        row1, col1 = corners[b]
        slope = (row1 - row0) / (col1 - col0)
        angle = math.degrees(math.atan(slope))
        rows.append(
            {
                "edge": edge,
                "axis": axis,
                "from": a,
                "to": b,
                "row0": row0,
                "col0": col0,
                "row1": row1,
                "col1": col1,
                "signed_slope_row_per_col": slope,
                "angle_deg_image_row": angle,
            }
        )

    at_slope = float(np.mean([row["signed_slope_row_per_col"] for row in rows if row["axis"] == "AT"]))
    ct_slope = float(np.mean([row["signed_slope_row_per_col"] for row in rows if row["axis"] == "CT"]))
    return at_slope, abs(ct_slope), rows


def load_existing_thin_only(input_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.load(input_dir / "baseline_no_destripe_alpha_raw.npy").astype(float)
    plume_mask = np.load(input_dir / "baseline_no_destripe_plume_mask.npy").astype(bool)
    thin_primary = np.load(
        input_dir
        / (
            "median_thin_eachiter_then_broad_median_stripe_map_thin_"
            "thin_detected_thin_high_alpha_primary_positive_y_minus_x_"
            "slope_0.9793_bin_2.0.npy"
        )
    ).astype(float)
    thin_orthogonal = np.load(
        input_dir
        / (
            "median_thin_eachiter_then_broad_median_stripe_map_thin_"
            "thin_detected_thin_high_alpha_orthogonal_to_primary_y_plus_x_"
            "slope_1.0212_bin_2.0.npy"
        )
    ).astype(float)
    existing_thin_only = raw - thin_primary - thin_orthogonal
    existing_thin_stripe = thin_primary + thin_orthogonal
    return raw, plume_mask, existing_thin_only, existing_thin_stripe


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_metrics(
    helpers,
    name: str,
    image: np.ndarray,
    plume_mask: np.ndarray,
    baseline_threshold: float,
) -> dict[str, float | int | str]:
    non_plume = ~plume_mask

    def profile_rstd(slope: float, bin_width: float) -> float:
        _, prof = helpers.directional_profile(
            image,
            slope=slope,
            bin_width=bin_width,
            protected_mask=plume_mask,
        )
        return helpers.robust_std(prof)

    row: dict[str, float | int | str] = {
        "method": name,
        "robust_std_all": helpers.robust_std(image),
        "robust_std_non_plume": helpers.robust_std(image, non_plume),
        "p95_abs_non_plume": float(np.nanpercentile(np.abs(image[non_plume]), 95.0)),
        "profile_rstd_detected_broad_slope_1p257": profile_rstd(DETECTED_BROAD_SLOPE, 18.0),
        "profile_rstd_ct_meta_slope_1p267": profile_rstd(CT_MIRROR_SLOPE, 18.0),
        "profile_rstd_detected_thin_slope_0p979": profile_rstd(DETECTED_THIN_SLOPE, 2.0),
        "profile_rstd_at_meta_slope_0p977": profile_rstd(AT_META_SLOPE, 2.0),
        "plume_mean": float(np.nanmean(image[plume_mask])) if plume_mask.any() else float("nan"),
        "plume_p95": float(np.nanpercentile(image[plume_mask], 95.0)) if plume_mask.any() else float("nan"),
        "plume_max": float(np.nanmax(image[plume_mask])) if plume_mask.any() else float("nan"),
    }
    if np.isfinite(baseline_threshold):
        row["count_above_baseline_threshold"] = int(np.sum(image > baseline_threshold))
        row["non_plume_count_above_baseline_threshold"] = int(np.sum((image > baseline_threshold) & non_plume))
    else:
        row["count_above_baseline_threshold"] = -1
        row["non_plume_count_above_baseline_threshold"] = -1
    row["balanced_score"] = (
        float(row["robust_std_non_plume"])
        + 0.6 * float(row["profile_rstd_detected_broad_slope_1p257"])
        + 0.6 * float(row["profile_rstd_detected_thin_slope_0p979"])
        + 0.4 * float(row["profile_rstd_ct_meta_slope_1p267"])
        + 0.4 * float(row["profile_rstd_at_meta_slope_0p977"])
    )
    return row


def color_limits(images: list[np.ndarray], percentile: float = 99.0) -> tuple[float, float]:
    vals = np.concatenate([arr[np.isfinite(arr)].ravel() for arr in images])
    lo, hi = np.nanpercentile(vals, [100.0 - percentile, percentile])
    span = max(abs(float(lo)), abs(float(hi)), 1.0e-9)
    return -span, span


def save_overview(
    path: Path,
    images: dict[str, np.ndarray],
    stripe_maps: dict[str, np.ndarray],
    plume_mask: np.ndarray,
) -> None:
    display = [
        ("raw_mf", images["raw_mf"]),
        ("existing_median", images["existing_median_thin_then_broad"]),
        ("previous_recommended", images["previous_takaku_recommended"]),
        (RECOMMENDED_NAME, images[RECOMMENDED_NAME]),
        ("recommended_removed_stripe", stripe_maps["recommended_total_stripe"]),
        ("new_minus_previous", images[RECOMMENDED_NAME] - images["previous_takaku_recommended"]),
    ]
    vmin, vmax = color_limits([v for _, v in display[:4]], percentile=99.0)
    stripe_v = max(
        float(np.nanpercentile(np.abs(stripe_maps["recommended_total_stripe"]), 99.0)),
        float(np.nanpercentile(np.abs(display[-1][1]), 99.0)),
        1.0e-9,
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, (name, image) in zip(axes.ravel(), display):
        if name in {"recommended_removed_stripe", "new_minus_previous"}:
            im = ax.imshow(image, cmap="coolwarm", vmin=-stripe_v, vmax=stripe_v)
        else:
            im = ax.imshow(image, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.contour(plume_mask, levels=[0.5], colors="black", linewidths=0.6)
        ax.set_title(name.replace("_", " "), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("QA/metadata-guided HISUI MF destriping")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_profile_plot(
    helpers,
    path: Path,
    images: dict[str, np.ndarray],
    plume_mask: np.ndarray,
) -> None:
    selected = [
        "raw_mf",
        "existing_median_thin_then_broad",
        "previous_takaku_recommended",
        RECOMMENDED_NAME,
    ]
    specs = [
        ("AT metadata slope 0.977", AT_META_SLOPE, 2.0),
        ("Detected thin slope 0.979", DETECTED_THIN_SLOPE, 2.0),
        ("Detected broad slope 1.257", DETECTED_BROAD_SLOPE, 18.0),
        ("CT metadata mirror slope 1.267", CT_MIRROR_SLOPE, 18.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (title, slope, bin_width) in zip(axes.ravel(), specs):
        for name in selected:
            x, y = helpers.directional_profile(
                images[name],
                slope=slope,
                bin_width=bin_width,
                protected_mask=plume_mask,
            )
            ax.plot(x, y, lw=1.05, label=name.replace("_", " "))
        ax.set_title(title)
        ax.set_xlabel("line coordinate")
        ax.set_ylabel("median residual MF")
        ax.grid(True, alpha=0.25)
    axes.ravel()[0].legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="QA/metadata-guided destriping for HISUI MF output.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--takaku-output-dir", type=Path, default=DEFAULT_TAKAKU_DIR)
    parser.add_argument("--takaku-script", type=Path, default=DEFAULT_TAKAKU_SCRIPT)
    parser.add_argument("--metadata-txt", type=Path, default=DEFAULT_METADATA_TXT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--canvas-size", type=int, default=512)
    args = parser.parse_args()

    global AT_META_SLOPE, CT_MIRROR_SLOPE

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    helpers = load_takaku_module(args.takaku_script)
    at_slope, ct_mirror_slope, edge_rows = metadata_footprint_slopes(args.metadata_txt)
    AT_META_SLOPE = at_slope
    CT_MIRROR_SLOPE = ct_mirror_slope
    write_csv(output_dir / "footprint_edge_slopes_used.csv", edge_rows)

    raw, plume_mask, existing_thin_only, existing_thin_stripe = load_existing_thin_only(args.input_dir)
    existing_median = np.load(args.input_dir / "median_thin_eachiter_then_broad_median_alpha_corrected.npy").astype(float)
    previous_recommended = np.load(args.takaku_output_dir / "recommended_alpha_corrected.npy").astype(float)
    baseline_threshold = helpers.read_baseline_threshold(args.input_dir / "statistic_quantitative_comparison.csv")

    recommended, recommended_ct_stripe, threshold_rows = helpers.wavelet_horizontal_destripe(
        existing_thin_only,
        plume_mask,
        slope=CT_MIRROR_SLOPE,
        levels_to_filter=(2, 3, 4, 5),
        threshold_scale=1.05,
        diff_fraction=0.25,
        canvas_size=args.canvas_size,
        operation_name=f"{RECOMMENDED_NAME}:ct_meta_dwt",
    )

    comparison_variant, comparison_stripe, comparison_threshold_rows = helpers.wavelet_horizontal_destripe(
        existing_thin_only,
        plume_mask,
        slope=DETECTED_BROAD_SLOPE,
        levels_to_filter=(2, 3, 4, 5),
        threshold_scale=1.05,
        diff_fraction=0.25,
        canvas_size=args.canvas_size,
        operation_name="comparison_existing_thin_then_detected_broad_dwt:broad",
    )
    threshold_rows.extend(comparison_threshold_rows)

    images: dict[str, np.ndarray] = {
        "raw_mf": raw,
        "existing_thin_only": existing_thin_only,
        "existing_median_thin_then_broad": existing_median,
        "previous_takaku_recommended": previous_recommended,
        "comparison_existing_thin_then_detected_broad_dwt": comparison_variant,
        RECOMMENDED_NAME: recommended,
    }
    stripe_maps = {
        "existing_thin_stripe": existing_thin_stripe,
        "recommended_ct_dwt_stripe": recommended_ct_stripe,
        "recommended_total_stripe": raw - recommended,
        "comparison_detected_broad_dwt_stripe": comparison_stripe,
    }

    metrics = [
        collect_metrics(helpers, name, image, plume_mask, baseline_threshold)
        for name, image in images.items()
    ]
    metrics = sorted(metrics, key=lambda row: float(row["balanced_score"]))

    for name, image in images.items():
        np.save(output_dir / f"{name}.npy", image)
    for name, stripe in stripe_maps.items():
        np.save(output_dir / f"{name}.npy", stripe)
    np.save(output_dir / "recommended_alpha_corrected.npy", recommended)
    np.save(output_dir / "recommended_stripe_estimate.npy", raw - recommended)

    write_csv(output_dir / "qa_guided_metrics.csv", metrics)
    write_csv(output_dir / "qa_guided_dwt_thresholds.csv", threshold_rows)
    save_overview(output_dir / "qa_guided_overview.png", images, stripe_maps, plume_mask)
    save_profile_plot(helpers, output_dir / "qa_guided_profiles.png", images, plume_mask)

    best = next(row for row in metrics if row["method"] == RECOMMENDED_NAME)
    previous = next(row for row in metrics if row["method"] == "previous_takaku_recommended")
    summary_lines = [
        "QA/metadata-guided HISUI MF destriping",
        f"input_dir: {args.input_dir}",
        f"output_dir: {output_dir}",
        f"at_meta_slope: {AT_META_SLOPE:.12f}",
        f"ct_mirror_slope: {CT_MIRROR_SLOPE:.12f}",
        f"detected_thin_slope: {DETECTED_THIN_SLOPE:.12f}",
        f"detected_broad_slope: {DETECTED_BROAD_SLOPE:.12f}",
        "recommended: existing thin-line median cleanup + CT metadata-slope DWT levels 2-5",
        "",
        "Recommended metrics:",
        (
            "  robust_std_non_plume={robust_std_non_plume:.6g}, "
            "detected_broad_profile={profile_rstd_detected_broad_slope_1p257:.6g}, "
            "ct_meta_profile={profile_rstd_ct_meta_slope_1p267:.6g}, "
            "detected_thin_profile={profile_rstd_detected_thin_slope_0p979:.6g}, "
            "plume_p95={plume_p95:.6g}"
        ).format(**best),
        "Previous Takaku-hybrid metrics:",
        (
            "  robust_std_non_plume={robust_std_non_plume:.6g}, "
            "detected_broad_profile={profile_rstd_detected_broad_slope_1p257:.6g}, "
            "ct_meta_profile={profile_rstd_ct_meta_slope_1p267:.6g}, "
            "detected_thin_profile={profile_rstd_detected_thin_slope_0p979:.6g}, "
            "plume_p95={plume_p95:.6g}"
        ).format(**previous),
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
