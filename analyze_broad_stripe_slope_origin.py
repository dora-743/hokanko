from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ALPHA = Path(
    r"D:\research\code\outputs_paper_sensor_geometry_destripe\baseline_no_destripe_alpha_corrected.npy"
)
DEFAULT_SIGNED_DETECTION_DIR = Path(r"D:\research\code\outputs_general_scene_slope_detection_signed")
DEFAULT_OUTPUT_DIR = Path(r"D:\research\code\outputs_broad_stripe_slope_origin")


def read_metadata(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    pattern = re.compile(r"^\s*([^=#]+?)\s*=\s*(.*?)\s*$")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pattern.match(line)
        if not m:
            continue
        key = m.group(1).strip()
        value = m.group(2).strip().strip('"')
        meta[key] = value
    return meta


def f(meta: dict[str, str], key: str) -> float:
    return float(meta[key])


def latlon_to_utm_wgs84(lat_deg: float, lon_deg: float, zone: int) -> tuple[float, float]:
    """Return UTM easting/northing in meters for the northern hemisphere."""
    a = 6378137.0
    f_inv = 298.257223563
    f0 = 1.0 / f_inv
    e2 = f0 * (2.0 - f0)
    ep2 = e2 / (1.0 - e2)
    k0 = 0.9996
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    tan_lat = math.tan(lat)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    t = tan_lat * tan_lat
    c = ep2 * cos_lat * cos_lat
    aa = cos_lat * (lon - lon0)
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
        * tan_lat
        * (
            aa**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * aa**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * ep2) * aa**6 / 720
        )
    )
    return easting, northing


def row_col_from_latlon(
    lat: float,
    lon: float,
    map_ul_lat: float,
    map_ul_lon: float,
    zone: int,
    grid_m: float,
) -> tuple[float, float]:
    e, n = latlon_to_utm_wgs84(lat, lon, zone)
    ul_e, ul_n = latlon_to_utm_wgs84(map_ul_lat, map_ul_lon, zone)
    row = (ul_n - n) / grid_m
    col = (e - ul_e) / grid_m
    return row, col


def slope_angle_from_points(p0: tuple[float, float], p1: tuple[float, float]) -> tuple[float, float]:
    dr = p1[0] - p0[0]
    dc = p1[1] - p0[1]
    slope = dr / dc
    angle = math.degrees(math.atan(slope))
    return slope, angle


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    out = np.full_like(values, np.nan, dtype=float)
    if not finite.any():
        return out
    lo = np.nanpercentile(values[finite], 1)
    hi = np.nanpercentile(values[finite], 99)
    if not np.isfinite(hi - lo) or hi <= lo:
        out[finite] = 0.0
        return out
    out[finite] = np.clip((values[finite] - lo) / (hi - lo), 0, 1)
    return out


def robust_z(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vals = values[valid & np.isfinite(values)]
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med))
    sigma = 1.4826 * mad if mad > 0 else np.nanstd(vals)
    return (values - med) / sigma


def nan_gaussian(values: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    filled = np.where(valid, values, 0.0)
    weight = valid.astype(float)
    smooth_num = ndimage.gaussian_filter(filled, sigma=sigma, mode="nearest")
    smooth_den = ndimage.gaussian_filter(weight, sigma=sigma, mode="nearest")
    return np.where(smooth_den > 1e-6, smooth_num / smooth_den, np.nan)


def line_ids(rows: np.ndarray, cols: np.ndarray, signed_slope: float, bin_width: float) -> np.ndarray:
    if signed_slope >= 0:
        coord = rows - signed_slope * cols
    else:
        coord = rows + abs(signed_slope) * cols
    return np.rint(coord / bin_width).astype(np.int64)


def top_count_score(
    rows: np.ndarray,
    cols: np.ndarray,
    slopes: np.ndarray,
    bin_width: float,
    top_k: int = 6,
) -> np.ndarray:
    scores = np.zeros(len(slopes), dtype=float)
    if rows.size == 0:
        return scores
    for i, slope in enumerate(slopes):
        ids = line_ids(rows, cols, slope, bin_width)
        ids = ids - ids.min()
        counts = np.bincount(ids)
        if counts.size == 0:
            continue
        scores[i] = np.sort(counts)[-top_k:].sum()
    return scores


def weighted_line_p99_score(
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    slopes: np.ndarray,
    bin_width: float,
    min_count: int,
    use_abs_mean: bool,
) -> np.ndarray:
    scores = np.zeros(len(slopes), dtype=float)
    for i, slope in enumerate(slopes):
        ids = line_ids(rows, cols, slope, bin_width)
        ids = ids - ids.min()
        counts = np.bincount(ids)
        sums = np.bincount(ids, weights=weights)
        ok = counts >= min_count
        if not ok.any():
            continue
        if use_abs_mean:
            line_values = np.abs(sums[ok] / counts[ok])
        else:
            line_values = sums[ok] / counts[ok]
        scores[i] = np.nanpercentile(line_values, 99)
    return scores


def nearest_row(df: pd.DataFrame, angle_col: str, target_angle: float) -> pd.Series:
    idx = (df[angle_col] - target_angle).abs().idxmin()
    return df.loc[idx]


def derive_footprint(meta: dict[str, str]) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    zone = int(f(meta, "UTMZone"))
    grid_m = f(meta, "GridCellSizeMeter")
    map_ul_lat = f(meta, "MapUpperLeftLatitudeDegree")
    map_ul_lon = f(meta, "MapUpperLeftLongitudeDegree")
    corners = {
        "UL": (
            f(meta, "ObservationUpperLeftLatitudeDegree"),
            f(meta, "ObservationUpperLeftLongitudeDegree"),
        ),
        "UR": (
            f(meta, "ObservationUpperRightLatitudeDegree"),
            f(meta, "ObservationUpperRightLongitudeDegree"),
        ),
        "LL": (
            f(meta, "ObservationLowerLeftLatitudeDegree"),
            f(meta, "ObservationLowerLeftLongitudeDegree"),
        ),
        "LR": (
            f(meta, "ObservationLowerRightLatitudeDegree"),
            f(meta, "ObservationLowerRightLongitudeDegree"),
        ),
    }
    rc = {
        name: row_col_from_latlon(lat, lon, map_ul_lat, map_ul_lon, zone, grid_m)
        for name, (lat, lon) in corners.items()
    }
    edges = [
        ("left_edge_UL_to_LL_inferred_AT", "UL", "LL"),
        ("right_edge_UR_to_LR_inferred_AT", "UR", "LR"),
        ("top_edge_UL_to_UR_inferred_CT", "UL", "UR"),
        ("bottom_edge_LL_to_LR_inferred_CT", "LL", "LR"),
    ]
    rows = []
    for label, a, b in edges:
        slope, angle = slope_angle_from_points(rc[a], rc[b])
        rows.append(
            {
                "edge": label,
                "from": a,
                "to": b,
                "row0": rc[a][0],
                "col0": rc[a][1],
                "row1": rc[b][0],
                "col1": rc[b][1],
                "signed_slope_row_per_col": slope,
                "angle_deg_image_row": angle,
            }
        )
    return pd.DataFrame(rows), rc


def build_key_slopes(edge_df: pd.DataFrame, selected_df: pd.DataFrame) -> pd.DataFrame:
    thin = selected_df[selected_df["detection_type"].eq("thin_high_alpha")].iloc[0]
    broad = selected_df[selected_df["detection_type"].eq("broad_offset")].iloc[0]
    at_mean = edge_df[edge_df["edge"].str.contains("AT")]["signed_slope_row_per_col"].mean()
    ct_mean = edge_df[edge_df["edge"].str.contains("CT")]["signed_slope_row_per_col"].mean()
    rows = [
        {
            "label": "detected_thin_high_alpha",
            "signed_slope": float(thin["signed_slope"]),
            "angle_deg": float(thin["angle_deg_signed"]),
            "interpretation": "Detected high-alpha thin-line direction",
        },
        {
            "label": "footprint_inferred_AT_mean",
            "signed_slope": float(at_mean),
            "angle_deg": math.degrees(math.atan(at_mean)),
            "interpretation": "Mean of left/right observation footprint edges, inferred along-track",
        },
        {
            "label": "detected_broad_offset",
            "signed_slope": float(broad["signed_slope"]),
            "angle_deg": float(broad["angle_deg_signed"]),
            "interpretation": "Detected broad MF offset direction",
        },
        {
            "label": "footprint_inferred_CT_mean_image_signed",
            "signed_slope": float(ct_mean),
            "angle_deg": math.degrees(math.atan(ct_mean)),
            "interpretation": "Mean of top/bottom observation footprint edges, inferred cross-track",
        },
        {
            "label": "footprint_inferred_CT_mirror_positive",
            "signed_slope": float(abs(ct_mean)),
            "angle_deg": math.degrees(math.atan(abs(ct_mean))),
            "interpretation": "Same absolute CT slope with image-row positive sign",
        },
        {
            "label": "perpendicular_to_AT_mean",
            "signed_slope": float(-1.0 / at_mean),
            "angle_deg": math.degrees(math.atan(-1.0 / at_mean)),
            "interpretation": "Direction perpendicular to inferred along-track mean",
        },
    ]
    return pd.DataFrame(rows)


def plot_existing_curves(
    thin_df: pd.DataFrame,
    broad_df: pd.DataFrame,
    key_df: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, df, title in [
        (axes[0], thin_df, "Existing thin high-alpha score"),
        (axes[1], broad_df, "Existing broad offset score"),
    ]:
        sdf = df.sort_values("angle_deg_signed")
        x = sdf["angle_deg_signed"].to_numpy()
        y = normalize01(sdf["score"].to_numpy())
        ax.plot(x, y, lw=1.2, color="black")
        ax.set_ylabel("normalized score")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        for _, row in key_df.iterrows():
            ax.axvline(row["angle_deg"], color="tab:red", alpha=0.18, lw=1)
    axes[-1].set_xlabel("signed image angle (deg); positive runs upper-left to lower-right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_hough_curves(score_df: pd.DataFrame, key_df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        ("high_alpha_top6_count", "High-alpha point Hough"),
        ("broad_abs_line_mean_p99", "High-pass residual line-mean"),
        ("broad_abs_line_mean_p99_no_extreme_alpha", "Residual line-mean, extreme alpha masked"),
        ("gradient_abs_line_mean_p99", "Gradient-weight line mean"),
        ("mask_boundary_top6_count", "Valid-mask boundary Hough"),
    ]
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 11), sharex=True)
    x = score_df["angle_deg"].to_numpy()
    colors = {
        "detected_thin_high_alpha": "tab:red",
        "footprint_inferred_AT_mean": "tab:green",
        "detected_broad_offset": "tab:orange",
        "footprint_inferred_CT_mean_image_signed": "tab:blue",
        "footprint_inferred_CT_mirror_positive": "tab:purple",
        "perpendicular_to_AT_mean": "tab:gray",
    }
    for ax, (col, title) in zip(axes, cols):
        ax.plot(x, normalize01(score_df[col].to_numpy()), lw=1.2, color="black")
        for _, row in key_df.iterrows():
            ax.axvline(
                row["angle_deg"],
                color=colors.get(row["label"], "tab:red"),
                alpha=0.35,
                lw=1.1,
            )
        ax.set_title(title)
        ax.set_ylabel("normalized score")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("signed image angle (deg); positive runs upper-left to lower-right")
    handles = [
        plt.Line2D([0], [0], color=colors.get(row["label"], "black"), lw=2, label=row["label"])
        for _, row in key_df.iterrows()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_footprint(
    alpha: np.ndarray,
    rc: dict[str, tuple[float, float]],
    key_df: pd.DataFrame,
    out_path: Path,
) -> None:
    valid = np.isfinite(alpha)
    lo, hi = np.nanpercentile(alpha[valid], [2, 98])
    step = 4
    img = alpha[::step, ::step]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, origin="upper", extent=[0, alpha.shape[1], alpha.shape[0], 0])
    poly_names = ["UL", "UR", "LR", "LL", "UL"]
    xs = [rc[name][1] for name in poly_names]
    ys = [rc[name][0] for name in poly_names]
    ax.plot(xs, ys, color="cyan", lw=2.0, label="metadata observation footprint")
    center_r = np.mean([p[0] for p in rc.values()])
    center_c = np.mean([p[1] for p in rc.values()])
    x0, x1 = 0, alpha.shape[1] - 1
    colors = {
        "detected_thin_high_alpha": "red",
        "footprint_inferred_AT_mean": "lime",
        "detected_broad_offset": "orange",
        "footprint_inferred_CT_mean_image_signed": "dodgerblue",
        "footprint_inferred_CT_mirror_positive": "violet",
    }
    for _, row in key_df.iterrows():
        label = row["label"]
        if label not in colors:
            continue
        m = row["signed_slope"]
        y0 = center_r + m * (x0 - center_c)
        y1 = center_r + m * (x1 - center_c)
        ax.plot([x0, x1], [y0, y1], color=colors[label], lw=1.6, alpha=0.9, label=label)
    ax.set_xlim(0, alpha.shape[1])
    ax.set_ylim(alpha.shape[0], 0)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_title("Footprint and candidate directions on baseline MF alpha")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_hough_diagnostics(alpha: np.ndarray, slopes: np.ndarray) -> pd.DataFrame:
    valid = np.isfinite(alpha)
    z = robust_z(alpha, valid)
    high = valid & (z > 4.0)
    high_rows, high_cols = np.nonzero(high)

    smooth = nan_gaussian(alpha, valid, sigma=28.0)
    hp = alpha - smooth
    hp_z = robust_z(hp, valid)

    sample = valid.copy()
    sample[::4, ::4] &= True
    keep = np.zeros_like(sample, dtype=bool)
    keep[::4, ::4] = True
    sample &= keep
    sample_rows, sample_cols = np.nonzero(sample)
    sample_hp = hp_z[sample_rows, sample_cols]

    no_extreme = sample & (np.abs(z) < 3.5)
    no_extreme_rows, no_extreme_cols = np.nonzero(no_extreme)
    no_extreme_hp = hp_z[no_extreme_rows, no_extreme_cols]

    sx = ndimage.sobel(np.where(valid, smooth, np.nanmedian(alpha[valid])), axis=1, mode="nearest")
    sy = ndimage.sobel(np.where(valid, smooth, np.nanmedian(alpha[valid])), axis=0, mode="nearest")
    grad = np.hypot(sx, sy)
    grad_z = robust_z(grad, valid)
    sample_grad = np.abs(grad_z[sample_rows, sample_cols])

    eroded = ndimage.binary_erosion(valid, structure=np.ones((3, 3), dtype=bool), border_value=0)
    boundary = valid & ~eroded
    b_rows, b_cols = np.nonzero(boundary)

    high_score = top_count_score(high_rows, high_cols, slopes, bin_width=2.0, top_k=6)
    boundary_score = top_count_score(b_rows, b_cols, slopes, bin_width=4.0, top_k=8)
    broad_score = weighted_line_p99_score(
        sample_rows,
        sample_cols,
        sample_hp,
        slopes,
        bin_width=18.0,
        min_count=40,
        use_abs_mean=True,
    )
    broad_no_extreme_score = weighted_line_p99_score(
        no_extreme_rows,
        no_extreme_cols,
        no_extreme_hp,
        slopes,
        bin_width=18.0,
        min_count=40,
        use_abs_mean=True,
    )
    grad_score = weighted_line_p99_score(
        sample_rows,
        sample_cols,
        sample_grad,
        slopes,
        bin_width=8.0,
        min_count=30,
        use_abs_mean=True,
    )
    return pd.DataFrame(
        {
            "signed_slope": slopes,
            "angle_deg": np.degrees(np.arctan(slopes)),
            "high_alpha_top6_count": high_score,
            "broad_abs_line_mean_p99": broad_score,
            "broad_abs_line_mean_p99_no_extreme_alpha": broad_no_extreme_score,
            "gradient_abs_line_mean_p99": grad_score,
            "mask_boundary_top6_count": boundary_score,
        }
    )


def attach_nearest_scores(
    key_df: pd.DataFrame,
    thin_df: pd.DataFrame,
    broad_df: pd.DataFrame,
    hough_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in key_df.iterrows():
        angle = row["angle_deg"]
        thin = nearest_row(thin_df, "angle_deg_signed", angle)
        broad = nearest_row(broad_df, "angle_deg_signed", angle)
        hough = nearest_row(hough_df, "angle_deg", angle)
        out = row.to_dict()
        out.update(
            {
                "nearest_existing_thin_angle_deg": thin["angle_deg_signed"],
                "existing_thin_score": thin["score"],
                "nearest_existing_broad_angle_deg": broad["angle_deg_signed"],
                "existing_broad_score": broad["score"],
                "existing_broad_peak_prominence": broad.get("peak_prominence", np.nan),
                "nearest_hough_angle_deg": hough["angle_deg"],
                "hough_high_alpha_top6_count": hough["high_alpha_top6_count"],
                "hough_broad_abs_line_mean_p99": hough["broad_abs_line_mean_p99"],
                "hough_broad_abs_line_mean_p99_no_extreme_alpha": hough[
                    "broad_abs_line_mean_p99_no_extreme_alpha"
                ],
                "hough_gradient_abs_line_mean_p99": hough["gradient_abs_line_mean_p99"],
                "hough_mask_boundary_top6_count": hough["mask_boundary_top6_count"],
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=Path, default=DEFAULT_ALPHA)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--signed-detection-dir", type=Path, default=DEFAULT_SIGNED_DETECTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--angle-step-deg", type=float, default=0.2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = read_metadata(args.metadata)
    edge_df, rc = derive_footprint(meta)
    selected_df = pd.read_csv(args.signed_detection_dir / "signed_slope_search_selected_primary.csv")
    thin_df = pd.read_csv(args.signed_detection_dir / "thin_signed_slope_search_all_candidates.csv")
    broad_df = pd.read_csv(args.signed_detection_dir / "broad_signed_slope_search_all_candidates.csv")

    key_df = build_key_slopes(edge_df, selected_df)
    angles = np.arange(-80.0, 80.0 + args.angle_step_deg / 2, args.angle_step_deg)
    slopes = np.tan(np.deg2rad(angles))
    alpha = np.load(args.alpha)
    hough_df = run_hough_diagnostics(alpha, slopes)

    comparison_df = attach_nearest_scores(key_df, thin_df, broad_df, hough_df)

    edge_df.to_csv(args.output_dir / "footprint_edge_slopes.csv", index=False)
    key_df.to_csv(args.output_dir / "key_slope_definitions.csv", index=False)
    hough_df.to_csv(args.output_dir / "independent_hough_like_scores.csv", index=False)
    comparison_df.to_csv(args.output_dir / "key_slope_comparison.csv", index=False)

    plot_existing_curves(
        thin_df,
        broad_df,
        key_df,
        args.output_dir / "existing_signed_score_curves_with_geometry.png",
    )
    plot_hough_curves(
        hough_df,
        key_df,
        args.output_dir / "independent_hough_like_score_curves.png",
    )
    plot_footprint(
        alpha,
        rc,
        key_df,
        args.output_dir / "footprint_and_candidate_directions.png",
    )

    print("Wrote", args.output_dir)
    print(comparison_df[["label", "signed_slope", "angle_deg", "existing_broad_score", "hough_broad_abs_line_mean_p99", "hough_mask_boundary_top6_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
