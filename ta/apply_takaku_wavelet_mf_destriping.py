from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


DEFAULT_INPUT_DIR = Path(
    r"D:\research\code\outputs_detected_slopes_orthogonal_thin_eachiter_then_broad_median"
)
DEFAULT_OUTPUT_DIR = Path(r"D:\research\code\takaku_wavelet_mf_destriping\outputs")

THIN_SLOPE = 0.9792723507257799
BROAD_SLOPE = 1.257172298918948

DISPLAY_NAMES = {
    "raw_mf": "raw MF",
    "existing_median_thin_then_broad": "existing median",
    "existing_thin_only": "existing thin only",
    "takaku_broad_only": "DWT broad",
    "dwt_broad_then_median_thin_primary": "DWT broad -> median thin",
    "dwt_broad_then_median_thin_primary_orthogonal": "DWT broad -> median thin + ortho",
    "hybrid_existing_thin_then_takaku_broad": "existing thin + DWT broad",
    "takaku_thin_only": "DWT thin",
    "takaku_thin_then_broad": "DWT thin + broad",
}


def robust_std(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    arr = np.asarray(values, dtype=float)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = np.nanmedian(arr)
    return float(1.4826 * np.nanmedian(np.abs(arr - med)))


def nearest_fill(image: np.ndarray, invalid_mask: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    invalid_mask = np.asarray(invalid_mask, dtype=bool)
    if not invalid_mask.any():
        return image.copy()
    valid = ~invalid_mask & np.isfinite(image)
    if not valid.any():
        return np.nan_to_num(image, nan=float(np.nanmedian(image)))
    _, inds = ndimage.distance_transform_edt(~valid, return_indices=True)
    filled = image.copy()
    filled[invalid_mask | ~np.isfinite(filled)] = image[tuple(inds[:, invalid_mask | ~np.isfinite(filled)])]
    return filled


def central_reflect_pad(image: np.ndarray, canvas_size: int) -> tuple[np.ndarray, tuple[slice, slice]]:
    h, w = image.shape
    if canvas_size < max(h, w):
        raise ValueError("canvas_size must be larger than the input image.")
    py0 = (canvas_size - h) // 2
    px0 = (canvas_size - w) // 2
    py1 = canvas_size - h - py0
    px1 = canvas_size - w - px0
    padded = np.pad(image, ((py0, py1), (px0, px1)), mode="reflect")
    crop = (slice(py0, py0 + h), slice(px0, px0 + w))
    return padded, crop


def haar_decompose2d(image: np.ndarray, levels: int) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    current = np.asarray(image, dtype=float)
    details: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _ in range(levels):
        a = current[0::2, 0::2]
        b = current[0::2, 1::2]
        c = current[1::2, 0::2]
        d = current[1::2, 1::2]
        approx = (a + b + c + d) / 2.0
        horizontal = (a + b - c - d) / 2.0
        vertical = (a - b + c - d) / 2.0
        diagonal = (a - b - c + d) / 2.0
        details.append((horizontal, vertical, diagonal))
        current = approx
    return current, details


def haar_reconstruct2d(
    approx: np.ndarray,
    details: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    current = approx
    for horizontal, vertical, diagonal in reversed(details):
        a = (current + horizontal + vertical + diagonal) / 2.0
        b = (current + horizontal - vertical - diagonal) / 2.0
        c = (current - horizontal + vertical - diagonal) / 2.0
        d = (current - horizontal - vertical + diagonal) / 2.0
        out = np.empty((current.shape[0] * 2, current.shape[1] * 2), dtype=float)
        out[0::2, 0::2] = a
        out[0::2, 1::2] = b
        out[1::2, 0::2] = c
        out[1::2, 1::2] = d
        current = out
    return current


def histogram_difference_threshold(
    target: np.ndarray,
    reference: np.ndarray,
    *,
    bins: int = 128,
    diff_fraction: float = 0.25,
) -> float:
    target_abs = np.abs(target[np.isfinite(target)]).ravel()
    reference_abs = np.abs(reference[np.isfinite(reference)]).ravel()
    if target_abs.size == 0 or reference_abs.size == 0:
        return 0.0
    upper = float(np.nanpercentile(np.concatenate([target_abs, reference_abs]), 99.5))
    if not np.isfinite(upper) or upper <= 0:
        return 0.0
    hist_t, edges = np.histogram(target_abs, bins=bins, range=(0.0, upper), density=True)
    hist_r, _ = np.histogram(reference_abs, bins=bins, range=(0.0, upper), density=True)
    diff = hist_t - hist_r
    diff_max = float(np.nanmax(diff))
    if not np.isfinite(diff_max) or diff_max <= 0:
        return 0.0
    centers = (edges[:-1] + edges[1:]) / 2.0
    keep = np.flatnonzero(diff >= diff_fraction * diff_max)
    if keep.size == 0:
        return 0.0
    threshold = float(centers[keep[-1]])
    return min(threshold, float(np.nanpercentile(target_abs, 90.0)))


def soft_threshold(coeff: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0 or not np.isfinite(threshold):
        return coeff.copy()
    return np.sign(coeff) * np.maximum(np.abs(coeff) - threshold, 0.0)


def choose_rotation_angle(image: np.ndarray, slope: float, canvas_size: int = 512) -> tuple[float, dict[str, float]]:
    padded, _ = central_reflect_pad(image, canvas_size)
    theta = math.degrees(math.atan(slope))
    scores: dict[str, float] = {}
    candidates = [theta, -theta]
    best_angle = candidates[0]
    best_score = -np.inf
    for angle in candidates:
        rotated = ndimage.rotate(padded, angle=angle, reshape=False, order=1, mode="reflect")
        row_profile = np.nanmedian(rotated - np.nanmedian(rotated), axis=1)
        col_profile = np.nanmedian(rotated - np.nanmedian(rotated), axis=0)
        score = robust_std(row_profile) / (robust_std(col_profile) + 1.0e-12)
        scores[f"{angle:.6f}"] = score
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle, scores


def wavelet_horizontal_destripe(
    image: np.ndarray,
    protected_mask: np.ndarray,
    *,
    slope: float,
    levels_to_filter: tuple[int, ...],
    max_level: int = 6,
    threshold_scale: float = 0.75,
    diff_fraction: float = 0.25,
    canvas_size: int = 512,
    operation_name: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | str]]]:
    expanded_mask = ndimage.binary_dilation(protected_mask, iterations=2)
    filled = nearest_fill(image, expanded_mask)
    padded, crop = central_reflect_pad(filled, canvas_size)

    angle, angle_scores = choose_rotation_angle(filled, slope=slope, canvas_size=canvas_size)
    rotated = ndimage.rotate(padded, angle=angle, reshape=False, order=1, mode="reflect")
    approx, details = haar_decompose2d(rotated, levels=max_level)
    new_details: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    threshold_rows: list[dict[str, float | str]] = []

    for level, (horizontal, vertical, diagonal) in enumerate(details, start=1):
        if level in levels_to_filter:
            raw_threshold = histogram_difference_threshold(
                horizontal,
                vertical,
                diff_fraction=diff_fraction,
            )
            threshold = threshold_scale * raw_threshold
            filtered_horizontal = soft_threshold(horizontal, threshold)
            changed_fraction = float(np.mean(np.abs(filtered_horizontal - horizontal) > 0))
        else:
            raw_threshold = 0.0
            threshold = 0.0
            filtered_horizontal = horizontal
            changed_fraction = 0.0

        threshold_rows.append(
            {
                "operation": operation_name,
                "slope": slope,
                "chosen_rotation_deg": angle,
                "angle_score_pos": angle_scores.get(f"{math.degrees(math.atan(slope)):.6f}", float("nan")),
                "angle_score_neg": angle_scores.get(f"{-math.degrees(math.atan(slope)):.6f}", float("nan")),
                "level": level,
                "filtered": int(level in levels_to_filter),
                "coeff_shape": f"{horizontal.shape[0]}x{horizontal.shape[1]}",
                "horizontal_robust_std": robust_std(horizontal),
                "vertical_robust_std": robust_std(vertical),
                "raw_threshold": raw_threshold,
                "applied_threshold": threshold,
                "changed_fraction": changed_fraction,
            }
        )
        new_details.append((filtered_horizontal, vertical, diagonal))

    reconstructed_rotated = haar_reconstruct2d(approx, new_details)
    stripe_rotated = rotated - reconstructed_rotated
    stripe_padded = ndimage.rotate(stripe_rotated, angle=-angle, reshape=False, order=1, mode="reflect")
    stripe = stripe_padded[crop]

    corrected = image - stripe
    return corrected, stripe, threshold_rows


def directional_profile(
    image: np.ndarray,
    *,
    slope: float,
    bin_width: float,
    protected_mask: np.ndarray | None = None,
    residual_sigma: float = 12.0,
    min_count: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    smooth = ndimage.gaussian_filter(image, sigma=residual_sigma, mode="nearest")
    residual = image - smooth
    yy, xx = np.indices(image.shape)
    coord = yy - slope * xx
    valid = np.isfinite(residual)
    if protected_mask is not None:
        valid &= ~protected_mask
    if not valid.any():
        return np.array([]), np.array([])
    bins = np.floor((coord[valid] - np.nanmin(coord[valid])) / bin_width).astype(int)
    max_bin = int(np.nanmax(bins))
    centers: list[float] = []
    medians: list[float] = []
    coords_valid = coord[valid]
    residual_valid = residual[valid]
    for b in range(max_bin + 1):
        selected = bins == b
        if int(np.sum(selected)) < min_count:
            continue
        centers.append(float(np.nanmedian(coords_valid[selected])))
        medians.append(float(np.nanmedian(residual_valid[selected])))
    return np.asarray(centers), np.asarray(medians)


def fixed_slope_line_id_map(
    shape: tuple[int, int],
    *,
    slope: float,
    line_bin_width: float,
    direction_key: str = "y_minus_x",
) -> np.ndarray:
    rows, cols = np.indices(shape)
    if direction_key == "y_minus_x":
        coord = rows - float(slope) * cols
    elif direction_key == "y_plus_x":
        coord = rows + float(slope) * cols
    else:
        raise ValueError("direction_key must be 'y_minus_x' or 'y_plus_x'.")
    return np.rint(coord / float(line_bin_width)).astype(np.int32)


def median_fixed_slope_destripe(
    image: np.ndarray,
    *,
    slope: float,
    line_bin_width: float = 2.0,
    min_pixels_per_line: int = 5,
    direction_key: str = "y_minus_x",
    valid_mask: np.ndarray | None = None,
    exclude_mask: np.ndarray | None = None,
    preserve_global_median: bool = True,
    operation_name: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | str]]]:
    alpha = np.asarray(image, dtype=float)
    finite = np.isfinite(alpha)
    valid = finite if valid_mask is None else np.asarray(valid_mask, dtype=bool) & finite
    estimate = valid.copy()
    if exclude_mask is not None:
        estimate &= ~np.asarray(exclude_mask, dtype=bool)
    if not np.any(estimate):
        estimate = valid.copy()

    ids = fixed_slope_line_id_map(
        alpha.shape,
        slope=slope,
        line_bin_width=line_bin_width,
        direction_key=direction_key,
    )
    id_values = np.arange(int(np.nanmin(ids[valid])), int(np.nanmax(ids[valid])) + 1, dtype=np.int32)
    global_median = float(np.nanmedian(alpha[estimate]))

    stripe_map = np.zeros_like(alpha, dtype=float)
    rows_out: list[dict[str, float | int | str]] = []
    for line_id in id_values:
        m = (ids == line_id) & estimate
        used_fallback = 0
        if int(np.sum(m)) < min_pixels_per_line:
            fallback = (ids == line_id) & valid
            if int(np.sum(fallback)) >= min_pixels_per_line:
                m = fallback
                used_fallback = 1

        count = int(np.sum(m))
        if count >= min_pixels_per_line:
            line_median = float(np.nanmedian(alpha[m]))
            offset = line_median - global_median if preserve_global_median else line_median
        else:
            line_median = float("nan")
            offset = 0.0
        stripe_map[ids == line_id] = offset
        rows_out.append(
            {
                "operation": operation_name,
                "direction_key": direction_key,
                "slope": float(slope),
                "signed_slope": float(slope) if direction_key == "y_minus_x" else -float(slope),
                "line_bin_width": float(line_bin_width),
                "line_id": int(line_id),
                "n_pixels_used": count,
                "used_fallback": used_fallback,
                "line_median": line_median,
                "global_median": global_median,
                "stripe_offset_subtracted": float(offset),
            }
        )

    stripe_map[~valid] = np.nan
    corrected = alpha - stripe_map
    corrected[~finite] = np.nan
    return corrected, stripe_map, rows_out


def read_baseline_threshold(metric_csv: Path) -> float:
    if not metric_csv.exists():
        return float("nan")
    with metric_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("method") or row.get("thin_stat")
            if label in {"baseline_no_destripe", "baseline"}:
                try:
                    return float(row.get("threshold", "nan"))
                except ValueError:
                    return float("nan")
    return float("nan")


def collect_metrics(
    name: str,
    image: np.ndarray,
    plume_mask: np.ndarray,
    baseline_threshold: float,
) -> dict[str, float | str]:
    non_plume = ~plume_mask
    _, broad_prof = directional_profile(
        image,
        slope=BROAD_SLOPE,
        bin_width=18.0,
        protected_mask=plume_mask,
    )
    _, thin_prof = directional_profile(
        image,
        slope=THIN_SLOPE,
        bin_width=2.0,
        protected_mask=plume_mask,
    )
    out: dict[str, float | str] = {
        "method": name,
        "robust_std_all": robust_std(image),
        "robust_std_non_plume": robust_std(image, non_plume),
        "p95_abs_non_plume": float(np.nanpercentile(np.abs(image[non_plume]), 95.0)),
        "broad_profile_robust_std": robust_std(broad_prof),
        "thin_profile_robust_std": robust_std(thin_prof),
        "plume_mean": float(np.nanmean(image[plume_mask])) if plume_mask.any() else float("nan"),
        "plume_p95": float(np.nanpercentile(image[plume_mask], 95.0)) if plume_mask.any() else float("nan"),
        "plume_max": float(np.nanmax(image[plume_mask])) if plume_mask.any() else float("nan"),
    }
    if np.isfinite(baseline_threshold):
        out["count_above_baseline_threshold"] = int(np.sum(image > baseline_threshold))
        out["non_plume_count_above_baseline_threshold"] = int(np.sum((image > baseline_threshold) & non_plume))
    else:
        out["count_above_baseline_threshold"] = float("nan")
        out["non_plume_count_above_baseline_threshold"] = float("nan")
    return out


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


def save_overview(
    output_path: Path,
    images: dict[str, np.ndarray],
    raw: np.ndarray,
    plume_mask: np.ndarray,
    *,
    draw_contours: bool = True,
) -> None:
    selected = [
        "raw_mf",
        "existing_median_thin_then_broad",
        "takaku_broad_only",
        "dwt_broad_then_median_thin_primary",
        "hybrid_existing_thin_then_takaku_broad",
        "takaku_thin_then_broad",
    ]
    selected = [name for name in selected if name in images]
    finite_raw = raw[np.isfinite(raw)]
    vmin = float(np.nanpercentile(finite_raw, 1.0))
    vmax = float(np.nanpercentile(finite_raw, 99.5))
    diff_abs = max(
        1.0e-9,
        float(np.nanpercentile(np.abs(np.stack([images[name] - raw for name in selected if name != "raw_mf"])), 99.0)),
    )

    fig, axes = plt.subplots(2, len(selected), figsize=(15, 7), constrained_layout=True)
    for i, name in enumerate(selected):
        ax = axes[0, i]
        im = ax.imshow(images[name], cmap="viridis", vmin=vmin, vmax=vmax)
        if draw_contours:
            ax.contour(plume_mask, levels=[0.5], colors="white", linewidths=0.6)
        ax.set_title(DISPLAY_NAMES.get(name, name), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == len(selected) - 1:
            fig.colorbar(im, ax=ax, shrink=0.75)

        ax = axes[1, i]
        diff = images[name] - raw
        im2 = ax.imshow(diff, cmap="coolwarm", vmin=-diff_abs, vmax=diff_abs)
        if draw_contours:
            ax.contour(plume_mask, levels=[0.5], colors="black", linewidths=0.5)
        ax.set_title(f"{DISPLAY_NAMES.get(name, name)} - raw", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == len(selected) - 1:
            fig.colorbar(im2, ax=ax, shrink=0.75)
    fig.suptitle("Takaku-inspired directional DWT destriping for HISUI MF ROI")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_single_panel(
    output_path: Path,
    image: np.ndarray,
    *,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 5.2), constrained_layout=True)
    im = ax.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_profiles(output_path: Path, images: dict[str, np.ndarray], plume_mask: np.ndarray) -> None:
    selected = [
        "raw_mf",
        "existing_median_thin_then_broad",
        "takaku_broad_only",
        "dwt_broad_then_median_thin_primary",
        "existing_thin_only",
        "hybrid_existing_thin_then_takaku_broad",
        "takaku_thin_only",
        "takaku_thin_then_broad",
    ]
    selected = [name for name in selected if name in images]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for name in selected:
        x, y = directional_profile(
            images[name],
            slope=BROAD_SLOPE,
            bin_width=18.0,
            protected_mask=plume_mask,
        )
        axes[0].plot(x, y, label=DISPLAY_NAMES.get(name, name), linewidth=1.3)
    axes[0].set_title("Broad-stripe direction profile: median residual by y - 1.257x")
    axes[0].set_xlabel("line coordinate")
    axes[0].set_ylabel("median residual MF")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    for name in selected:
        x, y = directional_profile(
            images[name],
            slope=THIN_SLOPE,
            bin_width=2.0,
            protected_mask=plume_mask,
        )
        axes[1].plot(x, y, label=DISPLAY_NAMES.get(name, name), linewidth=1.1)
    axes[1].set_title("Thin-stripe direction profile: median residual by y - 0.979x")
    axes[1].set_xlabel("line coordinate")
    axes[1].set_ylabel("median residual MF")
    axes[1].grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_stripe_maps(output_path: Path, stripe_maps: dict[str, np.ndarray], plume_mask: np.ndarray) -> None:
    ncols = 3
    nrows = int(math.ceil(len(stripe_maps) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    vmax = max(float(np.nanpercentile(np.abs(v), 99.0)) for v in stripe_maps.values())
    vmax = max(vmax, 1.0e-9)
    im = None
    for ax, (name, stripe) in zip(axes_flat, stripe_maps.items()):
        im = ax.imshow(stripe, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.contour(plume_mask, levels=[0.5], colors="black", linewidths=0.5)
        ax.set_title(name.replace("_", " "), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_flat[len(stripe_maps) :]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=list(axes_flat), shrink=0.75)
    fig.suptitle("Estimated stripe components")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Takaku-inspired directional DWT destriping for MF alpha images.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--canvas-size", type=int, default=512)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(input_dir / "baseline_no_destripe_alpha_raw.npy").astype(float)
    plume_mask = np.load(input_dir / "baseline_no_destripe_plume_mask.npy").astype(bool)
    existing = np.load(input_dir / "median_thin_eachiter_then_broad_median_alpha_corrected.npy").astype(float)
    existing_thin_primary = np.load(
        input_dir
        / "median_thin_eachiter_then_broad_median_stripe_map_thin_thin_detected_thin_high_alpha_primary_positive_y_minus_x_slope_0.9793_bin_2.0.npy"
    ).astype(float)
    existing_thin_orthogonal = np.load(
        input_dir
        / "median_thin_eachiter_then_broad_median_stripe_map_thin_thin_detected_thin_high_alpha_orthogonal_to_primary_y_plus_x_slope_1.0212_bin_2.0.npy"
    ).astype(float)
    existing_thin_only = raw - existing_thin_primary - existing_thin_orthogonal
    baseline_threshold = read_baseline_threshold(input_dir / "statistic_quantitative_comparison.csv")

    images: dict[str, np.ndarray] = {
        "raw_mf": raw,
        "existing_median_thin_then_broad": existing,
        "existing_thin_only": existing_thin_only,
    }
    stripe_maps: dict[str, np.ndarray] = {}
    thresholds: list[dict[str, float | str]] = []
    median_line_rows: list[dict[str, float | int | str]] = []

    broad, broad_stripe, rows = wavelet_horizontal_destripe(
        raw,
        plume_mask,
        slope=BROAD_SLOPE,
        levels_to_filter=(3, 4, 5),
        threshold_scale=0.75,
        diff_fraction=0.25,
        canvas_size=args.canvas_size,
        operation_name="takaku_broad_only:broad",
    )
    images["takaku_broad_only"] = broad
    stripe_maps["takaku_broad_only_stripe"] = broad_stripe
    thresholds.extend(rows)

    broad_then_thin_primary, broad_then_thin_primary_stripe, rows_median = median_fixed_slope_destripe(
        broad,
        slope=THIN_SLOPE,
        line_bin_width=2.0,
        min_pixels_per_line=5,
        direction_key="y_minus_x",
        valid_mask=np.isfinite(broad),
        exclude_mask=None,
        preserve_global_median=True,
        operation_name="dwt_broad_then_median_thin_primary:thin_primary",
    )
    images["dwt_broad_then_median_thin_primary"] = broad_then_thin_primary
    stripe_maps["dwt_broad_then_median_thin_primary_stripe"] = broad_then_thin_primary_stripe
    median_line_rows.extend(rows_median)

    broad_then_thin_ortho, broad_then_thin_ortho_stripe, rows_median = median_fixed_slope_destripe(
        broad_then_thin_primary,
        slope=1.0211663785451084,
        line_bin_width=2.0,
        min_pixels_per_line=5,
        direction_key="y_plus_x",
        valid_mask=np.isfinite(broad_then_thin_primary),
        exclude_mask=None,
        preserve_global_median=True,
        operation_name="dwt_broad_then_median_thin_primary_orthogonal:thin_orthogonal",
    )
    images["dwt_broad_then_median_thin_primary_orthogonal"] = broad_then_thin_ortho
    stripe_maps["dwt_broad_then_median_thin_orthogonal_stripe"] = broad_then_thin_ortho_stripe
    median_line_rows.extend(rows_median)

    hybrid, hybrid_broad_stripe, rows = wavelet_horizontal_destripe(
        existing_thin_only,
        plume_mask,
        slope=BROAD_SLOPE,
        levels_to_filter=(3, 4, 5),
        threshold_scale=0.75,
        diff_fraction=0.25,
        canvas_size=args.canvas_size,
        operation_name="hybrid_existing_thin_then_takaku_broad:broad",
    )
    images["hybrid_existing_thin_then_takaku_broad"] = hybrid
    stripe_maps["hybrid_existing_thin_then_takaku_broad_stripe"] = hybrid_broad_stripe
    thresholds.extend(rows)

    thin, thin_stripe, rows = wavelet_horizontal_destripe(
        raw,
        plume_mask,
        slope=THIN_SLOPE,
        levels_to_filter=(1, 2, 3),
        threshold_scale=0.55,
        diff_fraction=0.25,
        canvas_size=args.canvas_size,
        operation_name="takaku_thin_only:thin",
    )
    images["takaku_thin_only"] = thin
    stripe_maps["takaku_thin_only_stripe"] = thin_stripe
    thresholds.extend(rows)

    thin_step, thin_step_stripe, rows = wavelet_horizontal_destripe(
        raw,
        plume_mask,
        slope=THIN_SLOPE,
        levels_to_filter=(1, 2, 3),
        threshold_scale=0.55,
        diff_fraction=0.25,
        canvas_size=args.canvas_size,
        operation_name="takaku_thin_then_broad:thin",
    )
    thresholds.extend(rows)
    combined, broad_step_stripe, rows = wavelet_horizontal_destripe(
        thin_step,
        plume_mask,
        slope=BROAD_SLOPE,
        levels_to_filter=(3, 4, 5),
        threshold_scale=0.75,
        diff_fraction=0.25,
        canvas_size=args.canvas_size,
        operation_name="takaku_thin_then_broad:broad",
    )
    images["takaku_thin_then_broad"] = combined
    stripe_maps["takaku_thin_then_broad_thin_stripe"] = thin_step_stripe
    stripe_maps["takaku_thin_then_broad_broad_stripe"] = broad_step_stripe
    thresholds.extend(rows)

    metrics = [collect_metrics(name, image, plume_mask, baseline_threshold) for name, image in images.items()]

    for name, image in images.items():
        np.save(output_dir / f"{name}.npy", image)
    for name, stripe in stripe_maps.items():
        np.save(output_dir / f"{name}.npy", stripe)
    np.save(output_dir / "recommended_alpha_corrected.npy", images["hybrid_existing_thin_then_takaku_broad"])
    np.save(output_dir / "recommended_stripe_estimate.npy", raw - images["hybrid_existing_thin_then_takaku_broad"])
    np.save(output_dir / "requested_dwt_broad_then_median_thin_primary.npy", images["dwt_broad_then_median_thin_primary"])
    np.save(
        output_dir / "requested_dwt_broad_then_median_thin_primary_stripe_estimate.npy",
        raw - images["dwt_broad_then_median_thin_primary"],
    )
    np.save(
        output_dir / "requested_dwt_broad_then_median_thin_primary_orthogonal.npy",
        images["dwt_broad_then_median_thin_primary_orthogonal"],
    )

    write_csv(output_dir / "takaku_wavelet_metrics.csv", metrics)
    write_csv(output_dir / "takaku_wavelet_thresholds.csv", thresholds)
    write_csv(output_dir / "dwt_broad_then_thin_median_line_table.csv", median_line_rows)
    save_overview(output_dir / "takaku_wavelet_overview.png", images, raw, plume_mask)
    save_overview(
        output_dir / "takaku_wavelet_overview_no_contours.png",
        images,
        raw,
        plume_mask,
        draw_contours=False,
    )
    finite_raw = raw[np.isfinite(raw)]
    vmin = float(np.nanpercentile(finite_raw, 1.0))
    vmax = float(np.nanpercentile(finite_raw, 99.5))
    save_single_panel(
        output_dir / "existing_median_no_contours.png",
        images["existing_median_thin_then_broad"],
        title="existing median",
        vmin=vmin,
        vmax=vmax,
    )
    save_single_panel(
        output_dir / "requested_dwt_broad_then_median_thin_primary_no_contours.png",
        images["dwt_broad_then_median_thin_primary"],
        title="DWT broad -> median thin",
        vmin=vmin,
        vmax=vmax,
    )
    save_single_panel(
        output_dir / "dwt_thin_only_no_contours.png",
        images["takaku_thin_only"],
        title="DWT thin only",
        vmin=vmin,
        vmax=vmax,
    )
    save_profiles(output_dir / "takaku_wavelet_profiles.png", images, plume_mask)
    save_stripe_maps(output_dir / "takaku_wavelet_stripe_maps.png", stripe_maps, plume_mask)

    summary_lines = [
        "Takaku-inspired directional DWT destriping for HISUI MF ROI",
        f"input_dir: {input_dir}",
        f"output_dir: {output_dir}",
        f"thin_slope: {THIN_SLOPE:.12f}",
        f"broad_slope: {BROAD_SLOPE:.12f}",
        f"baseline_threshold: {baseline_threshold}",
        "recommended: hybrid_existing_thin_then_takaku_broad",
        "",
        "Metrics:",
    ]
    for row in metrics:
        summary_lines.append(
            "  {method}: robust_std_non_plume={robust_std_non_plume:.6g}, "
            "broad_profile_robust_std={broad_profile_robust_std:.6g}, "
            "thin_profile_robust_std={thin_profile_robust_std:.6g}, "
            "plume_p95={plume_p95:.6g}".format(**row)
        )
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
