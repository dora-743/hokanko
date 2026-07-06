"""
Detect stripe/noise slope families in a 2-D MF alpha image.

The detector intentionally uses two scores:
  1. thin high-alpha lines: high-alpha pixels concentrated on line bins.
     One slope family is selected.
  2. broad offset bands: the previous angle sweep destriping score. For each
     signed slope, group pixels by b = row +/- a * col, estimate the line-wise
     alpha offset, subtract that trial stripe map, and score the slope by the
     robust-std reduction. One slope family is selected by default.

Outputs:
  - thin_slope_search_all_candidates.csv
  - thin_slope_search_selected.csv
  - broad_slope_search_all_candidates.csv
  - broad_slope_search_selected.csv
  - signed_slope_search_all_candidates.csv
  - signed_slope_search_selected_primary.csv
  - positive_slope_search_all_candidates.csv (compatibility alias)
  - positive_slope_search_selected_primary.csv (compatibility alias)
  - detected_slope_directions.csv
  - detected_slope_directions.json
  - detected_six_slope_directions.csv/json (compatibility alias)
  - slope_detection_config.json
  - thin_slope_score_curve.png
  - broad_slope_score_curve.png
  - detected_slope_overlay.png
  - detected_six_slope_overlay.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Default settings edited most often in notebooks
# ============================================================

DEFAULT_ALPHA_PATH = Path(
    r"D:/research/code/outputs_paper_sensor_geometry_destripe/"
    r"baseline_no_destripe_alpha_corrected.npy"
)
DEFAULT_OUTPUT_DIR = Path(r"D:/research/code/outputs_general_scene_slope_detection")


@dataclass
class SlopeDetectionConfig:
    alpha_path: Path = DEFAULT_ALPHA_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    valid_mask_path: Optional[Path] = None
    csv_value_column: Optional[str] = None

    # Slopes are searched as angles because equal slope steps bias steep
    # directions. If search_negative_slopes=True, both +theta and -theta are
    # evaluated using row-a*col and row+a*col line coordinates respectively.
    angle_min_deg: float = 1.0
    angle_max_deg: float = 89.0
    angle_step_deg: float = 0.25
    search_negative_slopes: bool = True

    # Detection strategy:
    # - one thin high-alpha line family is found from concentration of high
    #   alpha pixels on signed-slope line bins.
    # - one broad offset family is found from the previous correction-gain
    #   score. Together these are the two primary positive slopes by default.
    thin_top_k: int = 1
    broad_top_k: int = 1

    # This merges nearby peaks, so 1.22 and 1.23 are treated as the same
    # slope family. A candidate is skipped when either separation is small.
    top_k_primary: int = 2
    min_angle_separation_deg: float = 1.0
    min_slope_separation: float = 0.05

    # Same basic line grouping as the earlier code for positive slopes:
    # b = row - a * col, line_id = round(b / line_bin_width).
    # Negative slopes use b = row + a * col with the same positive parameter a.
    line_bin_width: float = 18.0
    min_pixels_per_line: int = 80
    sample_step: int = 4

    # Thin-line detector settings. Use sample_step=1 when possible because
    # one- or two-pixel high-alpha lines can be missed by coarse sampling.
    thin_line_bin_width: float = 2.0
    thin_sample_step: int = 1
    thin_high_nsigma: float = 4.0
    thin_lines_per_slope: int = 6
    thin_min_high_pixels_per_line: int = 4

    # Broad-line detector settings. These default to the previous wide-band
    # destriping settings. If set to None, line_bin_width/min_pixels/sample_step
    # are used for backward compatibility.
    broad_line_bin_width: Optional[float] = None
    broad_min_pixels_per_line: Optional[int] = None
    broad_sample_step: Optional[int] = None

    statistic_method: str = "median"
    trim_fraction: float = 0.1
    mode_bins: int = 64
    sigma_clip_nsigma: float = 3.0
    sigma_clip_max_iter: int = 3

    # False by default to reproduce the earlier b = y - ax behavior.
    # If searching very shallow/steep slopes, True makes the bin width a
    # perpendicular pixel distance instead of raw b-coordinate units.
    normalize_line_coordinate: bool = False

    # Leave high alpha in by default so thin high-alpha stripe families can
    # be detected. Turn on when real plumes dominate the score.
    exclude_high_alpha_from_offset_estimate: bool = False
    exclude_high_nsigma: float = 4.0

    # Select true local peaks instead of simply taking the largest absolute
    # scores. This avoids choosing the high-angle end of a monotonic trend.
    select_local_peaks_only: bool = True
    peak_trend_window_deg: float = 8.0
    peak_local_window_deg: float = 1.0
    peak_edge_exclusion_deg: float = 1.0
    peak_min_prominence: Optional[float] = None

    overlay_lines_per_direction: int = 4
    overlay_percentile_low: float = 1.0
    overlay_percentile_high: float = 99.0
    random_seed: int = 0
    verbose: bool = True


# ============================================================
# Small robust/statistical helpers
# ============================================================


def robust_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    med = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - med)))
    rstd = 1.4826 * mad
    if not np.isfinite(rstd) or rstd <= 0:
        rstd = float(np.nanstd(values))
    return float(rstd)


def robust_threshold(values: np.ndarray, nsigma: float = 4.0) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    med = float(np.nanmedian(values))
    rstd = robust_std(values)
    return float(med + nsigma * rstd), med, rstd


def statistic_1d(
    values: np.ndarray,
    method: str = "median",
    trim_fraction: float = 0.1,
    mode_bins: int = 64,
    sigma_clip_nsigma: float = 3.0,
    sigma_clip_max_iter: int = 3,
) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")

    if method == "median":
        return float(np.nanmedian(values))
    if method == "mean":
        return float(np.nanmean(values))
    if method == "trimmed_mean":
        v = np.sort(values)
        k = int(np.floor(trim_fraction * v.size))
        if 2 * k >= v.size:
            return float(np.nanmean(v))
        return float(np.nanmean(v[k : v.size - k]))
    if method == "mode":
        if values.size == 1 or np.allclose(values, values[0]):
            return float(values[0])
        counts, edges = np.histogram(values, bins=int(mode_bins))
        idx = int(np.argmax(counts))
        return float(0.5 * (edges[idx] + edges[idx + 1]))
    if method == "sigma_clipped_mean":
        clipped = values.copy()
        for _ in range(int(sigma_clip_max_iter)):
            if clipped.size < 3:
                break
            med = float(np.nanmedian(clipped))
            rstd = robust_std(clipped)
            if not np.isfinite(rstd) or rstd <= 0:
                break
            keep = np.abs(clipped - med) <= sigma_clip_nsigma * rstd
            if np.all(keep) or not np.any(keep):
                break
            clipped = clipped[keep]
        return float(np.nanmean(clipped))
    raise ValueError(f"Unknown statistic method: {method}")


# ============================================================
# Image loading
# ============================================================


def _find_first_existing_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower_to_original = {str(c).lower(): str(c) for c in columns}
    for name in candidates:
        if name.lower() in lower_to_original:
            return lower_to_original[name.lower()]
    return None


def load_alpha_image(path: Path, csv_value_column: Optional[str] = None) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2-D npy array, got shape {arr.shape}")
        return arr.astype(float, copy=False)

    if suffix == ".npz":
        data = np.load(path)
        first_key = list(data.keys())[0]
        arr = data[first_key]
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2-D array in npz, got shape {arr.shape}")
        return arr.astype(float, copy=False)

    if suffix == ".csv":
        df = pd.read_csv(path)
        row_col = _find_first_existing_column(df.columns, ["row", "y", "line"])
        col_col = _find_first_existing_column(df.columns, ["col", "x", "sample"])
        value_col = csv_value_column
        if value_col is None:
            value_col = _find_first_existing_column(
                df.columns,
                [
                    "alpha_corrected",
                    "corrected_alpha",
                    "alpha",
                    "mf_alpha",
                    "raw_alpha",
                    "alpha_final",
                ],
            )
        if row_col is not None and col_col is not None and value_col is not None:
            rows = df[row_col].to_numpy(dtype=int)
            cols = df[col_col].to_numpy(dtype=int)
            values = df[value_col].to_numpy(dtype=float)
            out = np.full((int(rows.max()) + 1, int(cols.max()) + 1), np.nan, dtype=float)
            out[rows, cols] = values
            return out

        arr = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError(f"Could not parse numeric 2-D alpha image from {path}")
        return arr

    raise ValueError(f"Unsupported alpha image type: {suffix}")


def load_optional_valid_mask(path: Optional[Path], alpha: np.ndarray) -> np.ndarray:
    valid = np.isfinite(alpha)
    if path is None:
        return valid
    path = Path(path)
    mask = np.load(path)
    if mask.shape != alpha.shape:
        raise ValueError(f"valid mask shape {mask.shape} does not match alpha shape {alpha.shape}")
    return valid & np.asarray(mask, dtype=bool)


# ============================================================
# Line grouping and slope score
# ============================================================


def line_coordinate_map(
    shape: tuple[int, int],
    direction: str,
    slope: float,
    row_step: int = 1,
    col_step: int = 1,
    normalize: bool = False,
) -> np.ndarray:
    height, width = shape
    row_values = np.arange(0, height * row_step, row_step, dtype=float)
    col_values = np.arange(0, width * col_step, col_step, dtype=float)
    rows, cols = np.meshgrid(row_values, col_values, indexing="ij")

    if direction == "y_minus_x":
        coord = rows - float(slope) * cols
    elif direction == "y_plus_x":
        coord = rows + float(slope) * cols
    else:
        raise ValueError("direction must be y_minus_x or y_plus_x")

    if normalize:
        coord = coord / np.sqrt(1.0 + float(slope) ** 2)
    return coord


def line_id_map(
    shape: tuple[int, int],
    direction: str,
    slope: float,
    line_bin_width: float,
    row_step: int = 1,
    col_step: int = 1,
    normalize: bool = False,
) -> np.ndarray:
    if line_bin_width <= 0:
        raise ValueError("line_bin_width must be positive")
    coord = line_coordinate_map(
        shape=shape,
        direction=direction,
        slope=slope,
        row_step=row_step,
        col_step=col_step,
        normalize=normalize,
    )
    return np.rint(coord / float(line_bin_width)).astype(np.int32)


def signed_slope_to_existing_code(signed_slope: float) -> tuple[str, float]:
    """Convert a visual image slope to the existing line-coordinate convention."""
    m = float(signed_slope)
    if m >= 0:
        return "y_minus_x", m
    return "y_plus_x", abs(m)


def signed_slope_equation(signed_slope: float) -> str:
    m = float(signed_slope)
    if m >= 0:
        return f"row = {m:.8f} * col + b"
    return f"row = -{abs(m):.8f} * col + b"


def _group_stats_series(
    ids: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    method: str,
    trim_fraction: float,
    mode_bins: int,
    sigma_clip_nsigma: float,
    sigma_clip_max_iter: int,
) -> tuple[pd.Series, pd.Series]:
    flat_mask = np.asarray(mask, dtype=bool).ravel()
    flat_values = np.asarray(values, dtype=float).ravel()
    flat_ids = np.asarray(ids, dtype=np.int32).ravel()
    flat_mask &= np.isfinite(flat_values)

    if not np.any(flat_mask):
        empty_index = pd.Index([], dtype=np.int32, name="line_id")
        return pd.Series(dtype=float, index=empty_index), pd.Series(dtype=int, index=empty_index)

    df = pd.DataFrame(
        {
            "line_id": flat_ids[flat_mask],
            "alpha": flat_values[flat_mask],
        }
    )
    grouped = df.groupby("line_id", sort=True)["alpha"]
    counts = grouped.size().astype(int)
    if method == "median":
        stats = grouped.median()
    elif method == "mean":
        stats = grouped.mean()
    else:
        stats = grouped.apply(
            lambda s: statistic_1d(
                s.to_numpy(dtype=float, copy=False),
                method=method,
                trim_fraction=trim_fraction,
                mode_bins=mode_bins,
                sigma_clip_nsigma=sigma_clip_nsigma,
                sigma_clip_max_iter=sigma_clip_max_iter,
            )
        )
    return stats.astype(float), counts


def compute_line_stats(
    alpha: np.ndarray,
    line_ids: np.ndarray,
    estimate_mask: np.ndarray,
    valid_mask: np.ndarray,
    min_pixels_per_line: int,
    method: str,
    trim_fraction: float,
    mode_bins: int,
    sigma_clip_nsigma: float,
    sigma_clip_max_iter: int,
    fallback_to_valid: bool = True,
) -> pd.DataFrame:
    id_min = int(np.nanmin(line_ids))
    id_max = int(np.nanmax(line_ids))
    id_values = np.arange(id_min, id_max + 1, dtype=np.int32)
    index = pd.Index(id_values, name="line_id")

    est_stats, est_counts = _group_stats_series(
        line_ids,
        alpha,
        estimate_mask,
        method,
        trim_fraction,
        mode_bins,
        sigma_clip_nsigma,
        sigma_clip_max_iter,
    )
    stats = est_stats.reindex(index).to_numpy(dtype=float, copy=True)
    counts = est_counts.reindex(index, fill_value=0).to_numpy(dtype=int, copy=True)
    used_fallback = np.zeros(id_values.shape, dtype=bool)

    too_few = counts < int(min_pixels_per_line)
    if fallback_to_valid and np.any(too_few):
        valid_stats, valid_counts = _group_stats_series(
            line_ids,
            alpha,
            valid_mask,
            method,
            trim_fraction,
            mode_bins,
            sigma_clip_nsigma,
            sigma_clip_max_iter,
        )
        fallback_stats = valid_stats.reindex(index).to_numpy(dtype=float, copy=True)
        fallback_counts = valid_counts.reindex(index, fill_value=0).to_numpy(dtype=int, copy=True)
        can_use = too_few & (fallback_counts >= int(min_pixels_per_line))
        stats[can_use] = fallback_stats[can_use]
        counts[can_use] = fallback_counts[can_use]
        used_fallback[can_use] = True

    stats[counts < int(min_pixels_per_line)] = np.nan
    finite = np.isfinite(stats) & (counts >= int(min_pixels_per_line))
    center = float(np.nanmedian(stats[finite])) if np.any(finite) else float("nan")
    offsets = stats - center

    return pd.DataFrame(
        {
            "line_id": id_values,
            "line_stat": stats,
            "line_offset": offsets,
            "n_pixels_used": counts,
            "used_fallback": used_fallback,
        }
    )


def slope_search_score(
    alpha: np.ndarray,
    valid_mask: np.ndarray,
    estimate_mask: np.ndarray,
    slope: float,
    cfg: SlopeDetectionConfig,
    direction_key: str = "y_minus_x",
    line_bin_width: Optional[float] = None,
    min_pixels_per_line: Optional[int] = None,
    sample_step: Optional[int] = None,
) -> dict:
    if line_bin_width is None:
        line_bin_width = cfg.line_bin_width
    if min_pixels_per_line is None:
        min_pixels_per_line = cfg.min_pixels_per_line
    if sample_step is None:
        sample_step = cfg.sample_step

    sample_step = max(int(sample_step), 1)
    alpha_s = np.asarray(alpha, dtype=float)[::sample_step, ::sample_step]
    valid_s = np.asarray(valid_mask, dtype=bool)[::sample_step, ::sample_step]
    estimate_s = np.asarray(estimate_mask, dtype=bool)[::sample_step, ::sample_step]

    ids_s = line_id_map(
        shape=alpha_s.shape,
        direction=direction_key,
        slope=float(slope),
        line_bin_width=float(line_bin_width),
        row_step=sample_step,
        col_step=sample_step,
        normalize=cfg.normalize_line_coordinate,
    )
    signed_slope = float(slope) if direction_key == "y_minus_x" else -float(slope)
    signed_angle = float(np.degrees(np.arctan(signed_slope)))
    line_table = compute_line_stats(
        alpha=alpha_s,
        line_ids=ids_s,
        estimate_mask=estimate_s,
        valid_mask=valid_s,
        min_pixels_per_line=max(2, int(np.ceil(int(min_pixels_per_line) / sample_step))),
        method=cfg.statistic_method,
        trim_fraction=cfg.trim_fraction,
        mode_bins=cfg.mode_bins,
        sigma_clip_nsigma=cfg.sigma_clip_nsigma,
        sigma_clip_max_iter=cfg.sigma_clip_max_iter,
        fallback_to_valid=True,
    )

    finite = np.isfinite(line_table["line_offset"].to_numpy(dtype=float))
    if not np.any(finite):
        return {
            "slope": signed_slope,
            "angle_deg": signed_angle,
            "signed_slope": signed_slope,
            "angle_deg_signed": signed_angle,
            "angle_deg_0_180": signed_angle % 180.0,
            "direction_key_for_existing_code": direction_key,
            "slope_parameter_for_existing_code": float(slope),
            "equation_form": signed_slope_equation(signed_slope),
            "score": -np.inf,
            "before_robust_std": np.nan,
            "after_robust_std": np.nan,
            "n_lines_scored": 0,
            "line_stat_center": np.nan,
        }

    offsets = line_table["line_offset"].to_numpy(dtype=float)
    id_values = line_table["line_id"].to_numpy(dtype=np.int32)
    id_min = int(id_values[0])
    lookup = np.where(np.isfinite(offsets), offsets, 0.0)
    stripe_s = lookup[ids_s.astype(np.int64) - id_min]
    corrected_s = alpha_s - stripe_s

    score_mask = valid_s & np.isfinite(alpha_s) & np.isfinite(corrected_s)
    before = alpha_s[score_mask]
    after = corrected_s[score_mask]
    before_rstd = robust_std(before)
    after_rstd = robust_std(after)
    score = float(before_rstd - after_rstd)

    return {
        "slope": signed_slope,
        "angle_deg": signed_angle,
        "signed_slope": signed_slope,
        "angle_deg_signed": signed_angle,
        "angle_deg_0_180": signed_angle % 180.0,
        "direction_key_for_existing_code": direction_key,
        "slope_parameter_for_existing_code": float(slope),
        "equation_form": signed_slope_equation(signed_slope),
        "score": score,
        "before_robust_std": float(before_rstd),
        "after_robust_std": float(after_rstd),
        "n_lines_scored": int(np.sum(finite)),
        "line_stat_center": float(np.nanmedian(line_table["line_stat"].to_numpy(dtype=float)[finite])),
        "offset_abs_p95": float(np.nanpercentile(np.abs(offsets[finite]), 95)),
        "offset_abs_p99": float(np.nanpercentile(np.abs(offsets[finite]), 99)),
    }


def make_positive_slope_candidates(cfg: SlopeDetectionConfig) -> pd.DataFrame:
    if cfg.angle_min_deg <= 0 or cfg.angle_max_deg >= 90:
        raise ValueError("Use 0 < angle_min_deg < angle_max_deg < 90 for positive finite slopes.")
    if cfg.angle_step_deg <= 0:
        raise ValueError("angle_step_deg must be positive.")

    angles = np.arange(
        cfg.angle_min_deg,
        cfg.angle_max_deg + 0.5 * cfg.angle_step_deg,
        cfg.angle_step_deg,
        dtype=float,
    )
    angles = angles[(angles > 0) & (angles < 90)]
    slopes = np.tan(np.deg2rad(angles))
    df = pd.DataFrame({"angle_deg": angles, "slope": slopes})
    df["signed_slope"] = df["slope"]
    df["angle_deg_signed"] = df["angle_deg"]
    df["angle_deg_0_180"] = df["angle_deg"] % 180.0
    df["direction_key_for_existing_code"] = "y_minus_x"
    df["slope_parameter_for_existing_code"] = df["slope"]
    df["equation_form"] = [signed_slope_equation(m) for m in df["signed_slope"]]
    return df


def make_signed_slope_candidates(cfg: SlopeDetectionConfig) -> pd.DataFrame:
    positive = make_positive_slope_candidates(cfg)
    if not cfg.search_negative_slopes:
        return positive

    negative = positive.copy()
    negative["slope"] = -negative["slope"].astype(float)
    negative["signed_slope"] = negative["slope"]
    negative["angle_deg"] = -negative["angle_deg"].astype(float)
    negative["angle_deg_signed"] = negative["angle_deg"]
    negative["angle_deg_0_180"] = negative["angle_deg"] % 180.0
    negative["direction_key_for_existing_code"] = "y_plus_x"
    negative["slope_parameter_for_existing_code"] = negative["slope"].abs()
    negative["equation_form"] = [signed_slope_equation(m) for m in negative["signed_slope"]]

    out = pd.concat([negative, positive], ignore_index=True, sort=False)
    out = out.sort_values("angle_deg").reset_index(drop=True)
    return out


def annotate_peak_selection_scores(score_table: pd.DataFrame, cfg: SlopeDetectionConfig) -> pd.DataFrame:
    """Add detrended local-peak scores for selecting isolated peaks.

    The raw correction-gain score can have a broad monotonic trend with angle.
    We therefore estimate a slow background trend by a rolling median over
    angle, then select local maxima of score - trend.
    """
    if score_table is None or len(score_table) == 0:
        return pd.DataFrame() if score_table is None else score_table.copy()

    df = score_table.copy()
    for col in ["score_trend", "peak_prominence", "is_local_peak", "selection_score"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    ordered = df.sort_values("angle_deg").copy()
    angles = ordered["angle_deg"].to_numpy(dtype=float)
    scores = ordered["score"].to_numpy(dtype=float)
    finite = np.isfinite(scores)

    if angles.size < 3 or not np.any(finite):
        ordered["score_trend"] = np.nan
        ordered["peak_prominence"] = np.nan
        ordered["is_local_peak"] = False
        ordered["selection_score"] = ordered["score"] if not cfg.select_local_peaks_only else -np.inf
        return ordered.reindex(df.index).copy()

    diffs = np.diff(np.sort(np.unique(angles)))
    step = float(np.nanmedian(diffs)) if diffs.size else float(cfg.angle_step_deg)
    if not np.isfinite(step) or step <= 0:
        step = float(cfg.angle_step_deg)

    trend_window = max(3, int(round(float(cfg.peak_trend_window_deg) / step)))
    if trend_window % 2 == 0:
        trend_window += 1
    min_periods = max(3, trend_window // 3)

    score_series = pd.Series(np.where(finite, scores, np.nan))
    trend = (
        score_series.rolling(window=trend_window, center=True, min_periods=min_periods)
        .median()
        .interpolate(limit_direction="both")
        .to_numpy(dtype=float)
    )
    if np.any(~np.isfinite(trend)):
        fill = float(np.nanmedian(scores[finite]))
        trend[~np.isfinite(trend)] = fill

    prominence = scores - trend

    local_half = max(1, int(round(float(cfg.peak_local_window_deg) / step)))
    edge_exclusion = max(local_half, int(round(float(cfg.peak_edge_exclusion_deg) / step)))
    min_prom = 0.0 if cfg.peak_min_prominence is None else float(cfg.peak_min_prominence)

    is_peak = np.zeros(scores.shape, dtype=bool)
    for i in range(scores.size):
        if i < edge_exclusion or i >= scores.size - edge_exclusion:
            continue
        if not np.isfinite(prominence[i]):
            continue
        lo = max(0, i - local_half)
        hi = min(scores.size, i + local_half + 1)
        neighborhood = prominence[lo:hi]
        if not np.any(np.isfinite(neighborhood)):
            continue
        left = prominence[lo:i]
        right = prominence[i + 1:hi]
        left_max = np.nanmax(left) if left.size and np.any(np.isfinite(left)) else -np.inf
        right_max = np.nanmax(right) if right.size and np.any(np.isfinite(right)) else -np.inf
        is_peak[i] = (
            prominence[i] > left_max
            and prominence[i] >= right_max
            and prominence[i] >= min_prom
        )

    if cfg.select_local_peaks_only:
        selection_score = np.where(is_peak, prominence, -np.inf)
    else:
        selection_score = scores

    ordered["score_trend"] = trend
    ordered["peak_prominence"] = prominence
    ordered["is_local_peak"] = is_peak
    ordered["selection_score"] = selection_score

    return ordered.sort_index().copy()


def make_estimate_mask(alpha: np.ndarray, valid_mask: np.ndarray, cfg: SlopeDetectionConfig) -> tuple[np.ndarray, dict]:
    estimate = np.asarray(valid_mask, dtype=bool).copy()
    meta = {"exclude_high_alpha_from_offset_estimate": bool(cfg.exclude_high_alpha_from_offset_estimate)}

    if cfg.exclude_high_alpha_from_offset_estimate:
        threshold, med, rstd = robust_threshold(alpha[valid_mask], nsigma=cfg.exclude_high_nsigma)
        high = np.zeros_like(estimate, dtype=bool)
        high[valid_mask] = alpha[valid_mask] > threshold
        estimate &= ~high
        meta.update(
            {
                "exclude_high_threshold": float(threshold),
                "exclude_high_median": float(med),
                "exclude_high_robust_std": float(rstd),
                "n_high_excluded": int(np.sum(high)),
            }
        )
    else:
        meta.update(
            {
                "exclude_high_threshold": None,
                "exclude_high_median": None,
                "exclude_high_robust_std": None,
                "n_high_excluded": 0,
            }
        )
    meta["n_estimate_pixels"] = int(np.sum(estimate))
    return estimate, meta


def make_high_alpha_mask(alpha: np.ndarray, valid_mask: np.ndarray, cfg: SlopeDetectionConfig) -> tuple[np.ndarray, dict]:
    threshold, med, rstd = robust_threshold(alpha[valid_mask], nsigma=cfg.thin_high_nsigma)
    high = np.zeros_like(valid_mask, dtype=bool)
    high[valid_mask] = alpha[valid_mask] > threshold
    meta = {
        "thin_high_threshold": float(threshold),
        "thin_high_median": float(med),
        "thin_high_robust_std": float(rstd),
        "thin_high_nsigma": float(cfg.thin_high_nsigma),
        "n_high_alpha_pixels": int(np.sum(high)),
    }
    return high, meta


def thin_slope_search_score(
    alpha: np.ndarray,
    valid_mask: np.ndarray,
    high_mask: np.ndarray,
    slope: float,
    cfg: SlopeDetectionConfig,
    direction_key: str = "y_minus_x",
) -> dict:
    sample_step = max(int(cfg.thin_sample_step), 1)
    valid_s = np.asarray(valid_mask, dtype=bool)[::sample_step, ::sample_step]
    high_s = np.asarray(high_mask, dtype=bool)[::sample_step, ::sample_step] & valid_s

    ids_s = line_id_map(
        shape=valid_s.shape,
        direction=direction_key,
        slope=float(slope),
        line_bin_width=float(cfg.thin_line_bin_width),
        row_step=sample_step,
        col_step=sample_step,
        normalize=cfg.normalize_line_coordinate,
    )
    signed_slope = float(slope) if direction_key == "y_minus_x" else -float(slope)
    signed_angle = float(np.degrees(np.arctan(signed_slope)))

    flat_ids = ids_s.ravel().astype(np.int32, copy=False)
    flat_valid = valid_s.ravel()
    flat_high = high_s.ravel()
    if not np.any(flat_high):
        return {
            "slope": signed_slope,
            "angle_deg": signed_angle,
            "signed_slope": signed_slope,
            "angle_deg_signed": signed_angle,
            "angle_deg_0_180": signed_angle % 180.0,
            "direction_key_for_existing_code": direction_key,
            "slope_parameter_for_existing_code": float(slope),
            "equation_form": signed_slope_equation(signed_slope),
            "score": -np.inf,
            "sum_selected_high_pixels": 0,
            "top_high_pixels": 0,
            "n_selected_lines": 0,
            "n_high_alpha_pixels": 0,
            "max_line_density": np.nan,
        }

    valid_counts = pd.Series(flat_ids[flat_valid]).value_counts(sort=False)
    high_counts = pd.Series(flat_ids[flat_high]).value_counts(sort=False)
    line_table = pd.DataFrame(
        {
            "line_id": high_counts.index.astype(np.int32),
            "high_pixels": high_counts.to_numpy(dtype=int),
        }
    )
    line_table["valid_pixels"] = (
        valid_counts.reindex(line_table["line_id"].to_numpy(dtype=np.int32), fill_value=0)
        .to_numpy(dtype=int)
    )
    line_table = line_table[line_table["high_pixels"] >= int(cfg.thin_min_high_pixels_per_line)].copy()
    if len(line_table) == 0:
        return {
            "slope": signed_slope,
            "angle_deg": signed_angle,
            "signed_slope": signed_slope,
            "angle_deg_signed": signed_angle,
            "angle_deg_0_180": signed_angle % 180.0,
            "direction_key_for_existing_code": direction_key,
            "slope_parameter_for_existing_code": float(slope),
            "equation_form": signed_slope_equation(signed_slope),
            "score": -np.inf,
            "sum_selected_high_pixels": 0,
            "top_high_pixels": 0,
            "n_selected_lines": 0,
            "n_high_alpha_pixels": int(np.sum(flat_high)),
            "max_line_density": np.nan,
        }

    line_table["density"] = line_table["high_pixels"] / np.maximum(line_table["valid_pixels"], 1)
    line_table = line_table.sort_values(["high_pixels", "density"], ascending=False)

    selected_rows = []
    for _, row in line_table.iterrows():
        line_id = int(row["line_id"])
        if any(abs(line_id - int(prev["line_id"])) <= 1 for prev in selected_rows):
            continue
        selected_rows.append(row)
        if len(selected_rows) >= int(cfg.thin_lines_per_slope):
            break

    if not selected_rows:
        return {
            "slope": signed_slope,
            "angle_deg": signed_angle,
            "signed_slope": signed_slope,
            "angle_deg_signed": signed_angle,
            "angle_deg_0_180": signed_angle % 180.0,
            "direction_key_for_existing_code": direction_key,
            "slope_parameter_for_existing_code": float(slope),
            "equation_form": signed_slope_equation(signed_slope),
            "score": -np.inf,
            "sum_selected_high_pixels": 0,
            "top_high_pixels": 0,
            "n_selected_lines": 0,
            "n_high_alpha_pixels": int(np.sum(flat_high)),
            "max_line_density": np.nan,
        }

    selected = pd.DataFrame(selected_rows)
    sum_high = int(selected["high_pixels"].sum())
    top_high = int(selected["high_pixels"].max())
    mean_density = float(selected["density"].mean())
    max_density = float(selected["density"].max())
    global_density = float(np.sum(flat_high) / max(np.sum(flat_valid), 1))
    density_gain = mean_density / max(global_density, 1e-12)

    # The main term is concentration of high-alpha pixels on a few line bins.
    # The density gain reduces the tendency to prefer long line bins only.
    score = float(sum_high * np.log1p(max(density_gain, 0.0)))
    return {
        "slope": signed_slope,
        "angle_deg": signed_angle,
        "signed_slope": signed_slope,
        "angle_deg_signed": signed_angle,
        "angle_deg_0_180": signed_angle % 180.0,
        "direction_key_for_existing_code": direction_key,
        "slope_parameter_for_existing_code": float(slope),
        "equation_form": signed_slope_equation(signed_slope),
        "score": score,
        "sum_selected_high_pixels": sum_high,
        "top_high_pixels": top_high,
        "n_selected_lines": int(len(selected)),
        "n_high_alpha_pixels": int(np.sum(flat_high)),
        "mean_selected_line_density": mean_density,
        "max_line_density": max_density,
        "global_high_density": global_density,
        "density_gain": float(density_gain),
    }


def search_thin_positive_slopes(alpha: np.ndarray, valid_mask: np.ndarray, cfg: SlopeDetectionConfig) -> tuple[pd.DataFrame, dict]:
    high_mask, high_meta = make_high_alpha_mask(alpha, valid_mask, cfg)
    candidates = make_signed_slope_candidates(cfg)

    rows = []
    n = len(candidates)
    for i, row in candidates.iterrows():
        if cfg.verbose and (i == 0 or (i + 1) % 50 == 0 or i + 1 == n):
            print(f"Thin search {i + 1}/{n}: angle={row['angle_deg']:.2f} deg, slope={row['slope']:.5f}")
        rows.append(
            thin_slope_search_score(
                alpha,
                valid_mask,
                high_mask,
                float(row["slope_parameter_for_existing_code"]),
                cfg,
                direction_key=str(row["direction_key_for_existing_code"]),
            )
        )

    table = pd.DataFrame(rows)
    table = annotate_peak_selection_scores(table, cfg)
    sort_col = "selection_score" if cfg.select_local_peaks_only else "score"
    table = table.sort_values(sort_col, ascending=False).reset_index(drop=True)
    table.insert(0, "candidate_rank", np.arange(1, len(table) + 1))
    return table, high_meta


def search_positive_slopes(alpha: np.ndarray, valid_mask: np.ndarray, cfg: SlopeDetectionConfig) -> tuple[pd.DataFrame, dict]:
    estimate_mask, estimate_meta = make_estimate_mask(alpha, valid_mask, cfg)
    candidates = make_signed_slope_candidates(cfg)
    broad_line_bin_width = cfg.broad_line_bin_width if cfg.broad_line_bin_width is not None else cfg.line_bin_width
    broad_min_pixels = (
        cfg.broad_min_pixels_per_line
        if cfg.broad_min_pixels_per_line is not None
        else cfg.min_pixels_per_line
    )
    broad_sample_step = cfg.broad_sample_step if cfg.broad_sample_step is not None else cfg.sample_step

    rows = []
    n = len(candidates)
    for i, row in candidates.iterrows():
        if cfg.verbose and (i == 0 or (i + 1) % 50 == 0 or i + 1 == n):
            print(f"Broad search {i + 1}/{n}: angle={row['angle_deg']:.2f} deg, slope={row['slope']:.5f}")
        score_row = slope_search_score(
            alpha,
            valid_mask,
            estimate_mask,
            float(row["slope_parameter_for_existing_code"]),
            cfg,
            direction_key=str(row["direction_key_for_existing_code"]),
            line_bin_width=float(broad_line_bin_width),
            min_pixels_per_line=int(broad_min_pixels),
            sample_step=int(broad_sample_step),
        )
        rows.append(score_row)

    table = pd.DataFrame(rows)
    table = annotate_peak_selection_scores(table, cfg)
    sort_col = "selection_score" if cfg.select_local_peaks_only else "score"
    table = table.sort_values(sort_col, ascending=False).reset_index(drop=True)
    table.insert(0, "candidate_rank", np.arange(1, len(table) + 1))
    return table, estimate_meta


def select_separated_primary_slopes(
    search_table: pd.DataFrame,
    cfg: SlopeDetectionConfig,
    max_keep: Optional[int] = None,
    excluded_slopes: Optional[pd.DataFrame] = None,
    detection_type: str = "primary",
) -> pd.DataFrame:
    if max_keep is None:
        max_keep = cfg.top_k_primary

    excluded = []
    if excluded_slopes is not None and len(excluded_slopes) > 0:
        for _, row in excluded_slopes.iterrows():
            excluded.append(
                {
                    "slope": float(row["slope"]),
                    "angle_deg": float(row["angle_deg"]),
                }
            )

    score_col = "selection_score" if (
        cfg.select_local_peaks_only and "selection_score" in search_table.columns
    ) else "score"
    selected = []
    for _, row in search_table.sort_values(score_col, ascending=False).iterrows():
        score = float(row[score_col])
        if not np.isfinite(score):
            continue
        slope = float(row["slope"])
        angle = float(row["angle_deg"])

        too_close_to_excluded = False
        for chosen in excluded:
            if abs(slope - float(chosen["slope"])) < cfg.min_slope_separation:
                too_close_to_excluded = True
                break
            if abs(angle - float(chosen["angle_deg"])) < cfg.min_angle_separation_deg:
                too_close_to_excluded = True
                break
        if too_close_to_excluded:
            continue

        too_close = False
        for chosen in selected:
            chosen_slope = float(chosen["slope"])
            chosen_angle = float(chosen["angle_deg"])
            if abs(slope - chosen_slope) < cfg.min_slope_separation:
                too_close = True
                break
            if abs(angle - chosen_angle) < cfg.min_angle_separation_deg:
                too_close = True
                break
        if too_close:
            continue
        selected.append(row.to_dict())
        if len(selected) >= int(max_keep):
            break

    out = pd.DataFrame(selected)
    if len(out) == 0:
        return out
    out = out.reset_index(drop=True)
    out.insert(0, "selected_rank", np.arange(1, len(out) + 1))
    out.insert(1, "detection_type", detection_type)
    return out


def combine_thin_and_broad_slopes(thin_selected: pd.DataFrame, broad_selected: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if thin_selected is not None and len(thin_selected) > 0:
        frames.append(thin_selected.copy())
    if broad_selected is not None and len(broad_selected) > 0:
        frames.append(broad_selected.copy())
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.reset_index(drop=True)
    if "selected_rank" in out.columns:
        out = out.drop(columns=["selected_rank"])
    out.insert(0, "selected_rank", np.arange(1, len(out) + 1))
    return out


def build_six_direction_table(selected_primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, src in selected_primary.iterrows():
        rank = int(src["selected_rank"])
        a = float(src["slope"])
        primary_angle = float(np.degrees(np.arctan(a)))
        direction_key, slope_param = signed_slope_to_existing_code(a)
        orth_signed_slope = -1.0 / a
        orth_angle_signed = float(np.degrees(np.arctan(orth_signed_slope)))
        orth_angle_0_180 = (orth_angle_signed + 180.0) % 180.0
        orth_direction_key, orth_slope_param = signed_slope_to_existing_code(orth_signed_slope)

        common = {
            "family_rank": rank,
            "detection_type": str(src["detection_type"]) if "detection_type" in src else "primary",
            "parent_signed_slope": a,
            "parent_angle_deg_signed": primary_angle,
            "parent_angle_deg_0_180": primary_angle % 180.0,
            # Compatibility names for older downstream summaries.
            "parent_positive_slope": a,
            "parent_positive_angle_deg": primary_angle,
            "parent_score": float(src["score"]),
            "parent_selection_score": float(src["selection_score"]) if "selection_score" in src else float(src["score"]),
            "parent_peak_prominence": float(src["peak_prominence"]) if "peak_prominence" in src else np.nan,
            "parent_before_robust_std": float(src["before_robust_std"]) if "before_robust_std" in src else np.nan,
            "parent_after_robust_std": float(src["after_robust_std"]) if "after_robust_std" in src else np.nan,
        }
        rows.append(
            {
                **common,
                "direction_type": "primary_positive",
                "signed_slope": a,
                "angle_deg_signed": primary_angle,
                "angle_deg_0_180": primary_angle % 180.0,
                "direction_key_for_existing_code": direction_key,
                "slope_parameter_for_existing_code": slope_param,
                "equation_form": signed_slope_equation(a),
            }
        )
        rows.append(
            {
                **common,
                "direction_type": "orthogonal_to_primary",
                "signed_slope": orth_signed_slope,
                "angle_deg_signed": orth_angle_signed,
                "angle_deg_0_180": orth_angle_0_180,
                "direction_key_for_existing_code": orth_direction_key,
                "slope_parameter_for_existing_code": orth_slope_param,
                "equation_form": signed_slope_equation(orth_signed_slope),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Plotting helpers
# ============================================================


def line_segment_for_image(signed_slope: float, intercept: float, shape: tuple[int, int]) -> Optional[tuple[np.ndarray, np.ndarray]]:
    height, width = shape
    max_y = height - 1
    max_x = width - 1
    m = float(signed_slope)
    b = float(intercept)
    points = []

    for x in (0.0, float(max_x)):
        y = m * x + b
        if 0 <= y <= max_y:
            points.append((x, y))

    if abs(m) > 1e-12:
        for y in (0.0, float(max_y)):
            x = (y - b) / m
            if 0 <= x <= max_x:
                points.append((x, y))

    if len(points) < 2:
        return None

    # Use the farthest pair to avoid tiny duplicated boundary segments.
    best = None
    best_dist = -1.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p = np.array(points[i], dtype=float)
            q = np.array(points[j], dtype=float)
            dist = float(np.sum((p - q) ** 2))
            if dist > best_dist:
                best_dist = dist
                best = (p, q)
    if best is None:
        return None
    xs = np.array([best[0][0], best[1][0]], dtype=float)
    ys = np.array([best[0][1], best[1][1]], dtype=float)
    return xs, ys


def strongest_line_intercepts(
    alpha: np.ndarray,
    valid_mask: np.ndarray,
    direction_key: str,
    slope_parameter: float,
    cfg: SlopeDetectionConfig,
    max_lines: int,
    line_bin_width: Optional[float] = None,
    min_pixels_per_line: Optional[int] = None,
    sample_step: Optional[int] = None,
) -> pd.DataFrame:
    if line_bin_width is None:
        line_bin_width = cfg.line_bin_width
    if min_pixels_per_line is None:
        min_pixels_per_line = cfg.min_pixels_per_line
    if sample_step is None:
        sample_step = cfg.sample_step
    sample_step = max(int(sample_step), 1)
    alpha_s = alpha[::sample_step, ::sample_step]
    valid_s = valid_mask[::sample_step, ::sample_step]
    ids_s = line_id_map(
        shape=alpha_s.shape,
        direction=direction_key,
        slope=slope_parameter,
        line_bin_width=float(line_bin_width),
        row_step=sample_step,
        col_step=sample_step,
        normalize=cfg.normalize_line_coordinate,
    )
    line_table = compute_line_stats(
        alpha=alpha_s,
        line_ids=ids_s,
        estimate_mask=valid_s,
        valid_mask=valid_s,
        min_pixels_per_line=max(2, int(np.ceil(int(min_pixels_per_line) / sample_step))),
        method=cfg.statistic_method,
        trim_fraction=cfg.trim_fraction,
        mode_bins=cfg.mode_bins,
        sigma_clip_nsigma=cfg.sigma_clip_nsigma,
        sigma_clip_max_iter=cfg.sigma_clip_max_iter,
        fallback_to_valid=False,
    )
    finite = np.isfinite(line_table["line_offset"].to_numpy(dtype=float))
    line_table = line_table[finite].copy()
    if len(line_table) == 0:
        return line_table
    line_table["abs_offset"] = np.abs(line_table["line_offset"].astype(float))
    line_table = line_table.sort_values("abs_offset", ascending=False).head(int(max_lines)).copy()

    coord = line_table["line_id"].astype(float) * float(line_bin_width)
    if cfg.normalize_line_coordinate:
        coord = coord * np.sqrt(1.0 + float(slope_parameter) ** 2)
    line_table["intercept_for_plot"] = coord
    return line_table.reset_index(drop=True)


def plot_score_curve(search_table: pd.DataFrame, selected_primary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    plot_table = search_table.sort_values("angle_deg")
    ax.plot(plot_table["angle_deg"], plot_table["score"], color="0.25", lw=1.2, label="raw score")
    if "score_trend" in plot_table.columns:
        ax.plot(
            plot_table["angle_deg"],
            plot_table["score_trend"],
            color="tab:blue",
            lw=1.0,
            ls="--",
            alpha=0.8,
            label="rolling-median trend",
        )
    if selected_primary is not None and len(selected_primary) > 0:
        ax.scatter(selected_primary["angle_deg"], selected_primary["score"], color="crimson", s=60, zorder=3)
        for _, row in selected_primary.iterrows():
            label = f"{row['slope']:.4f}"
            if "peak_prominence" in row and np.isfinite(float(row["peak_prominence"])):
                label += f"\nprom={float(row['peak_prominence']):.2g}"
            ax.annotate(
                label,
                (row["angle_deg"], row["score"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
            )
    ax.set_xlabel("signed angle theta = atan(slope) [deg]")
    ax.set_ylabel("score = robust_std_before - robust_std_after")
    ax.set_title("Signed slope search")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_detected_direction_overlay(
    alpha: np.ndarray,
    valid_mask: np.ndarray,
    six_directions: pd.DataFrame,
    cfg: SlopeDetectionConfig,
    output_path: Path,
) -> pd.DataFrame:
    colors = ["tab:red", "tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown"]
    v = alpha[valid_mask & np.isfinite(alpha)]
    vmin, vmax = np.nanpercentile(v, [cfg.overlay_percentile_low, cfg.overlay_percentile_high])

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(alpha, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="alpha")

    overlay_rows = []
    for idx, (_, row) in enumerate(six_directions.iterrows()):
        color = colors[idx % len(colors)]
        direction_key = str(row["direction_key_for_existing_code"])
        slope_param = float(row["slope_parameter_for_existing_code"])
        signed_slope = float(row["signed_slope"])
        if str(row.get("detection_type", "")) == "thin_high_alpha":
            plot_line_bin_width = cfg.thin_line_bin_width
            plot_min_pixels = max(cfg.thin_min_high_pixels_per_line, 2)
            plot_sample_step = cfg.thin_sample_step
        else:
            plot_line_bin_width = cfg.broad_line_bin_width if cfg.broad_line_bin_width is not None else cfg.line_bin_width
            plot_min_pixels = (
                cfg.broad_min_pixels_per_line
                if cfg.broad_min_pixels_per_line is not None
                else cfg.min_pixels_per_line
            )
            plot_sample_step = cfg.broad_sample_step if cfg.broad_sample_step is not None else cfg.sample_step
        lines = strongest_line_intercepts(
            alpha=alpha,
            valid_mask=valid_mask,
            direction_key=direction_key,
            slope_parameter=slope_param,
            cfg=cfg,
            max_lines=cfg.overlay_lines_per_direction,
            line_bin_width=float(plot_line_bin_width),
            min_pixels_per_line=int(plot_min_pixels),
            sample_step=int(plot_sample_step),
        )
        for line_idx, line in lines.iterrows():
            intercept = float(line["intercept_for_plot"])
            segment = line_segment_for_image(signed_slope, intercept, alpha.shape)
            if segment is None:
                continue
            xs, ys = segment
            label = None
            if line_idx == 0:
                label = (
                    f"rank {int(row['family_rank'])} "
                    f"{row['direction_type']} m={signed_slope:.3f}"
                )
            ax.plot(xs, ys, color=color, lw=1.2, alpha=0.85, label=label)
            overlay_rows.append(
                {
                    "family_rank": int(row["family_rank"]),
                    "direction_type": row["direction_type"],
                    "signed_slope": signed_slope,
                    "direction_key_for_existing_code": direction_key,
                    "slope_parameter_for_existing_code": slope_param,
                    "line_rank_within_direction": int(line_idx + 1),
                    "intercept": intercept,
                    "line_offset": float(line["line_offset"]),
                    "abs_offset": float(line["abs_offset"]),
                    "n_pixels_used": int(line["n_pixels_used"]),
                    "equation": f"row = {signed_slope:.8f} * col + {intercept:.3f}",
                }
            )

    ax.set_title("Detected signed slopes and orthogonal directions")
    ax.set_xlim(0, alpha.shape[1] - 1)
    ax.set_ylim(alpha.shape[0] - 1, 0)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return pd.DataFrame(overlay_rows)


# ============================================================
# Main orchestration
# ============================================================


def _jsonable_config(cfg: SlopeDetectionConfig, extra: Optional[dict] = None) -> dict:
    data = asdict(cfg)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    if extra:
        data.update(extra)
    return data


def run_detection(cfg: SlopeDetectionConfig) -> dict:
    np.random.seed(int(cfg.random_seed))
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha = load_alpha_image(Path(cfg.alpha_path), cfg.csv_value_column)
    valid_mask = load_optional_valid_mask(cfg.valid_mask_path, alpha)
    if cfg.verbose:
        print(f"Loaded alpha image: shape={alpha.shape}, valid_pixels={int(np.sum(valid_mask))}")

    thin_search_table, thin_meta = search_thin_positive_slopes(alpha, valid_mask, cfg)
    thin_selected = select_separated_primary_slopes(
        thin_search_table,
        cfg,
        max_keep=cfg.thin_top_k,
        detection_type="thin_high_alpha",
    )

    broad_search_table, estimate_meta = search_positive_slopes(alpha, valid_mask, cfg)
    broad_selected = select_separated_primary_slopes(
        broad_search_table,
        cfg,
        max_keep=cfg.broad_top_k,
        excluded_slopes=thin_selected,
        detection_type="broad_offset",
    )

    selected_primary = combine_thin_and_broad_slopes(thin_selected, broad_selected)
    six_directions = build_six_direction_table(selected_primary)

    thin_search_table.to_csv(output_dir / "thin_signed_slope_search_all_candidates.csv", index=False)
    thin_search_table.to_csv(output_dir / "thin_slope_search_all_candidates.csv", index=False)
    thin_selected.to_csv(output_dir / "thin_slope_search_selected.csv", index=False)
    broad_search_table.to_csv(output_dir / "broad_signed_slope_search_all_candidates.csv", index=False)
    broad_search_table.to_csv(output_dir / "broad_slope_search_all_candidates.csv", index=False)
    broad_selected.to_csv(output_dir / "broad_slope_search_selected.csv", index=False)

    combined_search_table = pd.concat(
        [
            thin_search_table.assign(detection_type="thin_high_alpha"),
            broad_search_table.assign(detection_type="broad_offset"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined_search_table.to_csv(output_dir / "signed_slope_search_all_candidates.csv", index=False)
    combined_search_table.to_csv(output_dir / "positive_slope_search_all_candidates.csv", index=False)
    selected_primary.to_csv(output_dir / "signed_slope_search_selected_primary.csv", index=False)
    selected_primary.to_csv(output_dir / "positive_slope_search_selected_primary.csv", index=False)
    six_directions.to_csv(output_dir / "detected_slope_directions.csv", index=False)
    (output_dir / "detected_slope_directions.json").write_text(
        json.dumps(six_directions.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    # Compatibility aliases for older notebooks/scripts. The row count is now
    # variable: thin 1 + broad 1 gives four directions after orthogonal pairs.
    six_directions.to_csv(output_dir / "detected_six_slope_directions.csv", index=False)
    (output_dir / "detected_six_slope_directions.json").write_text(
        json.dumps(six_directions.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    config_payload = _jsonable_config(
        cfg,
        extra={
            "alpha_shape": list(alpha.shape),
            "n_valid_pixels": int(np.sum(valid_mask)),
            "thin_high_alpha_meta": thin_meta,
            "estimate_mask_meta": estimate_meta,
        },
    )
    (output_dir / "slope_detection_config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )

    if len(selected_primary) > 0:
        plot_score_curve(broad_search_table, broad_selected, output_dir / "broad_slope_score_curve.png")
        plot_score_curve(thin_search_table, thin_selected, output_dir / "thin_slope_score_curve.png")
        overlay_path = output_dir / "detected_slope_overlay.png"
        overlay_lines = plot_detected_direction_overlay(
            alpha,
            valid_mask,
            six_directions,
            cfg,
            overlay_path,
        )
        (output_dir / "detected_six_slope_overlay.png").write_bytes(overlay_path.read_bytes())
        overlay_lines.to_csv(output_dir / "detected_overlay_lines.csv", index=False)
    else:
        overlay_lines = pd.DataFrame()

    if cfg.verbose:
        print(f"\nSelected signed primary slopes (thin {cfg.thin_top_k} + broad {cfg.broad_top_k}):")
        if len(selected_primary) == 0:
            print("  none")
        else:
            print(selected_primary[["selected_rank", "detection_type", "slope", "angle_deg", "score"]].to_string(index=False))
        print(f"\nSaved outputs to: {output_dir}")

    return {
        "alpha": alpha,
        "valid_mask": valid_mask,
        "search_table": combined_search_table,
        "thin_search_table": thin_search_table,
        "broad_search_table": broad_search_table,
        "thin_selected": thin_selected,
        "broad_selected": broad_selected,
        "selected_primary": selected_primary,
        "six_directions": six_directions,
        "overlay_lines": overlay_lines,
        "output_dir": output_dir,
        "config": cfg,
    }


def parse_args() -> SlopeDetectionConfig:
    parser = argparse.ArgumentParser(description="Detect signed stripe slopes and their orthogonal directions.")
    parser.add_argument("--alpha", type=Path, default=DEFAULT_ALPHA_PATH, help="2-D alpha image path (.npy/.npz/.csv)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--valid-mask", type=Path, default=None)
    parser.add_argument("--csv-value-column", type=str, default=None)
    parser.add_argument("--angle-min-deg", type=float, default=1.0)
    parser.add_argument("--angle-max-deg", type=float, default=89.0)
    parser.add_argument("--angle-step-deg", type=float, default=0.25)
    parser.add_argument("--search-negative-slopes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--thin-top-k", type=int, default=1)
    parser.add_argument("--broad-top-k", type=int, default=1)
    parser.add_argument("--top-k-primary", type=int, default=2)
    parser.add_argument("--min-angle-separation-deg", type=float, default=1.0)
    parser.add_argument("--min-slope-separation", type=float, default=0.05)
    parser.add_argument("--line-bin-width", type=float, default=18.0)
    parser.add_argument("--min-pixels-per-line", type=int, default=80)
    parser.add_argument("--sample-step", type=int, default=4)
    parser.add_argument("--thin-line-bin-width", type=float, default=2.0)
    parser.add_argument("--thin-sample-step", type=int, default=1)
    parser.add_argument("--thin-high-nsigma", type=float, default=4.0)
    parser.add_argument("--thin-lines-per-slope", type=int, default=6)
    parser.add_argument("--thin-min-high-pixels-per-line", type=int, default=4)
    parser.add_argument("--broad-line-bin-width", type=float, default=None)
    parser.add_argument("--broad-min-pixels-per-line", type=int, default=None)
    parser.add_argument("--broad-sample-step", type=int, default=None)
    parser.add_argument(
        "--statistic-method",
        type=str,
        default="median",
        choices=["median", "mean", "trimmed_mean", "mode", "sigma_clipped_mean"],
    )
    parser.add_argument("--normalize-line-coordinate", action="store_true")
    parser.add_argument("--exclude-high-alpha-from-offset-estimate", action="store_true")
    parser.add_argument("--exclude-high-nsigma", type=float, default=4.0)
    parser.add_argument("--select-local-peaks-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--peak-trend-window-deg", type=float, default=8.0)
    parser.add_argument("--peak-local-window-deg", type=float, default=1.0)
    parser.add_argument("--peak-edge-exclusion-deg", type=float, default=1.0)
    parser.add_argument("--peak-min-prominence", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    return SlopeDetectionConfig(
        alpha_path=args.alpha,
        output_dir=args.output_dir,
        valid_mask_path=args.valid_mask,
        csv_value_column=args.csv_value_column,
        angle_min_deg=args.angle_min_deg,
        angle_max_deg=args.angle_max_deg,
        angle_step_deg=args.angle_step_deg,
        search_negative_slopes=bool(args.search_negative_slopes),
        thin_top_k=args.thin_top_k,
        broad_top_k=args.broad_top_k,
        top_k_primary=args.top_k_primary,
        min_angle_separation_deg=args.min_angle_separation_deg,
        min_slope_separation=args.min_slope_separation,
        line_bin_width=args.line_bin_width,
        min_pixels_per_line=args.min_pixels_per_line,
        sample_step=args.sample_step,
        thin_line_bin_width=args.thin_line_bin_width,
        thin_sample_step=args.thin_sample_step,
        thin_high_nsigma=args.thin_high_nsigma,
        thin_lines_per_slope=args.thin_lines_per_slope,
        thin_min_high_pixels_per_line=args.thin_min_high_pixels_per_line,
        broad_line_bin_width=args.broad_line_bin_width,
        broad_min_pixels_per_line=args.broad_min_pixels_per_line,
        broad_sample_step=args.broad_sample_step,
        statistic_method=args.statistic_method,
        normalize_line_coordinate=bool(args.normalize_line_coordinate),
        exclude_high_alpha_from_offset_estimate=bool(args.exclude_high_alpha_from_offset_estimate),
        exclude_high_nsigma=args.exclude_high_nsigma,
        select_local_peaks_only=bool(args.select_local_peaks_only),
        peak_trend_window_deg=args.peak_trend_window_deg,
        peak_local_window_deg=args.peak_local_window_deg,
        peak_edge_exclusion_deg=args.peak_edge_exclusion_deg,
        peak_min_prominence=args.peak_min_prominence,
        verbose=not bool(args.quiet),
    )


def main() -> dict:
    cfg = parse_args()
    return run_detection(cfg)


if __name__ == "__main__":
    main()
