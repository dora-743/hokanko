from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 0. User settings

ROI_CSV = Path(r"E:\refit\all_map_spectra.csv")
MODTRAN_CSV = Path(r"E:/refit/CH4c.csv")
METADATA_TXT = Path(
    r"E:/メタン/2025_HISUI_72_The Permian Basin-論文照合用/"
    r"HSHL1G_N320W1032_20221030160051_20231127193053/"
    r"HSHL1G_N320W1032_20221030160051_20231127193053.txt"
)
OUTPUT_DIR = Path("outputs_paper_sensor_geometry_destripe")

# Override the copied mojibake path above with the readable path supplied in
# this thread. Edit this line if you move the HISUI product directory.
METADATA_TXT = Path(
    r"E:/メタン/2025_HISUI_72_The Permian Basin-論文照合用/"
    r"HSHL1G_N320W1032_20221030160051_20231127193053/"
    r"HSHL1G_N320W1032_20221030160051_20231127193053.txt"
)

WL_MIN = 2100.0
WL_MAX = 2450.0
FWHM_NM = 12.5

UAS_ALPHA_MIN = 0.0
UAS_ALPHA_MAX = 0.5

N_ITER = 5
NSIGMA = 3.0
REG = 1e-6
RCOND = 1e-8

NODATA_VALUES = [0, -9999]
REQUIRE_POSITIVE = True
MIN_VALID_FRACTION = 1.0

# If the CSV y/x columns are local crop coordinates, set the crop's top-left
# pixel location in the original HISUI L1G image here. For a full-image CSV or
# a CSV whose y/x columns are already original image coordinates, keep 0.
IMAGE_Y_OFFSET_FOR_SENSOR_GEOMETRY = 0
IMAGE_X_OFFSET_FOR_SENSOR_GEOMETRY = 0

RUN_PLOTS = True
USE_METADATA_DIRECTION_SLOPES = False
USE_METADATA_SENSOR_GEOMETRY = True


DEFAULT_DESTRIPE_PARAMS = {
    # Paper-inspired mode:
    # Use an approximate sensor-geometry line/column coordinate made from
    # HISUI observation/map corner metadata. If a real per-pixel sensor
    # line/column product is available later, replace sensor_line_map and
    # sensor_col_map in these params with those arrays.
    "geometry_mode": "sensor_metadata",
    "directions": ["sensor_line", "sensor_column"],
    "sensor_geometry_metadata_txt": METADATA_TXT,
    "sensor_line_map": None,
    "sensor_col_map": None,

    # Bin width in sensor-line or sensor-column pixels. Increase to 2-5 if
    # each line has too few valid background pixels; decrease toward 1 to
    # follow fine striping.
    "line_bin_width": 2.0,

    # Koga & Iwasaki (2011)-style correction: fit alpha along each constant
    # sensor-line curve as a linear function of sensor-column, then subtract it.
    "fit_degree": 1,
    "robust_fit_nsigma": 3.0,
    "robust_fit_max_iter": 4,
    "min_pixels_per_line": 20,
    "preserve_global_stat": True,
    "smooth_half_window": 2,

    # Extra cleanup for broad bands that are wider than a few pixels.
    # This runs after the paper-style per-line linear correction. It bins the
    # approximate sensor geometry into wider bands and subtracts a robust band
    # offset, while still excluding plume-like high-alpha pixels.
    "broad_band_cleanup": True,
    "broad_band_directions": ["sensor_line", "sensor_column"],
    "broad_band_bin_width": 16.0,
    "broad_band_smooth_half_window": 2,
    "broad_band_method": "median",
    "broad_band_min_pixels_per_band": 200,

    # Hough/Radon-like image-space cleanup for remaining straight broad bands.
    # It tests many positive slopes y=a*x, keeps only the strongest directions,
    # and subtracts robust offsets along those bands.
    "angle_sweep_cleanup": True,
    "angle_sweep_direction": "y_minus_x",
    "angle_sweep_slope_min": 0.30,
    "angle_sweep_slope_max": 2.50,
    "angle_sweep_num_slopes": 89,
    "angle_sweep_top_k": 2,
    "angle_sweep_min_slope_separation": 0.05,
    "angle_sweep_line_bin_width": 18.0,
    "angle_sweep_smooth_half_window": 0,
    "angle_sweep_min_pixels_per_line": 200,
    "angle_sweep_sample_step": 6,
    "angle_sweep_metric": "correction_gain",
    "angle_sweep_score_z_min": 2.0,

    # These older image-space parameters are kept so you can still switch
    # geometry_mode back to "image_lines" for comparison.
    "direction_slopes": {"y_minus_x": 1.017, "y_plus_x": 1.23},
    "use_metadata_direction_slopes": True,
    "auto_slope": True,
    "slope_search_pixel_drift": 60.0,
    "slope_search_step_pixel_drift": 2.0,
    "slope_search_sample_step": 4,
    "slope_search_metric": "offset_abs_p99",
    "method": "median",
    "exclude_mode": "robust_high",
    "exclude_nsigma": NSIGMA,
    "recompute_exclude_each_direction": True,
    "fallback_to_valid": True,
    "trim_fraction": 0.1,
    "mode_bins": 64,
    "sigma_clip_nsigma": 3.0,
    "sigma_clip_max_iter": 3,
    "threshold_source": "corrected",
}


EXPERIMENTS = {
    "baseline_no_destripe": {
        "destripe_when": "none",
        "destripe_params": None,
    },
    "paper_sensor_line_final_only": {
        "destripe_when": "final_only",
        "destripe_params": {
            **DEFAULT_DESTRIPE_PARAMS,
            "directions": ["sensor_line"],
        },
    },
    "paper_sensor_line_then_column_final_only": {
        "destripe_when": "final_only",
        "destripe_params": {**DEFAULT_DESTRIPE_PARAMS},
    },
    "paper_sensor_line_then_column_each_iter": {
        "destripe_when": "each_iter",
        "destripe_params": {**DEFAULT_DESTRIPE_PARAMS},
    },
    "paper_sensor_line_then_column_each_iter_angle2475_focus": {
        "destripe_when": "each_iter",
        "destripe_params": {
            **DEFAULT_DESTRIPE_PARAMS,
            # Use this when visual inspection shows the remaining band is the
            # steeper y=a*x family around a=2.475, not the statistically
            # strongest a=1.25 family.
            "angle_sweep_slope_min": 2.35,
            "angle_sweep_slope_max": 2.60,
            "angle_sweep_num_slopes": 101,
            "angle_sweep_top_k": 1,
            "angle_sweep_min_slope_separation": 0.02,
            "angle_sweep_line_bin_width": 18.0,
            "angle_sweep_score_z_min": None,
        },
    },
    "paper_sensor_line_then_column_each_iter_angle125_focus": {
        "destripe_when": "each_iter",
        "destripe_params": {
            **DEFAULT_DESTRIPE_PARAMS,
            # Use this when visual inspection shows the remaining band belongs
            # to the statistically strongest y=a*x family around a=1.25-1.275.
            "angle_sweep_slope_min": 1.18,
            "angle_sweep_slope_max": 1.34,
            "angle_sweep_num_slopes": 101,
            "angle_sweep_top_k": 1,
            "angle_sweep_min_slope_separation": 0.015,
            "angle_sweep_line_bin_width": 18.0,
            "angle_sweep_score_z_min": None,
        },
    },
}

# Set to None to run all experiments. For a first large-image test, you may use:
# RUN_EXPERIMENT_NAMES = ["baseline_no_destripe", "paper_sensor_line_then_column_each_iter_angle125_focus"]
RUN_EXPERIMENT_NAMES = ["baseline_no_destripe", "paper_sensor_line_then_column_each_iter_angle125_focus"]


# 1. Data loading helpers

def get_wave_columns(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    wave_cols: list[str] = []
    wavelengths: list[float] = []
    pattern = re.compile(r"^wave_([0-9]+(?:\.[0-9]+)?)nm$")

    for col in df.columns:
        match = pattern.match(str(col))
        if match is not None:
            wave_cols.append(str(col))
            wavelengths.append(float(match.group(1)))

    if not wave_cols:
        raise ValueError("No wavelength columns found. Expected columns like wave_2300.0nm.")

    wavelengths_arr = np.asarray(wavelengths, dtype=float)
    order = np.argsort(wavelengths_arr)
    return [wave_cols[i] for i in order], wavelengths_arr[order]


def load_roi_spectra_csv(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "y" not in df.columns or "x" not in df.columns:
        raise ValueError("The CSV file must contain y and x columns.")

    wave_cols, wavelengths = get_wave_columns(df)
    spectra = df[wave_cols].to_numpy(dtype=float)
    return df, wavelengths, spectra


def spectra_to_cube(
    df: pd.DataFrame,
    spectra: np.ndarray,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys = np.sort(df["y"].unique())
    xs = np.sort(df["x"].unique())
    y_to_i = {y: i for i, y in enumerate(ys)}
    x_to_j = {x: j for j, x in enumerate(xs)}

    H, W, B = len(ys), len(xs), spectra.shape[1]
    cube = np.full((H, W, B), fill_value, dtype=float)

    y_idx = df["y"].map(y_to_i).to_numpy()
    x_idx = df["x"].map(x_to_j).to_numpy()
    cube[y_idx, x_idx, :] = spectra
    return cube, ys, xs


def select_bands(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    wl_min: float,
    wl_max: float,
    exclude_ranges: Optional[Sequence[tuple[float, float]]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    if exclude_ranges is not None:
        for start, end in exclude_ranges:
            mask &= ~((wavelengths >= start) & (wavelengths <= end))
    if not np.any(mask):
        raise ValueError("No wavelength bands selected.")
    return cube[:, :, mask], wavelengths[mask], mask


# 2. MODTRAN and UAS helpers

def load_ch4_modtran_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "wavelength" not in df.columns:
        raise ValueError("MODTRAN CSV must contain a wavelength column.")

    mod_wave = df["wavelength"].to_numpy(dtype=float)
    alpha_cols = [c for c in df.columns if c != "wavelength"]
    alpha_grid = np.asarray([float(c) for c in alpha_cols], dtype=float)
    order = np.argsort(alpha_grid)
    alpha_grid = alpha_grid[order]
    alpha_cols = [alpha_cols[i] for i in order]
    spectra_grid = df[alpha_cols].to_numpy(dtype=float).T
    return mod_wave, alpha_grid, spectra_grid


def gaussian_srf_resample(
    mod_wave: np.ndarray,
    mod_spectra: np.ndarray,
    sensor_wave: np.ndarray,
    fwhm_nm: float | np.ndarray,
) -> np.ndarray:
    mod_wave = np.asarray(mod_wave, dtype=float)
    sensor_wave = np.asarray(sensor_wave, dtype=float)
    mod_spectra = np.asarray(mod_spectra, dtype=float)

    if np.isscalar(fwhm_nm):
        fwhm_arr = np.full(sensor_wave.shape, float(fwhm_nm), dtype=float)
    else:
        fwhm_arr = np.asarray(fwhm_nm, dtype=float)
        if fwhm_arr.shape != sensor_wave.shape:
            raise ValueError("fwhm_nm must be scalar or same length as sensor_wave.")

    out = np.zeros((mod_spectra.shape[0], sensor_wave.size), dtype=float)
    for j, center in enumerate(sensor_wave):
        sigma = fwhm_arr[j] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        use = np.abs(mod_wave - center) <= 4.0 * sigma
        if np.sum(use) < 2:
            out[:, j] = np.interp(center, mod_wave, mod_spectra)
            continue
        weights = np.exp(-0.5 * ((mod_wave[use] - center) / sigma) ** 2)
        weights = weights / np.sum(weights)
        out[:, j] = mod_spectra[:, use] @ weights
    return out


def compute_uas_log_slope(
    alpha_grid: np.ndarray,
    spectra_grid: np.ndarray,
    alpha_min: Optional[float] = None,
    alpha_max: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    spectra_grid = np.asarray(spectra_grid, dtype=float)

    use = np.ones_like(alpha_grid, dtype=bool)
    if alpha_min is not None:
        use &= alpha_grid >= alpha_min
    if alpha_max is not None:
        use &= alpha_grid <= alpha_max

    alpha = alpha_grid[use]
    log_spectra = np.log(np.maximum(spectra_grid[use], 1e-30))
    if alpha.size < 2:
        raise ValueError("Need at least two alpha values to estimate UAS.")

    design = np.vstack([np.ones_like(alpha), alpha]).T
    coeff, _, _, _ = np.linalg.lstsq(design, log_spectra, rcond=None)
    intercept = coeff[0]
    slope = coeff[1]
    uas = -slope
    return uas, intercept


# 3. Matched Filter helpers

def make_valid_pixel_mask(
    cube: np.ndarray,
    nodata_values: Optional[Sequence[float]] = None,
    require_positive: bool = True,
    min_valid_fraction: float = 1.0,
) -> np.ndarray:
    valid_band = np.isfinite(cube)

    if nodata_values is not None:
        for value in nodata_values:
            valid_band &= cube != value

    if require_positive:
        valid_band &= cube > 0

    valid_fraction = np.mean(valid_band, axis=2)
    return valid_fraction >= min_valid_fraction


def estimate_background_mean_cov_from_cube(
    cube: np.ndarray,
    background_mask: np.ndarray,
    reg: float = 1e-6,
    min_pixels: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, W, B = cube.shape
    if min_pixels is None:
        min_pixels = max(B + 5, 30)

    background_mask = np.asarray(background_mask, dtype=bool)
    X = cube[background_mask]
    good = np.all(np.isfinite(X), axis=1)
    X = X[good]
    if X.shape[0] < min_pixels:
        raise ValueError(f"Too few background pixels: {X.shape[0]} < {min_pixels}")

    mu = np.mean(X, axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    scale = np.nanmean(np.diag(cov))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    cov = cov + reg * scale * np.eye(B)
    return mu, cov, X


def make_methane_target(mu: np.ndarray, uas: np.ndarray, positive_alpha: bool = True) -> np.ndarray:
    return -mu * uas if positive_alpha else mu * uas


def matched_filter_alpha_map(
    cube: np.ndarray,
    uas: np.ndarray,
    valid_mask: np.ndarray,
    background_mask: np.ndarray,
    reg: float = 1e-6,
    rcond: float = 1e-8,
    min_background_pixels: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    H, W, B = cube.shape
    uas = np.asarray(uas, dtype=float).reshape(-1)
    if uas.size != B:
        raise ValueError("UAS length must match cube band count.")

    valid_mask = np.asarray(valid_mask, dtype=bool)
    background_mask = np.asarray(background_mask, dtype=bool) & valid_mask

    mu, cov, _ = estimate_background_mean_cov_from_cube(
        cube=cube,
        background_mask=background_mask,
        reg=reg,
        min_pixels=min_background_pixels,
    )
    target = make_methane_target(mu, uas, positive_alpha=True)
    inv_cov = np.linalg.pinv(cov, rcond=rcond)

    denominator = float(target.T @ inv_cov @ target)
    if abs(denominator) < 1e-12:
        raise ValueError("Matched Filter denominator is too small.")

    alpha = np.full((H, W), np.nan, dtype=float)
    X = cube[valid_mask]
    diff = X - mu
    numerator = diff @ inv_cov @ target
    alpha[valid_mask] = numerator / denominator
    return alpha, mu, cov, target


def robust_threshold_from_alpha(alpha_values: np.ndarray, nsigma: float = 3.0) -> tuple[float, float, float]:
    values = np.asarray(alpha_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite alpha values for thresholding.")

    med = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - med)))
    robust_std = 1.4826 * mad
    if not np.isfinite(robust_std) or robust_std <= 0:
        robust_std = float(np.nanstd(values))
    threshold = med + nsigma * robust_std
    return float(threshold), med, float(robust_std)


def plume_mask_from_alpha(alpha_map: np.ndarray, valid_mask: np.ndarray, nsigma: float = 3.0) -> tuple[np.ndarray, dict]:
    threshold, med, robust_std = robust_threshold_from_alpha(alpha_map[valid_mask], nsigma=nsigma)
    plume = np.zeros_like(valid_mask, dtype=bool)
    plume[valid_mask] = alpha_map[valid_mask] > threshold
    meta = {
        "threshold": float(threshold),
        "median": float(med),
        "robust_std": float(robust_std),
        "n_plume": int(np.sum(plume)),
    }
    return plume, meta


# 4. Tilt-aware diagonal destriping helpers

# normalize directions to a list of strings, and validate them based on the geometry mode
def normalize_directions(destripe_params: Optional[dict]) -> list[str]:
    if destripe_params is None:
        return []
    dirs = destripe_params.get("directions", destripe_params.get("direction", "y_minus_x"))
    if isinstance(dirs, str):
        dirs = [dirs]
    dirs = list(dirs)
    if destripe_params.get("geometry_mode") in {"sensor_metadata", "sensor_geometry"}:
        return dirs
    for d in dirs:
        if d not in {"y_minus_x", "y_plus_x"}:
            raise ValueError("directions must contain y_minus_x and/or y_plus_x.")
    return dirs

# get the slope for a given direction from destripe_params, with a fallback to default if not specified
def get_direction_slope(destripe_params: Optional[dict], direction: str, default: float = 1.0) -> float:
    if destripe_params is None:
        return float(default)
    slopes = destripe_params.get("direction_slopes")
    if isinstance(slopes, dict) and direction in slopes:
        return float(slopes[direction])
    if "slope" in destripe_params:
        return float(destripe_params["slope"])
    if "line_slope" in destripe_params:
        return float(destripe_params["line_slope"])
    return float(default)

# compute the slope for a direction based on pixel drift and run length, as an alternative to using metadata or defaults
def slope_from_pixel_drift(drift_pixels: float, run_pixels: int) -> float:
    run = max(int(run_pixels) - 1, 1)
    return 1.0 + float(drift_pixels) / float(run)

# compute a coordinate map for lines in the image based on the specified direction and slope, which will be used for binning pixels along those lines
def line_coordinate_map(
    shape: tuple[int, int],
    direction: str = "y_minus_x",
    slope: float = 1.0,
    row_step: int = 1,
    col_step: int = 1,
) -> np.ndarray:
    H, W = shape
    row_values = np.arange(0, H * row_step, row_step, dtype=float)
    col_values = np.arange(0, W * col_step, col_step, dtype=float)
    rows, cols = np.meshgrid(row_values, col_values, indexing="ij")

    if direction == "y_minus_x":
        return rows - float(slope) * cols
    if direction == "y_plus_x":
        return rows + float(slope) * cols
    raise ValueError("direction must be y_minus_x or y_plus_x.")

# compute a line ID map by binning the line coordinates with a specified bin width, which will be used to group pixels for statistics and correction along those lines
def line_id_map(
    shape: tuple[int, int],
    direction: str = "y_minus_x",
    slope: float = 1.0,
    line_bin_width: float = 1.0,
    row_step: int = 1,
    col_step: int = 1,
) -> np.ndarray:
    if line_bin_width <= 0:
        raise ValueError("line_bin_width must be positive.")
    coord = line_coordinate_map(shape, direction, slope, row_step, col_step)
    return np.rint(coord / float(line_bin_width)).astype(np.int32)

# compute a smoothed 1D statistic along each line by applying a moving window and a specified statistic function, which will be used to smooth the per-line corrections and avoid introducing new striping artifacts
def moving_nanmedian_1d(values: np.ndarray, half_window: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if half_window <= 0:
        return values.copy()
    out = np.full_like(values, np.nan, dtype=float)
    for i in range(values.size):
        lo = max(0, i - half_window)
        hi = min(values.size, i + half_window + 1)
        win = values[lo:hi]
        finite = np.isfinite(win)
        if np.any(finite):
            out[i] = np.nanmedian(win[finite])
    return out

# compute a robust statistic for a 1D array of values using various methods, which will be used to compute the per-line or per-band corrections while being resistant to outliers such as plume pixels
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
        return np.nan

    if method == "median":
        return float(np.nanmedian(values))
    if method == "mean":
        return float(np.nanmean(values))
    if method == "trimmed_mean":
        v = np.sort(values)
        k = int(np.floor(trim_fraction * v.size))
        if 2 * k >= v.size:
            return float(np.nanmean(v))
        return float(np.nanmean(v[k:v.size - k]))
    if method == "mode":
        if values.size == 1 or np.allclose(values, values[0]):
            return float(values[0])
        counts, edges = np.histogram(values, bins=mode_bins)
        idx = int(np.argmax(counts))
        return float(0.5 * (edges[idx] + edges[idx + 1]))
    if method == "sigma_clipped_mean":
        clipped = values.copy()
        for _ in range(int(sigma_clip_max_iter)):
            if clipped.size < 3:
                break
            med = float(np.nanmedian(clipped))
            mad = float(np.nanmedian(np.abs(clipped - med)))
            robust_std = 1.4826 * mad
            if not np.isfinite(robust_std) or robust_std <= 0:
                break
            keep = np.abs(clipped - med) <= sigma_clip_nsigma * robust_std
            if np.all(keep) or not np.any(keep):
                break
            clipped = clipped[keep]
        return float(np.nanmean(clipped))
    raise ValueError("Unknown statistic method.")

# compute the per-line statistic and count of pixels used for each line ID, which will be used to determine the correction for each line and whether to fallback to a different mask if there are too few valid pixels
def _group_line_stats(
    alpha_flat: np.ndarray,
    line_ids_flat: np.ndarray,
    mask_flat: np.ndarray,
    method: str = "median",
    trim_fraction: float = 0.1,
    mode_bins: int = 64,
    sigma_clip_nsigma: float = 3.0,
    sigma_clip_max_iter: int = 3,
) -> tuple[pd.Series, pd.Series]:
    mask_flat = np.asarray(mask_flat, dtype=bool)
    if not np.any(mask_flat):
        empty = pd.Index([], dtype=np.int32, name="line_id")
        return pd.Series(dtype=float, index=empty), pd.Series(dtype=int, index=empty)

    df = pd.DataFrame({
        "line_id": line_ids_flat[mask_flat].astype(np.int32, copy=False),
        "alpha": alpha_flat[mask_flat].astype(float, copy=False),
    })
    grouped = df.groupby("line_id", sort=True)["alpha"]
    counts = grouped.size()

    if method == "mean":
        stats = grouped.mean()
    elif method == "median":
        stats = grouped.median()
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
    return stats.astype(float), counts.astype(int)

# compute the per-line statistics for the estimate mask, and optionally fallback to the valid mask if there are too few pixels, which will be used to get the correction values for each line while ensuring enough pixels are used for a reliable estimate
def compute_line_stats_fast(
    alpha: np.ndarray,
    line_ids: np.ndarray,
    estimate_mask: np.ndarray,
    valid_mask: np.ndarray,
    min_pixels_per_line: int = 5,
    fallback_to_valid: bool = True,
    method: str = "median",
    trim_fraction: float = 0.1,
    mode_bins: int = 64,
    sigma_clip_nsigma: float = 3.0,
    sigma_clip_max_iter: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alpha_flat = np.asarray(alpha, dtype=float).ravel()
    ids_flat = np.asarray(line_ids, dtype=np.int32).ravel()
    estimate_flat = np.asarray(estimate_mask, dtype=bool).ravel() & np.isfinite(alpha_flat)
    valid_flat = np.asarray(valid_mask, dtype=bool).ravel() & np.isfinite(alpha_flat)

    id_min = int(np.nanmin(ids_flat))
    id_max = int(np.nanmax(ids_flat))
    id_values = np.arange(id_min, id_max + 1, dtype=np.int32)
    index = pd.Index(id_values, name="line_id")

    est_stats_s, est_counts_s = _group_line_stats(
        alpha_flat,
        ids_flat,
        estimate_flat,
        method,
        trim_fraction,
        mode_bins,
        sigma_clip_nsigma,
        sigma_clip_max_iter,
    )
    raw_stats = est_stats_s.reindex(index).to_numpy(dtype=float, copy=True)
    counts_used = est_counts_s.reindex(index, fill_value=0).to_numpy(dtype=int, copy=True)
    used_fallback = np.zeros(id_values.shape, dtype=bool)

    too_few = counts_used < int(min_pixels_per_line)
    if fallback_to_valid and np.any(too_few):
        valid_stats_s, valid_counts_s = _group_line_stats(
            alpha_flat,
            ids_flat,
            valid_flat,
            method,
            trim_fraction,
            mode_bins,
            sigma_clip_nsigma,
            sigma_clip_max_iter,
        )
        valid_stats = valid_stats_s.reindex(index).to_numpy(dtype=float, copy=True)
        valid_counts = valid_counts_s.reindex(index, fill_value=0).to_numpy(dtype=int, copy=True)
        can_fallback = too_few & (valid_counts >= int(min_pixels_per_line))
        raw_stats[can_fallback] = valid_stats[can_fallback]
        counts_used[can_fallback] = valid_counts[can_fallback]
        used_fallback[can_fallback] = True

    raw_stats[counts_used < int(min_pixels_per_line)] = np.nan
    return id_values, raw_stats, counts_used, used_fallback

# compute a boolean mask of pixels to exclude from destriping based on the alpha map and valid mask, using either a robust high-alpha threshold or a provided plume mask, which will be used to ignore pixels that are likely affected by methane plumes when computing the destriping corrections
def make_exclude_mask_for_destriping(
    alpha_map: np.ndarray,
    valid_mask: np.ndarray,
    plume_mask: Optional[np.ndarray] = None,
    exclude_mode: str = "robust_high",
    exclude_nsigma: float = 4.0,
) -> tuple[np.ndarray, dict]:
    valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(alpha_map)
    exclude = np.zeros_like(valid_mask, dtype=bool)
    meta: dict = {"exclude_mode": exclude_mode}

    if exclude_mode == "none" or exclude_mode is None:
        meta["n_excluded"] = 0
        return exclude, meta

    if "high" in exclude_mode:
        threshold, med, robust_std = robust_threshold_from_alpha(alpha_map[valid_mask], nsigma=exclude_nsigma)
        high_mask = np.zeros_like(valid_mask, dtype=bool)
        high_mask[valid_mask] = alpha_map[valid_mask] > threshold
        exclude |= high_mask
        meta.update({
            "high_threshold": float(threshold),
            "high_median": float(med),
            "high_robust_std": float(robust_std),
            "n_high_excluded": int(np.sum(high_mask)),
        })

    if "plume" in exclude_mode:
        if plume_mask is not None:
            plume_mask = np.asarray(plume_mask, dtype=bool)
            exclude |= plume_mask
            meta["n_plume_excluded"] = int(np.sum(plume_mask))
        else:
            meta["n_plume_excluded"] = 0

    if exclude_mode not in {"robust_high", "previous_plume", "previous_plume_or_high", "none", None}:
        raise ValueError("Invalid exclude_mode.")

    exclude &= valid_mask
    meta["n_excluded"] = int(np.sum(exclude))
    return exclude, meta

# generate candidate slopes for the angle sweep cleanup based on the specified pixel drift and step, or use provided candidates, which will be used to test different line orientations for destriping and find the best one according to the scoring metric
def make_slope_candidates(
    shape: tuple[int, int],
    base_slope: float = 1.0,
    search_pixel_drift: float = 2.0,
    step_pixel_drift: float = 0.25,
    slope_candidates: Optional[Sequence[float]] = None,
) -> np.ndarray:
    if slope_candidates is not None:
        candidates = np.asarray(list(slope_candidates), dtype=float)
        if candidates.size == 0:
            raise ValueError("slope_candidates must not be empty.")
        return np.unique(np.sort(candidates))

    W = max(int(shape[1]) - 1, 1)
    half_width = abs(float(search_pixel_drift)) / float(W)
    step = abs(float(step_pixel_drift)) / float(W)
    if step <= 0:
        raise ValueError("slope_search_step_pixel_drift must be positive.")
    n = int(np.floor((2.0 * half_width) / step))
    candidates = float(base_slope) - half_width + step * np.arange(n + 1)
    candidates = np.r_[candidates, float(base_slope), float(base_slope) + half_width]
    return np.unique(np.round(candidates, 12))

# compute the score for a given slope by computing the per-line statistics, calculating the offsets from the center, and applying the specified metric to those offsets, which will be used to evaluate how well a particular line orientation removes striping artifacts while being robust to outliers
def slope_search_score(
    alpha: np.ndarray,
    valid: np.ndarray,
    estimate_mask: np.ndarray,
    direction: str,
    slope: float,
    line_bin_width: float = 1.0,
    method: str = "median",
    min_pixels_per_line: int = 5,
    trim_fraction: float = 0.1,
    mode_bins: int = 64,
    sigma_clip_nsigma: float = 3.0,
    sigma_clip_max_iter: int = 3,
    sample_step: int = 4,
    metric: str = "offset_abs_p99",
) -> dict:
    sample_step = max(int(sample_step), 1)
    alpha_s = np.asarray(alpha, dtype=float)[::sample_step, ::sample_step]
    valid_s = np.asarray(valid, dtype=bool)[::sample_step, ::sample_step]
    estimate_s = np.asarray(estimate_mask, dtype=bool)[::sample_step, ::sample_step]

    ids_s = line_id_map(
        alpha_s.shape,
        direction=direction,
        slope=slope,
        line_bin_width=line_bin_width,
        row_step=sample_step,
        col_step=sample_step,
    )

    id_values, raw_stats, counts_used, _ = compute_line_stats_fast(
        alpha=alpha_s,
        line_ids=ids_s,
        estimate_mask=estimate_s,
        valid_mask=valid_s,
        min_pixels_per_line=max(2, int(np.ceil(min_pixels_per_line / sample_step))),
        fallback_to_valid=True,
        method=method,
        trim_fraction=trim_fraction,
        mode_bins=mode_bins,
        sigma_clip_nsigma=sigma_clip_nsigma,
        sigma_clip_max_iter=sigma_clip_max_iter,
    )

    finite = np.isfinite(raw_stats) & (counts_used > 0)
    if not np.any(finite):
        return {"slope": float(slope), "score": -np.inf, "n_lines_scored": 0}

    stats = raw_stats[finite]
    center = float(np.nanmedian(stats))
    offsets = stats - center
    abs_offsets = np.abs(offsets[np.isfinite(offsets)])
    if abs_offsets.size == 0:
        return {"slope": float(slope), "score": -np.inf, "n_lines_scored": int(np.sum(finite))}

    if metric in {"correction_gain", "correction_ratio"}:
        offsets_all = np.where(np.isfinite(raw_stats), raw_stats - center, 0.0)
        id_min = int(id_values[0])
        stripe_s = offsets_all[ids_s.astype(np.int64) - id_min]
        corrected_s = alpha_s - stripe_s
        score_mask = valid_s & np.isfinite(alpha_s) & np.isfinite(corrected_s)
        before_vals = alpha_s[score_mask]
        after_vals = corrected_s[score_mask]

        before_med = float(np.nanmedian(before_vals))
        before_mad = float(np.nanmedian(np.abs(before_vals - before_med)))
        before_rstd = 1.4826 * before_mad
        after_med = float(np.nanmedian(after_vals))
        after_mad = float(np.nanmedian(np.abs(after_vals - after_med)))
        after_rstd = 1.4826 * after_mad
        if not np.isfinite(before_rstd) or before_rstd <= 0:
            before_rstd = float(np.nanstd(before_vals))
        if not np.isfinite(after_rstd) or after_rstd <= 0:
            after_rstd = float(np.nanstd(after_vals))

        if metric == "correction_gain":
            score = float(before_rstd - after_rstd)
        else:
            score = float(before_rstd / after_rstd) if after_rstd > 0 else np.inf
    elif metric == "offset_abs_p99":
        score = float(np.nanpercentile(abs_offsets, 99))
    elif metric == "offset_abs_p95":
        score = float(np.nanpercentile(abs_offsets, 95))
    elif metric == "offset_abs_p90":
        score = float(np.nanpercentile(abs_offsets, 90))
    elif metric == "offset_abs_mad":
        score = float(np.nanmedian(abs_offsets))
    elif metric == "offset_std":
        score = float(np.nanstd(offsets))
    else:
        raise ValueError(
            "Invalid slope_search_metric. Use correction_gain, correction_ratio, "
            "offset_abs_p99, offset_abs_p95, offset_abs_p90, offset_abs_mad, or offset_std."
        )

    return {
        "slope": float(slope),
        "score": score,
        "n_lines_scored": int(np.sum(finite)),
        "line_stat_center": center,
    }

# evaluate multiple slope candidates and choose the best one based on the scoring metric, which will be used to determine the optimal line orientation for destriping that minimizes striping artifacts while being robust to plume pixels
def choose_best_slope(
    alpha: np.ndarray,
    valid: np.ndarray,
    estimate_mask: np.ndarray,
    direction: str,
    base_slope: float,
    destripe_params: dict,
    method: str,
    min_pixels_per_line: int,
    trim_fraction: float,
    mode_bins: int,
    sigma_clip_nsigma: float,
    sigma_clip_max_iter: int,
) -> tuple[float, pd.DataFrame]:
    slope_candidates = destripe_params.get("slope_candidates")
    if isinstance(slope_candidates, dict):
        slope_candidates = slope_candidates.get(direction)

    candidates = make_slope_candidates(
        shape=alpha.shape,
        base_slope=base_slope,
        search_pixel_drift=destripe_params.get("slope_search_pixel_drift", 2.0),
        step_pixel_drift=destripe_params.get("slope_search_step_pixel_drift", 0.25),
        slope_candidates=slope_candidates,
    )

    rows = []
    for slope in candidates:
        rows.append(slope_search_score(
            alpha=alpha,
            valid=valid,
            estimate_mask=estimate_mask,
            direction=direction,
            slope=float(slope),
            line_bin_width=destripe_params.get("line_bin_width", 1.0),
            method=method,
            min_pixels_per_line=min_pixels_per_line,
            trim_fraction=trim_fraction,
            mode_bins=mode_bins,
            sigma_clip_nsigma=sigma_clip_nsigma,
            sigma_clip_max_iter=sigma_clip_max_iter,
            sample_step=destripe_params.get("slope_search_sample_step", 4),
            metric=destripe_params.get("slope_search_metric", "offset_abs_p99"),
        ))

    search_table = pd.DataFrame(rows)
    if len(search_table) == 0 or not np.any(np.isfinite(search_table["score"])):
        return float(base_slope), search_table
    best_idx = search_table["score"].astype(float).idxmax()
    return float(search_table.loc[best_idx, "slope"]), search_table

# generate a linear set of slope candidates between specified min and max, or use provided candidates, ensuring they are positive and sorted, which will be used as an alternative to the pixel-drift-based candidates for testing line orientations in the angle sweep cleanup
def make_linear_slope_candidates(
    slope_min: float,
    slope_max: float,
    num_slopes: int,
    slope_candidates: Optional[Sequence[float]] = None,
) -> np.ndarray:
    if slope_candidates is not None:
        candidates = np.asarray(list(slope_candidates), dtype=float)
    else:
        if slope_max <= slope_min:
            raise ValueError("angle_sweep_slope_max must be greater than angle_sweep_slope_min.")
        candidates = np.linspace(float(slope_min), float(slope_max), int(num_slopes))
    candidates = candidates[np.isfinite(candidates) & (candidates > 0)]
    if candidates.size == 0:
        raise ValueError("No positive angle-sweep slope candidates.")
    return np.unique(np.sort(candidates))

# select the top slope candidates based on their scores, ensuring they are sufficiently separated in slope and optionally above a score threshold, which will be used to pick the best line orientations for the angle sweep cleanup while avoiding similar slopes that may produce similar corrections
def select_top_slope_candidates(
    search_table: pd.DataFrame,
    top_k: int = 2,
    min_slope_separation: float = 0.05,
    score_z_min: Optional[float] = 2.0,
) -> pd.DataFrame:
    if search_table is None or len(search_table) == 0:
        return pd.DataFrame()
    table = search_table[np.isfinite(search_table["score"])].copy()
    if len(table) == 0:
        return pd.DataFrame()

    scores = table["score"].to_numpy(dtype=float)
    score_floor = -np.inf
    if score_z_min is not None:
        med = float(np.nanmedian(scores))
        mad = float(np.nanmedian(np.abs(scores - med)))
        rstd = 1.4826 * mad
        if np.isfinite(rstd) and rstd > 0:
            score_floor = med + float(score_z_min) * rstd

    table = table.sort_values("score", ascending=False)
    selected = []
    for _, row in table.iterrows():
        slope = float(row["slope"])
        score = float(row["score"])
        if score < score_floor and selected:
            continue
        if any(abs(slope - float(prev["slope"])) < float(min_slope_separation) for prev in selected):
            continue
        selected.append(row)
        if len(selected) >= int(top_k):
            break

    if not selected:
        selected = [table.iloc[0]]
    return pd.DataFrame(selected).reset_index(drop=True)

# main function to perform angle sweep directional cleanup by testing multiple slopes, computing the corrections, and accumulating the results, which will be used to remove remaining broad band striping artifacts that are oriented along a particular angle in the image
def angle_sweep_directional_cleanup(
    alpha_map: np.ndarray,
    valid_mask: np.ndarray,
    plume_mask: Optional[np.ndarray] = None,
    destripe_params: Optional[dict] = None,
    nsigma: float = 4.0,
) -> dict:
    """Find and remove remaining straight broad bands by sweeping y=a*x slopes."""
    if destripe_params is None:
        destripe_params = {}

    direction = destripe_params.get("angle_sweep_direction", "y_minus_x")
    if direction not in {"y_minus_x", "y_plus_x"}:
        raise ValueError("angle_sweep_direction must be y_minus_x or y_plus_x.")

    current = np.asarray(alpha_map, dtype=float).copy()
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(current)
    exclude_mask, exclude_meta = make_exclude_mask_for_destriping(
        alpha_map=current,
        valid_mask=valid,
        plume_mask=plume_mask,
        exclude_mode=destripe_params.get("exclude_mode", "robust_high"),
        exclude_nsigma=destripe_params.get("exclude_nsigma", nsigma),
    )
    estimate_mask = valid & (~exclude_mask)
    if not np.any(estimate_mask):
        estimate_mask = valid

    candidates = make_linear_slope_candidates(
        slope_min=destripe_params.get("angle_sweep_slope_min", 0.3),
        slope_max=destripe_params.get("angle_sweep_slope_max", 2.5),
        num_slopes=destripe_params.get("angle_sweep_num_slopes", 89),
        slope_candidates=destripe_params.get("angle_sweep_slope_candidates"),
    )

    method = destripe_params.get("broad_band_method", destripe_params.get("method", "median"))
    min_pixels_per_line = destripe_params.get("angle_sweep_min_pixels_per_line", 200)
    trim_fraction = destripe_params.get("trim_fraction", 0.1)
    mode_bins = destripe_params.get("mode_bins", 64)
    sigma_clip_nsigma = destripe_params.get("sigma_clip_nsigma", 3.0)
    sigma_clip_max_iter = destripe_params.get("sigma_clip_max_iter", 3)
    line_bin_width = destripe_params.get("angle_sweep_line_bin_width", 18.0)

    rows = []
    for slope in candidates:
        rows.append(slope_search_score(
            alpha=current,
            valid=valid,
            estimate_mask=estimate_mask,
            direction=direction,
            slope=float(slope),
            line_bin_width=line_bin_width,
            method=method,
            min_pixels_per_line=min_pixels_per_line,
            trim_fraction=trim_fraction,
            mode_bins=mode_bins,
            sigma_clip_nsigma=sigma_clip_nsigma,
            sigma_clip_max_iter=sigma_clip_max_iter,
            sample_step=destripe_params.get("angle_sweep_sample_step", 6),
            metric=destripe_params.get("angle_sweep_metric", "correction_gain"),
        ))
    search_table = pd.DataFrame(rows)
    if len(search_table) > 0:
        search_table.insert(0, "direction", direction)
        search_table.insert(1, "line_bin_width", float(line_bin_width))
        search_table.insert(2, "correction_type", "angle_sweep_score")

    selected = select_top_slope_candidates(
        search_table=search_table,
        top_k=destripe_params.get("angle_sweep_top_k", 2),
        min_slope_separation=destripe_params.get("angle_sweep_min_slope_separation", 0.05),
        score_z_min=destripe_params.get("angle_sweep_score_z_min", 2.0),
    )

    total_stripe = np.zeros_like(current, dtype=float)
    directional_maps: dict[str, np.ndarray] = {}
    line_tables = []

    for pass_index, row in selected.iterrows():
        slope = float(row["slope"])
        exclude_mask, exclude_meta = make_exclude_mask_for_destriping(
            alpha_map=current,
            valid_mask=valid_mask,
            plume_mask=plume_mask,
            exclude_mode=destripe_params.get("exclude_mode", "robust_high"),
            exclude_nsigma=destripe_params.get("exclude_nsigma", nsigma),
        )
        out = destripe_by_directional_lines(
            alpha_map=current,
            valid_mask=valid_mask,
            exclude_mask=exclude_mask,
            direction=direction,
            method=method,
            min_pixels_per_line=min_pixels_per_line,
            preserve_global_stat=destripe_params.get("preserve_global_stat", True),
            smooth_half_window=destripe_params.get("angle_sweep_smooth_half_window", 0),
            fallback_to_valid=destripe_params.get("fallback_to_valid", True),
            trim_fraction=trim_fraction,
            mode_bins=mode_bins,
            sigma_clip_nsigma=sigma_clip_nsigma,
            sigma_clip_max_iter=sigma_clip_max_iter,
            slope=slope,
            line_bin_width=line_bin_width,
        )
        current = out["corrected"]
        total_stripe += out["stripe_map"]
        map_key = f"angle_sweep_{direction}_slope_{slope:.4f}"
        directional_maps[map_key] = out["stripe_map"].copy()

        table = out["line_table"].copy()
        table["pass_index"] = int(pass_index) + 1
        table["correction_type"] = "angle_sweep"
        table["angle_sweep_score"] = float(row.get("score", np.nan))
        table["exclude_threshold"] = exclude_meta.get("threshold")
        table["exclude_mode"] = exclude_meta.get("exclude_mode")
        line_tables.append(table)

    line_table = pd.concat(line_tables, ignore_index=True) if line_tables else pd.DataFrame()
    return {
        "corrected": current,
        "stripe_map": total_stripe,
        "directional_stripe_maps": directional_maps,
        "line_table": line_table,
        "selected_slopes": selected,
        "search_table": search_table,
        "exclude_meta": exclude_meta,
    }

# perform destriping by computing and subtracting the per-line offsets based on the specified direction and method, which will be used to remove striping artifacts along the given line orientation while preserving the overall image statistics if desired
def destripe_by_directional_lines(
    alpha_map: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    exclude_mask: Optional[np.ndarray] = None,
    direction: str = "y_minus_x",
    method: str = "median",
    min_pixels_per_line: int = 5,
    preserve_global_stat: bool = True,
    smooth_half_window: int = 0,
    fallback_to_valid: bool = True,
    trim_fraction: float = 0.1,
    mode_bins: int = 64,
    sigma_clip_nsigma: float = 3.0,
    sigma_clip_max_iter: int = 3,
    slope: float = 1.0,
    line_bin_width: float = 1.0,
) -> dict:
    alpha = np.asarray(alpha_map, dtype=float)
    if alpha.ndim != 2:
        raise ValueError("alpha_map must be 2D.")

    finite = np.isfinite(alpha)
    valid = finite.copy() if valid_mask is None else np.asarray(valid_mask, dtype=bool) & finite
    if not np.any(valid):
        raise ValueError("No valid finite pixels for destriping.")

    estimate_mask = valid.copy() if exclude_mask is None else valid & (~np.asarray(exclude_mask, dtype=bool))
    if not np.any(estimate_mask):
        if fallback_to_valid:
            estimate_mask = valid.copy()
        else:
            raise ValueError("No pixels remain after exclusion for stripe estimation.")

    ids = line_id_map(alpha.shape, direction=direction, slope=slope, line_bin_width=line_bin_width)

    global_stat = statistic_1d(
        alpha[estimate_mask],
        method=method,
        trim_fraction=trim_fraction,
        mode_bins=mode_bins,
        sigma_clip_nsigma=sigma_clip_nsigma,
        sigma_clip_max_iter=sigma_clip_max_iter,
    )

    id_values, raw_stats, counts_used, used_fallback = compute_line_stats_fast(
        alpha=alpha,
        line_ids=ids,
        estimate_mask=estimate_mask,
        valid_mask=valid,
        min_pixels_per_line=min_pixels_per_line,
        fallback_to_valid=fallback_to_valid,
        method=method,
        trim_fraction=trim_fraction,
        mode_bins=mode_bins,
        sigma_clip_nsigma=sigma_clip_nsigma,
        sigma_clip_max_iter=sigma_clip_max_iter,
    )

    smoothed_stats = moving_nanmedian_1d(raw_stats, half_window=smooth_half_window)
    offsets = smoothed_stats - global_stat if preserve_global_stat and np.isfinite(global_stat) else smoothed_stats
    offsets_filled = np.where(np.isfinite(offsets), offsets, 0.0)

    id_min = int(id_values[0])
    stripe_map = offsets_filled[ids.astype(np.int64) - id_min]
    stripe_map = np.asarray(stripe_map, dtype=float)

    corrected = alpha - stripe_map
    corrected[~finite] = np.nan

    line_table = pd.DataFrame({
        "line_id": id_values,
        "line_coordinate_center": id_values.astype(float) * float(line_bin_width),
        "n_pixels_used": counts_used,
        "used_fallback_to_valid": used_fallback,
        "line_stat_raw": raw_stats,
        "line_stat_after_smoothing": smoothed_stats,
        "stripe_offset_subtracted": offsets_filled,
        "global_stat": global_stat,
        "direction": direction,
        "slope": float(slope),
        "line_bin_width": float(line_bin_width),
        "method": method,
    })

    return {
        "corrected": corrected,
        "stripe_map": stripe_map,
        "line_table": line_table,
        "global_stat": global_stat,
        "estimate_mask": estimate_mask,
    }

# normalize the sensor geometry directions from the destriping parameters, allowing for aliases and ensuring they are valid, which will be used to determine which line orientations to use for sensor-geometry-based destriping while providing flexibility in how the directions are specified
def normalize_sensor_geometry_directions(destripe_params: Optional[dict]) -> list[str]:
    if destripe_params is None:
        return []
    directions = destripe_params.get("directions", ["sensor_line"])
    if isinstance(directions, str):
        directions = [directions]

    aliases = {
        "line": "sensor_line",
        "sensor_line": "sensor_line",
        "along_track": "sensor_line",
        "scan_line": "sensor_line",
        "column": "sensor_column",
        "sensor_col": "sensor_column",
        "sensor_column": "sensor_column",
        "cross_track": "sensor_column",
    }
    out = []
    for direction in directions:
        key = aliases.get(str(direction), str(direction))
        if key not in {"sensor_line", "sensor_column"}:
            raise ValueError(f"Unknown sensor-geometry direction: {direction}")
        out.append(key)
    return out

#  perform a robust polynomial fit to the given coordinates and values, iteratively excluding outliers based on the residuals, which will be used to model and remove any broad trends in the per-line statistics that may be due to sensor geometry rather than striping artifacts
def _robust_polyfit_1d(
    coord: np.ndarray,
    values: np.ndarray,
    degree: int = 1,
    nsigma: float = 3.0,
    max_iter: int = 4,
) -> dict:
    coord = np.asarray(coord, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(coord) & np.isfinite(values)
    coord = coord[ok]
    values = values[ok]

    min_needed = max(int(degree) + 1, 2)
    if values.size < min_needed:
        return {
            "coeff": None,
            "center": np.nan,
            "scale": np.nan,
            "n_initial": int(values.size),
            "n_used": 0,
            "robust_std": np.nan,
        }

    center = float(np.nanmedian(coord))
    q25, q75 = np.nanpercentile(coord, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(coord))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    x = (coord - center) / scale
    X = np.vander(x, N=int(degree) + 1, increasing=True)
    use = np.ones(values.size, dtype=bool)

    coeff = None
    robust_std = np.nan
    for _ in range(max(1, int(max_iter))):
        if np.sum(use) < min_needed:
            break
        coeff = np.linalg.lstsq(X[use], values[use], rcond=None)[0]
        residual = values - (X @ coeff)
        med = np.nanmedian(residual[use])
        mad = np.nanmedian(np.abs(residual[use] - med))
        robust_std = float(1.4826 * mad)
        if not np.isfinite(robust_std) or robust_std <= 0:
            robust_std = float(np.nanstd(residual[use]))
        if not np.isfinite(robust_std) or robust_std <= 0:
            break
        new_use = np.abs(residual - med) <= float(nsigma) * robust_std
        if np.array_equal(new_use, use):
            break
        use = new_use

    if coeff is None or np.sum(use) < min_needed:
        return {
            "coeff": None,
            "center": center,
            "scale": scale,
            "n_initial": int(values.size),
            "n_used": int(np.sum(use)),
            "robust_std": robust_std,
        }

    coeff = np.linalg.lstsq(X[use], values[use], rcond=None)[0]
    residual = values[use] - (X[use] @ coeff)
    mad = np.nanmedian(np.abs(residual - np.nanmedian(residual)))
    robust_std = float(1.4826 * mad) if np.isfinite(mad) else np.nan

    return {
        "coeff": coeff,
        "center": center,
        "scale": scale,
        "n_initial": int(values.size),
        "n_used": int(np.sum(use)),
        "robust_std": robust_std,
    }

# predict the values at the given coordinates using the polynomial coefficients and normalization parameters, which will be used to compute the modeled trend in the per-line statistics based on the fitted polynomial for sensor-geometry-based destriping
def _poly_predict(coord: np.ndarray, coeff: np.ndarray, center: float, scale: float) -> np.ndarray:
    x = (np.asarray(coord, dtype=float) - float(center)) / float(scale)
    X = np.vander(x, N=len(coeff), increasing=True)
    return X @ coeff

# main function to perform destriping based on sensor geometry by computing per-line statistics along the specified sensor lines or columns, fitting a robust polynomial to those statistics, and subtracting the modeled trend from the image, which will be used to remove striping artifacts that are correlated with the sensor readout geometry while preserving the overall image statistics if desired
def destripe_by_sensor_geometry_lines(
    alpha_map: np.ndarray,
    sensor_line_map: np.ndarray,
    sensor_col_map: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    exclude_mask: Optional[np.ndarray] = None,
    direction: str = "sensor_line",
    method: str = "median",
    min_pixels_per_line: int = 20,
    preserve_global_stat: bool = True,
    smooth_half_window: int = 2,
    fallback_to_valid: bool = True,
    line_bin_width: float = 2.0,
    fit_degree: int = 1,
    robust_fit_nsigma: float = 3.0,
    robust_fit_max_iter: int = 4,
    trim_fraction: float = 0.1,
    mode_bins: int = 64,
    sigma_clip_nsigma: float = 3.0,
    sigma_clip_max_iter: int = 3,
) -> dict:
    alpha = np.asarray(alpha_map, dtype=float)
    sensor_line = np.asarray(sensor_line_map, dtype=float)
    sensor_col = np.asarray(sensor_col_map, dtype=float)
    if alpha.ndim != 2:
        raise ValueError("alpha_map must be 2D.")
    if sensor_line.shape != alpha.shape or sensor_col.shape != alpha.shape:
        raise ValueError("sensor_line_map and sensor_col_map must match alpha_map shape.")
    if line_bin_width <= 0:
        raise ValueError("line_bin_width must be positive.")

    direction = normalize_sensor_geometry_directions({"directions": [direction]})[0]
    if direction == "sensor_line":
        group_coord = sensor_line
        fit_coord = sensor_col
    else:
        group_coord = sensor_col
        fit_coord = sensor_line

    finite = np.isfinite(alpha)
    valid = finite.copy() if valid_mask is None else np.asarray(valid_mask, dtype=bool) & finite
    valid &= np.isfinite(group_coord) & np.isfinite(fit_coord)
    if not np.any(valid):
        raise ValueError("No valid finite pixels for sensor-geometry destriping.")

    estimate_mask = valid.copy() if exclude_mask is None else valid & (~np.asarray(exclude_mask, dtype=bool))
    if not np.any(estimate_mask):
        if fallback_to_valid:
            estimate_mask = valid.copy()
        else:
            raise ValueError("No pixels remain after exclusion for stripe estimation.")

    global_stat = statistic_1d(
        alpha[estimate_mask],
        method=method,
        trim_fraction=trim_fraction,
        mode_bins=mode_bins,
        sigma_clip_nsigma=sigma_clip_nsigma,
        sigma_clip_max_iter=sigma_clip_max_iter,
    )
    if not np.isfinite(global_stat):
        global_stat = 0.0

    ids = np.full(alpha.shape, np.iinfo(np.int64).min, dtype=np.int64)
    ids[valid] = np.rint(group_coord[valid] / float(line_bin_width)).astype(np.int64)

    alpha_flat = alpha.ravel()
    fit_flat = fit_coord.ravel()
    valid_flat = valid.ravel()
    estimate_flat = estimate_mask.ravel()
    ids_flat = ids.ravel()

    flat_idx = np.flatnonzero(valid_flat)
    ordered = flat_idx[np.argsort(ids_flat[flat_idx], kind="mergesort")]
    ordered_ids = ids_flat[ordered]
    id_values, starts, counts = np.unique(ordered_ids, return_index=True, return_counts=True)

    n_groups = len(id_values)
    coeffs = np.full((n_groups, int(fit_degree) + 1), np.nan, dtype=float)
    centers = np.full(n_groups, np.nan, dtype=float)
    scales = np.full(n_groups, np.nan, dtype=float)
    n_pixels_valid = np.zeros(n_groups, dtype=int)
    n_pixels_used = np.zeros(n_groups, dtype=int)
    n_pixels_initial = np.zeros(n_groups, dtype=int)
    used_fallback = np.zeros(n_groups, dtype=bool)
    fit_rstd = np.full(n_groups, np.nan, dtype=float)

    for i, (start, count) in enumerate(zip(starts, counts)):
        group_idx = ordered[start:start + count]
        fit_idx = group_idx[estimate_flat[group_idx]]
        if fit_idx.size < min_pixels_per_line and fallback_to_valid:
            fit_idx = group_idx
            used_fallback[i] = True
        n_pixels_valid[i] = int(group_idx.size)

        if fit_idx.size < max(int(fit_degree) + 1, min_pixels_per_line):
            continue

        fit = _robust_polyfit_1d(
            coord=fit_flat[fit_idx],
            values=alpha_flat[fit_idx],
            degree=int(fit_degree),
            nsigma=float(robust_fit_nsigma),
            max_iter=int(robust_fit_max_iter),
        )
        if fit["coeff"] is None:
            continue

        coeffs[i, :] = fit["coeff"]
        centers[i] = fit["center"]
        scales[i] = fit["scale"]
        n_pixels_initial[i] = fit["n_initial"]
        n_pixels_used[i] = fit["n_used"]
        fit_rstd[i] = fit["robust_std"]

    coeffs_raw = coeffs.copy()
    if smooth_half_window > 0 and n_groups > 0:
        for j in range(coeffs.shape[1]):
            coeffs[:, j] = moving_nanmedian_1d(coeffs[:, j], half_window=int(smooth_half_window))

    stripe_flat = np.zeros(alpha.size, dtype=float)
    for i, (start, count) in enumerate(zip(starts, counts)):
        if not np.all(np.isfinite(coeffs[i, :])) or not np.isfinite(centers[i]) or not np.isfinite(scales[i]):
            continue
        group_idx = ordered[start:start + count]
        model = _poly_predict(fit_flat[group_idx], coeffs[i, :], centers[i], scales[i])
        offset = model - float(global_stat) if preserve_global_stat else model
        stripe_flat[group_idx] = offset

    stripe_map = stripe_flat.reshape(alpha.shape)
    corrected = alpha - stripe_map
    corrected[~finite] = np.nan

    line_table = pd.DataFrame({
        "line_id": id_values,
        "line_coordinate_center": id_values.astype(float) * float(line_bin_width),
        "direction": direction,
        "line_bin_width": float(line_bin_width),
        "fit_degree": int(fit_degree),
        "n_pixels_valid": n_pixels_valid,
        "n_pixels_initial": n_pixels_initial,
        "n_pixels_used": n_pixels_used,
        "used_fallback_to_valid": used_fallback,
        "global_stat": float(global_stat),
        "fit_robust_std": fit_rstd,
        "fit_center": centers,
        "fit_scale": scales,
        "method_for_global_stat": method,
    })
    for j in range(coeffs.shape[1]):
        line_table[f"coef_{j}_raw"] = coeffs_raw[:, j]
        line_table[f"coef_{j}_after_smoothing"] = coeffs[:, j]
    if coeffs.shape[1] >= 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            line_table["linear_slope_per_sensor_pixel_raw"] = coeffs_raw[:, 1] / scales
            line_table["linear_slope_per_sensor_pixel_after_smoothing"] = coeffs[:, 1] / scales
        line_table["stripe_offset_subtracted"] = coeffs[:, 0] - float(global_stat)
    else:
        line_table["stripe_offset_subtracted"] = coeffs[:, 0] - float(global_stat)

    return {
        "corrected": corrected,
        "stripe_map": stripe_map,
        "line_table": line_table,
        "global_stat": float(global_stat),
        "estimate_mask": estimate_mask,
    }

# main function to perform destriping based on broad sensor geometry bands by computing per-band statistics along the specified sensor lines or columns, smoothing those statistics, and subtracting the smoothed trend from the image, which will be used to remove broad band striping artifacts that are correlated with the sensor readout geometry while preserving the overall image statistics if desired
def destripe_by_sensor_geometry_broad_bands(
    alpha_map: np.ndarray,
    sensor_line_map: np.ndarray,
    sensor_col_map: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    exclude_mask: Optional[np.ndarray] = None,
    direction: str = "sensor_line",
    method: str = "median",
    min_pixels_per_band: int = 200,
    preserve_global_stat: bool = True,
    smooth_half_window: int = 2,
    fallback_to_valid: bool = True,
    band_bin_width: float = 16.0,
    trim_fraction: float = 0.1,
    mode_bins: int = 64,
    sigma_clip_nsigma: float = 3.0,
    sigma_clip_max_iter: int = 3,
) -> dict:
    """Remove broad sensor-coordinate bands after fine line destriping."""
    alpha = np.asarray(alpha_map, dtype=float)
    sensor_line = np.asarray(sensor_line_map, dtype=float)
    sensor_col = np.asarray(sensor_col_map, dtype=float)
    if alpha.ndim != 2:
        raise ValueError("alpha_map must be 2D.")
    if sensor_line.shape != alpha.shape or sensor_col.shape != alpha.shape:
        raise ValueError("sensor_line_map and sensor_col_map must match alpha_map shape.")
    if band_bin_width <= 0:
        raise ValueError("band_bin_width must be positive.")

    direction = normalize_sensor_geometry_directions({"directions": [direction]})[0]
    group_coord = sensor_line if direction == "sensor_line" else sensor_col

    finite = np.isfinite(alpha)
    valid = finite.copy() if valid_mask is None else np.asarray(valid_mask, dtype=bool) & finite
    valid &= np.isfinite(group_coord)
    if not np.any(valid):
        raise ValueError("No valid finite pixels for broad-band destriping.")

    estimate_mask = valid.copy() if exclude_mask is None else valid & (~np.asarray(exclude_mask, dtype=bool))
    if not np.any(estimate_mask):
        if fallback_to_valid:
            estimate_mask = valid.copy()
        else:
            raise ValueError("No pixels remain after exclusion for broad-band estimation.")

    ids = np.zeros(alpha.shape, dtype=np.int32)
    ids[valid] = np.rint(group_coord[valid] / float(band_bin_width)).astype(np.int32)

    global_stat = statistic_1d(
        alpha[estimate_mask],
        method=method,
        trim_fraction=trim_fraction,
        mode_bins=mode_bins,
        sigma_clip_nsigma=sigma_clip_nsigma,
        sigma_clip_max_iter=sigma_clip_max_iter,
    )
    if not np.isfinite(global_stat):
        global_stat = 0.0

    id_values, raw_stats, counts_used, used_fallback = compute_line_stats_fast(
        alpha=alpha,
        line_ids=ids,
        estimate_mask=estimate_mask,
        valid_mask=valid,
        min_pixels_per_line=min_pixels_per_band,
        fallback_to_valid=fallback_to_valid,
        method=method,
        trim_fraction=trim_fraction,
        mode_bins=mode_bins,
        sigma_clip_nsigma=sigma_clip_nsigma,
        sigma_clip_max_iter=sigma_clip_max_iter,
    )

    smoothed_stats = moving_nanmedian_1d(raw_stats, half_window=smooth_half_window)
    offsets = smoothed_stats - global_stat if preserve_global_stat else smoothed_stats
    offsets_filled = np.where(np.isfinite(offsets), offsets, 0.0)

    id_min = int(id_values[0])
    stripe_map = offsets_filled[ids.astype(np.int64) - id_min]
    stripe_map = np.asarray(stripe_map, dtype=float)
    stripe_map[~valid] = 0.0

    corrected = alpha - stripe_map
    corrected[~finite] = np.nan

    line_table = pd.DataFrame({
        "line_id": id_values,
        "line_coordinate_center": id_values.astype(float) * float(band_bin_width),
        "n_pixels_used": counts_used,
        "used_fallback_to_valid": used_fallback,
        "line_stat_raw": raw_stats,
        "line_stat_after_smoothing": smoothed_stats,
        "stripe_offset_subtracted": offsets_filled,
        "global_stat": float(global_stat),
        "direction": direction,
        "correction_type": "broad_band",
        "line_bin_width": float(band_bin_width),
        "method": method,
    })

    return {
        "corrected": corrected,
        "stripe_map": stripe_map,
        "line_table": line_table,
        "global_stat": float(global_stat),
        "estimate_mask": estimate_mask,
    }

# main function to perform destriping by sequentially applying sensor-geometry-based line destriping along the specified directions, optionally followed by broad-band cleanup, and accumulating the results, which will be used to remove striping artifacts that are correlated with the sensor readout geometry in multiple passes while preserving the overall image statistics if desired
def destripe_by_sequential_sensor_geometry(
    alpha_map: np.ndarray,
    valid_mask: np.ndarray,
    plume_mask: Optional[np.ndarray] = None,
    destripe_params: Optional[dict] = None,
    nsigma: float = 4.0,
) -> dict:
    if destripe_params is None:
        destripe_params = {}

    directions = normalize_sensor_geometry_directions(destripe_params)
    if len(directions) == 0:
        zero = np.zeros_like(alpha_map, dtype=float)
        return {
            "corrected": np.asarray(alpha_map, dtype=float).copy(),
            "stripe_map": zero,
            "directional_stripe_maps": {},
            "line_table": pd.DataFrame(),
            "exclude_meta": [],
            "slope_search_tables": {},
        }

    sensor_line = destripe_params.get("sensor_line_map")
    sensor_col = destripe_params.get("sensor_col_map")
    if sensor_line is None or sensor_col is None:
        raise ValueError(
            "sensor_line_map and sensor_col_map are required for geometry_mode='sensor_metadata'. "
            "Call apply_metadata_sensor_geometry_to_experiments(...) in main, or provide real "
            "per-pixel sensor geometry arrays."
        )

    current = np.asarray(alpha_map, dtype=float).copy()
    total_stripe = np.zeros_like(current, dtype=float)
    directional_stripe_maps: dict[str, np.ndarray] = {}
    line_tables = []
    exclude_metas = []
    slope_search_tables: dict[str, pd.DataFrame] = {}

    recompute_exclude = destripe_params.get("recompute_exclude_each_direction", True)
    fixed_exclude_mask = None
    fixed_exclude_meta = None
    if not recompute_exclude:
        fixed_exclude_mask, fixed_exclude_meta = make_exclude_mask_for_destriping(
            alpha_map=current,
            valid_mask=valid_mask,
            plume_mask=plume_mask,
            exclude_mode=destripe_params.get("exclude_mode", "robust_high"),
            exclude_nsigma=destripe_params.get("exclude_nsigma", nsigma),
        )

    for pass_index, direction in enumerate(directions, start=1):
        if recompute_exclude:
            exclude_mask, exclude_meta = make_exclude_mask_for_destriping(
                alpha_map=current,
                valid_mask=valid_mask,
                plume_mask=plume_mask,
                exclude_mode=destripe_params.get("exclude_mode", "robust_high"),
                exclude_nsigma=destripe_params.get("exclude_nsigma", nsigma),
            )
        else:
            exclude_mask = fixed_exclude_mask
            exclude_meta = dict(fixed_exclude_meta)

        out = destripe_by_sensor_geometry_lines(
            alpha_map=current,
            sensor_line_map=sensor_line,
            sensor_col_map=sensor_col,
            valid_mask=valid_mask,
            exclude_mask=exclude_mask,
            direction=direction,
            method=destripe_params.get("method", "median"),
            min_pixels_per_line=destripe_params.get("min_pixels_per_line", 20),
            preserve_global_stat=destripe_params.get("preserve_global_stat", True),
            smooth_half_window=destripe_params.get("smooth_half_window", 2),
            fallback_to_valid=destripe_params.get("fallback_to_valid", True),
            line_bin_width=destripe_params.get("line_bin_width", 2.0),
            fit_degree=destripe_params.get("fit_degree", 1),
            robust_fit_nsigma=destripe_params.get("robust_fit_nsigma", 3.0),
            robust_fit_max_iter=destripe_params.get("robust_fit_max_iter", 4),
            trim_fraction=destripe_params.get("trim_fraction", 0.1),
            mode_bins=destripe_params.get("mode_bins", 64),
            sigma_clip_nsigma=destripe_params.get("sigma_clip_nsigma", 3.0),
            sigma_clip_max_iter=destripe_params.get("sigma_clip_max_iter", 3),
        )

        stripe_map = out["stripe_map"]
        current = out["corrected"]
        total_stripe += stripe_map
        map_key = direction if direction not in directional_stripe_maps else f"{direction}_{pass_index}"
        directional_stripe_maps[map_key] = stripe_map.copy()

        table = out["line_table"].copy()
        table["pass_index"] = pass_index
        table["correction_type"] = "fine_line_linear"
        table["exclude_threshold"] = exclude_meta.get("threshold")
        table["exclude_mode"] = exclude_meta.get("exclude_mode")
        line_tables.append(table)
        exclude_meta["pass_index"] = pass_index
        exclude_meta["direction"] = direction
        exclude_metas.append(exclude_meta)

    if destripe_params.get("broad_band_cleanup", False):
        broad_directions = destripe_params.get("broad_band_directions", ["sensor_line"])
        broad_directions = normalize_sensor_geometry_directions({"directions": broad_directions})
        next_pass_index = len(directions) + 1
        for broad_i, direction in enumerate(broad_directions, start=next_pass_index):
            exclude_mask, exclude_meta = make_exclude_mask_for_destriping(
                alpha_map=current,
                valid_mask=valid_mask,
                plume_mask=plume_mask,
                exclude_mode=destripe_params.get("exclude_mode", "robust_high"),
                exclude_nsigma=destripe_params.get("exclude_nsigma", nsigma),
            )

            out = destripe_by_sensor_geometry_broad_bands(
                alpha_map=current,
                sensor_line_map=sensor_line,
                sensor_col_map=sensor_col,
                valid_mask=valid_mask,
                exclude_mask=exclude_mask,
                direction=direction,
                method=destripe_params.get("broad_band_method", destripe_params.get("method", "median")),
                min_pixels_per_band=destripe_params.get("broad_band_min_pixels_per_band", 200),
                preserve_global_stat=destripe_params.get("preserve_global_stat", True),
                smooth_half_window=destripe_params.get("broad_band_smooth_half_window", 2),
                fallback_to_valid=destripe_params.get("fallback_to_valid", True),
                band_bin_width=destripe_params.get("broad_band_bin_width", 16.0),
                trim_fraction=destripe_params.get("trim_fraction", 0.1),
                mode_bins=destripe_params.get("mode_bins", 64),
                sigma_clip_nsigma=destripe_params.get("sigma_clip_nsigma", 3.0),
                sigma_clip_max_iter=destripe_params.get("sigma_clip_max_iter", 3),
            )

            stripe_map = out["stripe_map"]
            current = out["corrected"]
            total_stripe += stripe_map
            map_key = f"broad_band_{direction}"
            if map_key in directional_stripe_maps:
                map_key = f"{map_key}_{broad_i}"
            directional_stripe_maps[map_key] = stripe_map.copy()

            table = out["line_table"].copy()
            table["pass_index"] = broad_i
            table["exclude_threshold"] = exclude_meta.get("threshold")
            table["exclude_mode"] = exclude_meta.get("exclude_mode")
            line_tables.append(table)

            exclude_meta["pass_index"] = broad_i
            exclude_meta["direction"] = direction
            exclude_meta["correction_type"] = "broad_band"
            exclude_metas.append(exclude_meta)

    if destripe_params.get("angle_sweep_cleanup", False):
        out = angle_sweep_directional_cleanup(
            alpha_map=current,
            valid_mask=valid_mask,
            plume_mask=plume_mask,
            destripe_params=destripe_params,
            nsigma=nsigma,
        )
        current = out["corrected"]
        total_stripe += out["stripe_map"]
        for key, value in out["directional_stripe_maps"].items():
            directional_stripe_maps[key] = value.copy()
        if out["line_table"] is not None and len(out["line_table"]) > 0:
            table = out["line_table"].copy()
            table["pass_index"] = table["pass_index"] + len(directions)
            if destripe_params.get("broad_band_cleanup", False):
                table["pass_index"] = table["pass_index"] + len(
                    normalize_sensor_geometry_directions({
                        "directions": destripe_params.get("broad_band_directions", ["sensor_line"])
                    })
                )
            line_tables.append(table)
        slope_search_tables["angle_sweep_all_candidates"] = out["search_table"].copy()
        slope_search_tables["angle_sweep_selected_slopes"] = out["selected_slopes"].copy()

        meta = dict(out.get("exclude_meta", {}))
        meta["correction_type"] = "angle_sweep"
        if len(out["selected_slopes"]) > 0:
            meta["selected_slopes"] = ",".join(f"{float(v):.6f}" for v in out["selected_slopes"]["slope"])
        exclude_metas.append(meta)

    line_table = pd.concat(line_tables, ignore_index=True) if line_tables else pd.DataFrame()

    return {
        "corrected": current,
        "stripe_map": total_stripe,
        "directional_stripe_maps": directional_stripe_maps,
        "line_table": line_table,
        "exclude_meta": exclude_metas,
        "slope_search_tables": slope_search_tables,
    }

# main function to perform destriping by sequentially applying sensor-geometry-based line destriping along the specified directions, optionally followed by broad-band cleanup and angle sweep cleanup, and accumulating the results, which will be used to remove striping artifacts that are correlated with the sensor readout geometry in multiple passes while preserving the overall image statistics if desired
def destripe_by_sequential_directions(
    alpha_map: np.ndarray,
    valid_mask: np.ndarray,
    plume_mask: Optional[np.ndarray] = None,
    destripe_params: Optional[dict] = None,
    nsigma: float = 4.0,
) -> dict:
    if destripe_params is None:
        destripe_params = {}

    if destripe_params.get("geometry_mode") in {"sensor_metadata", "sensor_geometry"}:
        return destripe_by_sequential_sensor_geometry(
            alpha_map=alpha_map,
            valid_mask=valid_mask,
            plume_mask=plume_mask,
            destripe_params=destripe_params,
            nsigma=nsigma,
        )

    directions = normalize_directions(destripe_params)
    if len(directions) == 0:
        zero = np.zeros_like(alpha_map, dtype=float)
        return {
            "corrected": np.asarray(alpha_map, dtype=float).copy(),
            "stripe_map": zero,
            "directional_stripe_maps": {},
            "line_table": pd.DataFrame(),
            "exclude_meta": [],
            "slope_search_tables": {},
        }

    current = np.asarray(alpha_map, dtype=float).copy()
    total_stripe = np.zeros_like(current, dtype=float)
    directional_stripe_maps: dict[str, np.ndarray] = {}
    line_tables = []
    exclude_metas = []
    slope_search_tables: dict[str, pd.DataFrame] = {}

    recompute_exclude = destripe_params.get("recompute_exclude_each_direction", True)
    fixed_exclude_mask = None
    fixed_exclude_meta = None
    if not recompute_exclude:
        fixed_exclude_mask, fixed_exclude_meta = make_exclude_mask_for_destriping(
            alpha_map=current,
            valid_mask=valid_mask,
            plume_mask=plume_mask,
            exclude_mode=destripe_params.get("exclude_mode", "robust_high"),
            exclude_nsigma=destripe_params.get("exclude_nsigma", nsigma),
        )

    for pass_index, direction in enumerate(directions, start=1):
        if recompute_exclude:
            exclude_mask, exclude_meta = make_exclude_mask_for_destriping(
                alpha_map=current,
                valid_mask=valid_mask,
                plume_mask=plume_mask,
                exclude_mode=destripe_params.get("exclude_mode", "robust_high"),
                exclude_nsigma=destripe_params.get("exclude_nsigma", nsigma),
            )
        else:
            exclude_mask = fixed_exclude_mask
            exclude_meta = dict(fixed_exclude_meta)

        method = destripe_params.get("method", "median")
        min_pixels_per_line = destripe_params.get("min_pixels_per_line", 5)
        trim_fraction = destripe_params.get("trim_fraction", 0.1)
        mode_bins = destripe_params.get("mode_bins", 64)
        sigma_clip_nsigma = destripe_params.get("sigma_clip_nsigma", 3.0)
        sigma_clip_max_iter = destripe_params.get("sigma_clip_max_iter", 3)
        base_slope = get_direction_slope(destripe_params, direction, default=1.0)
        slope = float(base_slope)
        search_table = pd.DataFrame()

        if destripe_params.get("auto_slope", False):
            valid_for_search = np.asarray(valid_mask, dtype=bool) & np.isfinite(current)
            estimate_for_search = valid_for_search & (~np.asarray(exclude_mask, dtype=bool))
            if not np.any(estimate_for_search):
                estimate_for_search = valid_for_search

            slope, search_table = choose_best_slope(
                alpha=current,
                valid=valid_for_search,
                estimate_mask=estimate_for_search,
                direction=direction,
                base_slope=base_slope,
                destripe_params=destripe_params,
                method=method,
                min_pixels_per_line=min_pixels_per_line,
                trim_fraction=trim_fraction,
                mode_bins=mode_bins,
                sigma_clip_nsigma=sigma_clip_nsigma,
                sigma_clip_max_iter=sigma_clip_max_iter,
            )
            if len(search_table) > 0:
                search_table = search_table.copy()
                search_table.insert(0, "pass_index", pass_index)
                search_table.insert(1, "direction", direction)
                search_table.insert(2, "base_slope", float(base_slope))
            slope_search_tables[f"pass{pass_index}_{direction}"] = search_table

        out = destripe_by_directional_lines(
            alpha_map=current,
            valid_mask=valid_mask,
            exclude_mask=exclude_mask,
            direction=direction,
            method=method,
            min_pixels_per_line=min_pixels_per_line,
            preserve_global_stat=destripe_params.get("preserve_global_stat", True),
            smooth_half_window=destripe_params.get("smooth_half_window", 0),
            fallback_to_valid=destripe_params.get("fallback_to_valid", True),
            trim_fraction=trim_fraction,
            mode_bins=mode_bins,
            sigma_clip_nsigma=sigma_clip_nsigma,
            sigma_clip_max_iter=sigma_clip_max_iter,
            slope=slope,
            line_bin_width=destripe_params.get("line_bin_width", 1.0),
        )

        current = out["corrected"]
        total_stripe = total_stripe + out["stripe_map"]
        directional_stripe_maps[direction] = out["stripe_map"].copy()

        table = out["line_table"].copy()
        table.insert(0, "pass_index", pass_index)
        table.insert(1, "base_slope", float(base_slope))
        table.insert(2, "auto_slope", bool(destripe_params.get("auto_slope", False)))
        line_tables.append(table)

        exclude_meta = dict(exclude_meta)
        exclude_meta.update({
            "pass_index": pass_index,
            "direction": direction,
            "base_slope": float(base_slope),
            "selected_slope": float(slope),
            "auto_slope": bool(destripe_params.get("auto_slope", False)),
        })
        exclude_metas.append(exclude_meta)

    line_table_all = pd.concat(line_tables, ignore_index=True) if line_tables else pd.DataFrame()
    return {
        "corrected": current,
        "stripe_map": total_stripe,
        "directional_stripe_maps": directional_stripe_maps,
        "line_table": line_table_all,
        "exclude_meta": exclude_metas,
        "slope_search_tables": slope_search_tables,
    }


# 5. Iterative MF with optional tilted destriping

# Decide whether to apply destriping at the current iteration based on the destripe_when parameter, which can be "none", "each_iter", "final_only", or a list of iteration numbers to apply destriping, which will be used to control the application of destriping in the iterative matched filtering process
def should_apply_destriping(iteration_number: int, destripe_when) -> bool:
    if destripe_when is None or destripe_when == "none":
        return False
    if destripe_when == "each_iter":
        return True
    if destripe_when == "final_only":
        return False
    if isinstance(destripe_when, (list, tuple, set, np.ndarray)):
        return int(iteration_number) in {int(v) for v in destripe_when}
    raise ValueError("Invalid destripe_when.")

# Main function to run iterative matched filtering with optional destriping at specified iterations, which will be used to perform iterative background estimation and plume detection while optionally applying destriping in between iterations to improve the quality of the matched filter results in the presence of striping artifacts correlated with the sensor geometry
def run_iterative_mf_with_optional_destriping(
    cube: np.ndarray,
    uas: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    initial_background_mask: Optional[np.ndarray] = None,
    n_iter: int = 5,
    nsigma: float = 4.0,
    reg: float = 1e-6,
    rcond: float = 1e-8,
    min_background_pixels: Optional[int] = None,
    destripe_when="none",
    destripe_params: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    H, W, B = cube.shape

    if valid_mask is None:
        valid_mask = np.all(np.isfinite(cube), axis=2)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if initial_background_mask is None:
        background_mask = valid_mask.copy()
    else:
        background_mask = valid_mask & np.asarray(initial_background_mask, dtype=bool)

    if min_background_pixels is None:
        min_background_pixels = max(B + 5, 30)

    if destripe_params is None:
        destripe_params = {}

    alpha_raw_history = []
    alpha_corrected_history = []
    alpha_used_history = []
    stripe_history = []
    directional_stripe_history = []
    slope_search_history = []
    plume_mask_history = []
    background_mask_history = []
    threshold_meta_history = []
    line_table_history = []
    exclude_meta_history = []
    mu_history = []
    cov_history = []

    prev_plume_mask = None
    converged_iter = None
    directions = normalize_directions(destripe_params) if destripe_params else []

    for it in range(1, n_iter + 1):
        n_bg = int(np.sum(background_mask))
        if n_bg < min_background_pixels:
            raise ValueError(f"Background pixels too few at iter {it}: {n_bg} < {min_background_pixels}")

        alpha_raw, mu, cov, _ = matched_filter_alpha_map(
            cube=cube,
            uas=uas,
            valid_mask=valid_mask,
            background_mask=background_mask,
            reg=reg,
            rcond=rcond,
            min_background_pixels=min_background_pixels,
        )

        alpha_corrected = alpha_raw.copy()
        alpha_used = alpha_raw.copy()
        stripe_map = np.zeros((H, W), dtype=float)
        directional_stripe_maps = {}
        slope_search_tables = {}
        line_table = pd.DataFrame()
        exclude_meta = []

        apply_now = should_apply_destriping(it, destripe_when)
        if apply_now:
            out = destripe_by_sequential_directions(
                alpha_map=alpha_raw,
                valid_mask=valid_mask,
                plume_mask=prev_plume_mask,
                destripe_params=destripe_params,
                nsigma=nsigma,
            )
            alpha_corrected = out["corrected"]
            stripe_map = out["stripe_map"]
            directional_stripe_maps = out["directional_stripe_maps"]
            slope_search_tables = out.get("slope_search_tables", {})
            line_table = out["line_table"]
            exclude_meta = out["exclude_meta"]

            threshold_source = destripe_params.get("threshold_source", "corrected")
            if threshold_source == "corrected":
                alpha_used = alpha_corrected.copy()
            elif threshold_source == "raw":
                alpha_used = alpha_raw.copy()
            else:
                raise ValueError("threshold_source must be corrected or raw.")

        plume_mask, threshold_meta = plume_mask_from_alpha(alpha_used, valid_mask, nsigma=nsigma)
        new_background_mask = valid_mask & (~plume_mask)

        alpha_raw_history.append(alpha_raw.copy())
        alpha_corrected_history.append(alpha_corrected.copy())
        alpha_used_history.append(alpha_used.copy())
        stripe_history.append(stripe_map.copy())
        directional_stripe_history.append({k: v.copy() for k, v in directional_stripe_maps.items()})
        slope_search_history.append({k: v.copy() for k, v in slope_search_tables.items()})
        plume_mask_history.append(plume_mask.copy())
        background_mask_history.append(background_mask.copy())
        threshold_meta_history.append(threshold_meta.copy())
        line_table_history.append(line_table.copy())
        exclude_meta_history.append(exclude_meta.copy() if hasattr(exclude_meta, "copy") else exclude_meta)
        mu_history.append(mu.copy())
        cov_history.append(cov.copy())

        if verbose:
            print(
                f"iter {it:02d} | bg={n_bg:8d} | "
                f"thr={threshold_meta['threshold']:+.6e} | "
                f"med={threshold_meta['median']:+.6e} | "
                f"rstd={threshold_meta['robust_std']:.6e} | "
                f"plume={threshold_meta['n_plume']:8d} | "
                f"destripe={apply_now} | directions={directions if apply_now else []}"
            )

        if prev_plume_mask is not None and np.array_equal(plume_mask, prev_plume_mask):
            converged_iter = it
            background_mask = new_background_mask
            if verbose:
                print(f"Converged at iteration {it}.")
            break

        prev_plume_mask = plume_mask.copy()
        background_mask = new_background_mask

    final_only_post = None
    if destripe_when == "final_only":
        final_raw = alpha_raw_history[-1]
        final_loop_plume = plume_mask_history[-1]

        out = destripe_by_sequential_directions(
            alpha_map=final_raw,
            valid_mask=valid_mask,
            plume_mask=final_loop_plume,
            destripe_params=destripe_params,
            nsigma=nsigma,
        )
        final_corr = out["corrected"]
        final_plume, final_meta = plume_mask_from_alpha(final_corr, valid_mask, nsigma=nsigma)

        alpha_corrected_history[-1] = final_corr.copy()
        alpha_used_history[-1] = final_corr.copy()
        stripe_history[-1] = out["stripe_map"].copy()
        directional_stripe_history[-1] = {k: v.copy() for k, v in out["directional_stripe_maps"].items()}
        slope_search_history[-1] = {k: v.copy() for k, v in out.get("slope_search_tables", {}).items()}
        plume_mask_history[-1] = final_plume.copy()
        threshold_meta_history[-1] = final_meta.copy()
        line_table_history[-1] = out["line_table"].copy()
        exclude_meta_history[-1] = out["exclude_meta"]
        final_only_post = {"threshold_meta": final_meta, "exclude_meta": out["exclude_meta"]}

    result = {
        "alpha_raw_history": alpha_raw_history,
        "alpha_corrected_history": alpha_corrected_history,
        "alpha_used_history": alpha_used_history,
        "stripe_history": stripe_history,
        "directional_stripe_history": directional_stripe_history,
        "slope_search_history": slope_search_history,
        "plume_mask_history": plume_mask_history,
        "background_mask_history": background_mask_history,
        "threshold_meta_history": threshold_meta_history,
        "line_table_history": line_table_history,
        "exclude_meta_history": exclude_meta_history,
        "mu_history": mu_history,
        "cov_history": cov_history,
        "valid_mask": valid_mask,
        "background_mask_final": background_mask,
        "converged_iter": converged_iter,
        "destripe_when": destripe_when,
        "destripe_params": destripe_params,
        "final_only_post": final_only_post,
    }

    result["alpha_final_raw"] = alpha_raw_history[-1]
    result["alpha_final_corrected"] = alpha_corrected_history[-1]
    result["alpha_final_used"] = alpha_used_history[-1]
    result["stripe_map_final"] = stripe_history[-1]
    result["directional_stripe_maps_final"] = directional_stripe_history[-1]
    result["slope_search_tables_final"] = slope_search_history[-1]
    result["plume_mask_final"] = plume_mask_history[-1]
    result["threshold_meta_final"] = threshold_meta_history[-1]
    result["line_table_final"] = line_table_history[-1]
    return result


# 6. Plotting and saving helpers

# Extract finite values from an image, optionally applying a mask, which will be used to compute robust limits for plotting and thresholding while ignoring non-finite values and optionally focusing on a specific region of interest defined by the mask
def finite_values(img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    arr = np.asarray(img, dtype=float)
    if mask is None:
        vals = arr[np.isfinite(arr)]
    else:
        vals = arr[np.isfinite(arr) & np.asarray(mask, dtype=bool)]
    return vals

# Compute robust limits (e.g., 2nd and 98th percentiles) from one or more images, optionally applying a mask, which will be used to determine appropriate display ranges for plotting while being robust to outliers and optionally focusing on a specific region of interest defined by the mask
def robust_limits(imgs, mask: Optional[np.ndarray] = None, q_low: float = 2, q_high: float = 98) -> tuple[float, float]:
    if isinstance(imgs, (list, tuple)):
        vals = np.concatenate([finite_values(img, mask=mask) for img in imgs])
    else:
        vals = finite_values(imgs, mask=mask)
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(vals, [q_low, q_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        med = np.nanmedian(vals)
        return float(med - 1), float(med + 1)
    return float(lo), float(hi)

# Summarize results from multiple experiments into a DataFrame, which will be used to create a summary table of the results from multiple runs of the iterative matched filtering and destriping process, including key metrics and parameters for each experiment for easy comparison and analysis
def summarize_results_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        meta = res["threshold_meta_final"]
        params = res.get("destripe_params")
        line_table = res.get("line_table_final")
        if line_table is not None and len(line_table) > 0 and "slope" in line_table.columns:
            slope_parts = []
            slope_rows = line_table[np.isfinite(line_table["slope"])][["pass_index", "direction", "slope"]]
            for _, row in slope_rows.drop_duplicates().iterrows():
                slope_parts.append(f"{row['direction']}:{float(row['slope']):.8f}")
            selected_slopes = " -> ".join(slope_parts)
        else:
            selected_slopes = None
        rows.append({
            "name": name,
            "destripe_when": res["destripe_when"],
            "method": None if params is None else params.get("method"),
            "directions": None if params is None else " -> ".join(normalize_directions(params)),
            "auto_slope": None if params is None else params.get("auto_slope"),
            "selected_slopes": selected_slopes,
            "threshold": meta["threshold"],
            "median": meta["median"],
            "robust_std": meta["robust_std"],
            "plume_pixels": int(np.sum(res["plume_mask_final"])),
            "valid_pixels": int(np.sum(res["valid_mask"])),
            "converged_iter": res["converged_iter"],
        })
    return pd.DataFrame(rows)

# Plot raw, corrected, and plume mask images for multiple experiments in a grid layout, which will be used to visually compare the results of multiple runs of the iterative matched filtering and destriping process by showing the raw alpha map, the corrected alpha map after destriping, and the final plume mask for each experiment in a grid format for easy side-by-side comparison
def plot_experiment_grid(results: dict[str, dict], names: Optional[Sequence[str]] = None) -> None:
    if names is None:
        names = list(results.keys())
    names = [n for n in names if n in results]
    if not names:
        return

    valid = next(iter(results.values()))["valid_mask"]
    maps = []
    for name in names:
        maps.append(results[name]["alpha_final_raw"])
        maps.append(results[name]["alpha_final_corrected"])
    vmin, vmax = robust_limits(maps, mask=valid)

    n = len(names)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    if n == 1:
        axes = axes.reshape(3, 1)

    for j, name in enumerate(names):
        res = results[name]
        raw = res["alpha_final_raw"]
        corr = res["alpha_final_corrected"]
        plume = res["plume_mask_final"]

        axes[0, j].imshow(raw, origin="upper", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(f"{name}\nraw")
        axes[1, j].imshow(corr, origin="upper", vmin=vmin, vmax=vmax)
        axes[1, j].set_title("corrected")
        axes[2, j].imshow(plume, origin="upper")
        axes[2, j].set_title("plume mask")

        for i in range(3):
            axes[i, j].set_xlabel("x")
            axes[i, j].set_ylabel("y")

    plt.tight_layout()
    plt.show()

# Plot raw, corrected, stripe map, and plume mask for a single experiment, along with directional stripe maps and line tables if available, which will be used to visually analyze the results of a single run of the iterative matched filtering and destriping process by showing the key outputs including the raw alpha map, the corrected alpha map after destriping, the estimated stripe map, and the final plume mask, as well as any directional stripe maps and line tables that provide additional insights into the destriping process
def plot_single_result(res: dict, title_prefix: str = "result") -> None:
    valid = res["valid_mask"]
    raw = res["alpha_final_raw"]
    corr = res["alpha_final_corrected"]
    stripe = res["stripe_map_final"]
    plume = res["plume_mask_final"]
    directional_maps = res.get("directional_stripe_maps_final", {})

    vmin, vmax = robust_limits([raw, corr], mask=valid)
    svmin, svmax = robust_limits(stripe, mask=valid)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    im0 = axes[0].imshow(raw, origin="upper", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{title_prefix}\nraw alpha")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(stripe, origin="upper", vmin=svmin, vmax=svmax)
    axes[1].set_title("total estimated stripe")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(corr, origin="upper", vmin=vmin, vmax=vmax)
    axes[2].set_title("corrected alpha")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(plume, origin="upper")
    axes[3].set_title("plume mask")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    plt.tight_layout()
    plt.show()

    if directional_maps:
        n = len(directional_maps)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
        if n == 1:
            axes = [axes]
        for ax, (direction, smap) in zip(axes, directional_maps.items()):
            lo, hi = robust_limits(smap, mask=valid)
            im = ax.imshow(smap, origin="upper", vmin=lo, vmax=hi)
            ax.set_title(f"stripe: {direction}")
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.show()

    line_table = res.get("line_table_final")
    if line_table is not None and len(line_table) > 0:
        plt.figure(figsize=(10, 4))
        group_cols = ["direction"]
        if "correction_type" in line_table.columns:
            group_cols.append("correction_type")
        if "slope" in line_table.columns:
            group_cols.append("slope")
        for key, df_dir in line_table.groupby(group_cols, dropna=False):
            if isinstance(key, tuple) and len(key) >= 3 and "slope" in group_cols:
                if np.isfinite(float(key[2])):
                    label = f"{key[0]}, {key[1]}, slope={float(key[2]):.6f}"
                else:
                    label = f"{key[0]}, {key[1]}"
            elif isinstance(key, tuple) and len(key) >= 2:
                label = ", ".join(str(k) for k in key)
            elif isinstance(key, tuple) and len(key) == 1:
                label = str(key[0])
            else:
                label = str(key)
            plt.plot(df_dir["line_id"], df_dir["stripe_offset_subtracted"], marker=".", linewidth=1, label=label)
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("line_id")
        plt.ylabel("subtracted offset")
        plt.title(f"{title_prefix}: stripe offsets")
        plt.grid(True)
        plt.legend()
        plt.show()

# Save outputs for a single case to .npy files and CSV, which will be used to persist the results of a single run of the iterative matched filtering and destriping process by saving the key outputs including the raw alpha map, the corrected alpha map, the estimated stripe map, the plume mask, and any directional stripe maps to .npy files, as well as saving a pixel-wise summary and line tables to CSV files for further analysis and record-keeping
def save_case_outputs(
    result: dict,
    case_name: str,
    output_dir: str | Path,
    ys: Optional[np.ndarray] = None,
    xs: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = result["alpha_final_raw"]
    corr = result["alpha_final_corrected"]
    stripe = result["stripe_map_final"]
    plume = result["plume_mask_final"]
    valid = result["valid_mask"]
    directional_maps = result.get("directional_stripe_maps_final", {})
    params = result.get("destripe_params") or {}

    np.save(output_dir / f"{case_name}_alpha_raw.npy", raw)
    np.save(output_dir / f"{case_name}_alpha_corrected.npy", corr)
    np.save(output_dir / f"{case_name}_stripe_map_total.npy", stripe)
    np.save(output_dir / f"{case_name}_plume_mask.npy", plume)

    for direction, smap in directional_maps.items():
        np.save(output_dir / f"{case_name}_stripe_map_{direction}.npy", smap)

    H, W = raw.shape
    rows, cols = np.indices((H, W))
    y_values = np.asarray(ys)[rows.ravel()] if ys is not None and len(ys) == H else rows.ravel()
    x_values = np.asarray(xs)[cols.ravel()] if xs is not None and len(xs) == W else cols.ravel()

    pixel_df = pd.DataFrame({
        "row": rows.ravel(),
        "col": cols.ravel(),
        "y": y_values,
        "x": x_values,
        "is_valid": valid.ravel(),
        "alpha_raw": raw.ravel(),
        "stripe_offset_total_subtracted": stripe.ravel(),
        "alpha_corrected": corr.ravel(),
        "is_plume": plume.ravel(),
    })
    for direction, smap in directional_maps.items():
        pixel_df[f"stripe_offset_{direction}"] = smap.ravel()

    sensor_line = params.get("sensor_line_map")
    sensor_col = params.get("sensor_col_map")
    if isinstance(sensor_line, np.ndarray) and sensor_line.shape == raw.shape:
        pixel_df["sensor_line_approx"] = sensor_line.ravel()
    if isinstance(sensor_col, np.ndarray) and sensor_col.shape == raw.shape:
        pixel_df["sensor_col_approx"] = sensor_col.ravel()

    pixel_df.to_csv(output_dir / f"{case_name}_pixel_results.csv", index=False)

    line_table = result.get("line_table_final")
    if line_table is not None and len(line_table) > 0:
        line_table.to_csv(output_dir / f"{case_name}_line_table.csv", index=False)

    slope_tables = result.get("slope_search_tables_final", {})
    if isinstance(slope_tables, dict):
        for key, table in slope_tables.items():
            if table is not None and len(table) > 0:
                safe_key = str(key).replace("/", "_").replace("\\", "_")
                table.to_csv(output_dir / f"{case_name}_{safe_key}_slope_search.csv", index=False)

    return pixel_df


# 7. HISUI metadata direction helpers

# Parse simple key = value lines from a HISUI metadata txt file, which will be used to extract relevant metadata from the HISUI product that may be needed for estimating the sensor geometry and observation footprint for destriping purposes
def parse_hisui_metadata_txt(path: str | Path) -> dict[str, str]:
    """Parse simple key = value lines from a HISUI metadata txt file."""
    path = Path(path)
    meta: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"')
        if key:
            meta[key] = value
    return meta

# Extract a float value from metadata by key, raising KeyError if not found, which will be used to retrieve specific numeric values from the HISUI metadata that are needed for geometry calculations while ensuring that missing keys are properly handled
def _metadata_float(meta: dict[str, str], key: str) -> float:
    if key not in meta:
        raise KeyError(f"Metadata key not found: {key}")
    return float(meta[key])

# Extract an integer value from metadata by key, allowing for float representations of integers, and raising KeyError if not found, which will be used to retrieve specific integer values from the HISUI metadata that are needed for geometry calculations while being flexible to handle cases where the metadata may represent integers as floats
def _metadata_int(meta: dict[str, str], key: str) -> int:
    if key not in meta:
        raise KeyError(f"Metadata key not found: {key}")
    return int(float(meta[key]))

# Extract latitude and longitude from metadata for a given prefix, which will be used to retrieve the geographic coordinates of specific points (e.g., corners of the map and observation footprint) from the HISUI metadata for use in estimating the sensor geometry and performing destriping
def _metadata_latlon(meta: dict[str, str], prefix: str) -> tuple[float, float]:
    return (
        _metadata_float(meta, f"{prefix}LatitudeDegree"),
        _metadata_float(meta, f"{prefix}LongitudeDegree"),
    )

# Compute local east/north delta in meters from two lat/lon points, which will be used to convert geographic coordinates into a local Cartesian coordinate system that approximates the east and north directions in meters, which is useful for estimating the sensor geometry and performing destriping based on the observation footprint
def _local_delta_east_north_m(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Approximate local east/north delta in meters from two lat/lon points."""
    lat1, lon1 = a
    lat2, lon2 = b
    lat0 = np.deg2rad((lat1 + lat2) / 2.0)
    north = (lat2 - lat1) * 111_320.0
    east = (lon2 - lon1) * 111_320.0 * np.cos(lat0)
    return float(east), float(north)

# Compute the image slope for a north-up image based on the local east/north delta between two points, which will be used to estimate the slope of the observation footprint in the image coordinates, where row is positive south and column is positive east, which is important for performing tilted destriping that accounts for the oblique observation geometry
def _image_slope_row_per_col(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Return image slope for a north-up image: row is positive south, col is positive east."""
    east, north = _local_delta_east_north_m(a, b)
    south = -north
    if abs(east) < 1e-12:
        return np.inf
    return float(south / east)

# Fit an affine transform from source [x, y] to destination [u, v], where the source is augmented with a constant 1 for the affine term, which will be used to estimate a simple linear transformation that maps local east/north coordinates to image column/row coordinates based on the corners of the map and observation footprint, which is a key step in performing destriping that accounts for the sensor geometry
def _fit_affine_2d(src_xy: np.ndarray, dst_uv: np.ndarray) -> np.ndarray:
    """Fit [x, y, 1] @ coeff = [u, v]."""
    src_xy = np.asarray(src_xy, dtype=float)
    dst_uv = np.asarray(dst_uv, dtype=float)
    X = np.column_stack([src_xy[:, 0], src_xy[:, 1], np.ones(src_xy.shape[0])])
    coeff = np.linalg.lstsq(X, dst_uv, rcond=None)[0]
    return coeff

# Apply an affine transform to x and y coordinates using the given coefficients, which will be used to convert local east/north coordinates into image column/row coordinates based on the fitted affine transform, which is necessary for estimating where the observation footprint corners fall in the image pixels for destriping purposes
def _apply_affine_2d(x: np.ndarray, y: np.ndarray, coeff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = coeff[0, 0] * x + coeff[1, 0] * y + coeff[2, 0]
    v = coeff[0, 1] * x + coeff[1, 1] * y + coeff[2, 1]
    return u, v

# Estimate where HISUI observation-footprint corners fall in L1G image pixels by fitting an affine transform from local lat/lon coordinates to image row/column based on the map corners, and then applying that transform to the observation footprint corners, which will be used to determine the approximate pixel locations of the observation footprint corners in the L1G image, which is important for performing destriping that accounts for the sensor geometry
def estimate_observation_corner_pixels_from_hisui_metadata(path: str | Path) -> dict:
    meta = parse_hisui_metadata_txt(path)
    image_lines = _metadata_int(meta, "ImageLines")
    image_samples = _metadata_int(meta, "ImageSamples")

    map_ul = _metadata_latlon(meta, "MapUpperLeft")
    map_ur = _metadata_latlon(meta, "MapUpperRight")
    map_ll = _metadata_latlon(meta, "MapLowerLeft")
    map_lr = _metadata_latlon(meta, "MapLowerRight")

    map_src = np.asarray([
        _local_delta_east_north_m(map_ul, map_ul),
        _local_delta_east_north_m(map_ul, map_ur),
        _local_delta_east_north_m(map_ul, map_ll),
        _local_delta_east_north_m(map_ul, map_lr),
    ], dtype=float)
    map_dst_col_row = np.asarray([
        [0.0, 0.0],
        [image_samples - 1.0, 0.0],
        [0.0, image_lines - 1.0],
        [image_samples - 1.0, image_lines - 1.0],
    ], dtype=float)
    en_to_col_row = _fit_affine_2d(map_src, map_dst_col_row)

    obs_names = ["ObservationUpperLeft", "ObservationUpperRight", "ObservationLowerLeft", "ObservationLowerRight"]
    obs_latlon = {name: _metadata_latlon(meta, name) for name in obs_names}
    obs_en = np.asarray([_local_delta_east_north_m(map_ul, obs_latlon[name]) for name in obs_names], dtype=float)
    obs_col, obs_row = _apply_affine_2d(obs_en[:, 0], obs_en[:, 1], en_to_col_row)

    obs_pixels = {
        name: {"col": float(col), "row": float(row)}
        for name, col, row in zip(obs_names, obs_col, obs_row)
    }
    return {
        "image_lines": image_lines,
        "image_samples": image_samples,
        "observation_corner_pixels": obs_pixels,
        "en_to_col_row_affine": en_to_col_row,
    }

# Create approximate sensor geometry maps from HISUI metadata, which will be used to generate pixel-wise maps of the approximate sensor line and column coordinates for each image pixel based on the estimated observation footprint corners from the HISUI metadata, which can then be used for destriping that accounts for the sensor geometry
def make_approx_sensor_geometry_maps_from_hisui_metadata(
    shape: tuple[int, int],
    metadata_txt: str | Path,
    image_y_values: Optional[np.ndarray] = None,
    image_x_values: Optional[np.ndarray] = None,
) -> dict:
    H, W = shape
    info = estimate_observation_corner_pixels_from_hisui_metadata(metadata_txt)
    image_lines = int(info["image_lines"])
    image_samples = int(info["image_samples"])
    obs = info["observation_corner_pixels"]

    src_col_row = np.asarray([
        [obs["ObservationUpperLeft"]["col"], obs["ObservationUpperLeft"]["row"]],
        [obs["ObservationUpperRight"]["col"], obs["ObservationUpperRight"]["row"]],
        [obs["ObservationLowerLeft"]["col"], obs["ObservationLowerLeft"]["row"]],
        [obs["ObservationLowerRight"]["col"], obs["ObservationLowerRight"]["row"]],
    ], dtype=float)
    dst_sensor_col_line = np.asarray([
        [0.0, 0.0],
        [image_samples - 1.0, 0.0],
        [0.0, image_lines - 1.0],
        [image_samples - 1.0, image_lines - 1.0],
    ], dtype=float)
    image_to_sensor = _fit_affine_2d(src_col_row, dst_sensor_col_line)

    if image_y_values is not None and len(image_y_values) == H:
        rows_1d = np.asarray(image_y_values, dtype=float)
    else:
        rows_1d = np.arange(H, dtype=float)
    if image_x_values is not None and len(image_x_values) == W:
        cols_1d = np.asarray(image_x_values, dtype=float)
    else:
        cols_1d = np.arange(W, dtype=float)

    cols, rows = np.meshgrid(cols_1d, rows_1d)
    sensor_col, sensor_line = _apply_affine_2d(cols, rows, image_to_sensor)
    inside = (
        (sensor_line >= 0)
        & (sensor_line <= image_lines - 1)
        & (sensor_col >= 0)
        & (sensor_col <= image_samples - 1)
    )

    info["image_to_sensor_affine"] = image_to_sensor
    info["sensor_line_range"] = (float(np.nanmin(sensor_line)), float(np.nanmax(sensor_line)))
    info["sensor_col_range"] = (float(np.nanmin(sensor_col)), float(np.nanmax(sensor_col)))
    info["inside_observation_fraction"] = float(np.mean(inside))
    return {
        "sensor_line_map": sensor_line,
        "sensor_col_map": sensor_col,
        "inside_observation_mask": inside,
        "info": info,
    }

# Apply metadata-derived sensor geometry maps to experiments that request them, which will be used to update the destriping parameters in the experiments with the approximate sensor geometry maps derived from the HISUI metadata for those experiments that have requested to use the sensor geometry information for destriping, allowing for more accurate destriping that accounts for the sensor geometry
def apply_metadata_sensor_geometry_to_experiments(
    experiments: dict[str, dict],
    metadata_txt: str | Path,
    shape: tuple[int, int],
    ys: Optional[np.ndarray] = None,
    xs: Optional[np.ndarray] = None,
) -> Optional[dict]:
    metadata_txt = Path(metadata_txt)
    if not metadata_txt.exists():
        print(f"Metadata file not found; sensor geometry maps were not created: {metadata_txt}")
        return None

    geom = make_approx_sensor_geometry_maps_from_hisui_metadata(
        shape=shape,
        metadata_txt=metadata_txt,
        image_y_values=ys,
        image_x_values=xs,
    )
    info = geom["info"]
    print("Approximate sensor geometry from HISUI metadata:")
    print(f"  sensor_line range: {info['sensor_line_range'][0]:.2f} - {info['sensor_line_range'][1]:.2f}")
    print(f"  sensor_col  range: {info['sensor_col_range'][0]:.2f} - {info['sensor_col_range'][1]:.2f}")
    print(f"  inside observation fraction: {info['inside_observation_fraction']:.4f}")
    print("  observation corners in image pixels:")
    for key, value in info["observation_corner_pixels"].items():
        print(f"    {key}: row={value['row']:.2f}, col={value['col']:.2f}")

    for cfg in experiments.values():
        params = cfg.get("destripe_params")
        if params is None:
            continue
        if params.get("geometry_mode") in {"sensor_metadata", "sensor_geometry"}:
            params["sensor_line_map"] = geom["sensor_line_map"]
            params["sensor_col_map"] = geom["sensor_col_map"]
            params["inside_observation_mask"] = geom["inside_observation_mask"]
            params["sensor_geometry_info"] = info
    return geom

# Estimate direction slopes for destriping from HISUI metadata by computing the slopes of the observation footprint edges in the image coordinates, which will be used to derive the approximate slopes of the observation footprint in the image coordinates based on the corners of the observation footprint from the HISUI metadata, which can then be used for destriping that accounts for tilted stripes due to the oblique observation geometry
def estimate_direction_slopes_from_hisui_metadata(path: str | Path) -> dict:

    meta = parse_hisui_metadata_txt(path)

    obs_ul = (
        _metadata_float(meta, "ObservationUpperLeftLatitudeDegree"),
        _metadata_float(meta, "ObservationUpperLeftLongitudeDegree"),
    )
    obs_ur = (
        _metadata_float(meta, "ObservationUpperRightLatitudeDegree"),
        _metadata_float(meta, "ObservationUpperRightLongitudeDegree"),
    )
    obs_ll = (
        _metadata_float(meta, "ObservationLowerLeftLatitudeDegree"),
        _metadata_float(meta, "ObservationLowerLeftLongitudeDegree"),
    )
    obs_lr = (
        _metadata_float(meta, "ObservationLowerRightLatitudeDegree"),
        _metadata_float(meta, "ObservationLowerRightLongitudeDegree"),
    )

    along_left = _image_slope_row_per_col(obs_ul, obs_ll)
    along_right = _image_slope_row_per_col(obs_ur, obs_lr)
    cross_top = _image_slope_row_per_col(obs_ul, obs_ur)
    cross_bottom = _image_slope_row_per_col(obs_ll, obs_lr)

    y_minus_x_slope = float(np.nanmean([along_left, along_right]))
    y_plus_x_slope = float(np.nanmean([abs(cross_top), abs(cross_bottom)]))

    return {
        "direction_slopes": {
            "y_minus_x": y_minus_x_slope,
            "y_plus_x": y_plus_x_slope,
        },
        "raw_slopes": {
            "observation_ul_to_ll": along_left,
            "observation_ur_to_lr": along_right,
            "observation_ul_to_ur": cross_top,
            "observation_ll_to_lr": cross_bottom,
        },
    }

# Apply metadata-derived direction slopes to experiments that request them, which will be used to update the destriping parameters in the experiments with the direction slopes derived from the HISUI metadata for those experiments that have requested to use the metadata-derived direction slopes for destriping, allowing for destriping that accounts for tilted stripes based on the observation geometry
def apply_metadata_direction_slopes_to_experiments(
    experiments: dict[str, dict],
    metadata_txt: str | Path,
) -> Optional[dict]:
    metadata_txt = Path(metadata_txt)
    if not metadata_txt.exists():
        print(f"Metadata file not found; using configured direction_slopes: {metadata_txt}")
        return None

    info = estimate_direction_slopes_from_hisui_metadata(metadata_txt)
    slopes = info["direction_slopes"]
    print("Metadata-derived direction slopes:")
    print(f"  y_minus_x: {slopes['y_minus_x']:.8f}")
    print(f"  y_plus_x : {slopes['y_plus_x']:.8f}")
    print("Raw observation-edge slopes:")
    for key, value in info["raw_slopes"].items():
        print(f"  {key}: {value:.8f}")

    for cfg in experiments.values():
        params = cfg.get("destripe_params")
        if params is None:
            continue
        if params.get("use_metadata_direction_slopes", True):
            params["direction_slopes"] = dict(slopes)
            params["metadata_direction_slopes"] = dict(slopes)
    return info


# 8. Main workflow

def main() -> dict[str, dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    roi_df, wavelengths, spectra = load_roi_spectra_csv(ROI_CSV)
    cube, ys, xs = spectra_to_cube(roi_df, spectra)
    print(f"ROI table shape: {roi_df.shape}")
    print(f"Cube shape: {cube.shape}")
    print(f"Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} nm")

    valid_mask = make_valid_pixel_mask(
        cube,
        nodata_values=NODATA_VALUES,
        require_positive=REQUIRE_POSITIVE,
        min_valid_fraction=MIN_VALID_FRACTION,
    )
    print(f"Valid pixels: {int(np.sum(valid_mask))} / {valid_mask.size}")

    cube_sel, wavelengths_sel, _ = select_bands(cube, wavelengths, WL_MIN, WL_MAX)
    print(f"Selected cube shape: {cube_sel.shape}")
    print(f"Selected wavelength range: {wavelengths_sel[0]:.2f} - {wavelengths_sel[-1]:.2f} nm")

    mod_wave, alpha_grid, mod_spectra = load_ch4_modtran_csv(MODTRAN_CSV)
    mod_resampled = gaussian_srf_resample(
        mod_wave=mod_wave,
        mod_spectra=mod_spectra,
        sensor_wave=wavelengths_sel,
        fwhm_nm=FWHM_NM,
    )
    uas, _ = compute_uas_log_slope(
        alpha_grid=alpha_grid,
        spectra_grid=mod_resampled,
        alpha_min=UAS_ALPHA_MIN,
        alpha_max=UAS_ALPHA_MAX,
    )

    if USE_METADATA_DIRECTION_SLOPES:
        apply_metadata_direction_slopes_to_experiments(EXPERIMENTS, METADATA_TXT)
    if USE_METADATA_SENSOR_GEOMETRY:
        ys_for_sensor_geometry = np.asarray(ys, dtype=float) + float(IMAGE_Y_OFFSET_FOR_SENSOR_GEOMETRY)
        xs_for_sensor_geometry = np.asarray(xs, dtype=float) + float(IMAGE_X_OFFSET_FOR_SENSOR_GEOMETRY)
        apply_metadata_sensor_geometry_to_experiments(
            EXPERIMENTS,
            METADATA_TXT,
            shape=cube_sel.shape[:2],
            ys=ys_for_sensor_geometry,
            xs=xs_for_sensor_geometry,
        )

    if RUN_EXPERIMENT_NAMES is None:
        experiment_items = list(EXPERIMENTS.items())
    else:
        experiment_items = [(name, EXPERIMENTS[name]) for name in RUN_EXPERIMENT_NAMES]

    results: dict[str, dict] = {}
    for name, cfg in experiment_items:
        print("\n" + "=" * 80)
        print(f"Running experiment: {name}")
        print("=" * 80)
        res = run_iterative_mf_with_optional_destriping(
            cube=cube_sel,
            uas=uas,
            valid_mask=valid_mask,
            n_iter=N_ITER,
            nsigma=NSIGMA,
            reg=REG,
            rcond=RCOND,
            destripe_when=cfg["destripe_when"],
            destripe_params=cfg["destripe_params"],
            verbose=True,
        )
        results[name] = res

    summary_df = summarize_results_table(results)
    summary_df.to_csv(OUTPUT_DIR / "summary_df.csv", index=False)
    print("\nSummary")
    print(summary_df.to_string(index=False))
    print(f"Saved: {OUTPUT_DIR / 'summary_df.csv'}")

    for case_name, result in results.items():
        save_case_outputs(result, case_name, OUTPUT_DIR, ys=ys, xs=xs)
        print(f"Saved outputs: {case_name}")

    if RUN_PLOTS:
        plot_experiment_grid(results, names=list(results.keys()))
        for name, result in results.items():
            if name != "baseline_no_destripe":
                plot_single_result(result, title_prefix=name)

    return results


if __name__ == "__main__":
    main()
