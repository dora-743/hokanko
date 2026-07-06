# ============================================
# 0. Imports and user settings
# ============================================
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Input files
# ----------------------------
# ROI CSV: columns = y, x, wave_XXXXnm, wave_XXXXnm, ...
ROI_CSV = Path(r"D:\research\code\all_roi_spectra200x200.csv")

# MODTRAN CSV: columns = wavelength, 0.0, 0.5, 1.0, ...
MODTRAN_CSV = Path(r"E:/refit/CH4c.csv")

# Slopes detected from the general-scene slope detector.
DETECTED_SLOPE_DIRECTIONS_CSV = Path(
    r"D:\research\code\outputs_general_scene_slope_detection\detected_slope_directions.csv"
)
if not DETECTED_SLOPE_DIRECTIONS_CSV.exists():
    DETECTED_SLOPE_DIRECTIONS_CSV = Path(
        r"D:\research\code\outputs_general_scene_slope_detection\detected_six_slope_directions.csv"
    )

OUTPUT_DIR = Path(r"D:\research\code\outputs_detected_slopes_orthogonal_thin_eachiter_then_broad_median")

# ----------------------------
# Wavelength / UAS settings
# ----------------------------
WL_MIN = 2100.0
WL_MAX = 2450.0
FWHM_NM = 12.5

# Alpha range used to build the UAS template.
UAS_ALPHA_MIN = 0.0
UAS_ALPHA_MAX = 0.5

# ----------------------------
# Iterative MF settings
# ----------------------------
N_ITER = 5
NSIGMA = 3.0
REG = 1e-6
RCOND = 1e-8

# ----------------------------
# Valid pixel settings
# ----------------------------
NODATA_VALUES = [0, -9999]
REQUIRE_POSITIVE = True
MIN_VALID_FRACTION = 1.0

# ============================================================
# Detected-slope direction loading
# ============================================================
USE_DETECTED_DIRECTION_TYPES = ["primary_positive", "orthogonal_to_primary"]

DETECTED_THIN_LINE_BIN_WIDTH = 2.0
DETECTED_THIN_MIN_PIXELS_PER_LINE = 5
DETECTED_THIN_EXCLUDE_MODE = "none"  # thin high-alpha stripe itself should remain in offset estimation

DETECTED_BROAD_LINE_BIN_WIDTH = 18.0
DETECTED_BROAD_MIN_PIXELS_PER_LINE = 80
DETECTED_BROAD_EXCLUDE_MODE = "robust_high"

DETECTED_DIRECTION_COLUMNS_REQUIRED = [
    "detection_type",
    "direction_type",
    "direction_key_for_existing_code",
    "slope_parameter_for_existing_code",
]


def load_detected_direction_records(
    csv_path: Path,
    detection_types: Sequence[str],
    line_bin_width: float,
    min_pixels_per_line: int,
    exclude_mode: str,
    cleanup_stage: str,
) -> list[dict]:
    """Load slope directions produced by detect_general_scene_stripe_slopes.py."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    missing = [c for c in DETECTED_DIRECTION_COLUMNS_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Detected slope CSV is missing columns: {missing}")

    keep = df["detection_type"].isin(list(detection_types))
    keep &= df["direction_type"].isin(USE_DETECTED_DIRECTION_TYPES)
    df = df[keep].copy()
    if len(df) == 0:
        raise ValueError(f"No detected directions found for detection_types={detection_types}")

    direction_type_order = {"primary_positive": 0, "orthogonal_to_primary": 1}
    df["_direction_type_order"] = df["direction_type"].map(direction_type_order).fillna(99).astype(int)
    sort_cols = [c for c in ["family_rank", "_direction_type_order"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    records = []
    for _, row in df.iterrows():
        direction_key = str(row["direction_key_for_existing_code"])
        if direction_key not in {"y_minus_x", "y_plus_x"}:
            raise ValueError(f"Unsupported direction_key_for_existing_code: {direction_key}")

        slope_parameter = float(row["slope_parameter_for_existing_code"])
        signed_slope = float(row["signed_slope"]) if "signed_slope" in df.columns else np.nan
        family_rank = int(row["family_rank"]) if "family_rank" in df.columns and pd.notna(row["family_rank"]) else -1
        record = {
            "family_rank": family_rank,
            "detection_type": str(row["detection_type"]),
            "direction_type": str(row["direction_type"]),
            "direction_key": direction_key,
            "slope_parameter": slope_parameter,
            "signed_slope": signed_slope,
            "line_bin_width": float(line_bin_width),
            "min_pixels_per_line": int(min_pixels_per_line),
            "exclude_mode": exclude_mode,
            "cleanup_stage": cleanup_stage,
        }
        for optional_col in [
            "parent_positive_slope",
            "parent_positive_angle_deg",
            "parent_score",
            "parent_peak_prominence",
            "angle_deg_0_180",
            "equation_form",
        ]:
            if optional_col in df.columns:
                record[optional_col] = row[optional_col]
        records.append(record)
    return records


DETECTED_THIN_DIRECTION_RECORDS = load_detected_direction_records(
    DETECTED_SLOPE_DIRECTIONS_CSV,
    detection_types=["thin_high_alpha"],
    line_bin_width=DETECTED_THIN_LINE_BIN_WIDTH,
    min_pixels_per_line=DETECTED_THIN_MIN_PIXELS_PER_LINE,
    exclude_mode=DETECTED_THIN_EXCLUDE_MODE,
    cleanup_stage="thin_detected",
)

DETECTED_BROAD_DIRECTION_RECORDS = load_detected_direction_records(
    DETECTED_SLOPE_DIRECTIONS_CSV,
    detection_types=["broad_offset"],
    line_bin_width=DETECTED_BROAD_LINE_BIN_WIDTH,
    min_pixels_per_line=DETECTED_BROAD_MIN_PIXELS_PER_LINE,
    exclude_mode=DETECTED_BROAD_EXCLUDE_MODE,
    cleanup_stage="broad_median",
)

print("Detected thin directions:")
display(pd.DataFrame(DETECTED_THIN_DIRECTION_RECORDS))
print("Detected broad directions:")
display(pd.DataFrame(DETECTED_BROAD_DIRECTION_RECORDS))

# ============================================================
# Thin-line destriping settings
# ============================================================
# Thin-line removal now uses the detected thin high-alpha slope and its orthogonal direction.
# The statistic method is changed by experiment.
DEFAULT_THIN_DESTRIPE_PARAMS = {
    "directions": DETECTED_THIN_DIRECTION_RECORDS,
    "method": "median",
    "min_pixels_per_line": DETECTED_THIN_MIN_PIXELS_PER_LINE,
    "preserve_global_stat": True,
    "smooth_half_window": 0,
    "exclude_mode": DETECTED_THIN_EXCLUDE_MODE,
    "exclude_nsigma": NSIGMA,
    "recompute_exclude_each_direction": True,
    "fallback_to_valid": True,
    "trim_fraction": 0.1,
    "mode_bins": 64,
    "sigma_clip_nsigma": 3.0,
    "sigma_clip_max_iter": 3,
    "threshold_source": "corrected",
}

# These statistics are used only for the thin-line stage.
STRIPE_STAT_METHODS = [
    ("median", {}),
    ("mean", {}),
    ("trimmed_mean", {"trim_fraction": 0.1}),
    ("mode", {"mode_bins": 64}),
    ("sigma_clipped_mean", {"sigma_clip_nsigma": 3.0, "sigma_clip_max_iter": 3}),
]

# ============================================================
# Broad sensor-noise cleanup settings
# ============================================================
# Broad cleanup is applied after detected thin-line cleanup in every iteration.
# It uses the detected broad offset slope and its orthogonal direction, median only.
BROAD_METHOD = "median"          # fixed by design
BROAD_LINE_BIN_WIDTH = DETECTED_BROAD_LINE_BIN_WIDTH
BROAD_MIN_PIXELS_PER_LINE = DETECTED_BROAD_MIN_PIXELS_PER_LINE
BROAD_EXCLUDE_MODE = DETECTED_BROAD_EXCLUDE_MODE
BROAD_EXCLUDE_NSIGMA = NSIGMA
BROAD_PRESERVE_GLOBAL_STAT = True
BROAD_FALLBACK_TO_VALID = True
BROAD_SMOOTH_HALF_WINDOW = 0

BROAD_CLEANUP_PARAMS = {
    "directions": DETECTED_BROAD_DIRECTION_RECORDS,
    "method": BROAD_METHOD,
    "line_bin_width": BROAD_LINE_BIN_WIDTH,
    "min_pixels_per_line": BROAD_MIN_PIXELS_PER_LINE,
    "exclude_mode": BROAD_EXCLUDE_MODE,
    "exclude_nsigma": BROAD_EXCLUDE_NSIGMA,
    "preserve_global_stat": BROAD_PRESERVE_GLOBAL_STAT,
    "fallback_to_valid": BROAD_FALLBACK_TO_VALID,
    "smooth_half_window": BROAD_SMOOTH_HALF_WINDOW,
}

# ============================================================
# Experiment settings
# ============================================================
# All destriping experiments are each_iter only:
#   each iteration: detected thin cleanup with selected statistic -> detected broad median cleanup -> thresholding
EXPERIMENTS = {
    "baseline_no_destripe": {
        "destripe_when": "none",
        "destripe_params": None,
    },
}

for method, overrides in STRIPE_STAT_METHODS:
    EXPERIMENTS[f"{method}_thin_eachiter_then_broad_median"] = {
        "destripe_when": "each_iter",
        "destripe_params": {
            "thin_cleanup_params": {
                **DEFAULT_THIN_DESTRIPE_PARAMS,
                "method": method,
                **overrides,
            },
            "broad_cleanup_params": BROAD_CLEANUP_PARAMS,
            "threshold_source": "corrected",
        },
    }



# %% cell 2

# ============================================
# 1. Data loading helpers
# ============================================

def get_wave_columns(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Return wavelength columns sorted by wavelength."""
    wave_cols: list[str] = []
    wavelengths: list[float] = []

    pattern = re.compile(r"^wave_([0-9]+(?:\.[0-9]+)?)nm$")
    for col in df.columns:
        match = pattern.match(col)
        if match is not None:
            wave_cols.append(col)
            wavelengths.append(float(match.group(1)))

    if len(wave_cols) == 0:
        raise ValueError("No wavelength columns like wave_2300nm were found.")

    wavelengths_arr = np.asarray(wavelengths, dtype=float)
    order = np.argsort(wavelengths_arr)
    wavelengths_arr = wavelengths_arr[order]
    wave_cols_sorted = [wave_cols[i] for i in order]
    return wave_cols_sorted, wavelengths_arr


def load_roi_spectra_csv(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load ROI spectra CSV with y, x, and wave_XXXXnm columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ROI CSV not found: {path}")

    df = pd.read_csv(path)
    if "y" not in df.columns or "x" not in df.columns:
        raise ValueError("ROI CSV must contain columns 'y' and 'x'.")

    wave_cols, wavelengths = get_wave_columns(df)
    spectra = df[wave_cols].to_numpy(dtype=float)
    return df, wavelengths, spectra


def spectra_to_cube(
    df: pd.DataFrame,
    spectra: np.ndarray,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert table spectra to image cube [H, W, B]."""
    ys = np.sort(df["y"].unique())
    xs = np.sort(df["x"].unique())

    y_to_row = {y: i for i, y in enumerate(ys)}
    x_to_col = {x: j for j, x in enumerate(xs)}

    H, W = len(ys), len(xs)
    B = spectra.shape[1]
    cube = np.full((H, W, B), fill_value, dtype=float)

    for row_idx, row in df.iterrows():
        r = y_to_row[row["y"]]
        c = x_to_col[row["x"]]
        cube[r, c, :] = spectra[row_idx, :]

    return cube, ys, xs


def band_mask(
    wavelengths: np.ndarray,
    wl_min: Optional[float] = None,
    wl_max: Optional[float] = None,
    exclude_ranges: Optional[Sequence[tuple[float, float]]] = None,
) -> np.ndarray:
    mask = np.ones_like(wavelengths, dtype=bool)

    if wl_min is not None:
        mask &= wavelengths >= wl_min
    if wl_max is not None:
        mask &= wavelengths <= wl_max

    if exclude_ranges is not None:
        for a, b in exclude_ranges:
            mask &= ~((wavelengths >= a) & (wavelengths <= b))

    return mask


def select_bands(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    wl_min: float,
    wl_max: float,
    exclude_ranges: Optional[Sequence[tuple[float, float]]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = band_mask(wavelengths, wl_min=wl_min, wl_max=wl_max, exclude_ranges=exclude_ranges)
    return cube[:, :, mask], wavelengths[mask], mask



# %% cell 3

# ============================================
# 2. Plotting helpers
# ============================================

def finite_values(img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    arr = np.asarray(img, dtype=float)
    if mask is None:
        vals = arr[np.isfinite(arr)]
    else:
        vals = arr[np.isfinite(arr) & np.asarray(mask, dtype=bool)]
    return vals


def robust_limits(
    images: Union[np.ndarray, Sequence[np.ndarray]],
    mask: Optional[np.ndarray] = None,
    q_low: float = 2,
    q_high: float = 98,
) -> tuple[float, float]:
    if isinstance(images, np.ndarray):
        images = [images]

    vals_list = []
    for img in images:
        vals = finite_values(img, mask=mask)
        if vals.size > 0:
            vals_list.append(vals)

    if len(vals_list) == 0:
        return -1.0, 1.0

    vals_all = np.concatenate(vals_list)
    lo, hi = np.nanpercentile(vals_all, [q_low, q_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(vals_all)), float(np.nanmax(vals_all))
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def robust_scale_image(img: np.ndarray, pmin: float = 2, pmax: float = 98) -> np.ndarray:
    vals = finite_values(img)
    if vals.size == 0:
        return np.zeros_like(img, dtype=float)
    lo, hi = np.nanpercentile(vals, [pmin, pmax])
    if hi <= lo:
        return np.zeros_like(img, dtype=float)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def make_rgb_from_cube(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    r_wl: float = 650,
    g_wl: float = 550,
    b_wl: float = 460,
) -> np.ndarray:
    idx_r = int(np.argmin(np.abs(wavelengths - r_wl)))
    idx_g = int(np.argmin(np.abs(wavelengths - g_wl)))
    idx_b = int(np.argmin(np.abs(wavelengths - b_wl)))

    r = robust_scale_image(cube[:, :, idx_r])
    g = robust_scale_image(cube[:, :, idx_g])
    b = robust_scale_image(cube[:, :, idx_b])
    return np.dstack([r, g, b])


def plot_map(
    img: np.ndarray,
    title: str = "Map",
    mask: Optional[np.ndarray] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    colorbar_label: Optional[str] = None,
    figsize: tuple[float, float] = (5, 5),
):
    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = robust_limits(img, mask=mask)
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    plt.figure(figsize=figsize)
    im = plt.imshow(img, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(im, label=colorbar_label)
    plt.show()


def plot_mean_spectrum(cube: np.ndarray, wavelengths: np.ndarray, mask: Optional[np.ndarray] = None, xlim=None):
    if mask is None:
        X = cube.reshape(-1, cube.shape[2])
        spec = np.nanmean(X, axis=0)
    else:
        spec = np.nanmean(cube[mask], axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, spec, marker="o", ms=3)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Radiance / Reflectance")
    plt.title("Mean spectrum")
    if xlim is not None:
        plt.xlim(*xlim)
    plt.grid(True)
    plt.show()


def plot_uas(wavelengths: np.ndarray, uas: np.ndarray, title: str = "UAS", xlim=None):
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, uas, marker="o", ms=3)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("UAS")
    plt.title(title)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.grid(True)
    plt.show()



# %% cell 4

# ============================================
# 3. MODTRAN / UAS helpers
# ============================================

def load_ch4_modtran_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load MODTRAN spectra CSV.

    Expected format:
        wavelength, 0.0, 0.5, 1.0, ...

    Returns:
        mod_wave: (n_mod_wave,)
        alpha_grid: (n_alpha,)
        spectra_grid: (n_alpha, n_mod_wave)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MODTRAN CSV not found: {path}")

    df_mod = pd.read_csv(path)
    if "wavelength" not in df_mod.columns:
        raise ValueError("MODTRAN CSV must contain a 'wavelength' column.")

    mod_wave = df_mod["wavelength"].to_numpy(dtype=float)
    alpha_cols = [c for c in df_mod.columns if c != "wavelength"]

    try:
        alpha_grid = np.array([float(c) for c in alpha_cols], dtype=float)
    except ValueError as exc:
        raise ValueError("MODTRAN columns other than 'wavelength' must be numeric alpha values.") from exc

    order = np.argsort(alpha_grid)
    alpha_grid = alpha_grid[order]
    alpha_cols = [alpha_cols[i] for i in order]

    spectra_grid = df_mod[alpha_cols].to_numpy(dtype=float).T
    return mod_wave, alpha_grid, spectra_grid


def gaussian_srf_resample(
    mod_wave: np.ndarray,
    mod_spectra: np.ndarray,
    sensor_wave: np.ndarray,
    fwhm_nm: Union[float, np.ndarray],
) -> np.ndarray:
    """Resample high-resolution MODTRAN spectra to sensor wavelengths by Gaussian SRF."""
    mod_wave = np.asarray(mod_wave, dtype=float)
    mod_spectra = np.asarray(mod_spectra, dtype=float)
    sensor_wave = np.asarray(sensor_wave, dtype=float)

    if np.isscalar(fwhm_nm):
        fwhm_arr = np.full(sensor_wave.shape, float(fwhm_nm), dtype=float)
    else:
        fwhm_arr = np.asarray(fwhm_nm, dtype=float)
        if fwhm_arr.shape != sensor_wave.shape:
            raise ValueError("fwhm_nm must be scalar or same shape as sensor_wave.")

    out = np.full((mod_spectra.shape[0], sensor_wave.size), np.nan, dtype=float)

    for j, center in enumerate(sensor_wave):
        sigma = fwhm_arr[j] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        use = np.abs(mod_wave - center) <= 4.0 * sigma

        if np.sum(use) < 2:
            # fallback: simple interpolation for each alpha spectrum
            for i in range(mod_spectra.shape[0]):
                out[i, j] = np.interp(center, mod_wave, mod_spectra[i])
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
    """Compute UAS by fitting log(spectrum) = intercept - UAS * alpha."""
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    spectra_grid = np.asarray(spectra_grid, dtype=float)

    use = np.ones_like(alpha_grid, dtype=bool)
    if alpha_min is not None:
        use &= alpha_grid >= alpha_min
    if alpha_max is not None:
        use &= alpha_grid <= alpha_max

    a = alpha_grid[use]
    if a.size < 2:
        raise ValueError("Need at least two alpha values to compute UAS.")

    Y = np.log(np.maximum(spectra_grid[use], 1e-30))
    A = np.vstack([np.ones_like(a), a]).T
    coeff, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)

    intercept = coeff[0]
    slope = coeff[1]
    uas = -slope
    return uas, intercept



# %% cell 5

# ============================================
# 4. Matched Filter helpers
# ============================================

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
    """Estimate background mean and covariance from selected pixels."""
    H, W, B = cube.shape

    if min_pixels is None:
        min_pixels = max(B + 5, 30)

    X = cube[np.asarray(background_mask, dtype=bool)]
    X = X[np.all(np.isfinite(X), axis=1)]

    if X.shape[0] < min_pixels:
        raise ValueError(f"Too few background pixels: {X.shape[0]} < {min_pixels}")

    mu = np.mean(X, axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)

    # A small diagonal regularization. Scale-aware enough for most reflectance/radiance cases.
    scale = np.nanmean(np.diag(cov))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    cov = cov + reg * scale * np.eye(B)

    return mu, cov, X


def make_methane_target(mu: np.ndarray, uas: np.ndarray, positive_alpha: bool = True) -> np.ndarray:
    """MF target spectrum. positive_alpha=True follows the convention used in the original notebook."""
    mu = np.asarray(mu, dtype=float).reshape(-1)
    uas = np.asarray(uas, dtype=float).reshape(-1)
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
    """Compute MF alpha map using current background pixels."""
    H, W, B = cube.shape
    uas = np.asarray(uas, dtype=float).reshape(-1)
    if uas.size != B:
        raise ValueError(f"uas length {uas.size} does not match cube bands {B}.")

    valid_mask = np.asarray(valid_mask, dtype=bool)
    background_mask = valid_mask & np.asarray(background_mask, dtype=bool)

    mu, cov, _ = estimate_background_mean_cov_from_cube(
        cube=cube,
        background_mask=background_mask,
        reg=reg,
        min_pixels=min_background_pixels,
    )

    target = make_methane_target(mu, uas, positive_alpha=True)
    inv_cov = np.linalg.pinv(cov, rcond=rcond)

    denom = float(target.T @ inv_cov @ target)
    if abs(denom) < 1e-15:
        raise ValueError("MF denominator is too small. Check UAS, covariance, and wavelength selection.")

    alpha_map = np.full((H, W), np.nan, dtype=float)
    X = cube[valid_mask]
    diff = X - mu
    alpha_values = (diff @ inv_cov @ target) / denom
    alpha_map[valid_mask] = alpha_values

    return alpha_map, mu, cov, target


def robust_threshold_from_alpha(alpha_values: np.ndarray, nsigma: float = 4.0) -> tuple[float, float, float]:
    """Median + nsigma * 1.4826*MAD threshold."""
    values = np.asarray(alpha_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite alpha values for thresholding.")

    med = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - med)))
    robust_std = 1.4826 * mad + 1e-12
    threshold = med + nsigma * robust_std
    return float(threshold), med, float(robust_std)


def plume_mask_from_alpha(
    alpha_map: np.ndarray,
    valid_mask: np.ndarray,
    nsigma: float = 4.0,
) -> tuple[np.ndarray, dict]:
    threshold, med, robust_std = robust_threshold_from_alpha(alpha_map[valid_mask], nsigma=nsigma)
    plume_mask = np.zeros_like(valid_mask, dtype=bool)
    plume_mask[valid_mask] = alpha_map[valid_mask] > threshold

    meta = {
        "threshold": threshold,
        "median": med,
        "robust_std": robust_std,
        "n_plume": int(np.sum(plume_mask)),
    }
    return plume_mask, meta



# %% cell 6
# ============================================
# 5. Improved diagonal destriping helpers
# ============================================

def line_id_map(shape: tuple[int, int], direction: str = "y_minus_x") -> np.ndarray:
    """Return line IDs for directional line grouping.

    Image coordinate convention:
        row = y, positive downward
        col = x, positive rightward

    direction="y_minus_x": lines parallel to y=x,  row - col = const
    direction="y_plus_x" : lines parallel to y=-x, row + col = const
    """
    rows, cols = np.indices(shape)

    if direction == "y_minus_x":
        return rows - cols
    if direction == "y_plus_x":
        return rows + cols

    raise ValueError("direction must be 'y_minus_x' or 'y_plus_x'.")


def normalize_directions(destripe_params: Optional[dict]) -> list[str]:
    """Accept either 'directions' or old-style 'direction'."""
    if destripe_params is None:
        return []

    if "directions" in destripe_params:
        dirs = destripe_params["directions"]
    else:
        dirs = destripe_params.get("direction", "y_minus_x")

    if isinstance(dirs, str):
        dirs = [dirs]

    dirs = list(dirs)
    for d in dirs:
        if d not in {"y_minus_x", "y_plus_x"}:
            raise ValueError("directions must contain only 'y_minus_x' and/or 'y_plus_x'.")
    return dirs


def moving_nanmedian_1d(values: np.ndarray, half_window: int = 0) -> np.ndarray:
    """Median-smooth a 1D array while ignoring NaN."""
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


def statistic_1d(
    values: np.ndarray,
    method: str = "median",
    trim_fraction: float = 0.1,
    mode_bins: int = 64,
    sigma_clip_nsigma: float = 3.0,
    sigma_clip_max_iter: int = 3,
) -> float:
    """Compute 1D statistic for stripe offset estimation."""
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
        return float(np.nanmean(v[k:v.size-k]))

    if method == "mode":
        # Continuous values do not have an exact mode, so use the center of the most populated histogram bin.
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

    raise ValueError("method must be 'median', 'mean', 'trimmed_mean', 'mode', or 'sigma_clipped_mean'.")



def make_exclude_mask_for_destriping(
    alpha_map: np.ndarray,
    valid_mask: np.ndarray,
    plume_mask: Optional[np.ndarray] = None,
    exclude_mode: str = "robust_high",
    exclude_nsigma: float = 4.0,
) -> tuple[np.ndarray, dict]:
    """Build mask of pixels excluded from stripe estimation."""
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
        raise ValueError(
            "exclude_mode must be 'none', 'robust_high', 'previous_plume', or 'previous_plume_or_high'."
        )

    exclude &= valid_mask
    meta["n_excluded"] = int(np.sum(exclude))
    return exclude, meta


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
) -> dict:
    """Estimate directional stripe offsets and subtract them from alpha_map.

    If preserve_global_stat=True:
        corrected = alpha - (line_stat - global_stat)

    If preserve_global_stat=False:
        corrected = alpha - line_stat
    """
    alpha = np.asarray(alpha_map, dtype=float)
    if alpha.ndim != 2:
        raise ValueError("alpha_map must be 2D.")

    finite = np.isfinite(alpha)
    if valid_mask is None:
        valid = finite.copy()
    else:
        valid = np.asarray(valid_mask, dtype=bool) & finite

    if not np.any(valid):
        raise ValueError("No valid finite pixels for destriping.")

    if exclude_mask is None:
        estimate_mask = valid.copy()
    else:
        estimate_mask = valid & (~np.asarray(exclude_mask, dtype=bool))

    if not np.any(estimate_mask):
        if fallback_to_valid:
            estimate_mask = valid.copy()
        else:
            raise ValueError("No pixels remain after exclusion for stripe estimation.")

    ids = line_id_map(alpha.shape, direction=direction)
    id_values = np.arange(int(np.nanmin(ids)), int(np.nanmax(ids)) + 1)

    global_stat = statistic_1d(
        alpha[estimate_mask],
        method=method,
        trim_fraction=trim_fraction,
        mode_bins=mode_bins,
        sigma_clip_nsigma=sigma_clip_nsigma,
        sigma_clip_max_iter=sigma_clip_max_iter,
    )

    raw_stats = np.full(id_values.shape, np.nan, dtype=float)
    counts_used = np.zeros(id_values.shape, dtype=int)
    used_fallback = np.zeros(id_values.shape, dtype=bool)

    for k, line_id in enumerate(id_values):
        m = (ids == line_id) & estimate_mask
        count = int(np.sum(m))

        if count < min_pixels_per_line and fallback_to_valid:
            mf = (ids == line_id) & valid
            count_f = int(np.sum(mf))
            if count_f >= min_pixels_per_line:
                m = mf
                count = count_f
                used_fallback[k] = True

        counts_used[k] = count
        if count >= min_pixels_per_line:
            raw_stats[k] = statistic_1d(
                alpha[m],
                method=method,
                trim_fraction=trim_fraction,
                mode_bins=mode_bins,
                sigma_clip_nsigma=sigma_clip_nsigma,
                sigma_clip_max_iter=sigma_clip_max_iter,
            )

    smoothed_stats = moving_nanmedian_1d(raw_stats, half_window=smooth_half_window)

    if preserve_global_stat and np.isfinite(global_stat):
        offsets = smoothed_stats - global_stat
    else:
        offsets = smoothed_stats.copy()

    # Too-short or all-NaN lines are left uncorrected.
    offsets_filled = np.where(np.isfinite(offsets), offsets, 0.0)

    stripe_map = np.zeros_like(alpha, dtype=float)
    for line_id, offset in zip(id_values, offsets_filled):
        stripe_map[ids == line_id] = float(offset)

    corrected = alpha - stripe_map
    corrected[~finite] = np.nan

    line_table = pd.DataFrame({
        "line_id": id_values,
        "n_pixels_used": counts_used,
        "used_fallback_to_valid": used_fallback,
        "line_stat_raw": raw_stats,
        "line_stat_after_smoothing": smoothed_stats,
        "stripe_offset_subtracted": offsets_filled,
        "global_stat": global_stat,
        "direction": direction,
        "method": method,
    })

    return {
        "corrected": corrected,
        "stripe_map": stripe_map,
        "line_table": line_table,
        "global_stat": global_stat,
        "estimate_mask": estimate_mask,
    }


def destripe_by_sequential_directions(
    alpha_map: np.ndarray,
    valid_mask: np.ndarray,
    plume_mask: Optional[np.ndarray] = None,
    destripe_params: Optional[dict] = None,
    nsigma: float = 4.0,
) -> dict:
    """Apply directional destriping sequentially.

    Example:
        directions=["y_minus_x", "y_plus_x"]

    This means:
        1. remove stripes parallel to y=x
        2. from the corrected result, remove stripes parallel to y=-x
    """
    if destripe_params is None:
        destripe_params = {}

    directions = normalize_directions(destripe_params)
    if len(directions) == 0:
        zero = np.zeros_like(alpha_map, dtype=float)
        return {
            "corrected": np.asarray(alpha_map, dtype=float).copy(),
            "stripe_map": zero,
            "directional_stripe_maps": {},
            "line_table": pd.DataFrame(),
            "exclude_meta": [],
        }

    current = np.asarray(alpha_map, dtype=float).copy()
    total_stripe = np.zeros_like(current, dtype=float)
    directional_stripe_maps: dict[str, np.ndarray] = {}
    line_tables = []
    exclude_metas = []

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

        out = destripe_by_directional_lines(
            alpha_map=current,
            valid_mask=valid_mask,
            exclude_mask=exclude_mask,
            direction=direction,
            method=destripe_params.get("method", "median"),
            min_pixels_per_line=destripe_params.get("min_pixels_per_line", 5),
            preserve_global_stat=destripe_params.get("preserve_global_stat", True),
            smooth_half_window=destripe_params.get("smooth_half_window", 0),
            fallback_to_valid=destripe_params.get("fallback_to_valid", True),
            trim_fraction=destripe_params.get("trim_fraction", 0.1),
            mode_bins=destripe_params.get("mode_bins", 64),
            sigma_clip_nsigma=destripe_params.get("sigma_clip_nsigma", 3.0),
            sigma_clip_max_iter=destripe_params.get("sigma_clip_max_iter", 3),
        )

        current = out["corrected"]
        total_stripe = total_stripe + out["stripe_map"]
        directional_stripe_maps[direction] = out["stripe_map"].copy()

        table = out["line_table"].copy()
        table.insert(0, "pass_index", pass_index)
        line_tables.append(table)

        exclude_meta = dict(exclude_meta)
        exclude_meta.update({
            "pass_index": pass_index,
            "direction": direction,
        })
        exclude_metas.append(exclude_meta)

    line_table_all = pd.concat(line_tables, ignore_index=True) if len(line_tables) > 0 else pd.DataFrame()

    return {
        "corrected": current,
        "stripe_map": total_stripe,
        "directional_stripe_maps": directional_stripe_maps,
        "line_table": line_table_all,
        "exclude_meta": exclude_metas,
    }




# ============================================================
# 5b. Detected fixed-direction cleanup helpers
# ============================================================

def fixed_slope_line_id_map(
    shape: tuple[int, int],
    slope: float,
    line_bin_width: float = 18.0,
    direction_key: str = "y_minus_x",
) -> np.ndarray:
    """Return line IDs for fixed-slope cleanup.

    direction_key="y_minus_x": row - slope * col = const, i.e. row = +slope * col + b
    direction_key="y_plus_x" : row + slope * col = const, i.e. row = -slope * col + b
    """
    rows, cols = np.indices(shape)
    if direction_key == "y_minus_x":
        continuous_id = (rows - float(slope) * cols) / float(line_bin_width)
    elif direction_key == "y_plus_x":
        continuous_id = (rows + float(slope) * cols) / float(line_bin_width)
    else:
        raise ValueError("direction_key must be 'y_minus_x' or 'y_plus_x'.")
    return np.rint(continuous_id).astype(np.int32)


def _first_present(mapping: dict, keys: Sequence[str], default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            value = mapping[key]
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            return value
    return default


def normalize_fixed_direction_records(cleanup_params: Optional[dict]) -> list[dict]:
    """Normalize detected direction records and legacy slope lists into one schema."""
    if cleanup_params is None:
        cleanup_params = {}

    raw_records = cleanup_params.get("directions", None)
    if raw_records is None:
        raw_records = [
            {
                "detection_type": "legacy_slope",
                "direction_type": "primary_positive",
                "direction_key": "y_minus_x",
                "slope_parameter": float(s),
                "signed_slope": float(s),
            }
            for s in cleanup_params.get("slopes", [])
        ]

    if isinstance(raw_records, dict):
        raw_records = [raw_records]
    if isinstance(raw_records, str):
        raw_records = [{"direction_key": raw_records, "slope_parameter": 1.0}]

    records = []
    for i, rec in enumerate(list(raw_records), start=1):
        if isinstance(rec, str):
            rec = {"direction_key": rec, "slope_parameter": 1.0}
        rec = dict(rec)
        direction_key = str(_first_present(
            rec,
            ["direction_key", "direction_key_for_existing_code", "direction"],
            "y_minus_x",
        ))
        slope_parameter = float(_first_present(
            rec,
            ["slope_parameter", "slope_parameter_for_existing_code", "slope"],
            1.0,
        ))
        if direction_key not in {"y_minus_x", "y_plus_x"}:
            raise ValueError(f"Unsupported direction_key: {direction_key}")

        line_bin_width = float(_first_present(
            rec,
            ["line_bin_width"],
            cleanup_params.get("line_bin_width", 18.0),
        ))
        min_pixels_per_line = int(_first_present(
            rec,
            ["min_pixels_per_line"],
            cleanup_params.get("min_pixels_per_line", 80),
        ))
        method = str(_first_present(rec, ["method"], cleanup_params.get("method", "median")))
        exclude_mode = _first_present(rec, ["exclude_mode"], cleanup_params.get("exclude_mode", "robust_high"))
        cleanup_stage = str(_first_present(rec, ["cleanup_stage"], cleanup_params.get("cleanup_stage", "detected_fixed_direction")))

        out = {
            **rec,
            "pass_index": i,
            "direction_key": direction_key,
            "slope_parameter": slope_parameter,
            "signed_slope": float(rec.get("signed_slope", slope_parameter if direction_key == "y_minus_x" else -slope_parameter)),
            "line_bin_width": line_bin_width,
            "min_pixels_per_line": min_pixels_per_line,
            "method": method,
            "exclude_mode": exclude_mode,
            "cleanup_stage": cleanup_stage,
            "detection_type": str(rec.get("detection_type", cleanup_stage)),
            "direction_type": str(rec.get("direction_type", "primary_positive")),
            "family_rank": int(rec.get("family_rank", i)) if pd.notna(rec.get("family_rank", i)) else i,
        }
        records.append(out)
    return records


def summarize_cleanup_direction_records(cleanup_params: Optional[dict]) -> str:
    records = normalize_fixed_direction_records(cleanup_params)
    if len(records) == 0:
        return ""
    parts = []
    for rec in records:
        sign = "+" if rec["direction_key"] == "y_minus_x" else "-"
        parts.append(
            f"{rec.get('detection_type', 'dir')}:{rec.get('direction_type', '')}:"
            f"m={sign}{float(rec['slope_parameter']):.4f}:bin={float(rec['line_bin_width']):.1f}"
        )
    return " -> ".join(parts)


def destripe_by_fixed_slope_lines(
    alpha_map: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    exclude_mask: Optional[np.ndarray] = None,
    direction_key: str = "y_minus_x",
    slope: float = 1.0,
    line_bin_width: float = 18.0,
    min_pixels_per_line: int = 80,
    method: str = "median",
    preserve_global_stat: bool = True,
    smooth_half_window: int = 0,
    fallback_to_valid: bool = True,
    trim_fraction: float = 0.1,
    mode_bins: int = 64,
    sigma_clip_nsigma: float = 3.0,
    sigma_clip_max_iter: int = 3,
    metadata: Optional[dict] = None,
) -> dict:
    """Remove stripe bands along fixed-slope line groups."""
    alpha = np.asarray(alpha_map, dtype=float)
    if alpha.ndim != 2:
        raise ValueError("alpha_map must be 2D.")

    finite = np.isfinite(alpha)
    if valid_mask is None:
        valid = finite.copy()
    else:
        valid = np.asarray(valid_mask, dtype=bool) & finite

    if not np.any(valid):
        raise ValueError("No valid finite pixels for fixed-direction cleanup.")

    if exclude_mask is None:
        estimate_mask = valid.copy()
    else:
        estimate_mask = valid & (~np.asarray(exclude_mask, dtype=bool))

    if not np.any(estimate_mask):
        if fallback_to_valid:
            estimate_mask = valid.copy()
        else:
            raise ValueError("No pixels remain after exclusion for fixed-direction cleanup.")

    ids = fixed_slope_line_id_map(
        alpha.shape,
        slope=slope,
        line_bin_width=line_bin_width,
        direction_key=direction_key,
    )
    id_values = np.arange(int(np.nanmin(ids[valid])), int(np.nanmax(ids[valid])) + 1, dtype=np.int32)

    global_stat = statistic_1d(
        alpha[estimate_mask],
        method=method,
        trim_fraction=trim_fraction,
        mode_bins=mode_bins,
        sigma_clip_nsigma=sigma_clip_nsigma,
        sigma_clip_max_iter=sigma_clip_max_iter,
    )

    raw_stats = np.full(id_values.shape, np.nan, dtype=float)
    counts_used = np.zeros(id_values.shape, dtype=int)
    used_fallback = np.zeros(id_values.shape, dtype=bool)

    for k, line_id in enumerate(id_values):
        m = (ids == line_id) & estimate_mask
        count = int(np.sum(m))

        if count < min_pixels_per_line and fallback_to_valid:
            mf = (ids == line_id) & valid
            count_f = int(np.sum(mf))
            if count_f >= min_pixels_per_line:
                m = mf
                count = count_f
                used_fallback[k] = True

        counts_used[k] = count
        if count >= min_pixels_per_line:
            raw_stats[k] = statistic_1d(
                alpha[m],
                method=method,
                trim_fraction=trim_fraction,
                mode_bins=mode_bins,
                sigma_clip_nsigma=sigma_clip_nsigma,
                sigma_clip_max_iter=sigma_clip_max_iter,
            )

    smoothed_stats = moving_nanmedian_1d(raw_stats, half_window=smooth_half_window)

    if preserve_global_stat and np.isfinite(global_stat):
        offsets = smoothed_stats - global_stat
    else:
        offsets = smoothed_stats.copy()

    offsets_filled = np.where(np.isfinite(offsets), offsets, 0.0)

    stripe_map = np.zeros_like(alpha, dtype=float)
    for line_id, offset in zip(id_values, offsets_filled):
        stripe_map[ids == line_id] = float(offset)
    stripe_map[~valid] = np.nan

    corrected = alpha - stripe_map
    corrected[~finite] = np.nan

    metadata = {} if metadata is None else dict(metadata)
    cleanup_stage = metadata.get("cleanup_stage", "detected_fixed_direction")
    detection_type = metadata.get("detection_type", cleanup_stage)
    direction_type = metadata.get("direction_type", "primary_positive")
    direction_name = (
        f"{cleanup_stage}_{detection_type}_{direction_type}_{direction_key}_"
        f"slope_{float(slope):.4f}_bin_{float(line_bin_width):.1f}"
    )

    line_table = pd.DataFrame({
        "line_id": id_values,
        "n_pixels_used": counts_used,
        "used_fallback_to_valid": used_fallback,
        "line_stat_raw": raw_stats,
        "line_stat_after_smoothing": smoothed_stats,
        "stripe_offset_subtracted": offsets_filled,
        "global_stat": global_stat,
        "direction": direction_name,
        "direction_key": direction_key,
        "method": method,
        "cleanup_stage": cleanup_stage,
        "detection_type": detection_type,
        "direction_type": direction_type,
        "family_rank": metadata.get("family_rank", np.nan),
        "signed_slope": metadata.get("signed_slope", np.nan),
        "slope_parameter": float(slope),
        "line_bin_width": float(line_bin_width),
    })

    return {
        "corrected": corrected,
        "stripe_map": stripe_map,
        "line_table": line_table,
        "global_stat": global_stat,
        "estimate_mask": estimate_mask,
        "direction": direction_name,
    }


def destripe_by_fixed_line_directions(
    alpha_map: np.ndarray,
    valid_mask: np.ndarray,
    plume_mask: Optional[np.ndarray] = None,
    cleanup_params: Optional[dict] = None,
    nsigma: float = 4.0,
) -> dict:
    """Sequential cleanup using detected fixed directions, including orthogonal directions."""
    if cleanup_params is None:
        cleanup_params = {}

    records = normalize_fixed_direction_records(cleanup_params)
    if len(records) == 0:
        zero = np.zeros_like(alpha_map, dtype=float)
        zero[~np.asarray(valid_mask, dtype=bool)] = np.nan
        return {
            "corrected": np.asarray(alpha_map, dtype=float).copy(),
            "stripe_map": zero,
            "directional_stripe_maps": {},
            "line_table": pd.DataFrame(),
            "exclude_meta": [],
        }

    current = np.asarray(alpha_map, dtype=float).copy()
    total_stripe = np.zeros_like(current, dtype=float)
    total_stripe[~np.asarray(valid_mask, dtype=bool)] = np.nan
    directional_maps: dict[str, np.ndarray] = {}
    line_tables = []
    exclude_metas = []

    for pass_index, rec in enumerate(records, start=1):
        exclude_mask, exclude_meta = make_exclude_mask_for_destriping(
            alpha_map=current,
            valid_mask=valid_mask,
            plume_mask=plume_mask,
            exclude_mode=rec.get("exclude_mode", cleanup_params.get("exclude_mode", "robust_high")),
            exclude_nsigma=cleanup_params.get("exclude_nsigma", nsigma),
        )

        out = destripe_by_fixed_slope_lines(
            alpha_map=current,
            valid_mask=valid_mask,
            exclude_mask=exclude_mask,
            direction_key=rec["direction_key"],
            slope=float(rec["slope_parameter"]),
            line_bin_width=float(rec.get("line_bin_width", cleanup_params.get("line_bin_width", 18.0))),
            min_pixels_per_line=int(rec.get("min_pixels_per_line", cleanup_params.get("min_pixels_per_line", 80))),
            method=rec.get("method", cleanup_params.get("method", "median")),
            preserve_global_stat=cleanup_params.get("preserve_global_stat", True),
            smooth_half_window=cleanup_params.get("smooth_half_window", 0),
            fallback_to_valid=cleanup_params.get("fallback_to_valid", True),
            trim_fraction=cleanup_params.get("trim_fraction", 0.1),
            mode_bins=cleanup_params.get("mode_bins", 64),
            sigma_clip_nsigma=cleanup_params.get("sigma_clip_nsigma", 3.0),
            sigma_clip_max_iter=cleanup_params.get("sigma_clip_max_iter", 3),
            metadata={**rec, "pass_index": pass_index},
        )

        current = out["corrected"]
        total_stripe = total_stripe + out["stripe_map"]
        key = out["direction"]
        directional_maps[key] = out["stripe_map"].copy()

        table = out["line_table"].copy()
        table.insert(0, "pass_index", pass_index)
        line_tables.append(table)

        exclude_meta = dict(exclude_meta)
        exclude_meta.update({
            "pass_index": pass_index,
            "cleanup_stage": rec.get("cleanup_stage", cleanup_params.get("cleanup_stage", "detected_fixed_direction")),
            "detection_type": rec.get("detection_type"),
            "direction_type": rec.get("direction_type"),
            "direction_key": rec.get("direction_key"),
            "slope_parameter": float(rec.get("slope_parameter")),
            "signed_slope": rec.get("signed_slope"),
            "line_bin_width": float(rec.get("line_bin_width", cleanup_params.get("line_bin_width", 18.0))),
            "method": rec.get("method", cleanup_params.get("method", "median")),
        })
        exclude_metas.append(exclude_meta)

    line_table_all = pd.concat(line_tables, ignore_index=True) if len(line_tables) else pd.DataFrame()

    return {
        "corrected": current,
        "stripe_map": total_stripe,
        "directional_stripe_maps": directional_maps,
        "line_table": line_table_all,
        "exclude_meta": exclude_metas,
    }


# Backward-compatible wrapper name used by earlier notebooks.
def destripe_by_broad_fixed_slopes_median(
    alpha_map: np.ndarray,
    valid_mask: np.ndarray,
    plume_mask: Optional[np.ndarray] = None,
    broad_params: Optional[dict] = None,
    nsigma: float = 4.0,
) -> dict:
    return destripe_by_fixed_line_directions(
        alpha_map=alpha_map,
        valid_mask=valid_mask,
        plume_mask=plume_mask,
        cleanup_params=broad_params,
        nsigma=nsigma,
    )



# %% cell 7
# ============================================
# 6. Iterative MF with thin each-iter cleanup, then broad median cleanup
# ============================================

def should_apply_destriping(iteration_number: int, destripe_when) -> bool:
    if destripe_when is None or destripe_when == "none":
        return False
    if destripe_when == "each_iter":
        return True
    if destripe_when == "final_only":
        return False
    if isinstance(destripe_when, (list, tuple, set, np.ndarray)):
        return int(iteration_number) in {int(v) for v in destripe_when}
    raise ValueError("destripe_when must be 'none', 'each_iter', 'final_only', or a list of iteration numbers.")


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
    destripe_when = "none",
    destripe_params: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """Run Iterative MF.

    If destriping is enabled, the order at each iteration is:
        1. detected thin high-alpha cleanup with primary + orthogonal directions
        2. detected broad offset cleanup with primary + orthogonal directions, median only
        3. thresholding / plume mask update
    """
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
    plume_mask_history = []
    background_mask_history = []
    threshold_meta_history = []
    line_table_history = []
    exclude_meta_history = []
    mu_history = []
    cov_history = []

    prev_plume_mask = None
    converged_iter = None

    thin_params = destripe_params.get("thin_cleanup_params", destripe_params) if destripe_params else None
    broad_params = destripe_params.get("broad_cleanup_params", None) if destripe_params else None
    thin_directions_summary = summarize_cleanup_direction_records(thin_params) if thin_params else ""
    broad_directions_summary = summarize_cleanup_direction_records(broad_params) if broad_params else ""

    for it in range(1, n_iter + 1):
        n_bg = int(np.sum(background_mask))
        if n_bg < min_background_pixels:
            raise ValueError(f"Background pixels too few at iter {it}: {n_bg} < {min_background_pixels}")

        alpha_raw, mu, cov, target = matched_filter_alpha_map(
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
        line_table = pd.DataFrame()
        exclude_meta = []

        apply_now = should_apply_destriping(it, destripe_when)
        if apply_now:
            # 1. Thin-line cleanup: detected thin high-alpha slope + its orthogonal direction.
            thin_out = destripe_by_fixed_line_directions(
                alpha_map=alpha_raw,
                valid_mask=valid_mask,
                plume_mask=prev_plume_mask,
                cleanup_params=thin_params,
                nsigma=nsigma,
            )
            alpha_after_thin = thin_out["corrected"]

            # 2. Broad sensor-noise cleanup: detected broad offset slope + its orthogonal direction.
            broad_out = destripe_by_fixed_line_directions(
                alpha_map=alpha_after_thin,
                valid_mask=valid_mask,
                plume_mask=prev_plume_mask,
                cleanup_params=broad_params,
                nsigma=nsigma,
            )

            alpha_corrected = broad_out["corrected"]
            stripe_map = thin_out["stripe_map"] + broad_out["stripe_map"]

            # Prefix keys so thin and broad maps are not confused.
            directional_stripe_maps = {
                f"thin_{k}": v.copy()
                for k, v in thin_out["directional_stripe_maps"].items()
            }
            directional_stripe_maps.update({
                f"broad_{k}": v.copy()
                for k, v in broad_out["directional_stripe_maps"].items()
            })

            line_tables = []
            if thin_out["line_table"] is not None and len(thin_out["line_table"]) > 0:
                t = thin_out["line_table"].copy()
                t["cleanup_stage"] = "thin_detected"
                line_tables.append(t)
            if broad_out["line_table"] is not None and len(broad_out["line_table"]) > 0:
                line_tables.append(broad_out["line_table"].copy())
            line_table = pd.concat(line_tables, ignore_index=True) if len(line_tables) else pd.DataFrame()

            exclude_meta = [
                {"cleanup_stage": "thin_detected", "items": thin_out["exclude_meta"]},
                {"cleanup_stage": "broad_median", "items": broad_out["exclude_meta"]},
            ]

            threshold_source = destripe_params.get("threshold_source", "corrected")
            if threshold_source == "corrected":
                alpha_used = alpha_corrected.copy()
            elif threshold_source == "raw":
                alpha_used = alpha_raw.copy()
            else:
                raise ValueError("threshold_source must be 'corrected' or 'raw'.")

        plume_mask, threshold_meta = plume_mask_from_alpha(alpha_used, valid_mask, nsigma=nsigma)
        new_background_mask = valid_mask & (~plume_mask)

        alpha_raw_history.append(alpha_raw.copy())
        alpha_corrected_history.append(alpha_corrected.copy())
        alpha_used_history.append(alpha_used.copy())
        stripe_history.append(stripe_map.copy())
        directional_stripe_history.append({k: v.copy() for k, v in directional_stripe_maps.items()})
        plume_mask_history.append(plume_mask.copy())
        background_mask_history.append(background_mask.copy())
        threshold_meta_history.append(threshold_meta.copy())
        line_table_history.append(line_table.copy())
        exclude_meta_history.append(exclude_meta.copy() if hasattr(exclude_meta, 'copy') else exclude_meta)
        mu_history.append(mu.copy())
        cov_history.append(cov.copy())

        if verbose:
            print(
                f"iter {it:02d} | bg={n_bg:6d} | "
                f"thr={threshold_meta['threshold']:+.6e} | "
                f"med={threshold_meta['median']:+.6e} | "
                f"rstd={threshold_meta['robust_std']:.6e} | "
                f"plume={threshold_meta['n_plume']:6d} | "
                f"detected_cleanup={apply_now} | "
                f"thin_dirs={thin_directions_summary if apply_now else ''} | "
                f"broad_dirs={broad_directions_summary if apply_now else ''}"
            )

        if prev_plume_mask is not None and np.array_equal(plume_mask, prev_plume_mask):
            converged_iter = it
            background_mask = new_background_mask
            if verbose:
                print(f"Converged at iteration {it}.")
            break

        prev_plume_mask = plume_mask.copy()
        background_mask = new_background_mask

    result = {
        "alpha_raw_history": alpha_raw_history,
        "alpha_corrected_history": alpha_corrected_history,
        "alpha_used_history": alpha_used_history,
        "stripe_history": stripe_history,
        "directional_stripe_history": directional_stripe_history,
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
        "final_only_post": None,
    }

    # Convenience aliases
    result["alpha_final_raw"] = alpha_raw_history[-1]
    result["alpha_final_corrected"] = alpha_corrected_history[-1]
    result["alpha_final_used"] = alpha_used_history[-1]
    result["stripe_map_final"] = stripe_history[-1]
    result["directional_stripe_maps_final"] = directional_stripe_history[-1]
    result["plume_mask_final"] = plume_mask_history[-1]
    result["threshold_meta_final"] = threshold_meta_history[-1]
    result["line_table_final"] = line_table_history[-1]

    return result




# %% cell 8
# ============================================
# 7. Result comparison / saving helpers
# ============================================

def get_result_directions(res: dict):
    params = res.get("destripe_params")
    if params is None:
        return None
    thin_params = params.get("thin_cleanup_params", params)
    return summarize_cleanup_direction_records(thin_params)


def summarize_results_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        meta = res["threshold_meta_final"]
        params = res.get("destripe_params")
        thin_params = None if params is None else params.get("thin_cleanup_params", params)
        broad_params = None if params is None else params.get("broad_cleanup_params")
        rows.append({
            "name": name,
            "destripe_when": res["destripe_when"],
            "thin_method": None if thin_params is None else thin_params.get("method"),
            "thin_directions": None if thin_params is None else summarize_cleanup_direction_records(thin_params),
            "thin_exclude_mode": None if thin_params is None else thin_params.get("exclude_mode"),
            "broad_method": None if broad_params is None else broad_params.get("method"),
            "broad_directions": None if broad_params is None else summarize_cleanup_direction_records(broad_params),
            "broad_line_bin_width": None if broad_params is None else broad_params.get("line_bin_width"),
            "broad_exclude_mode": None if broad_params is None else broad_params.get("exclude_mode"),
            "converged_iter": res["converged_iter"],
            "threshold": meta["threshold"],
            "median": meta["median"],
            "robust_std": meta["robust_std"],
            "plume_pixels": int(np.sum(res["plume_mask_final"])),
            "valid_pixels": int(np.sum(res["valid_mask"])),
        })
    return pd.DataFrame(rows)


def plot_single_result(res: dict, title_prefix: str = "result"):
    valid = res["valid_mask"]
    raw = res["alpha_final_raw"]
    corr = res["alpha_final_corrected"]
    stripe = res["stripe_map_final"]
    plume = res["plume_mask_final"]
    directional_maps = res.get("directional_stripe_maps_final", {})

    vmin, vmax = robust_limits([raw, corr], mask=valid, q_low=2, q_high=98)
    svmin, svmax = robust_limits(stripe, mask=valid, q_low=2, q_high=98)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    im0 = axes[0].imshow(raw, origin="upper", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{title_prefix}\nraw alpha")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, label="alpha")

    im1 = axes[1].imshow(stripe, origin="upper", vmin=svmin, vmax=svmax)
    axes[1].set_title("total estimated stripe")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, label="offset")

    im2 = axes[2].imshow(corr, origin="upper", vmin=vmin, vmax=vmax)
    axes[2].set_title("corrected alpha")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, label="alpha")

    im3 = axes[3].imshow(plume, origin="upper")
    axes[3].set_title("plume mask")
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("y")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, label="candidate")

    plt.tight_layout()
    plt.show()

    # Direction-wise stripe maps
    if len(directional_maps) > 0:
        n = len(directional_maps)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
        if n == 1:
            axes = [axes]
        for ax, (direction, smap) in zip(axes, directional_maps.items()):
            lo, hi = robust_limits(smap, mask=valid, q_low=2, q_high=98)
            im = ax.imshow(smap, origin="upper", vmin=lo, vmax=hi)
            ax.set_title(f"stripe: {direction}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(im, ax=ax, fraction=0.046, label="offset")
        plt.tight_layout()
        plt.show()

    line_table = res["line_table_final"]
    if line_table is not None and len(line_table) > 0:
        plt.figure(figsize=(8, 4))
        for direction, df_dir in line_table.groupby("direction"):
            plt.plot(
                df_dir["line_id"],
                df_dir["stripe_offset_subtracted"],
                marker=".",
                linewidth=1,
                label=direction,
            )
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("line_id")
        plt.ylabel("subtracted offset")
        plt.title(f"{title_prefix}: directional stripe offset")
        plt.grid(True)
        plt.legend()
        plt.show()


def plot_experiment_grid(results: dict[str, dict], names: Optional[Sequence[str]] = None):
    if names is None:
        names = list(results.keys())

    valid = next(iter(results.values()))["valid_mask"]
    maps = []
    for name in names:
        maps.append(results[name]["alpha_final_raw"])
        maps.append(results[name]["alpha_final_corrected"])
    vmin, vmax = robust_limits(maps, mask=valid, q_low=2, q_high=98)

    n = len(names)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    if n == 1:
        axes = axes.reshape(3, 1)

    for j, name in enumerate(names):
        res = results[name]
        raw = res["alpha_final_raw"]
        corr = res["alpha_final_corrected"]
        plume = res["plume_mask_final"]

        im0 = axes[0, j].imshow(raw, origin="upper", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(f"{name}\nraw")
        axes[0, j].set_xlabel("x")
        axes[0, j].set_ylabel("y")

        im1 = axes[1, j].imshow(corr, origin="upper", vmin=vmin, vmax=vmax)
        axes[1, j].set_title("corrected")
        axes[1, j].set_xlabel("x")
        axes[1, j].set_ylabel("y")

        im2 = axes[2, j].imshow(plume, origin="upper")
        axes[2, j].set_title("plume mask")
        axes[2, j].set_xlabel("x")
        axes[2, j].set_ylabel("y")

    fig.colorbar(im1, ax=axes[0:2, :].ravel().tolist(), fraction=0.02, label="alpha")
    plt.tight_layout()
    plt.show()



def stat_case_name(method: str, when: str = "each_iter") -> str:
    # This notebook is each_iter only.
    return f"{method}_thin_eachiter_then_broad_median"


def available_stat_methods(results: dict[str, dict], methods: Optional[Sequence[str]] = None) -> list[str]:
    if methods is None:
        methods = [m for m, _ in STRIPE_STAT_METHODS]
    out = []
    for method in methods:
        if stat_case_name(method) in results:
            out.append(method)
    return out


def plot_eachiter_comparison_by_stat(
    results: dict[str, dict],
    methods: Optional[Sequence[str]] = None,
    include_baseline: bool = True,
):
    """Show one overview image for each_iter thin-stat + broad-median cases."""
    methods = available_stat_methods(results, methods)
    names = [stat_case_name(method) for method in methods if stat_case_name(method) in results]
    if include_baseline and "baseline_no_destripe" in results:
        names = ["baseline_no_destripe"] + names

    if len(names) == 0:
        print("No each_iter thin+Broad cases found.")
        return

    print("Overview: thin each_iter -> broad median / " + ", ".join(names))
    plot_experiment_grid(results, names=names)


def plot_stat_method_results_separately(
    results: dict[str, dict],
    methods: Optional[Sequence[str]] = None,
    show_single_result: bool = True,
):
    """For each thin statistic, show the detailed result image."""
    methods = available_stat_methods(results, methods)

    for method in methods:
        name = stat_case_name(method)
        if name not in results:
            continue

        print("\n" + "-" * 80)
        print(f"Thin statistic: {method} -> broad median")
        print("-" * 80)

        if show_single_result:
            plot_single_result(results[name], title_prefix=name)


def plot_stat_threshold_histories(results: dict[str, dict], methods: Optional[Sequence[str]] = None):
    """Show threshold histories for each thin statistic."""
    methods = available_stat_methods(results, methods)
    plt.figure(figsize=(10, 6))
    for method in methods:
        name = stat_case_name(method)
        if name not in results:
            continue
        th = [m["threshold"] for m in results[name]["threshold_meta_history"]]
        plt.plot(np.arange(1, len(th) + 1), th, marker="o", label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Threshold")
    plt.title("Threshold history: thin each_iter -> broad median")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.show()


def plot_threshold_histories(results: dict[str, dict]):
    plt.figure(figsize=(8, 5))
    for name, res in results.items():
        th = [m["threshold"] for m in res["threshold_meta_history"]]
        plt.plot(np.arange(1, len(th) + 1), th, marker="o", label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Threshold")
    plt.title("Threshold history")
    plt.grid(True)
    plt.legend()
    plt.show()


def save_case_outputs(
    result: dict,
    case_name: str,
    output_dir: str | Path,
    ys: Optional[np.ndarray] = None,
    xs: Optional[np.ndarray] = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = result["alpha_final_raw"]
    corr = result["alpha_final_corrected"]
    stripe = result["stripe_map_final"]
    plume = result["plume_mask_final"]
    valid = result["valid_mask"]
    directional_maps = result.get("directional_stripe_maps_final", {})

    np.save(output_dir / f"{case_name}_alpha_raw.npy", raw)
    np.save(output_dir / f"{case_name}_alpha_corrected.npy", corr)
    np.save(output_dir / f"{case_name}_stripe_map_total.npy", stripe)
    np.save(output_dir / f"{case_name}_plume_mask.npy", plume)

    for direction, smap in directional_maps.items():
        np.save(output_dir / f"{case_name}_stripe_map_{direction}.npy", smap)

    H, W = raw.shape
    rows, cols = np.indices((H, W))

    if ys is not None and len(ys) == H:
        y_values = np.asarray(ys)[rows.ravel()]
    else:
        y_values = rows.ravel()

    if xs is not None and len(xs) == W:
        x_values = np.asarray(xs)[cols.ravel()]
    else:
        x_values = cols.ravel()

    y_minus_x_ids = line_id_map(raw.shape, direction="y_minus_x")
    y_plus_x_ids = line_id_map(raw.shape, direction="y_plus_x")

    pixel_df = pd.DataFrame({
        "row": rows.ravel(),
        "col": cols.ravel(),
        "y": y_values,
        "x": x_values,
        "line_id_y_minus_x": y_minus_x_ids.ravel(),
        "line_id_y_plus_x": y_plus_x_ids.ravel(),
        "is_valid": valid.ravel(),
        "alpha_raw": raw.ravel(),
        "stripe_offset_total_subtracted": stripe.ravel(),
        "alpha_corrected": corr.ravel(),
        "is_plume": plume.ravel(),
    })

    for direction, smap in directional_maps.items():
        pixel_df[f"stripe_offset_{direction}"] = smap.ravel()

    pixel_df.to_csv(output_dir / f"{case_name}_pixel_results.csv", index=False)

    line_table = result["line_table_final"]
    if line_table is not None and len(line_table) > 0:
        line_table.to_csv(output_dir / f"{case_name}_line_table.csv", index=False)

    return pixel_df





# %% cell 9

# ============================================
# 8. Load data and prepare cube/UAS
# ============================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load ROI spectra
roi_df, wavelengths, spectra = load_roi_spectra_csv(ROI_CSV)
cube, ys, xs = spectra_to_cube(roi_df, spectra)

print(f"ROI table shape: {roi_df.shape}")
print(f"Cube shape: {cube.shape}")
print(f"Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} nm")

# Quick visual checks
plot_mean_spectrum(cube, wavelengths, xlim=(WL_MIN, WL_MAX))

try:
    rgb = make_rgb_from_cube(cube, wavelengths)
    plt.figure(figsize=(5, 5))
    plt.imshow(rgb, origin="upper")
    plt.title("RGB preview")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
except Exception as exc:
    print(f"RGB preview skipped: {exc}")

# Load MODTRAN and compute UAS
mod_wave, alpha_grid, mod_spectra = load_ch4_modtran_csv(MODTRAN_CSV)
print(f"MODTRAN wavelength range: {mod_wave[0]:.2f} - {mod_wave[-1]:.2f} nm")
print("alpha_grid:", alpha_grid)

mod_sensor = gaussian_srf_resample(
    mod_wave=mod_wave,
    mod_spectra=mod_spectra,
    sensor_wave=wavelengths,
    fwhm_nm=FWHM_NM,
)

uas_all, intercept = compute_uas_log_slope(
    alpha_grid=alpha_grid,
    spectra_grid=mod_sensor,
    alpha_min=UAS_ALPHA_MIN,
    alpha_max=UAS_ALPHA_MAX,
)

plot_uas(wavelengths, uas_all, title="CH4 UAS from MODTRAN", xlim=(WL_MIN, WL_MAX))

# Select SWIR / methane absorption range
cube_sel, wave_sel, band_sel = select_bands(cube, wavelengths, wl_min=WL_MIN, wl_max=WL_MAX)
uas_sel = uas_all[band_sel]

valid_mask = make_valid_pixel_mask(
    cube_sel,
    nodata_values=NODATA_VALUES,
    require_positive=REQUIRE_POSITIVE,
    min_valid_fraction=MIN_VALID_FRACTION,
)

print(f"Selected cube shape: {cube_sel.shape}")
print(f"Valid pixels: {int(np.sum(valid_mask))}")

plot_map(valid_mask.astype(float), title="Valid pixel mask", cmap="gray", colorbar_label="valid")



# %% cell 10
# ============================================
# 9. Run experiments
# ============================================

results = {}

for name, cfg in EXPERIMENTS.items():
    print("\n" + "=" * 80)
    print(f"Running experiment: {name}")
    print("=" * 80)

    res = run_iterative_mf_with_optional_destriping(
        cube=cube_sel,
        uas=uas_sel,
        valid_mask=valid_mask,
        initial_background_mask=None,
        n_iter=N_ITER,
        nsigma=NSIGMA,
        reg=REG,
        rcond=RCOND,
        min_background_pixels=None,
        destripe_when=cfg.get("destripe_when", "none"),
        destripe_params=cfg.get("destripe_params", None),
        verbose=True,
    )
    results[name] = res

summary_df = summarize_results_table(results)
summary_df




# %% cell 11
# ============================================
# 10. Compare results
# ============================================

summary_df = summarize_results_table(results)
display(summary_df)

# Threshold histories.
plot_threshold_histories(results)
plot_stat_threshold_histories(results)

# One overview image across thin statistics, all with broad median after thin cleanup.
plot_eachiter_comparison_by_stat(results)

# Detailed result for each thin statistic.
plot_stat_method_results_separately(
    results,
    methods=[m for m, _ in STRIPE_STAT_METHODS],
    show_single_result=True,
)




# %% cell 12
# ============================================
# 11. Save selected outputs
# ============================================

cases_to_save = ["baseline_no_destripe"]
for method, _ in STRIPE_STAT_METHODS:
    cases_to_save.append(f"{method}_thin_eachiter_then_broad_median")

for case_name in cases_to_save:
    if case_name in results:
        save_case_outputs(
            result=results[case_name],
            case_name=case_name,
            output_dir=OUTPUT_DIR,
            ys=ys,
            xs=xs,
        )
        print(f"Saved: {case_name}")
    else:
        print(f"Skipped missing case: {case_name}")

summary_df.to_csv(OUTPUT_DIR / "experiment_summary.csv", index=False)
np.save(OUTPUT_DIR / "selected_wavelengths.npy", wave_sel)
np.save(OUTPUT_DIR / "selected_uas.npy", uas_sel)

# Save detected direction settings for reproducibility.
pd.DataFrame(DETECTED_THIN_DIRECTION_RECORDS).to_csv(OUTPUT_DIR / "detected_thin_directions_used.csv", index=False)
pd.DataFrame(DETECTED_BROAD_DIRECTION_RECORDS).to_csv(OUTPUT_DIR / "detected_broad_directions_used.csv", index=False)
pd.read_csv(DETECTED_SLOPE_DIRECTIONS_CSV).to_csv(OUTPUT_DIR / "detected_slope_directions_source.csv", index=False)

print(f"All selected outputs were saved to: {OUTPUT_DIR.resolve()}")




# %% cell 13
summary_df.to_csv(OUTPUT_DIR / "summary_df.csv", index=False)