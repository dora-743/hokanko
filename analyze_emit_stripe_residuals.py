from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "This script needs h5py to read EMIT NetCDF4/HDF5 files. "
            "Install it with: python -m pip install h5py"
        ) from exc
    return h5py


def robust_std(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return np.nan
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med))
    if mad > 0:
        return float(1.4826 * mad)
    return float(np.nanstd(vals))


def robust_z(values: np.ndarray) -> np.ndarray:
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    sigma = 1.4826 * mad if mad > 0 else np.nanstd(values)
    return (values - med) / sigma


def list_datasets(h5) -> list[tuple[str, tuple[int, ...], str]]:
    found: list[tuple[str, tuple[int, ...], str]] = []

    def visit(name, obj):
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            found.append((name, tuple(obj.shape), str(obj.dtype)))

    h5.visititems(visit)
    return found


def choose_cube_dataset(datasets: list[tuple[str, tuple[int, ...], str]], requested: str | None) -> str:
    if requested:
        return requested
    candidates = []
    for name, shape, dtype in datasets:
        low = name.lower()
        if len(shape) == 3 and any(token in low for token in ["radiance", "reflectance", "rfl"]):
            candidates.append((name, shape, dtype))
    if not candidates:
        candidates = [(name, shape, dtype) for name, shape, dtype in datasets if len(shape) == 3]
    if not candidates:
        raise SystemExit("No 3-D radiance/reflectance-like dataset found. Run with --list-datasets.")
    candidates.sort(key=lambda item: np.prod(item[1]), reverse=True)
    return candidates[0][0]


def find_wavelengths(h5, n_bands: int, requested: str | None) -> np.ndarray:
    if requested:
        return np.asarray(h5[requested][...], dtype=float)
    for name, shape, dtype in list_datasets(h5):
        low = name.lower()
        if "wavelength" in low and np.prod(shape) == n_bands:
            return np.asarray(h5[name][...], dtype=float).reshape(-1)
    return np.arange(n_bands, dtype=float)


def infer_band_axis(shape: tuple[int, int, int], wavelengths: np.ndarray) -> int:
    for axis, size in enumerate(shape):
        if size == wavelengths.size:
            return axis
    return int(np.argmin(shape))


def parse_wavelengths(text: str | None, wavelengths: np.ndarray) -> np.ndarray:
    if text:
        values = [float(x.strip()) for x in text.split(",") if x.strip()]
    else:
        values = [1650, 2000, 2100, 2200, 2300, 2350, 2400]
    idx = [int(np.nanargmin(np.abs(wavelengths - value))) for value in values]
    return np.asarray(sorted(set(idx)), dtype=int)


def read_selected_bands(dataset, band_axis: int, band_idx: np.ndarray) -> np.ndarray:
    # Return rows x cols x bands.
    if band_axis == 2:
        data = dataset[:, :, band_idx]
    elif band_axis == 0:
        data = dataset[band_idx, :, :]
        data = np.moveaxis(data, 0, 2)
    elif band_axis == 1:
        data = dataset[:, band_idx, :]
        data = np.moveaxis(data, 1, 2)
    else:
        raise ValueError(f"Unexpected band axis: {band_axis}")
    arr = np.asarray(data, dtype=np.float32)
    arr[arr <= -9990] = np.nan
    return arr


def line_ids(rows: np.ndarray, cols: np.ndarray, signed_slope: float, bin_width: float) -> np.ndarray:
    if signed_slope >= 0:
        coord = rows - signed_slope * cols
    else:
        coord = rows + abs(signed_slope) * cols
    return np.rint(coord / bin_width).astype(np.int64)


def direction_score(image: np.ndarray, slopes: np.ndarray, sample_step: int = 3) -> pd.DataFrame:
    valid = np.isfinite(image)
    smooth = ndimage.gaussian_filter(np.nan_to_num(image, nan=np.nanmedian(image)), sigma=18, mode="nearest")
    residual = image - smooth
    keep = np.zeros_like(valid, dtype=bool)
    keep[::sample_step, ::sample_step] = True
    rows, cols = np.nonzero(valid & keep)
    weights = robust_z(residual[rows, cols])
    scores = []
    for slope in slopes:
        ids = line_ids(rows, cols, slope, bin_width=10.0)
        ids = ids - ids.min()
        counts = np.bincount(ids)
        sums = np.bincount(ids, weights=weights)
        ok = counts >= 30
        score = np.nanpercentile(np.abs(sums[ok] / counts[ok]), 99) if ok.any() else np.nan
        scores.append(score)
    return pd.DataFrame({"angle_deg": np.degrees(np.arctan(slopes)), "signed_slope": slopes, "score": scores})


def column_bias_profile(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Remove per-row median first, then summarize cross-track residual by column.
    row_centered = image - np.nanmedian(image, axis=1, keepdims=True)
    col_profile = np.nanmedian(row_centered, axis=0)
    return np.arange(image.shape[1]), col_profile


def simple_absorption_index(cube: np.ndarray, wavelengths: np.ndarray, band_idx: np.ndarray) -> np.ndarray:
    # A conservative visualization product, not a published methane retrieval.
    selected_waves = wavelengths[band_idx]
    left = np.where((selected_waves >= 2100) & (selected_waves <= 2200))[0]
    absorption = np.where((selected_waves >= 2280) & (selected_waves <= 2360))[0]
    right = np.where((selected_waves >= 2380) & (selected_waves <= 2450))[0]
    if left.size == 0 or absorption.size == 0 or right.size == 0:
        mid = cube.shape[2] // 2
        return cube[:, :, mid]
    continuum = 0.5 * (np.nanmedian(cube[:, :, left], axis=2) + np.nanmedian(cube[:, :, right], axis=2))
    center = np.nanmedian(cube[:, :, absorption], axis=2)
    return continuum - center


def plot_outputs(image: np.ndarray, score_df: pd.DataFrame, col_x: np.ndarray, col_profile: np.ndarray, out_dir: Path) -> None:
    valid = np.isfinite(image)
    lo, hi = np.nanpercentile(image[valid], [2, 98])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(image, cmap="gray", vmin=lo, vmax=hi, origin="upper")
    axes[0].set_title("simple absorption / selected-band image")
    axes[0].set_xlabel("cross-track column")
    axes[0].set_ylabel("down-track row")
    axes[1].plot(col_x, col_profile, color="black", lw=1)
    axes[1].set_title("column median residual profile")
    axes[1].set_xlabel("cross-track column")
    axes[1].set_ylabel("median(row-centered image)")
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(score_df["angle_deg"], score_df["score"], color="black", lw=1)
    axes[2].set_title("line-direction residual score")
    axes[2].set_xlabel("signed angle (deg)")
    axes[2].set_ylabel("score")
    axes[2].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "emit_stripe_residual_diagnostics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("emit_nc", type=Path, help="Local EMIT L1B_RAD or L2A_RFL NetCDF4 file")
    parser.add_argument("--dataset", default=None, help="HDF5 dataset path for radiance/reflectance")
    parser.add_argument("--wavelength-dataset", default=None)
    parser.add_argument("--wavelengths", default=None, help="Comma-separated wavelengths to load")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\research\code\outputs_emit_striping_investigation"))
    args = parser.parse_args()

    h5py = require_h5py()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.emit_nc, "r") as h5:
        datasets = list_datasets(h5)
        if args.list_datasets:
            for name, shape, dtype in datasets:
                print(name, shape, dtype)
            return
        cube_path = choose_cube_dataset(datasets, args.dataset)
        cube_ds = h5[cube_path]
        shape = tuple(cube_ds.shape)
        provisional_band_axis = int(np.argmin(shape))
        provisional_n_bands = shape[provisional_band_axis]
        wavelengths = find_wavelengths(h5, provisional_n_bands, args.wavelength_dataset)
        band_axis = infer_band_axis(shape, wavelengths)
        band_idx = parse_wavelengths(args.wavelengths, wavelengths)
        cube = read_selected_bands(cube_ds, band_axis, band_idx)

    image = simple_absorption_index(cube, wavelengths, band_idx)
    col_x, col_profile = column_bias_profile(image)
    angles = np.arange(-80, 80.5, 0.5)
    slopes = np.tan(np.deg2rad(angles))
    score_df = direction_score(image, slopes)

    metrics = pd.DataFrame(
        [
            {
                "input_file": str(args.emit_nc),
                "dataset": cube_path,
                "shape": "x".join(map(str, shape)),
                "band_axis": band_axis,
                "selected_band_indices_0based": ",".join(map(str, band_idx)),
                "selected_wavelengths": ",".join(f"{wavelengths[i]:.3f}" for i in band_idx),
                "image_robust_std": robust_std(image),
                "column_profile_robust_std": robust_std(col_profile),
                "column_profile_p95_abs": float(np.nanpercentile(np.abs(col_profile), 95)),
                "best_direction_angle_deg": float(score_df.loc[score_df["score"].idxmax(), "angle_deg"]),
                "best_direction_score": float(score_df["score"].max()),
            }
        ]
    )
    metrics.to_csv(args.output_dir / "emit_stripe_residual_metrics.csv", index=False)
    pd.DataFrame({"column": col_x, "column_median_residual": col_profile}).to_csv(
        args.output_dir / "emit_column_bias_profile.csv", index=False
    )
    score_df.to_csv(args.output_dir / "emit_direction_scores.csv", index=False)
    plot_outputs(image, score_df, col_x, col_profile, args.output_dir)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
