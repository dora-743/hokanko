from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def robust_std(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return np.nan
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med))
    if mad > 0:
        return float(1.4826 * mad)
    return float(np.nanstd(vals))


def column_median_residual(image: np.ndarray) -> np.ndarray:
    row_centered = image - np.nanmedian(image, axis=1, keepdims=True)
    return np.nanmedian(row_centered, axis=0)


def column_log_residual(image: np.ndarray) -> np.ndarray:
    positive = np.where(image > 0, image, np.nan)
    logged = np.log(positive)
    row_centered = logged - np.nanmedian(logged, axis=1, keepdims=True)
    return np.nanmedian(row_centered, axis=0)


def nearest_band_indices(wavelengths: np.ndarray, requested_nm: list[float]) -> np.ndarray:
    return np.asarray(
        sorted({int(np.nanargmin(np.abs(wavelengths - wave))) for wave in requested_nm}),
        dtype=int,
    )


def simple_absorption_index(cube: np.ndarray, wavelengths: np.ndarray, band_idx: np.ndarray) -> np.ndarray:
    selected_waves = wavelengths[band_idx]
    left = np.where((selected_waves >= 2100) & (selected_waves <= 2200))[0]
    absorption = np.where((selected_waves >= 2280) & (selected_waves <= 2360))[0]
    right = np.where((selected_waves >= 2380) & (selected_waves <= 2450))[0]
    if left.size == 0 or absorption.size == 0 or right.size == 0:
        return cube[:, :, cube.shape[2] // 2]
    continuum = 0.5 * (np.nanmedian(cube[:, :, left], axis=2) + np.nanmedian(cube[:, :, right], axis=2))
    center = np.nanmedian(cube[:, :, absorption], axis=2)
    return continuum - center


def corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("l1b_nc", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\research\code\outputs_emit_flatfield_before_after"))
    parser.add_argument(
        "--wavelengths",
        default="1650,2000,2100,2200,2300,2350,2400",
        help="Comma-separated wavelengths to compare",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested = [float(x.strip()) for x in args.wavelengths.split(",") if x.strip()]
    with h5py.File(args.l1b_nc, "r") as h5:
        wavelengths = np.asarray(h5["sensor_band_parameters/wavelengths"][...], dtype=float)
        band_idx = nearest_band_indices(wavelengths, requested)
        post = np.asarray(h5["radiance"][:, :, band_idx], dtype=np.float32)
        flat = np.asarray(h5["flat_field_update"][:, band_idx], dtype=np.float32)

    post[post <= -9990] = np.nan
    flat[flat <= 0] = np.nan
    pre = post / flat[None, :, :]

    rows = []
    profile_frames = []
    for j, band in enumerate(band_idx):
        post_prof = column_median_residual(post[:, :, j])
        pre_prof = column_median_residual(pre[:, :, j])
        post_log_prof = column_log_residual(post[:, :, j])
        pre_log_prof = column_log_residual(pre[:, :, j])
        flat_prof = flat[:, j]
        flat_dev = flat_prof - np.nanmedian(flat_prof)
        row = {
            "band_idx0": int(band),
            "wavelength_nm": float(wavelengths[band]),
            "flat_median": float(np.nanmedian(flat_prof)),
            "flat_robust_std": robust_std(flat_prof),
            "flat_p01": float(np.nanpercentile(flat_prof, 1)),
            "flat_p99": float(np.nanpercentile(flat_prof, 99)),
            "pre_column_robust_std": robust_std(pre_prof),
            "post_column_robust_std": robust_std(post_prof),
            "reduction_fraction": 1.0 - robust_std(post_prof) / robust_std(pre_prof),
            "pre_abs_p95": float(np.nanpercentile(np.abs(pre_prof), 95)),
            "post_abs_p95": float(np.nanpercentile(np.abs(post_prof), 95)),
            "pre_log_column_robust_std": robust_std(pre_log_prof),
            "post_log_column_robust_std": robust_std(post_log_prof),
            "log_reduction_fraction": 1.0 - robust_std(post_log_prof) / robust_std(pre_log_prof),
            "pre_log_abs_p95": float(np.nanpercentile(np.abs(pre_log_prof), 95)),
            "post_log_abs_p95": float(np.nanpercentile(np.abs(post_log_prof), 95)),
            "pre_log_profile_corr_with_inverse_flat": corr(pre_log_prof, 1.0 / flat_prof),
            "post_log_profile_corr_with_inverse_flat": corr(post_log_prof, 1.0 / flat_prof),
            "pre_profile_corr_with_inverse_flat": corr(pre_prof, 1.0 / flat_prof),
            "post_profile_corr_with_inverse_flat": corr(post_prof, 1.0 / flat_prof),
            "pre_minus_post_corr_with_inverse_flat": corr(pre_prof - post_prof, 1.0 / flat_prof),
            "pre_minus_post_corr_with_flat_dev": corr(pre_prof - post_prof, flat_dev),
        }
        rows.append(row)
        profile_frames.append(
            pd.DataFrame(
                {
                    "column": np.arange(post.shape[1]),
                    "band_idx0": int(band),
                    "wavelength_nm": float(wavelengths[band]),
                    "pre_column_profile": pre_prof,
                    "post_column_profile": post_prof,
                    "pre_minus_post_profile": pre_prof - post_prof,
                    "pre_log_column_profile": pre_log_prof,
                    "post_log_column_profile": post_log_prof,
                    "pre_minus_post_log_profile": pre_log_prof - post_log_prof,
                    "flat_field_update": flat_prof,
                    "inverse_flat": 1.0 / flat_prof,
                }
            )
        )

    summary = pd.DataFrame(rows)
    profiles = pd.concat(profile_frames, ignore_index=True)
    summary.to_csv(args.output_dir / "emit_flatfield_before_after_band_summary.csv", index=False)
    profiles.to_csv(args.output_dir / "emit_flatfield_before_after_column_profiles.csv", index=False)

    pre_img = simple_absorption_index(pre, wavelengths, band_idx)
    post_img = simple_absorption_index(post, wavelengths, band_idx)
    pre_img_prof = column_median_residual(pre_img)
    post_img_prof = column_median_residual(post_img)
    pre_img_log_prof = column_log_residual(np.abs(pre_img) + np.nanpercentile(np.abs(pre_img), 5))
    post_img_log_prof = column_log_residual(np.abs(post_img) + np.nanpercentile(np.abs(post_img), 5))
    image_summary = pd.DataFrame(
        [
            {
                "product": "simple_absorption_index",
                "pre_column_robust_std": robust_std(pre_img_prof),
                "post_column_robust_std": robust_std(post_img_prof),
                "reduction_fraction": 1.0 - robust_std(post_img_prof) / robust_std(pre_img_prof),
                "pre_abs_p95": float(np.nanpercentile(np.abs(pre_img_prof), 95)),
                "post_abs_p95": float(np.nanpercentile(np.abs(post_img_prof), 95)),
                "pre_log_column_robust_std": robust_std(pre_img_log_prof),
                "post_log_column_robust_std": robust_std(post_img_log_prof),
            }
        ]
    )
    image_summary.to_csv(args.output_dir / "emit_flatfield_before_after_image_summary.csv", index=False)
    pd.DataFrame(
        {
            "column": np.arange(post.shape[1]),
            "pre_column_profile": pre_img_prof,
            "post_column_profile": post_img_prof,
            "pre_minus_post_profile": pre_img_prof - post_img_prof,
            "pre_log_column_profile": pre_img_log_prof,
            "post_log_column_profile": post_img_log_prof,
        }
    ).to_csv(args.output_dir / "emit_flatfield_before_after_image_profiles.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for j, band in enumerate(band_idx):
        wave = wavelengths[band]
        if wave < 1950:
            continue
        sub = profiles[profiles["band_idx0"].eq(int(band))]
        axes[0].plot(sub["column"], sub["pre_column_profile"], lw=0.8, alpha=0.65, label=f"{wave:.0f} nm pre")
        axes[0].plot(sub["column"], sub["post_column_profile"], lw=0.8, alpha=0.65, linestyle="--", label=f"{wave:.0f} nm post")
        axes[1].plot(sub["column"], sub["pre_minus_post_profile"], lw=0.8, alpha=0.75, label=f"{wave:.0f} nm")
        axes[2].plot(sub["column"], sub["flat_field_update"], lw=0.8, alpha=0.75, label=f"{wave:.0f} nm")
    axes[0].set_ylabel("column residual")
    axes[0].set_title("Column residual profiles before/after applying EMIT flat_field_update")
    axes[1].set_ylabel("pre - post")
    axes[2].set_ylabel("flat field")
    axes[2].set_xlabel("cross-track column")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "emit_flatfield_before_after_band_profiles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for j, band in enumerate(band_idx):
        wave = wavelengths[band]
        if wave < 1800:
            continue
        sub = profiles[profiles["band_idx0"].eq(int(band))]
        axes[0].plot(sub["column"], sub["pre_log_column_profile"], lw=0.8, alpha=0.65, label=f"{wave:.0f} nm pre")
        axes[0].plot(sub["column"], sub["post_log_column_profile"], lw=0.8, alpha=0.65, linestyle="--", label=f"{wave:.0f} nm post")
        axes[1].plot(sub["column"], sub["pre_minus_post_log_profile"], lw=0.8, alpha=0.75, label=f"{wave:.0f} nm")
        axes[2].plot(sub["column"], -np.log(sub["flat_field_update"]), lw=0.8, alpha=0.75, label=f"{wave:.0f} nm")
    axes[0].set_ylabel("log column residual")
    axes[0].set_title("Relative/log column residual profiles before/after flat-field destriping")
    axes[1].set_ylabel("pre - post log")
    axes[2].set_ylabel("-log(flat)")
    axes[2].set_xlabel("cross-track column")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "emit_flatfield_before_after_log_profiles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    valid = np.isfinite(post_img)
    lo, hi = np.nanpercentile(post_img[valid], [2, 98])
    axes[0].imshow(pre_img, cmap="gray", vmin=lo, vmax=hi, origin="upper")
    axes[0].set_title("pseudo pre-destripe image")
    axes[1].imshow(post_img, cmap="gray", vmin=lo, vmax=hi, origin="upper")
    axes[1].set_title("distributed post-destripe image")
    axes[2].plot(pre_img_prof, label="pseudo pre", color="tab:red")
    axes[2].plot(post_img_prof, label="post", color="black")
    axes[2].set_title("column median residual")
    axes[2].set_xlabel("cross-track column")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "emit_flatfield_before_after_image.png", dpi=180)
    plt.close(fig)

    print(summary.to_string(index=False))
    print(image_summary.to_string(index=False))


if __name__ == "__main__":
    main()
