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


def corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def column_median_residual(image: np.ndarray) -> np.ndarray:
    row_centered = image - np.nanmedian(image, axis=1, keepdims=True)
    return np.nanmedian(row_centered, axis=0)


def read_target_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(path)
    wave_col = next((c for c in table.columns if "wave" in c.lower()), table.columns[0])
    value_cols = [c for c in table.columns if c != wave_col]
    if not value_cols:
        raise SystemExit(f"No target value column found in {path}")
    return table[wave_col].to_numpy(float), table[value_cols[0]].to_numpy(float)


def continuum_residual_operator(wavelengths: np.ndarray) -> np.ndarray:
    x = wavelengths.astype(float)
    x = (x - np.nanmean(x)) / np.nanstd(x)
    design = np.column_stack([np.ones_like(x), x])
    projection = design @ np.linalg.pinv(design)
    return np.eye(wavelengths.size) - projection


def transform_spectra(flat_spectra: np.ndarray, residual_op: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.all(np.isfinite(flat_spectra), axis=1) & np.all(flat_spectra > 0, axis=1)
    out = np.full(flat_spectra.shape, np.nan, dtype=np.float32)
    if valid.any():
        logged = np.log(flat_spectra[valid].astype(np.float64))
        out[valid] = (logged @ residual_op).astype(np.float32)
    return out, valid


def read_post_chunk(dataset, row0: int, row1: int, band_idx: np.ndarray) -> np.ndarray:
    chunk = np.asarray(dataset[row0:row1, :, band_idx], dtype=np.float32)
    chunk[chunk <= -9990] = np.nan
    return chunk


def collect_background_sample(
    dataset,
    band_idx: np.ndarray,
    residual_op: np.ndarray,
    max_sample: int,
    chunk_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_rows = dataset.shape[0]
    collected: list[np.ndarray] = []
    per_chunk = max(2000, max_sample // max(1, int(np.ceil(n_rows / chunk_rows))))
    for row0 in range(0, n_rows, chunk_rows):
        row1 = min(n_rows, row0 + chunk_rows)
        chunk = read_post_chunk(dataset, row0, row1, band_idx)
        spectra = chunk.reshape(-1, chunk.shape[-1])
        transformed, valid = transform_spectra(spectra, residual_op)
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            continue
        if idx.size > per_chunk:
            idx = rng.choice(idx, size=per_chunk, replace=False)
        collected.append(transformed[idx])
    if not collected:
        raise SystemExit("No valid background spectra were found.")
    sample = np.vstack(collected)
    if sample.shape[0] > max_sample:
        keep = rng.choice(sample.shape[0], size=max_sample, replace=False)
        sample = sample[keep]
    return sample.astype(np.float64)


def matched_filter_weights(
    sample: np.ndarray,
    target: np.ndarray,
    shrinkage: float,
    regularization: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mu = np.nanmean(sample, axis=0)
    centered = sample - mu
    cov = np.cov(centered, rowvar=False)
    diag = np.diag(cov).copy()
    diag = np.where(np.isfinite(diag) & (diag > 0), diag, np.nanmedian(diag[diag > 0]))
    cov = (1.0 - shrinkage) * cov + shrinkage * np.diag(diag)
    scale = float(np.nanmedian(diag[diag > 0]))
    cov = cov + regularization * scale * np.eye(cov.shape[0])
    weights = np.linalg.solve(cov, target)
    denom = float(target @ weights)
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        raise SystemExit("Matched-filter denominator is too small; target/covariance is ill-conditioned.")
    return weights.astype(np.float64), denom, mu.astype(np.float64), cov.astype(np.float64)


def apply_mf_to_transformed(
    transformed: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
    denom: float,
    mu: np.ndarray,
) -> np.ndarray:
    alpha = np.full(transformed.shape[0], np.nan, dtype=np.float32)
    if valid.any():
        alpha[valid] = (((transformed[valid].astype(np.float64) - mu) @ weights) / denom).astype(np.float32)
    return alpha


def image_limits(*images: np.ndarray, pct: tuple[float, float] = (2, 98)) -> tuple[float, float]:
    vals = np.concatenate([img[np.isfinite(img)].reshape(-1) for img in images if np.isfinite(img).any()])
    lo, hi = np.nanpercentile(vals, pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return -1.0, 1.0
    return float(lo), float(hi)


def symmetric_limits(image: np.ndarray, pct: float = 98) -> tuple[float, float]:
    vals = image[np.isfinite(image)]
    if vals.size == 0:
        return -1.0, 1.0
    scale = float(np.nanpercentile(np.abs(vals - np.nanmedian(vals)), pct))
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return -scale, scale


def plot_outputs(
    out_dir: Path,
    pre_alpha: np.ndarray,
    post_alpha: np.ndarray,
    delta_alpha: np.ndarray,
    col_profiles: pd.DataFrame,
    weights_table: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    vmin, vmax = image_limits(pre_alpha, post_alpha)
    dmin, dmax = symmetric_limits(delta_alpha)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    im0 = axes[0, 0].imshow(pre_alpha, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    axes[0, 0].set_title("pseudo pre-destripe MF")
    axes[0, 0].set_xlabel("cross-track column")
    axes[0, 0].set_ylabel("down-track row")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(post_alpha, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    axes[0, 1].set_title("distributed post-destripe MF")
    axes[0, 1].set_xlabel("cross-track column")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(delta_alpha, cmap="RdBu_r", vmin=dmin, vmax=dmax, origin="upper")
    axes[0, 2].set_title("pre - post MF")
    axes[0, 2].set_xlabel("cross-track column")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].plot(col_profiles["column"], col_profiles["pre_column_profile"], color="tab:red", lw=1.0, label="pseudo pre")
    axes[1, 0].plot(col_profiles["column"], col_profiles["post_column_profile"], color="black", lw=1.0, label="post")
    axes[1, 0].set_title("MF column median residual")
    axes[1, 0].set_xlabel("cross-track column")
    axes[1, 0].set_ylabel("row-centered median")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(col_profiles["column"], col_profiles["pre_minus_post_profile"], color="tab:blue", lw=1.0, label="measured pre - post")
    axes[1, 1].plot(
        col_profiles["column"],
        col_profiles["flat_predicted_profile"],
        color="tab:orange",
        lw=1.0,
        alpha=0.8,
        label="predicted from flat field",
    )
    axes[1, 1].set_title("Flat-field-predicted MF stripe")
    axes[1, 1].set_xlabel("cross-track column")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(frameon=False)

    axes[1, 2].plot(weights_table["wavelength_nm"], weights_table["target_residual"], color="tab:green", lw=1.0, label="target residual")
    ax2 = axes[1, 2].twinx()
    ax2.plot(weights_table["wavelength_nm"], weights_table["mf_weight"], color="tab:purple", lw=0.9, alpha=0.8, label="MF weight")
    axes[1, 2].set_title("Target and MF weights")
    axes[1, 2].set_xlabel("wavelength (nm)")
    axes[1, 2].grid(True, alpha=0.25)
    lines1, labels1 = axes[1, 2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 2].legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8)

    title = (
        "EMIT pseudo-pre vs post MF; "
        f"column RSTD pre={metrics.loc[0, 'pre_column_robust_std']:.4g}, "
        f"post={metrics.loc[0, 'post_column_robust_std']:.4g}"
    )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_dir / "emit_mf_pre_post_overview.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(col_profiles["column"], col_profiles["pre_column_profile"], color="tab:red", lw=1.0, label="pseudo pre")
    axes[0].plot(col_profiles["column"], col_profiles["post_column_profile"], color="black", lw=1.0, label="post")
    axes[0].set_ylabel("MF column residual")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(col_profiles["column"], col_profiles["pre_minus_post_profile"], color="tab:blue", lw=1.0, label="measured pre - post")
    axes[1].plot(col_profiles["column"], col_profiles["flat_predicted_profile"], color="tab:orange", lw=1.0, label="predicted from flat field")
    axes[1].set_xlabel("cross-track column")
    axes[1].set_ylabel("MF delta")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "emit_mf_column_profiles.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the same log-continuum matched filter to EMIT distributed radiance "
            "and to a pseudo pre-destripe radiance reconstructed as radiance / flat_field_update."
        )
    )
    parser.add_argument("l1b_nc", type=Path, help="EMIT L1B_RAD NetCDF4 file")
    parser.add_argument("--target-csv", type=Path, required=True, help="CSV with wavelength and target radiance-difference columns")
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\research\code\outputs_emit_mf_pre_post"))
    parser.add_argument("--min-wavelength", type=float, default=2000.0)
    parser.add_argument("--max-wavelength", type=float, default=2475.0)
    parser.add_argument("--max-sample", type=int, default=100_000)
    parser.add_argument("--chunk-rows", type=int, default=64)
    parser.add_argument("--shrinkage", type=float, default=0.05)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    target_wave, target_values = read_target_csv(args.target_csv)
    with h5py.File(args.l1b_nc, "r") as h5:
        wavelengths = np.asarray(h5["sensor_band_parameters/wavelengths"][...], dtype=float)
        band_idx = np.where((wavelengths >= args.min_wavelength) & (wavelengths <= args.max_wavelength))[0]
        band_idx = band_idx[(wavelengths[band_idx] >= np.nanmin(target_wave)) & (wavelengths[band_idx] <= np.nanmax(target_wave))]
        if band_idx.size < 5:
            raise SystemExit("Too few overlapping EMIT bands for the requested target wavelength range.")
        selected_waves = wavelengths[band_idx]
        target_interp = np.interp(selected_waves, target_wave, target_values)
        residual_op = continuum_residual_operator(selected_waves)
        target_residual = (target_interp.astype(np.float64) @ residual_op).astype(np.float64)
        target_norm = np.linalg.norm(target_residual)
        if not np.isfinite(target_norm) or target_norm == 0:
            raise SystemExit("Target residual has zero norm after continuum removal.")
        target_residual = target_residual / target_norm

        radiance = h5["radiance"]
        flat = np.asarray(h5["flat_field_update"][:, band_idx], dtype=np.float32)
        flat[flat <= 0] = np.nan

        sample = collect_background_sample(
            radiance,
            band_idx,
            residual_op,
            max_sample=args.max_sample,
            chunk_rows=args.chunk_rows,
            rng=rng,
        )
        weights, denom, mu, cov = matched_filter_weights(
            sample,
            target_residual,
            shrinkage=args.shrinkage,
            regularization=args.regularization,
        )

        n_rows, n_cols = radiance.shape[:2]
        post_alpha = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        pre_alpha = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

        for row0 in range(0, n_rows, args.chunk_rows):
            row1 = min(n_rows, row0 + args.chunk_rows)
            post_chunk = read_post_chunk(radiance, row0, row1, band_idx)
            pre_chunk = post_chunk / flat[None, :, :]

            post_spectra = post_chunk.reshape(-1, post_chunk.shape[-1])
            pre_spectra = pre_chunk.reshape(-1, pre_chunk.shape[-1])
            post_transformed, post_valid = transform_spectra(post_spectra, residual_op)
            pre_transformed, pre_valid = transform_spectra(pre_spectra, residual_op)
            post_alpha[row0:row1] = apply_mf_to_transformed(post_transformed, post_valid, weights, denom, mu).reshape(row1 - row0, n_cols)
            pre_alpha[row0:row1] = apply_mf_to_transformed(pre_transformed, pre_valid, weights, denom, mu).reshape(row1 - row0, n_cols)

    delta_alpha = pre_alpha - post_alpha
    pre_profile = column_median_residual(pre_alpha)
    post_profile = column_median_residual(post_alpha)
    delta_profile = column_median_residual(delta_alpha)

    delta_log_flat = -np.log(flat.astype(np.float64))
    flat_residual = delta_log_flat @ residual_op
    flat_pred = (flat_residual @ weights) / denom
    flat_pred_profile = flat_pred - np.nanmedian(flat_pred)

    col_profiles = pd.DataFrame(
        {
            "column": np.arange(pre_alpha.shape[1]),
            "pre_column_profile": pre_profile,
            "post_column_profile": post_profile,
            "pre_minus_post_profile": delta_profile,
            "flat_predicted_profile": flat_pred_profile,
        }
    )
    metrics = pd.DataFrame(
        [
            {
                "input_file": str(args.l1b_nc),
                "target_csv": str(args.target_csv),
                "n_bands": int(band_idx.size),
                "min_wavelength_nm": float(selected_waves.min()),
                "max_wavelength_nm": float(selected_waves.max()),
                "background_sample_size": int(sample.shape[0]),
                "pre_image_robust_std": robust_std(pre_alpha),
                "post_image_robust_std": robust_std(post_alpha),
                "pre_column_robust_std": robust_std(pre_profile),
                "post_column_robust_std": robust_std(post_profile),
                "column_reduction_fraction": 1.0 - robust_std(post_profile) / robust_std(pre_profile),
                "pre_column_p95_abs": float(np.nanpercentile(np.abs(pre_profile), 95)),
                "post_column_p95_abs": float(np.nanpercentile(np.abs(post_profile), 95)),
                "pre_minus_post_column_robust_std": robust_std(delta_profile),
                "flat_predicted_column_robust_std": robust_std(flat_pred_profile),
                "pre_minus_post_corr_with_flat_prediction": corr(delta_profile, flat_pred_profile),
                "pre_column_corr_with_flat_prediction": corr(pre_profile, flat_pred_profile),
                "post_column_corr_with_flat_prediction": corr(post_profile, flat_pred_profile),
                "mf_denom": denom,
                "cov_condition_number": float(np.linalg.cond(cov)),
            }
        ]
    )

    weights_table = pd.DataFrame(
        {
            "band_idx0": band_idx,
            "wavelength_nm": selected_waves,
            "target_interpolated": target_interp,
            "target_residual": target_residual,
            "mf_weight": weights,
            "flat_field_median": np.nanmedian(flat, axis=0),
            "flat_field_robust_std": [robust_std(flat[:, j]) for j in range(flat.shape[1])],
        }
    )

    metrics.to_csv(args.output_dir / "emit_mf_pre_post_metrics.csv", index=False)
    col_profiles.to_csv(args.output_dir / "emit_mf_column_profiles.csv", index=False)
    weights_table.to_csv(args.output_dir / "emit_mf_band_weights.csv", index=False)
    np.savez_compressed(
        args.output_dir / "emit_mf_products.npz",
        pre_alpha=pre_alpha,
        post_alpha=post_alpha,
        delta_alpha=delta_alpha,
        selected_wavelengths=selected_waves,
        band_idx0=band_idx,
        target_residual=target_residual,
        mf_weight=weights,
    )
    plot_outputs(args.output_dir, pre_alpha, post_alpha, delta_alpha, col_profiles, weights_table, metrics)

    print(metrics.to_string(index=False))
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
