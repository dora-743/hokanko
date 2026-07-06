from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
import tifffile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ALPHA = Path(
    r"D:\research\code\outputs_paper_sensor_geometry_destripe\baseline_no_destripe_alpha_corrected.npy"
)
DEFAULT_OUTPUT_DIR = Path(r"D:\research\code\outputs_l1g_band_directionality")
KEY_ANGLES = {
    "thin_plus44": 44.4,
    "at_footprint_plus44": 44.34,
    "broad_plus51": 51.5,
    "ct_footprint_minus52": -51.71,
    "ct_mirror_plus52": 51.71,
}


def nan_gaussian(values: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    filled = np.where(valid, values, 0.0)
    weight = valid.astype(float)
    smooth_num = ndimage.gaussian_filter(filled, sigma=sigma, mode="nearest")
    smooth_den = ndimage.gaussian_filter(weight, sigma=sigma, mode="nearest")
    return np.where(smooth_den > 1e-6, smooth_num / smooth_den, np.nan)


def robust_z(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    vals = values[valid & np.isfinite(values)]
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med))
    sigma = 1.4826 * mad if mad > 0 else np.nanstd(vals)
    return (values - med) / sigma


def line_ids(rows: np.ndarray, cols: np.ndarray, signed_slope: float, bin_width: float) -> np.ndarray:
    if signed_slope >= 0:
        coord = rows - signed_slope * cols
    else:
        coord = rows + abs(signed_slope) * cols
    return np.rint(coord / bin_width).astype(np.int64)


def weighted_line_p99_score(
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    slopes: np.ndarray,
    bin_width: float = 18.0,
    min_count: int = 40,
) -> np.ndarray:
    scores = np.zeros(len(slopes), dtype=float)
    for i, slope in enumerate(slopes):
        ids = line_ids(rows, cols, slope, bin_width)
        ids = ids - ids.min()
        counts = np.bincount(ids)
        sums = np.bincount(ids, weights=weights)
        ok = counts >= min_count
        if ok.any():
            scores[i] = np.nanpercentile(np.abs(sums[ok] / counts[ok]), 99)
    return scores


def read_selected_bands(tif_path: Path, band_indices0: list[int]) -> np.ndarray:
    with tifffile.TiffFile(str(tif_path)) as tif:
        page = tif.pages[0]
        height, width, samples = page.shape
        tile_h = int(page.tilelength)
        tile_w = int(page.tilewidth)
        ntiles_x = math.ceil(width / tile_w)
        offsets = page.dataoffsets
        bytecounts = page.databytecounts
        out = np.empty((len(band_indices0), height, width), dtype=np.uint16)
        with open(tif_path, "rb") as fh:
            for tile_i, (offset, bytecount) in enumerate(zip(offsets, bytecounts)):
                ty = tile_i // ntiles_x
                tx = tile_i % ntiles_x
                r0 = ty * tile_h
                c0 = tx * tile_w
                r1 = min(r0 + tile_h, height)
                c1 = min(c0 + tile_w, width)
                fh.seek(offset)
                buf = fh.read(bytecount)
                tile = np.frombuffer(buf, dtype="<u2").reshape(tile_h, tile_w, samples)
                selected = tile[: r1 - r0, : c1 - c0, band_indices0]
                out[:, r0:r1, c0:c1] = np.moveaxis(selected, -1, 0)
    return out


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    out = np.full_like(values, np.nan, dtype=float)
    if not finite.any():
        return out
    lo, hi = np.nanpercentile(values[finite], [1, 99])
    if hi <= lo:
        out[finite] = 0.0
    else:
        out[finite] = np.clip((values[finite] - lo) / (hi - lo), 0, 1)
    return out


def score_band(refl: np.ndarray, alpha_valid: np.ndarray, slopes: np.ndarray) -> np.ndarray:
    valid = alpha_valid & np.isfinite(refl) & (refl > -0.5) & (refl < 2.0)
    smooth = nan_gaussian(refl, valid, sigma=28.0)
    hp = refl - smooth
    hp_z = robust_z(hp, valid)
    refl_z = robust_z(refl, valid)
    keep = np.zeros_like(valid, dtype=bool)
    keep[::4, ::4] = True
    sample = valid & keep & (np.abs(refl_z) < 3.5)
    rows, cols = np.nonzero(sample)
    weights = hp_z[rows, cols]
    return weighted_line_p99_score(rows, cols, weights, slopes)


def nearest_score(curve_df: pd.DataFrame, angle: float) -> float:
    idx = (curve_df["angle_deg"] - angle).abs().idxmin()
    return float(curve_df.loc[idx, "score"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif", type=Path, required=True)
    parser.add_argument("--band-csv", type=Path, required=True)
    parser.add_argument("--alpha", type=Path, default=DEFAULT_ALPHA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bands",
        type=str,
        default="59,67,115,147,155,163,171,175,179,185",
        help="1-based HISUI band numbers, comma-separated",
    )
    parser.add_argument("--angle-step-deg", type=float, default=0.5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    band_numbers = [int(x.strip()) for x in args.bands.split(",") if x.strip()]
    band_indices0 = [b - 1 for b in band_numbers]
    band_df = pd.read_csv(args.band_csv)
    alpha_valid = np.isfinite(np.load(args.alpha))

    raw = read_selected_bands(args.tif, band_indices0)
    angles = np.arange(-80.0, 80.0 + args.angle_step_deg / 2, args.angle_step_deg)
    slopes = np.tan(np.deg2rad(angles))

    curve_rows = []
    summary_rows = []
    for i, band_no in enumerate(band_numbers):
        info = band_df.loc[band_df["BandNo"].eq(band_no)].iloc[0]
        refl = raw[i].astype(np.float32) * float(info["ReflectanceMulti"]) + float(info["ReflectanceAdd"])
        score = score_band(refl, alpha_valid, slopes)
        curve = pd.DataFrame({"angle_deg": angles, "signed_slope": slopes, "score": score})
        curve["band_no"] = band_no
        curve["center_wavelength_nm"] = float(info["CenterWavelengthNanometer"])
        curve_rows.append(curve)

        pos_window = curve[(curve["angle_deg"] >= 35) & (curve["angle_deg"] <= 65)]
        neg_window = curve[(curve["angle_deg"] >= -65) & (curve["angle_deg"] <= -35)]
        pos_peak = pos_window.loc[pos_window["score"].idxmax()]
        neg_peak = neg_window.loc[neg_window["score"].idxmax()]
        row = {
            "band_no": band_no,
            "center_wavelength_nm": float(info["CenterWavelengthNanometer"]),
            "pos_35_65_peak_angle_deg": float(pos_peak["angle_deg"]),
            "pos_35_65_peak_score": float(pos_peak["score"]),
            "neg_65_35_peak_angle_deg": float(neg_peak["angle_deg"]),
            "neg_65_35_peak_score": float(neg_peak["score"]),
        }
        for label, angle in KEY_ANGLES.items():
            row[f"score_at_{label}"] = nearest_score(curve, angle)
        summary_rows.append(row)

    curves = pd.concat(curve_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    curves.to_csv(args.output_dir / "band_direction_score_curves.csv", index=False)
    summary.to_csv(args.output_dir / "band_direction_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    for band_no in band_numbers:
        sub = curves[curves["band_no"].eq(band_no)]
        label = f"B{band_no} {sub['center_wavelength_nm'].iloc[0]:.1f} nm"
        ax.plot(sub["angle_deg"], normalize01(sub["score"]), lw=1.0, alpha=0.8, label=label)
    for label, angle in KEY_ANGLES.items():
        ax.axvline(angle, lw=1.2, alpha=0.35, label=label)
    ax.set_xlabel("signed image angle (deg)")
    ax.set_ylabel("normalized residual line-mean score")
    ax.set_title("L1G single-band direction scores before MF")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "band_direction_score_curves.png", dpi=180)
    plt.close(fig)

    print("Wrote", args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
