from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from apply_takaku_wavelet_mf_destriping import (
    BROAD_SLOPE,
    THIN_SLOPE,
    directional_profile,
    robust_std,
)


INPUT_DIR = Path(r"D:\research\code\outputs_detected_slopes_orthogonal_thin_eachiter_then_broad_median")
OUTPUT_DIR = Path(r"D:\research\code\takaku_wavelet_mf_destriping\outputs_broad_dwt_then_thin_median")

THIN_MEDIAN_PRIMARY = (
    "median_thin_eachiter_then_broad_median_stripe_map_thin_"
    "thin_detected_thin_high_alpha_primary_positive_y_minus_x_slope_0.9793_bin_2.0.npy"
)
BROAD_MEDIAN_PRIMARY = (
    "median_thin_eachiter_then_broad_median_stripe_map_broad_"
    "broad_median_broad_offset_primary_positive_y_minus_x_slope_1.2572_bin_18.0.npy"
)


def plot_three_way_profiles() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = np.load(INPUT_DIR / "baseline_no_destripe_alpha_raw.npy").astype(float)
    plume_mask = np.load(INPUT_DIR / "baseline_no_destripe_plume_mask.npy").astype(bool)

    median_thin_only = raw - np.load(INPUT_DIR / THIN_MEDIAN_PRIMARY).astype(float)
    median_broad_only = raw - np.load(INPUT_DIR / BROAD_MEDIAN_PRIMARY).astype(float)

    dwt_thin_only = np.load(OUTPUT_DIR / "takaku_thin_only.npy").astype(float)
    dwt_broad_only = np.load(OUTPUT_DIR / "takaku_broad_only.npy").astype(float)

    broad_cases = {
        "raw": raw,
        "MEDIAN only": median_broad_only,
        "DWT only": dwt_broad_only,
    }
    thin_cases = {
        "raw": raw,
        "MEDIAN only": median_thin_only,
        "DWT only": dwt_thin_only,
    }

    colors = {
        "raw": "tab:blue",
        "MEDIAN only": "tab:orange",
        "DWT only": "tab:green",
    }

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    metric_rows: list[dict[str, str | float]] = []

    for label, image in broad_cases.items():
        x, y = directional_profile(
            image,
            slope=BROAD_SLOPE,
            bin_width=18.0,
            protected_mask=plume_mask,
        )
        axes[0].plot(x, y, label=label, color=colors[label], linewidth=1.8)
        metric_rows.append(
            {
                "direction": "broad_1.257",
                "method": label,
                "profile_robust_std": robust_std(y),
                "raw_reduction_pct": np.nan,
            }
        )

    for label, image in thin_cases.items():
        x, y = directional_profile(
            image,
            slope=THIN_SLOPE,
            bin_width=2.0,
            protected_mask=plume_mask,
        )
        axes[1].plot(x, y, label=label, color=colors[label], linewidth=1.5)
        metric_rows.append(
            {
                "direction": "thin_0.979",
                "method": label,
                "profile_robust_std": robust_std(y),
                "raw_reduction_pct": np.nan,
            }
        )

    for direction in ("broad_1.257", "thin_0.979"):
        raw_std = next(r["profile_robust_std"] for r in metric_rows if r["direction"] == direction and r["method"] == "raw")
        for row in metric_rows:
            if row["direction"] == direction:
                row["raw_reduction_pct"] = 100.0 * (1.0 - float(row["profile_robust_std"]) / float(raw_std))

    axes[0].set_title("Broad-stripe direction: raw vs MEDIAN only vs DWT only")
    axes[0].set_xlabel("line coordinate: y - 1.257x")
    axes[0].set_ylabel("median residual MF")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Thin-stripe direction: raw vs MEDIAN only vs DWT only")
    axes[1].set_xlabel("line coordinate: y - 0.979x")
    axes[1].set_ylabel("median residual MF")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    out_png = OUTPUT_DIR / "raw_median_dwt_only_profiles.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    out_csv = OUTPUT_DIR / "raw_median_dwt_only_profile_metrics.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["direction", "method", "profile_robust_std", "raw_reduction_pct"])
        writer.writeheader()
        writer.writerows(metric_rows)

    print(f"saved: {out_png}")
    print(f"saved: {out_csv}")
    for row in metric_rows:
        print(
            f"{row['direction']} | {row['method']}: "
            f"rstd={float(row['profile_robust_std']):.6g}, "
            f"reduction={float(row['raw_reduction_pct']):.1f}%"
        )


if __name__ == "__main__":
    plot_three_way_profiles()
