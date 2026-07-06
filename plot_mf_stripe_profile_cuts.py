from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(r"D:\research\code\outputs_paper_sensor_geometry_destripe")
DEFAULT_OUTPUT_DIR = Path(r"D:\research\code\outputs_mf_stripe_profile_cuts")


@dataclass
class Profile:
    name: str
    family: str
    x: np.ndarray
    row: np.ndarray
    col: np.ndarray
    baseline: np.ndarray
    corrected: np.ndarray
    stripe_total: np.ndarray
    stripe_angle: np.ndarray
    valid: np.ndarray


def robust_std(values: np.ndarray) -> float:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return np.nan
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med))
    if mad > 0:
        return float(1.4826 * mad)
    return float(np.nanstd(vals))


def nan_gaussian1d(values: np.ndarray, sigma: float) -> np.ndarray:
    valid = np.isfinite(values)
    if valid.sum() < 5:
        return np.full_like(values, np.nan, dtype=float)
    filled = np.where(valid, values, 0.0)
    weight = valid.astype(float)
    num = ndimage.gaussian_filter1d(filled, sigma=sigma, mode="nearest")
    den = ndimage.gaussian_filter1d(weight, sigma=sigma, mode="nearest")
    return np.where(den > 1e-6, num / den, np.nan)


def highpass(values: np.ndarray, sigma: float = 45.0) -> np.ndarray:
    return values - nan_gaussian1d(values, sigma=sigma)


def load_arrays(base_dir: Path) -> dict[str, np.ndarray]:
    paths = {
        "baseline": base_dir / "baseline_no_destripe_alpha_corrected.npy",
        "corrected": base_dir / "paper_sensor_line_then_column_each_iter_angle125_focus_alpha_corrected.npy",
        "stripe_total": base_dir / "paper_sensor_line_then_column_each_iter_angle125_focus_stripe_map_total.npy",
        "stripe_angle": base_dir
        / "paper_sensor_line_then_column_each_iter_angle125_focus_stripe_map_angle_sweep_y_minus_x_slope_1.2520.npy",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required arrays:\n" + "\n".join(missing))
    return {key: np.load(path) for key, path in paths.items()}


def profile_from_row(arrays: dict[str, np.ndarray], row: int) -> Profile:
    baseline = arrays["baseline"][row, :].astype(float)
    corrected = arrays["corrected"][row, :].astype(float)
    stripe_total = arrays["stripe_total"][row, :].astype(float)
    stripe_angle = arrays["stripe_angle"][row, :].astype(float)
    valid = np.isfinite(baseline) & np.isfinite(corrected)
    cols = np.arange(baseline.size, dtype=float)
    return Profile(
        name=f"row_{row:04d}",
        family="y_const",
        x=cols,
        row=np.full_like(cols, row, dtype=float),
        col=cols,
        baseline=baseline,
        corrected=corrected,
        stripe_total=stripe_total,
        stripe_angle=stripe_angle,
        valid=valid,
    )


def sample_array(array: np.ndarray, rows: np.ndarray, cols: np.ndarray, order: int = 1) -> np.ndarray:
    return ndimage.map_coordinates(
        array.astype(float),
        [rows, cols],
        order=order,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )


def make_direction_profiles(
    arrays: dict[str, np.ndarray],
    family: str,
    slope: float,
    n_profiles: int,
    quantile_low: float = 0.18,
    quantile_high: float = 0.82,
) -> list[Profile]:
    baseline = arrays["baseline"]
    valid_mask = np.isfinite(baseline) & np.isfinite(arrays["corrected"])
    valid_rows, valid_cols = np.nonzero(valid_mask)
    center = np.array([np.nanmedian(valid_rows), np.nanmedian(valid_cols)], dtype=float)

    v = np.array([slope, 1.0], dtype=float)
    v /= np.linalg.norm(v)
    n = np.array([-1.0, slope], dtype=float)
    n /= np.linalg.norm(n)

    points = np.column_stack([valid_rows.astype(float), valid_cols.astype(float)])
    rel = points - center
    offsets = rel @ n
    ts = rel @ v
    chosen_offsets = np.quantile(offsets, np.linspace(quantile_low, quantile_high, n_profiles))
    t_grid = np.arange(np.floor(ts.min()) - 20, np.ceil(ts.max()) + 21, 1.0)

    profiles: list[Profile] = []
    for i, offset in enumerate(chosen_offsets, start=1):
        base_point = center + offset * n
        rows = base_point[0] + t_grid * v[0]
        cols = base_point[1] + t_grid * v[1]
        inside = (rows >= 0) & (rows <= baseline.shape[0] - 1) & (cols >= 0) & (cols <= baseline.shape[1] - 1)
        sampled_valid = sample_array(valid_mask.astype(float), rows, cols, order=0) > 0.5
        valid = inside & sampled_valid
        prof = Profile(
            name=f"{family}_{i:02d}",
            family=family,
            x=t_grid,
            row=rows,
            col=cols,
            baseline=sample_array(arrays["baseline"], rows, cols),
            corrected=sample_array(arrays["corrected"], rows, cols),
            stripe_total=sample_array(arrays["stripe_total"], rows, cols),
            stripe_angle=sample_array(arrays["stripe_angle"], rows, cols),
            valid=valid,
        )
        profiles.append(prof)
    return profiles


def finite_segments(valid: np.ndarray, min_len: int = 30) -> list[slice]:
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [slice(int(s), int(e) + 1) for s, e in zip(starts, ends) if e - s + 1 >= min_len]


def longest_valid_slice(profile: Profile) -> slice | None:
    segs = finite_segments(profile.valid)
    if not segs:
        return None
    return max(segs, key=lambda s: s.stop - s.start)


def profile_metrics(profile: Profile) -> dict[str, float | str | int]:
    seg = longest_valid_slice(profile)
    if seg is None:
        return {"family": profile.family, "profile": profile.name, "n_valid": 0}
    b = profile.baseline[seg]
    c = profile.corrected[seg]
    st = profile.stripe_total[seg]
    sa = profile.stripe_angle[seg]
    removed = b - c
    b_hp = highpass(b)
    c_hp = highpass(c)
    removed_hp = highpass(removed)
    ok = np.isfinite(removed) & np.isfinite(st)
    corr = float(np.corrcoef(removed[ok], st[ok])[0, 1]) if ok.sum() > 5 else np.nan
    return {
        "family": profile.family,
        "profile": profile.name,
        "n_valid": int(np.isfinite(b).sum()),
        "row_start": float(profile.row[seg][0]),
        "row_end": float(profile.row[seg][-1]),
        "col_start": float(profile.col[seg][0]),
        "col_end": float(profile.col[seg][-1]),
        "baseline_highpass_robust_std": robust_std(b_hp),
        "corrected_highpass_robust_std": robust_std(c_hp),
        "highpass_reduction_fraction": 1.0 - robust_std(c_hp) / robust_std(b_hp),
        "removed_robust_std": robust_std(removed),
        "removed_highpass_robust_std": robust_std(removed_hp),
        "stripe_total_robust_std": robust_std(st),
        "stripe_angle_robust_std": robust_std(sa),
        "stripe_angle_fraction_of_total": robust_std(sa) / robust_std(st) if robust_std(st) else np.nan,
        "removed_vs_stripe_total_corr": corr,
    }


def profile_samples(profile: Profile) -> pd.DataFrame:
    seg = longest_valid_slice(profile)
    if seg is None:
        return pd.DataFrame()
    x = profile.x[seg] - profile.x[seg][0]
    b = profile.baseline[seg]
    c = profile.corrected[seg]
    removed = b - c
    return pd.DataFrame(
        {
            "family": profile.family,
            "profile": profile.name,
            "distance_px": x,
            "row": profile.row[seg],
            "col": profile.col[seg],
            "baseline_alpha": b,
            "corrected_alpha": c,
            "removed_alpha": removed,
            "stripe_map_total": profile.stripe_total[seg],
            "stripe_map_angle_1p252": profile.stripe_angle[seg],
            "baseline_highpass": highpass(b),
            "corrected_highpass": highpass(c),
            "removed_highpass": highpass(removed),
        }
    )


def plot_profile(profile: Profile, out_path: Path) -> None:
    seg = longest_valid_slice(profile)
    if seg is None:
        return
    x = profile.x[seg]
    x = x - x[0]
    b = profile.baseline[seg]
    c = profile.corrected[seg]
    st = profile.stripe_total[seg]
    sa = profile.stripe_angle[seg]
    removed = b - c
    b_hp = highpass(b)
    c_hp = highpass(c)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(x, b, lw=0.9, label="baseline MF alpha", color="tab:blue")
    axes[0].plot(x, c, lw=0.9, label="after destripe", color="tab:orange")
    axes[0].set_ylabel("alpha")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, b_hp, lw=0.9, label="baseline high-pass", color="tab:blue")
    axes[1].plot(x, c_hp, lw=0.9, label="after high-pass", color="tab:orange")
    axes[1].set_ylabel("alpha high-pass")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(x, removed, lw=1.0, label="removed = baseline - after", color="black")
    axes[2].plot(x, st, lw=0.9, label="stripe_map_total", color="tab:green", alpha=0.9)
    axes[2].plot(x, sa, lw=0.8, label="slope 1.252 component", color="tab:red", alpha=0.8)
    axes[2].set_xlabel("distance along cut (pixels)")
    axes[2].set_ylabel("removed alpha")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(f"{profile.family}: {profile.name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_family_overlay(profiles: list[Profile], out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    for profile in profiles:
        seg = longest_valid_slice(profile)
        if seg is None:
            continue
        x = profile.x[seg] - profile.x[seg][0]
        b_hp = highpass(profile.baseline[seg])
        c_hp = highpass(profile.corrected[seg])
        removed = profile.baseline[seg] - profile.corrected[seg]
        axes[0].plot(x, b_hp, lw=0.7, alpha=0.55, label=f"{profile.name} baseline")
        axes[0].plot(x, c_hp, lw=0.7, alpha=0.55, linestyle="--", label=f"{profile.name} after")
        axes[1].plot(x, removed, lw=0.8, alpha=0.75, label=profile.name)
    axes[0].set_ylabel("alpha high-pass")
    axes[1].set_ylabel("removed alpha")
    axes[1].set_xlabel("distance along cut (pixels)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=2, fontsize=6, frameon=False)
    axes[1].legend(ncol=3, fontsize=7, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_overview(arrays: dict[str, np.ndarray], profiles: list[Profile], out_path: Path) -> None:
    alpha = arrays["baseline"]
    valid = np.isfinite(alpha)
    lo, hi = np.nanpercentile(alpha[valid], [2, 98])
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(alpha[::4, ::4], cmap="gray", vmin=lo, vmax=hi, origin="upper", extent=[0, alpha.shape[1], alpha.shape[0], 0])
    colors = {"y_const": "tab:cyan", "ct_direction": "tab:orange", "broad_perpendicular": "tab:red"}
    for profile in profiles:
        seg = longest_valid_slice(profile)
        if seg is None:
            continue
        ax.plot(
            profile.col[seg],
            profile.row[seg],
            lw=1.4,
            alpha=0.85,
            color=colors.get(profile.family, "white"),
            label=profile.family,
        )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
    ax.set_xlim(0, alpha.shape[1])
    ax.set_ylim(alpha.shape[0], 0)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.set_title("MF profile cut locations")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def choose_y_rows(baseline: np.ndarray, n_rows: int) -> list[int]:
    valid = np.isfinite(baseline)
    counts = valid.sum(axis=1)
    rows = np.where(counts > 500)[0]
    return [int(np.quantile(rows, q)) for q in np.linspace(0.15, 0.9, n_rows)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-y-rows", type=int, default=6)
    parser.add_argument("--n-direction-cuts", type=int, default=5)
    parser.add_argument("--ct-slope", type=float, default=-1.26672)
    parser.add_argument("--broad-slope", type=float, default=1.257172298918948)
    args = parser.parse_args()

    arrays = load_arrays(args.base_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["individual_profiles"]:
        (args.output_dir / sub).mkdir(exist_ok=True)

    y_rows = choose_y_rows(arrays["baseline"], args.n_y_rows)
    y_profiles = [profile_from_row(arrays, row) for row in y_rows]
    ct_profiles = make_direction_profiles(
        arrays,
        family="ct_direction",
        slope=args.ct_slope,
        n_profiles=args.n_direction_cuts,
    )
    perp_profiles = make_direction_profiles(
        arrays,
        family="broad_perpendicular",
        slope=-1.0 / args.broad_slope,
        n_profiles=args.n_direction_cuts,
    )
    profiles = y_profiles + ct_profiles + perp_profiles

    metrics = pd.DataFrame([profile_metrics(profile) for profile in profiles])
    metrics.to_csv(args.output_dir / "profile_cuts_summary.csv", index=False)
    family_summary = (
        metrics.groupby("family", as_index=False)
        .agg(
            n_profiles=("profile", "count"),
            mean_n_valid=("n_valid", "mean"),
            median_baseline_highpass_robust_std=("baseline_highpass_robust_std", "median"),
            median_corrected_highpass_robust_std=("corrected_highpass_robust_std", "median"),
            median_highpass_reduction_fraction=("highpass_reduction_fraction", "median"),
            median_removed_robust_std=("removed_robust_std", "median"),
            median_stripe_total_robust_std=("stripe_total_robust_std", "median"),
            median_stripe_angle_fraction_of_total=("stripe_angle_fraction_of_total", "median"),
            median_removed_vs_stripe_total_corr=("removed_vs_stripe_total_corr", "median"),
        )
    )
    family_summary.to_csv(args.output_dir / "profile_cuts_family_summary.csv", index=False)
    samples = pd.concat([profile_samples(profile) for profile in profiles], ignore_index=True)
    samples.to_csv(args.output_dir / "profile_cut_samples.csv", index=False)

    for profile in profiles:
        plot_profile(profile, args.output_dir / "individual_profiles" / f"{profile.name}.png")
    plot_family_overlay(y_profiles, args.output_dir / "y_const_profile_overlay.png", "y=const MF profiles")
    plot_family_overlay(ct_profiles, args.output_dir / "ct_direction_profile_overlay.png", "CT-direction MF profiles")
    plot_family_overlay(
        perp_profiles,
        args.output_dir / "broad_perpendicular_profile_overlay.png",
        "Profiles perpendicular to broad +1.257 stripe direction",
    )
    plot_overview(arrays, profiles, args.output_dir / "profile_cut_locations.png")

    print("Wrote", args.output_dir)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
