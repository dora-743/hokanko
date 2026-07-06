from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

import numpy as np
import pandas as pd



# 0. User settings
# This file is intended to live next to:
#   paper_sensor_geometry_iterative_mf_destriping_full.py
#   analyze_remaining_plume_lines_and_spectra.py
PIPELINE_DIR = Path(__file__).resolve().parent
MF_CODE = PIPELINE_DIR / "paper_sensor_geometry_iterative_mf_destriping_full.py"
LINE_ANALYSIS_CODE = PIPELINE_DIR / "analyze_remaining_plume_lines_and_spectra.py"

# Main data used by the Iterative MF workflow.
ROI_CSV = Path(r"D:/research/code/all_roi_spectra200x200.csv")
MODTRAN_CSV = Path(r"E:/refit/CH4c.csv")

# Leave None to use the path already configured in the MF code.
METADATA_TXT_OVERRIDE: Optional[Path] = None

# All MF and line-analysis outputs are read/written here.
RESULT_DIR = Path(r"D:/research/code/outputs_paper_sensor_geometry_destripe")

# The MF case to run and then analyze.
CASE_NAME = "paper_sensor_line_then_column_each_iter_angle125_focus"

# A previous/alternative mask can help detect the remaining one-pixel lines.
REFERENCE_CASE_NAME = "paper_sensor_line_then_column_each_iter_angle2475_focus"
USE_REFERENCE_MASK_FOR_LINE_DETECTION = True

# Spectrum source. The CSV is expected to have:
# y,x,wave_405.00nm,wave_415.00nm,...
SPECTRA_SOURCE = "map_csv"
MAP_SPECTRA_CSV = Path(r"E:/refit/all_map_spectra.csv")
MAP_CSV_CHUNKSIZE = 20_000

# Turn RUN_MF_STAGE off if the angle125 MF result already exists and you only
# want to rerun the remaining-line/spectrum analysis.
RUN_MF_STAGE = True
RUN_LINE_SPECTRUM_STAGE = True
RUN_PLOTS = True

# The destripe focus used inside the MF stage.
ANGLE125_SWEEP_MIN = 1.18
ANGLE125_SWEEP_MAX = 1.34
ANGLE125_SWEEP_NUM = 101
ANGLE125_SWEEP_TOP_K = 1
ANGLE125_LINE_BIN_WIDTH = 18.0

# The remaining thin lines looked closer to y=x than to a=1.252, so this
# second search estimates their line equations and angles in a narrow range.
LINE_SLOPES_TO_CHECK = [float(v) for v in np.linspace(0.90, 1.10, 81)]
ALLOW_LINE_SLOPE_REFIT = False
N_LINES_TO_KEEP = 4
N_LINE_PIXELS_PER_LINE = 6
N_CONTROL_PIXELS_PER_LINE = 6

# Output directory for the final integrated analysis.
COMBINED_OUTPUT_DIR = RESULT_DIR / f"{CASE_NAME}_combined_remaining_line_spectra"


# 1. Module loading and configuration

def load_python_file(path: Path, module_name: str) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Required code file was not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def configure_mf_module(mf: ModuleType) -> None:
    mf.ROI_CSV = ROI_CSV
    mf.MODTRAN_CSV = MODTRAN_CSV
    mf.OUTPUT_DIR = RESULT_DIR
    mf.RUN_PLOTS = RUN_PLOTS
    mf.RUN_EXPERIMENT_NAMES = ["baseline_no_destripe", CASE_NAME]

    if METADATA_TXT_OVERRIDE is not None:
        mf.METADATA_TXT = METADATA_TXT_OVERRIDE

    # Make sure every destriping config uses the active metadata path.
    for cfg in mf.EXPERIMENTS.values():
        params = cfg.get("destripe_params")
        if params is not None:
            params["sensor_geometry_metadata_txt"] = mf.METADATA_TXT

    # Force the current angle125-focus settings so this file is the single
    # place to adjust the broad residual-band correction.
    params = mf.EXPERIMENTS[CASE_NAME]["destripe_params"]
    params.update(
        {
            "angle_sweep_cleanup": True,
            "angle_sweep_slope_min": ANGLE125_SWEEP_MIN,
            "angle_sweep_slope_max": ANGLE125_SWEEP_MAX,
            "angle_sweep_num_slopes": ANGLE125_SWEEP_NUM,
            "angle_sweep_top_k": ANGLE125_SWEEP_TOP_K,
            "angle_sweep_min_slope_separation": 0.015,
            "angle_sweep_line_bin_width": ANGLE125_LINE_BIN_WIDTH,
            "angle_sweep_score_z_min": None,
        }
    )


def configure_line_module(line: ModuleType) -> None:
    line.RESULT_DIR = RESULT_DIR
    line.CASE_NAME = CASE_NAME
    line.REFERENCE_CASE_NAME = REFERENCE_CASE_NAME
    line.USE_REFERENCE_MASK_FOR_LINE_DETECTION = USE_REFERENCE_MASK_FOR_LINE_DETECTION
    line.SPECTRA_SOURCE = SPECTRA_SOURCE
    line.MAP_SPECTRA_CSV = MAP_SPECTRA_CSV
    line.MAP_CSV_CHUNKSIZE = MAP_CSV_CHUNKSIZE
    line.OUTPUT_DIR = COMBINED_OUTPUT_DIR

    line.LINE_SLOPES_TO_CHECK = LINE_SLOPES_TO_CHECK
    line.ALLOW_LINE_SLOPE_REFIT = ALLOW_LINE_SLOPE_REFIT
    line.N_LINES_TO_KEEP = N_LINES_TO_KEEP
    line.N_LINE_PIXELS_PER_LINE = N_LINE_PIXELS_PER_LINE
    line.N_CONTROL_PIXELS_PER_LINE = N_CONTROL_PIXELS_PER_LINE


# 2. Angle diagnostics for detected remaining lines

def add_angle_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "slope" not in out.columns:
        return out

    slope = out["slope"].astype(float)
    angle = np.degrees(np.arctan(slope))
    out["angle_deg_from_x_axis"] = angle
    out["angle_deg_from_y_eq_x"] = angle - 45.0
    out["angle_deg_from_vertical_axis"] = 90.0 - angle
    return out


def summarize_angle_candidates(candidate_table: pd.DataFrame) -> pd.DataFrame:
    candidates = add_angle_diagnostics(candidate_table)
    candidates["slope_rounded"] = candidates["slope"].astype(float).round(6)

    rows = []
    for slope, df_slope in candidates.groupby("slope_rounded", sort=False):
        angle = math.degrees(math.atan(float(slope)))
        rows.append(
            {
                "slope": float(slope),
                "angle_deg_from_x_axis": angle,
                "angle_deg_from_y_eq_x": angle - 45.0,
                "n_candidate_lines": int(len(df_slope)),
                "max_line_rank_score": float(df_slope["line_rank_score"].max()),
                "sum_target_pixels_on_line": int(df_slope["target_pixels_on_line"].sum()),
                "sum_reference_pixels_on_line": int(df_slope["reference_pixels_on_line"].sum()),
                "max_col_span": int(df_slope["col_span"].max()),
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["max_line_rank_score", "sum_target_pixels_on_line", "max_col_span"],
        ascending=False,
    ).reset_index(drop=True)


def save_angle_outputs(
    line_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    line_angles = add_angle_diagnostics(line_table)
    candidate_angles = add_angle_diagnostics(candidate_table)
    angle_summary = summarize_angle_candidates(candidate_table)

    line_angles.to_csv(output_dir / "detected_remaining_line_equations_with_angles.csv", index=False)
    candidate_angles.to_csv(output_dir / "detected_remaining_line_candidates_with_angles.csv", index=False)
    angle_summary.to_csv(output_dir / "detected_remaining_line_angle_summary.csv", index=False)

    print("\nDetected remaining-line angles")
    cols = [
        "line_id",
        "equation",
        "slope",
        "angle_deg_from_x_axis",
        "angle_deg_from_y_eq_x",
        "target_pixels_on_line",
        "reference_pixels_on_line",
    ]
    available_cols = [c for c in cols if c in line_angles.columns]
    print(line_angles[available_cols].to_string(index=False))
    print(f"Saved: {output_dir / 'detected_remaining_line_equations_with_angles.csv'}")
    print(f"Saved: {output_dir / 'detected_remaining_line_angle_summary.csv'}")
    return line_angles, angle_summary



# 3. Integrated pipeline

def run_mf_stage() -> dict:
    mf = load_python_file(MF_CODE, "paper_sensor_geometry_iterative_mf_destriping_full")
    configure_mf_module(mf)
    print("\n" + "=" * 80)
    print("Stage 1: Iterative MF with paper/sensor-geometry destriping + angle125 focus")
    print("=" * 80)
    return mf.main()


def run_line_spectrum_stage() -> dict[str, pd.DataFrame]:
    line = load_python_file(LINE_ANALYSIS_CODE, "analyze_remaining_plume_lines_and_spectra")
    configure_line_module(line)

    print("\n" + "=" * 80)
    print("Stage 2: Remaining-line angle detection, sample pixels, and spectra")
    print("=" * 80)

    COMBINED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    line_table, sample_table, valid_mask, candidate_table = line.detect_remaining_lines()
    line_table, angle_summary = save_angle_outputs(line_table, candidate_table, COMBINED_OUTPUT_DIR)

    candidate_table.head(line.N_CANDIDATE_LINES_TO_SAVE).to_csv(
        COMBINED_OUTPUT_DIR / "remaining_line_candidates_ranked.csv",
        index=False,
    )
    sample_table.to_csv(COMBINED_OUTPUT_DIR / "remaining_line_sample_pixels.csv", index=False)

    target_plume_mask = np.load(RESULT_DIR / f"{CASE_NAME}_plume_mask.npy").astype(bool)
    reference_case = REFERENCE_CASE_NAME if USE_REFERENCE_MASK_FOR_LINE_DETECTION else CASE_NAME
    reference_plume_mask = np.load(RESULT_DIR / f"{reference_case}_plume_mask.npy").astype(bool)

    spectra = pd.DataFrame()
    if SPECTRA_SOURCE == "map_csv":
        spectra = line.extract_spectra_from_map_csv(sample_table)
        spectra_path = COMBINED_OUTPUT_DIR / "remaining_line_sample_spectra_from_map_csv.csv"
    elif SPECTRA_SOURCE == "hisui_tif":
        spectra = line.extract_hisui_spectra(sample_table)
        spectra_path = COMBINED_OUTPUT_DIR / "remaining_line_sample_spectra_radiance.csv"
    else:
        raise ValueError("SPECTRA_SOURCE must be 'map_csv' or 'hisui_tif'.")

    spectra.to_csv(spectra_path, index=False)
    line.plot_spectra(spectra, COMBINED_OUTPUT_DIR)
    line.plot_line_overlay(
        target_plume_mask,
        line_table,
        sample_table,
        COMBINED_OUTPUT_DIR,
        title_suffix=": target corrected mask",
    )
    line.plot_line_overlay(
        reference_plume_mask,
        line_table,
        sample_table,
        COMBINED_OUTPUT_DIR / "reference_overlay",
        title_suffix=": reference mask",
    )

    print(f"Saved: {COMBINED_OUTPUT_DIR / 'remaining_line_sample_pixels.csv'}")
    print(f"Saved: {spectra_path}")
    print(f"Saved: {COMBINED_OUTPUT_DIR / 'remaining_line_spectra_all_wavelengths.png'}")
    print(f"Saved: {COMBINED_OUTPUT_DIR / 'remaining_line_spectra_swir_mf_range.png'}")
    print(f"Saved: {COMBINED_OUTPUT_DIR / 'remaining_line_overlay.png'}")

    return {
        "line_table": line_table,
        "angle_summary": angle_summary,
        "candidate_table": candidate_table,
        "sample_table": sample_table,
        "spectra": spectra,
        "valid_mask": pd.DataFrame({"valid_pixels": [int(np.sum(valid_mask))]}),
    }


def main() -> dict:
    outputs: dict = {}
    if RUN_MF_STAGE:
        outputs["mf_results"] = run_mf_stage()
    else:
        print("Skipping Stage 1 because RUN_MF_STAGE = False.")

    if RUN_LINE_SPECTRUM_STAGE:
        outputs["line_spectrum_results"] = run_line_spectrum_stage()
    else:
        print("Skipping Stage 2 because RUN_LINE_SPECTRUM_STAGE = False.")

    return outputs


if __name__ == "__main__":
    main()
