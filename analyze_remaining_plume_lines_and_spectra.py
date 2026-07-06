from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# User settings

RESULT_DIR = Path(r"D:/research/code/outputs_paper_sensor_geometry_destripe")
CASE_NAME = "paper_sensor_line_then_column_each_iter_angle125_focus"
REFERENCE_CASE_NAME = "paper_sensor_line_then_column_each_iter_angle2475_focus"
USE_REFERENCE_MASK_FOR_LINE_DETECTION = True

# Prefer the same spectra table format used by the MF workflow:
# y,x,wave_405.00nm,wave_415.00nm,...
SPECTRA_SOURCE = "map_csv"  # "map_csv" or "hisui_tif"
MAP_SPECTRA_CSV = Path(r"E:/refit/all_map_spectra.csv")
MAP_CSV_CHUNKSIZE = 20_000

HISUI_TIF = Path(
    r"E:/メタン/2025_HISUI_72_The Permian Basin-論文照合用/"
    r"HSHL1G_N320W1032_20221030160051_20231127193053/"
    r"HSHL1G_N320W1032_20221030160051_20231127193053.tif"
)
BAND_CSV = Path(
    r"E:/メタン/2025_HISUI_72_The Permian Basin-論文照合用/"
    r"HSHL1G_N320W1032_20221030160051_20231127193053/"
    r"HSHL1G_N320W1032_20221030160051_20231127193053_B.csv"
)
METADATA_TXT = Path(
    r"E:/メタン/2025_HISUI_72_The Permian Basin-論文照合用/"
    r"HSHL1G_N320W1032_20221030160051_20231127193053/"
    r"HSHL1G_N320W1032_20221030160051_20231127193053.txt"
)

OUTPUT_DIR = RESULT_DIR / f"{CASE_NAME}_remaining_line_spectrum_analysis_ref_{REFERENCE_CASE_NAME}"

# The previous angle sweep selected a=1.252, but visual inspection suggests the
# remaining one-pixel lines are closer to y=x. Search a narrow range around 1.
LINE_SLOPES_TO_CHECK: Optional[list[float]] = [float(v) for v in np.linspace(0.90, 1.10, 81)]
ALLOW_LINE_SLOPE_REFIT = False

N_LINES_TO_KEEP = 4
N_LINE_PIXELS_PER_LINE = 6
N_CONTROL_PIXELS_PER_LINE = 6

# A 1-pixel-wide line often spreads across adjacent b bins after rounding.
B_BIN_WIDTH = 1.0
B_GROUP_GAP_BINS = 2
LINE_DISTANCE_PX = 1.6
MIN_PLUME_PIXELS_PER_LINE = 25
REFERENCE_MIN_PLUME_PIXELS_PER_LINE = 10
MIN_LINE_COL_SPAN = 80
N_CANDIDATE_LINES_TO_SAVE = 40
MIN_SELECTED_LINE_SEPARATION_PX = 20.0

# Controls are chosen near the same line, shifted perpendicular to it.
CONTROL_OFFSETS_PX = [20, -20, 35, -35, 55, -55, 80, -80]
MIN_CONTROL_DISTANCE_TO_ANY_LINE_PX = 6.0

# HISUI L1G metadata for this product. Bands 1-58 are VNIR; bands 59-185 are SWIR.
VNIR_LAST_BAND_NO = 58
RADIANCE_MULTI_VNIR = 1.0e-2
RADIANCE_ADD_VNIR = -10.0
RADIANCE_MULTI_SWIR = 3.2e-3
RADIANCE_ADD_SWIR = -3.2


# Line detection helpers

# Robust line fitting with optional iterative outlier rejection based on distance to the line, which will be used to fit lines to the detected plume pixels for each candidate line slope, allowing for more accurate line fitting that is less sensitive to outliers and can better capture the true line of plume pixels even when there are some noisy pixels that are not part of the main line structure
def robust_polyfit_line(cols: np.ndarray, rows: np.ndarray, a0: float, b0: float, max_iter: int = 5) -> tuple[float, float, np.ndarray]:
    cols = np.asarray(cols, dtype=float)
    rows = np.asarray(rows, dtype=float)
    keep = np.isfinite(cols) & np.isfinite(rows)
    a = float(a0)
    b = float(b0)

    if not ALLOW_LINE_SLOPE_REFIT:
        dist = np.abs(rows - (a * cols + b)) / math.sqrt(a * a + 1.0)
        keep = keep & (dist <= LINE_DISTANCE_PX)
        return a, b, keep

    for _ in range(max_iter):
        if np.sum(keep) < 3:
            break
        a, b = np.polyfit(cols[keep], rows[keep], deg=1)
        dist = np.abs(rows - (a * cols + b)) / math.sqrt(a * a + 1.0)
        new_keep = dist <= LINE_DISTANCE_PX
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    return float(a), float(b), keep

# Group sorted integer IDs into contiguous groups where gaps between IDs are no more than max_gap, which will be used to group the candidate line b bin IDs into contiguous groups that likely belong to the same line structure, allowing for more robust line detection that can account for small gaps in the b bins due to noise or rounding
def contiguous_groups(sorted_ids: np.ndarray, max_gap: int = 2) -> list[np.ndarray]:
    if sorted_ids.size == 0:
        return []
    groups = []
    start = 0
    for i in range(1, sorted_ids.size):
        if sorted_ids[i] - sorted_ids[i - 1] > max_gap:
            groups.append(sorted_ids[start:i])
            start = i
    groups.append(sorted_ids[start:])
    return groups

# Detect lines of plume pixels in the plume mask for a given slope by grouping pixels based on their b values (intercept-like value) and fitting lines to the groups, which will be used to detect candidate lines of plume pixels in the plume mask for each slope by grouping the pixels based on their b values and fitting lines to those groups, allowing for the identification of potential remaining plume lines that were not removed by the previous destriping steps
def detect_lines_for_slope(
    plume_mask: np.ndarray,
    slope: float,
    min_plume_pixels_per_line: Optional[int] = None,
) -> pd.DataFrame:
    min_pixels = MIN_PLUME_PIXELS_PER_LINE if min_plume_pixels_per_line is None else int(min_plume_pixels_per_line)
    rows, cols = np.nonzero(plume_mask)
    b_values = rows.astype(float) - float(slope) * cols.astype(float)
    b_ids = np.rint(b_values / B_BIN_WIDTH).astype(int)
    unique_ids, counts = np.unique(b_ids, return_counts=True)

    candidate_ids = unique_ids[counts >= min_pixels]
    candidate_ids = np.sort(candidate_ids)
    groups = contiguous_groups(candidate_ids, max_gap=B_GROUP_GAP_BINS)

    detected = []
    for group in groups:
        if group.size == 0:
            continue
        in_group = np.isin(b_ids, group)
        rr = rows[in_group]
        cc = cols[in_group]
        if rr.size < min_pixels:
            continue

        b0 = float(np.median(rr - float(slope) * cc))
        a_fit, b_fit, keep = robust_polyfit_line(cc, rr, a0=slope, b0=b0)
        rr_keep = rr[keep]
        cc_keep = cc[keep]
        if rr_keep.size < min_pixels:
            continue
        col_span = int(cc_keep.max() - cc_keep.min()) if cc_keep.size else 0
        if col_span < MIN_LINE_COL_SPAN:
            continue

        detected.append({
            "initial_slope": float(slope),
            "slope": a_fit,
            "intercept": b_fit,
            "angle_deg_from_x": math.degrees(math.atan(a_fit)),
            "n_plume_pixels_on_line": int(rr_keep.size),
            "col_min": int(cc_keep.min()),
            "col_max": int(cc_keep.max()),
            "row_min": int(rr_keep.min()),
            "row_max": int(rr_keep.max()),
            "col_span": col_span,
            "row_span": int(rr_keep.max() - rr_keep.min()),
            "b_group_min": float(group.min() * B_BIN_WIDTH),
            "b_group_max": float(group.max() * B_BIN_WIDTH),
        })

    if not detected:
        return pd.DataFrame()
    out = pd.DataFrame(detected)
    out = out.sort_values(["n_plume_pixels_on_line", "col_span"], ascending=False).reset_index(drop=True)
    out.insert(0, "line_id", np.arange(1, len(out) + 1))
    return out

# Calculate the distance from points to a line defined by slope and intercept, which will be used to calculate the distance of pixels to the detected lines for line fitting and control pixel selection, allowing for the identification of pixels that are close enough to the line to be considered part of the line structure or suitable for use as control pixels
def distance_to_line(rows: np.ndarray, cols: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return np.abs(rows - (float(slope) * cols + float(intercept))) / math.sqrt(float(slope) ** 2 + 1.0)

# Find pixels in the plume mask that are within a certain distance from the line defined by slope and intercept, which will be used to find pixels in the plume mask that are close to the detected lines for sampling and analysis, allowing for the extraction of spectra from pixels that are likely part of the remaining plume lines for further analysis
def pixels_on_line(plume_mask: np.ndarray, slope: float, intercept: float, max_distance_px: float = LINE_DISTANCE_PX) -> pd.DataFrame:
    rows, cols = np.nonzero(plume_mask)
    dist = distance_to_line(rows.astype(float), cols.astype(float), slope, intercept)
    use = dist <= max_distance_px
    df = pd.DataFrame({
        "row": rows[use].astype(int),
        "col": cols[use].astype(int),
        "distance_to_line_px": dist[use],
    })
    return df.sort_values(["col", "row"]).reset_index(drop=True)

# Evenly sample up to n pixels from the given DataFrame of pixels, which will be used to select a representative subset of pixels from the detected lines for spectral extraction and analysis, allowing for a manageable number of samples that are spread across the line structure for better representation of the line's spectral characteristics
def evenly_spaced_samples(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    idx = np.linspace(0, len(df) - 1, n).round().astype(int)
    return df.iloc[np.unique(idx)].copy().reset_index(drop=True)

# Choose control pixels near the line samples but shifted perpendicular to the line, which will be used to select control pixels that are near the detected lines but not on them for spectral extraction and analysis, allowing for a comparison of the spectra from the line pixels to nearby control pixels that are not part of the line structure to help identify spectral features that are specific to the plume lines
def choose_control_pixels(
    line_samples: pd.DataFrame,
    line_table: pd.DataFrame,
    valid_mask: np.ndarray,
    plume_mask: np.ndarray,
    n_per_line: int,
) -> pd.DataFrame:
    H, W = valid_mask.shape
    controls = []
    used = set()

    for _, sample in line_samples.iterrows():
        line_id = int(sample["line_id"])
        line = line_table.loc[line_table["line_id"] == line_id].iloc[0]
        a = float(line["slope"])
        norm = math.sqrt(a * a + 1.0)
        col0 = float(sample["col"])
        row0 = float(sample["row"])

        for offset in CONTROL_OFFSETS_PX:
            col = int(round(col0 - a / norm * float(offset)))
            row = int(round(row0 + 1.0 / norm * float(offset)))
            key = (row, col)
            if key in used:
                continue
            if row < 0 or row >= H or col < 0 or col >= W:
                continue
            if not valid_mask[row, col] or plume_mask[row, col]:
                continue

            min_dist = np.inf
            for _, line2 in line_table.iterrows():
                d = distance_to_line(np.array([row], dtype=float), np.array([col], dtype=float), line2["slope"], line2["intercept"])[0]
                min_dist = min(min_dist, float(d))
            if min_dist < MIN_CONTROL_DISTANCE_TO_ANY_LINE_PX:
                continue

            used.add(key)
            controls.append({
                "line_id": line_id,
                "sample_type": "control_off_line",
                "row": row,
                "col": col,
                "source_line_sample_row": int(sample["row"]),
                "source_line_sample_col": int(sample["col"]),
                "perpendicular_offset_px": float(offset),
                "distance_to_nearest_detected_line_px": min_dist,
            })
            break

        if sum(1 for c in controls if c["line_id"] == line_id) >= n_per_line:
            continue

    return pd.DataFrame(controls)

# Load the slopes selected from the previous angle sweep step from a CSV file, which will be used to load the candidate line slopes that were selected from the previous angle sweep step for line detection, allowing for the use of those slopes as candidate slopes for detecting remaining plume lines in this analysis
def load_angle_sweep_slopes(result_dir: Path, case_name: str) -> list[float]:
    path = result_dir / f"{case_name}_angle_sweep_selected_slopes_slope_search.csv"
    if not path.exists():
        raise FileNotFoundError(f"Angle sweep selected slopes file not found: {path}")
    df = pd.read_csv(path)
    slopes = [float(v) for v in df["slope"].dropna().to_numpy()]
    if not slopes:
        raise ValueError(f"No slopes found in {path}")
    return slopes

# Count the number of pixels in the mask that are within a certain distance from each line defined in the line table, which will be used to count the number of pixels in the plume mask and reference mask that are near each detected line for ranking and selection of the most likely remaining plume lines, allowing for a quantitative assessment of how well each detected line corresponds to actual plume pixels in the target and reference masks
def count_mask_pixels_near_lines(mask: np.ndarray, line_table: pd.DataFrame) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if rows.size == 0 or len(line_table) == 0:
        return np.zeros(len(line_table), dtype=int)
    counts = []
    rows_f = rows.astype(float)
    cols_f = cols.astype(float)
    for _, line in line_table.iterrows():
        dist = distance_to_line(rows_f, cols_f, line["slope"], line["intercept"])
        counts.append(int(np.sum(dist <= LINE_DISTANCE_PX)))
    return np.asarray(counts, dtype=int)

# Select non-duplicate lines from the candidate table by ensuring that the selected lines are separated by at least a certain distance at the center column of the image, which will be used to select a subset of the candidate lines that are not too close to each other for further analysis, allowing for a more diverse set of remaining plume lines to be analyzed without having multiple very similar lines that likely correspond to the same line structure
def select_nonduplicate_lines(candidate_table: pd.DataFrame, n_keep: int, image_width: int) -> pd.DataFrame:
    selected = []
    center_col = 0.5 * (float(image_width) - 1.0)
    for _, row in candidate_table.iterrows():
        y_center = float(row["slope"]) * center_col + float(row["intercept"])
        too_close = False
        for prev in selected:
            y_prev = float(prev["slope"]) * center_col + float(prev["intercept"])
            if abs(y_center - y_prev) < MIN_SELECTED_LINE_SEPARATION_PX:
                too_close = True
                break
        if too_close:
            continue
        selected.append(row)
        if len(selected) >= int(n_keep):
            break

    if not selected:
        return candidate_table.head(n_keep).copy().reset_index(drop=True)
    return pd.DataFrame(selected).reset_index(drop=True)

# Detect candidate lines of plume pixels in the target plume mask using the reference plume mask for line detection if configured, and prepare a table of detected lines and a table of sampled pixels on those lines and control pixels for spectral extraction, which will be used to perform the main analysis of detecting remaining plume lines in the target plume mask, extracting spectra from those lines, and comparing them to control pixels, allowing for the identification and analysis of potential remaining plume lines that were not removed by the previous destriping steps
def detect_remaining_lines() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_plume_path = RESULT_DIR / f"{CASE_NAME}_plume_mask.npy"
    reference_case = REFERENCE_CASE_NAME if USE_REFERENCE_MASK_FOR_LINE_DETECTION else CASE_NAME
    detection_plume_path = RESULT_DIR / f"{reference_case}_plume_mask.npy"
    alpha_path = RESULT_DIR / f"{CASE_NAME}_alpha_corrected.npy"
    target_plume_mask = np.load(target_plume_path).astype(bool)
    detection_plume_mask = np.load(detection_plume_path).astype(bool)
    valid_mask = np.isfinite(np.load(alpha_path))

    slopes = LINE_SLOPES_TO_CHECK if LINE_SLOPES_TO_CHECK is not None else load_angle_sweep_slopes(RESULT_DIR, CASE_NAME)
    all_lines = []
    min_pixels = REFERENCE_MIN_PLUME_PIXELS_PER_LINE if USE_REFERENCE_MASK_FOR_LINE_DETECTION else MIN_PLUME_PIXELS_PER_LINE
    for slope in slopes:
        lines = detect_lines_for_slope(
            detection_plume_mask,
            float(slope),
            min_plume_pixels_per_line=min_pixels,
        )
        if len(lines) > 0:
            all_lines.append(lines)
    if not all_lines:
        raise RuntimeError("No remaining line candidates were detected. Lower MIN_PLUME_PIXELS_PER_LINE or LINE_DISTANCE_PX.")

    candidate_table = pd.concat(all_lines, ignore_index=True)
    candidate_table["detection_case_name"] = reference_case
    candidate_table["target_case_name"] = CASE_NAME
    candidate_table["target_pixels_on_line"] = count_mask_pixels_near_lines(target_plume_mask, candidate_table)
    candidate_table["reference_pixels_on_line"] = count_mask_pixels_near_lines(detection_plume_mask, candidate_table)
    candidate_table["line_rank_score"] = (
        0.70 * candidate_table["target_pixels_on_line"].astype(float)
        + 0.30 * candidate_table["reference_pixels_on_line"].astype(float)
    )

    candidate_table = candidate_table.sort_values(
        ["line_rank_score", "target_pixels_on_line", "reference_pixels_on_line", "col_span"],
        ascending=False,
    ).reset_index(drop=True)
    candidate_table.insert(0, "candidate_rank", np.arange(1, len(candidate_table) + 1))
    candidate_table["equation"] = candidate_table.apply(lambda r: f"row = {r['slope']:.8f} * col + {r['intercept']:.3f}", axis=1)

    line_table = select_nonduplicate_lines(candidate_table, N_LINES_TO_KEEP, image_width=target_plume_mask.shape[1])
    line_table["line_id"] = np.arange(1, len(line_table) + 1)
    line_table["equation"] = line_table.apply(lambda r: f"row = {r['slope']:.8f} * col + {r['intercept']:.3f}", axis=1)

    sample_rows = []
    for _, line in line_table.iterrows():
        line_pixels = pixels_on_line(target_plume_mask, line["slope"], line["intercept"], max_distance_px=LINE_DISTANCE_PX)
        sample_source = "target_plume_mask"
        if len(line_pixels) < N_LINE_PIXELS_PER_LINE:
            line_pixels = pixels_on_line(detection_plume_mask, line["slope"], line["intercept"], max_distance_px=LINE_DISTANCE_PX)
            sample_source = "reference_plume_mask"
        samples = evenly_spaced_samples(line_pixels, N_LINE_PIXELS_PER_LINE)
        samples.insert(0, "line_id", int(line["line_id"]))
        samples.insert(1, "sample_type", "on_line_plume_mask")
        samples["sample_source_mask"] = sample_source
        sample_rows.append(samples)
    on_line_samples = pd.concat(sample_rows, ignore_index=True)

    controls = choose_control_pixels(
        line_samples=on_line_samples,
        line_table=line_table,
        valid_mask=valid_mask,
        plume_mask=target_plume_mask,
        n_per_line=N_CONTROL_PIXELS_PER_LINE,
    )

    sample_table = pd.concat([on_line_samples, controls], ignore_index=True, sort=False)
    sample_table.insert(0, "sample_id", [f"S{i:03d}" for i in range(1, len(sample_table) + 1)])
    return line_table, sample_table, valid_mask, candidate_table


# HISUI radiance extraction and plotting

# Read a metadata value from the HISUI metadata text file, which will be used to read specific metadata values from the HISUI metadata text file for use in the analysis, allowing for the incorporation of relevant metadata information such as observation geometry or calibration parameters into the analysis of the spectra extracted from the HISUI TIF if that source is used
def read_metadata_value(path: Path, key: str, default: Optional[float] = None) -> Optional[float]:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*$")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pattern.match(line)
        if m:
            value = m.group(1).strip().strip('"')
            try:
                return float(value)
            except ValueError:
                return value
    return default

# Load the HISUI band information from the CSV file, which will be used to load the band information for the HISUI TIF from the provided CSV file for use in converting DN values to radiance and for understanding the wavelength corresponding to each band when analyzing the spectra extracted from the HISUI TIF if that source is used
def load_band_info() -> pd.DataFrame:
    band = pd.read_csv(BAND_CSV)
    band.columns = [c.strip() for c in band.columns]
    band["BandNo"] = band["BandNo"].astype(int)
    band["wavelength_nm"] = band["CenterWavelengthNanometer"].astype(float)
    return band

# Parse the wave_*nm columns from the map spectra CSV header to extract the corresponding wavelengths, which will be used to identify the wave_*nm columns in the map spectra CSV and extract the corresponding wavelengths for those columns for use in analyzing the spectra extracted from the map CSV if that source is used
def parse_wave_columns(columns: Iterable[str]) -> tuple[list[str], np.ndarray]:
    wave_cols = []
    wavelengths = []
    pattern = re.compile(r"^wave_([0-9]+(?:\.[0-9]+)?)nm$")
    for col in columns:
        m = pattern.match(str(col))
        if m:
            wave_cols.append(str(col))
            wavelengths.append(float(m.group(1)))
    if not wave_cols:
        raise ValueError("No wave_*nm columns found in map spectra CSV.")
    wavelengths = np.asarray(wavelengths, dtype=float)
    order = np.argsort(wavelengths)
    return [wave_cols[i] for i in order], wavelengths[order]

# Extract spectra for the sampled pixels from the map spectra CSV, which will be used to extract the spectra for the sampled pixels from the map spectra CSV for analysis, allowing for the analysis of the spectral characteristics of the detected remaining plume lines using the spectra provided in the map CSV if that source is used
def extract_spectra_from_map_csv(sample_table: pd.DataFrame) -> pd.DataFrame:
    header = pd.read_csv(MAP_SPECTRA_CSV, nrows=0)
    if "y" not in header.columns or "x" not in header.columns:
        raise ValueError("MAP_SPECTRA_CSV must contain y and x columns.")
    wave_cols, wavelengths = parse_wave_columns(header.columns)

    sample_info = sample_table.copy()
    sample_info["row"] = sample_info["row"].astype(int)
    sample_info["col"] = sample_info["col"].astype(int)
    key_mult = 10_000_000
    sample_info["_pixel_key"] = sample_info["row"] * key_mult + sample_info["col"]
    target_keys = set(sample_info["_pixel_key"].astype(int).tolist())
    sample_by_key = sample_info.set_index("_pixel_key", drop=False)

    found_records = []
    found_keys: set[int] = set()
    usecols = ["y", "x"] + wave_cols

    for chunk in pd.read_csv(MAP_SPECTRA_CSV, usecols=usecols, chunksize=MAP_CSV_CHUNKSIZE):
        y = chunk["y"].astype(np.int64).to_numpy()
        x = chunk["x"].astype(np.int64).to_numpy()
        keys = y * key_mult + x
        keep = np.isin(keys, list(target_keys - found_keys))
        if not np.any(keep):
            continue

        for _, row in chunk.loc[keep].iterrows():
            key = int(row["y"]) * key_mult + int(row["x"])
            if key in found_keys:
                continue
            found_keys.add(key)
            samples_for_pixel = sample_by_key.loc[[key]]
            for _, sample in samples_for_pixel.iterrows():
                values = row[wave_cols].to_numpy(dtype=float)
                for wave, value in zip(wavelengths, values):
                    found_records.append({
                        "sample_id": sample["sample_id"],
                        "sample_type": sample["sample_type"],
                        "line_id": int(sample["line_id"]),
                        "row": int(sample["row"]),
                    "col": int(sample["col"]),
                    "wavelength_nm": float(wave),
                    "radiance_from_map_csv": float(value),
                })

        if found_keys >= target_keys:
            break

    missing_keys = sorted(target_keys - found_keys)
    if missing_keys:
        missing = sample_info[sample_info["_pixel_key"].isin(missing_keys)][["sample_id", "row", "col"]]
        print("Warning: some sampled pixels were not found in MAP_SPECTRA_CSV:")
        print(missing.to_string(index=False))

    spectra = pd.DataFrame(found_records)
    if len(spectra) == 0:
        raise RuntimeError("No sampled spectra were extracted from MAP_SPECTRA_CSV.")
    return spectra

# Convert DN values to radiance using the HISUI calibration parameters, which will be used to convert the DN values extracted from the HISUI TIF to radiance values for analysis, allowing for the analysis of the spectral characteristics of the detected remaining plume lines using the radiance values derived from the HISUI TIF if that source is used
def dn_to_radiance(dn: np.ndarray, band_no: np.ndarray) -> np.ndarray:
    dn = np.asarray(dn, dtype=float)
    band_no = np.asarray(band_no, dtype=int)
    out = np.empty_like(dn, dtype=float)
    is_vnir = band_no <= VNIR_LAST_BAND_NO
    out[is_vnir] = dn[is_vnir] * RADIANCE_MULTI_VNIR + RADIANCE_ADD_VNIR
    out[~is_vnir] = dn[~is_vnir] * RADIANCE_MULTI_SWIR + RADIANCE_ADD_SWIR
    return out

# Extract spectra for the sampled pixels from the HISUI Big GeoTIFF, which will be used to extract the spectra for the sampled pixels from the HISUI Big GeoTIFF for analysis, allowing for the analysis of the spectral characteristics of the detected remaining plume lines using the spectra extracted from the HISUI TIF if that source is used
def extract_hisui_spectra(sample_table: pd.DataFrame) -> pd.DataFrame:
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:
        raise ImportError(
            "rasterio is required to read the HISUI Big GeoTIFF. "
            "Run this script in the same Python/Jupyter environment that can read your HISUI tif, "
            "or install rasterio there."
        ) from exc

    band = load_band_info()
    records = []
    with rasterio.open(HISUI_TIF) as src:
        if src.count != len(band):
            raise ValueError(f"TIF band count ({src.count}) does not match band CSV rows ({len(band)}).")

        for _, sample in sample_table.iterrows():
            row = int(sample["row"])
            col = int(sample["col"])
            dn = src.read(window=Window(col, row, 1, 1))[:, 0, 0].astype(float)
            radiance = dn_to_radiance(dn, band["BandNo"].to_numpy())
            for i, (_, b) in enumerate(band.iterrows()):
                records.append({
                    "sample_id": sample["sample_id"],
                    "sample_type": sample["sample_type"],
                    "line_id": int(sample["line_id"]),
                    "row": row,
                    "col": col,
                    "band_no": int(b["BandNo"]),
                    "wavelength_nm": float(b["wavelength_nm"]),
                    "dn": float(dn[i]),
                    "radiance_w_m2_um_sr": float(radiance[i]),
                })

    return pd.DataFrame(records)

# Plot the spectra for the sampled pixels, showing individual spectra and mean spectra for each detected line, which will be used to visualize the spectra extracted from the sampled pixels for analysis, allowing for a visual comparison of the spectral characteristics of the detected remaining plume lines and their corresponding control pixels to help identify any distinctive spectral features associated with the plume lines
def plot_spectra(spectra: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, spectra["line_id"].nunique() + 1)))
    if "radiance_w_m2_um_sr" in spectra.columns:
        value_col = "radiance_w_m2_um_sr"
        y_label = "Radiance (W/m2/micron/sr)"
    else:
        value_col = "radiance_from_map_csv"
        y_label = "Radiance from all_map_spectra.csv"

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    for line_id, df_line in spectra.groupby("line_id"):
        color = colors[(int(line_id) - 1) % len(colors)]
        for sample_id, df_s in df_line[df_line["sample_type"] == "on_line_plume_mask"].groupby("sample_id"):
            axes[0].plot(df_s["wavelength_nm"], df_s[value_col], color=color, alpha=0.55, linewidth=1.0)
        for sample_id, df_s in df_line[df_line["sample_type"] == "control_off_line"].groupby("sample_id"):
            axes[0].plot(df_s["wavelength_nm"], df_s[value_col], color="0.55", alpha=0.35, linewidth=0.9, linestyle="--")

        mean_line = df_line[df_line["sample_type"] == "on_line_plume_mask"].groupby("wavelength_nm")[value_col].mean()
        mean_ctrl = df_line[df_line["sample_type"] == "control_off_line"].groupby("wavelength_nm")[value_col].mean()
        axes[1].plot(mean_line.index, mean_line.values, color=color, linewidth=1.8, label=f"line {line_id} on-line")
        if len(mean_ctrl) > 0:
            axes[1].plot(mean_ctrl.index, mean_ctrl.values, color=color, linewidth=1.4, linestyle="--", label=f"line {line_id} control")

    axes[0].set_title("Individual spectra")
    axes[1].set_title("Mean spectra by detected line")
    for ax in axes:
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "remaining_line_spectra_all_wavelengths.png", dpi=180)
    plt.close(fig)

    swir = spectra[(spectra["wavelength_nm"] >= 2100) & (spectra["wavelength_nm"] <= 2450)]
    fig, ax = plt.subplots(figsize=(10, 5))
    for line_id, df_line in swir.groupby("line_id"):
        color = colors[(int(line_id) - 1) % len(colors)]
        mean_line = df_line[df_line["sample_type"] == "on_line_plume_mask"].groupby("wavelength_nm")[value_col].mean()
        mean_ctrl = df_line[df_line["sample_type"] == "control_off_line"].groupby("wavelength_nm")[value_col].mean()
        ax.plot(mean_line.index, mean_line.values, color=color, linewidth=2.0, label=f"line {line_id} on-line")
        if len(mean_ctrl) > 0:
            ax.plot(mean_ctrl.index, mean_ctrl.values, color=color, linewidth=1.6, linestyle="--", label=f"line {line_id} control")
    ax.set_title("Mean spectra in MF wavelength range")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "remaining_line_spectra_swir_mf_range.png", dpi=180)
    plt.close(fig)

# Plot the detected lines overlaid on the plume mask, showing the sampled pixels, which will be used to visualize the detected remaining plume lines overlaid on the plume mask along with the sampled pixels for analysis, allowing for a visual confirmation of how well the detected lines correspond to actual plume pixels in the mask and where the sampled pixels are located in relation to the detected lines
def plot_line_overlay(
    plume_mask: np.ndarray,
    line_table: pd.DataFrame,
    sample_table: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    H, W = plume_mask.shape
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(plume_mask, origin="upper", cmap="gray_r", interpolation="nearest")
    xs = np.arange(W)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(line_table) + 1)))
    for _, line in line_table.iterrows():
        line_id = int(line["line_id"])
        ys = line["slope"] * xs + line["intercept"]
        ok = (ys >= 0) & (ys < H)
        ax.plot(xs[ok], ys[ok], color=colors[(line_id - 1) % len(colors)], linewidth=1.2, label=f"line {line_id}")
    on_line = sample_table[sample_table["sample_type"] == "on_line_plume_mask"]
    ctrl = sample_table[sample_table["sample_type"] == "control_off_line"]
    ax.scatter(on_line["col"], on_line["row"], s=22, c="red", marker="o", label="sample on line")
    if len(ctrl) > 0:
        ax.scatter(ctrl["col"], ctrl["row"], s=22, c="cyan", marker="x", label="control")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xlabel("col / x")
    ax.set_ylabel("row / y")
    ax.set_title(f"Detected residual plume-mask lines and sampled pixels{title_suffix}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "remaining_line_overlay.png", dpi=180)
    plt.close(fig)


def main() -> dict[str, pd.DataFrame]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    line_table, sample_table, valid_mask, candidate_table = detect_remaining_lines()
    target_plume_mask = np.load(RESULT_DIR / f"{CASE_NAME}_plume_mask.npy").astype(bool)
    reference_case = REFERENCE_CASE_NAME if USE_REFERENCE_MASK_FOR_LINE_DETECTION else CASE_NAME
    reference_plume_mask = np.load(RESULT_DIR / f"{reference_case}_plume_mask.npy").astype(bool)

    line_table.to_csv(OUTPUT_DIR / "remaining_line_equations.csv", index=False)
    candidate_table.head(N_CANDIDATE_LINES_TO_SAVE).to_csv(OUTPUT_DIR / "remaining_line_candidates_reference_ranked.csv", index=False)
    sample_table.to_csv(OUTPUT_DIR / "remaining_line_sample_pixels.csv", index=False)

    print("Detected line equations:")
    cols = [
        "line_id", "equation", "angle_deg_from_x", "target_pixels_on_line",
        "reference_pixels_on_line", "col_span",
    ]
    print(line_table[cols].to_string(index=False))
    print(f"Saved: {OUTPUT_DIR / 'remaining_line_equations.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'remaining_line_candidates_reference_ranked.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'remaining_line_sample_pixels.csv'}")

    try:
        if SPECTRA_SOURCE == "map_csv":
            spectra = extract_spectra_from_map_csv(sample_table)
            spectra_path = OUTPUT_DIR / "remaining_line_sample_spectra_from_map_csv.csv"
        elif SPECTRA_SOURCE == "hisui_tif":
            spectra = extract_hisui_spectra(sample_table)
            spectra_path = OUTPUT_DIR / "remaining_line_sample_spectra_radiance.csv"
        else:
            raise ValueError("SPECTRA_SOURCE must be 'map_csv' or 'hisui_tif'.")

        spectra.to_csv(spectra_path, index=False)
        plot_spectra(spectra, OUTPUT_DIR)
        plot_line_overlay(target_plume_mask, line_table, sample_table, OUTPUT_DIR, title_suffix=": target corrected mask")
        plot_line_overlay(reference_plume_mask, line_table, sample_table, OUTPUT_DIR / "reference_overlay", title_suffix=": reference baseline mask")
        print(f"Saved: {spectra_path}")
        print(f"Saved: {OUTPUT_DIR / 'remaining_line_spectra_all_wavelengths.png'}")
        print(f"Saved: {OUTPUT_DIR / 'remaining_line_spectra_swir_mf_range.png'}")
        print(f"Saved: {OUTPUT_DIR / 'remaining_line_overlay.png'}")
        print(f"Saved: {OUTPUT_DIR / 'reference_overlay' / 'remaining_line_overlay.png'}")
    except ImportError as exc:
        print(str(exc))
        print("Line equations and sample pixel CSVs were still saved.")
        spectra = pd.DataFrame()

    return {
        "line_table": line_table,
        "candidate_table": candidate_table,
        "sample_table": sample_table,
        "spectra": spectra,
    }


if __name__ == "__main__":
    main()
