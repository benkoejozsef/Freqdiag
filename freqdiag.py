#!/usr/bin/env python3
"""
freqdiag.py -- frequency-list diagnostics for modulation, beating, and coupling

Reads a frequency/amplitude/phase table and tests which Fourier patterns are
compatible with the observed significant frequencies.  The program is a
conservative diagnostic aid: it does not determine the physical model of the
star, but checks necessary frequency-, amplitude-, and phase-relation signatures
of several mathematical descriptions.

Typical use
-----------
./freqdiag.py frequencies.csv --baseline 200 --outdir diag_out

For VizieR/ASCII tables:
./freqdiag.py V1127_Aql.txt --separator whitespace --freq-col Freq --amp-col Amp \
  --phase-col Phi --phase-unit deg --baseline 150 --nmax 20 --outdir diag_V1127

Main output files
-----------------
freqdiag_report.md                  Human-readable report
freqdiag_model_scores.csv           Full model/submodel score table
freqdiag_report_scores.csv          Model rows shown in the Markdown report
freqdiag_feature_flags.csv           Feature/qualifier rows shown separately in the report
freqdiag_matches.csv                All expected frequencies and matches
freqdiag_frequency_assignments.csv  Input list with assignment tags
freqdiag_harmonics.csv              Detected primary harmonics
freqdiag_f0_candidates.csv          Automatic primary-frequency ranking
freqdiag_candidates.csv             Candidate close-frequency offsets
freqdiag_phase_tests.csv            Quadratic-coupling phase/amplitude tests
freqdiag.json                       Machine-readable summary
freqdiag_latex_table.tex            Compact LaTeX summary table

"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__version__ = "freqdiag-1.0.0-rc2"

TWOPI = 2.0 * np.pi


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class FreqMatch:
    category: str
    model: str
    label: str
    expected_frequency: float
    observed_frequency: Optional[float]
    delta_frequency: Optional[float]
    input_index: Optional[int]
    amplitude: Optional[float]
    phase_rad: Optional[float]
    matched: bool
    notes: str = ""


@dataclass
class Candidate:
    candidate_id: int
    fB_abs: float
    signs: str                    # right / left / both / manual
    right_frequency: Optional[float]
    left_frequency: Optional[float]
    right_input_index: Optional[int]
    left_input_index: Optional[int]
    right_amplitude: Optional[float]
    left_amplitude: Optional[float]
    right_phase_rad: Optional[float]
    left_phase_rad: Optional[float]
    source: str = "auto"
    family_id: int = 0
    family_fundamental: Optional[float] = None
    family_order: int = 1
    family_max_order: int = 1
    family_representative_id: int = 0
    family_member_ids: str = ""
    family_member_orders: str = ""
    family_representative: bool = True
    discovery_harmonics: str = ""
    discovery_harmonic_count: int = 0
    discovery_peak_count: int = 0
    discovery_total_amplitude: float = 0.0
    discovery_max_amplitude: float = 0.0

    @property
    def representative_frequency(self) -> Optional[float]:
        if self.right_amplitude is not None and self.left_amplitude is not None:
            return self.right_frequency if self.right_amplitude >= self.left_amplitude else self.left_frequency
        if self.right_frequency is not None:
            return self.right_frequency
        return self.left_frequency

    @property
    def representative_amplitude(self) -> Optional[float]:
        vals = [v for v in (self.right_amplitude, self.left_amplitude) if v is not None]
        if self.discovery_max_amplitude > 0:
            vals.append(float(self.discovery_max_amplitude))
        return max(vals) if vals else None

    def fprime(self, sign: int, f0: float) -> float:
        return f0 + sign * self.fB_abs


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def parse_float_list(text: Optional[str]) -> List[float]:
    if text is None or str(text).strip() == "":
        return []
    parts = re.split(r"[,;\s]+", str(text).strip())
    return [float(p) for p in parts if p]


def circular_wrap_rad(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + np.pi) % TWOPI - np.pi


def circular_mean_std(phases: Sequence[float]) -> Tuple[float, float, int]:
    arr = np.asarray([p for p in phases if np.isfinite(p)], dtype=float)
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, 0
    z = np.mean(np.exp(1j * arr))
    mean = float(np.angle(z))
    R = abs(z)
    if R <= 0:
        std = np.pi / math.sqrt(3)
    else:
        std = math.sqrt(max(0.0, -2.0 * math.log(R)))
    return mean, float(std), n


def robust_log_scatter(values: Sequence[float]) -> Tuple[float, float, int]:
    vals = np.asarray([v for v in values if np.isfinite(v) and v > 0], dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan, 0
    logs = np.log10(vals)
    return float(np.nanmedian(logs)), float(np.nanstd(logs)), int(len(vals))




def normalized_complex_scatter(values: Sequence[complex]) -> Tuple[float, int]:
    """Return RMS complex scatter normalized by a robust amplitude scale."""
    vals = np.asarray([z for z in values if np.isfinite(z.real) and np.isfinite(z.imag)], dtype=complex)
    n = len(vals)
    if n < 2:
        return np.nan, n
    centre = np.mean(vals)
    scale = float(np.nanmedian(np.abs(vals)))
    if not np.isfinite(scale) or scale <= 0:
        scale = abs(centre)
    if not np.isfinite(scale) or scale <= 0:
        return np.nan, n
    scatter = float(np.sqrt(np.nanmean(np.abs(vals - centre) ** 2)) / scale)
    return scatter, n


def summarize_coupling_phase_subset(df: pd.DataFrame) -> Dict[str, Any]:
    """Compact amplitude/phase consistency statistics for a subset of coupling matches."""
    if len(df) == 0:
        return {
            "n_phase_points": 0,
            "combination_phase_scatter_rad": np.nan,
            "log10_amp_ratio_scatter": np.nan,
            "coupling_complex_scatter": np.nan,
            "coupling_complex_points": 0,
        }
    _, std_all, nphase = circular_mean_std(df["combination_phase_rad"].dropna().to_numpy())
    group_stds = []
    for L_value, grp in df.groupby("L"):
        _mean_L, std_L, n_L = circular_mean_std(grp["combination_phase_rad"].dropna().to_numpy())
        if n_L >= 2 and np.isfinite(std_L):
            group_stds.append(std_L)
    phase_scatter = float(np.nanmedian(group_stds)) if group_stds else std_all
    _, amp_std, _nrat = robust_log_scatter(df["amplitude_ratio_side_over_parent_secondary"].to_numpy())
    complex_group_scats = []
    complex_points = 0
    for L_value, grp in df.groupby("L"):
        zvals = []
        for _, rr in grp.iterrows():
            ratio = rr.get("amplitude_ratio_side_over_parent_secondary", np.nan)
            phi = rr.get("combination_phase_rad", np.nan)
            if np.isfinite(ratio) and ratio > 0 and np.isfinite(phi):
                zvals.append(float(ratio) * np.exp(1j * float(phi)))
        scat, n_z = normalized_complex_scatter(zvals)
        if n_z >= 2 and np.isfinite(scat):
            complex_group_scats.append(scat)
            complex_points += n_z
    return {
        "n_phase_points": int(nphase),
        "combination_phase_scatter_rad": phase_scatter,
        "log10_amp_ratio_scatter": amp_std,
        "coupling_complex_scatter": float(np.nanmedian(complex_group_scats)) if complex_group_scats else np.nan,
        "coupling_complex_points": int(complex_points),
    }


def format_float(x: Any, ndigits: int = 8) -> str:
    if x is None:
        return "--"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(xf):
        return "--"
    return f"{xf:.{ndigits}g}"


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or not np.isfinite(float(x)):
            return default
        return int(x)
    except Exception:
        return default


def md_cell(x: Any) -> str:
    """Escape a value for use in a Markdown table cell."""
    if x is None:
        return ""
    return str(x).replace("|", r"\|").replace("\n", " ")


def markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    numeric_columns: Optional[Iterable[int]] = None,
) -> List[str]:
    """Return a source-aligned GitHub Markdown table."""
    numeric = set(numeric_columns or [])
    cells = [[md_cell(v) for v in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in cells:
        if len(row) != len(headers):
            raise ValueError("Markdown table row has a different column count from its header")
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value), 3)

    def padded(value: Any, i: int) -> str:
        s = str(value)
        return s.rjust(widths[i]) if i in numeric else s.ljust(widths[i])

    out = ["| " + " | ".join(padded(h, i) for i, h in enumerate(headers)) + " |"]
    separators = []
    for i, width in enumerate(widths):
        if i in numeric:
            separators.append("-" * max(2, width - 1) + ":")
        else:
            separators.append(":" + "-" * max(2, width - 1))
    out.append("|" + "|".join(separators) + "|")
    for row in cells:
        out.append("| " + " | ".join(padded(value, i) for i, value in enumerate(row)) + " |")
    return out


def model_short_code(model: str) -> str:
    fixed = {
        "linear_beating_sinusoidal_secondary": "LB-SIN",
        "linear_beating_nonsinusoidal_secondary": "LB-NON",
        "equidistant_multiplet_frequency_grid": "MULTIPLET-GRID",
        "ambiguous_modulation_vs_coupling": "AMBIG",
        "general_combined_AM_FM_PM_modulation": "MOD-AMPM",
        "general_quadratic_coupling_secondary_secondary": "QC-SECSEC",
        "non_sinusoidal_modulation": "MOD-NON",
        "combined_AM_FM_PM_indicator": "MOD-MIX",
        "AM_like_sideband_scaling_indicator": "MOD-AM",
        "FM_PM_like_sideband_scaling_indicator": "MOD-PM",
    }
    if model in fixed:
        return fixed[model]
    suffix = "-R" if model.endswith("_right_fprime") else ("-L" if model.endswith("_left_fprime") else "")
    if model.startswith("quadratic_coupling_sinusoidal_secondary_"):
        return "QC-SIN" + suffix
    if model.startswith("quadratic_coupling_nonsinusoidal_secondary_"):
        return "QC-NON" + suffix
    if model.startswith("general_nonlinear_coupling_"):
        return "QC-GEN" + suffix
    return model


PHYSICAL_SELECTION_CODES = {
    "modulation_strong": "MOD-STRONG",
    "modulation_lean": "MOD-LEAN",
    "blazhko_modlike": "BL-MODLIKE",
    "coupling_lean": "COUP-LEAN",
    "coupling_strong": "COUP-STRONG",
    "ambiguous": "AMBIG",
}


def physical_selection_code(value: Any) -> str:
    """Return a compact report label for the five physical decision classes."""
    return PHYSICAL_SELECTION_CODES.get(str(value), "")


# -----------------------------------------------------------------------------
# Input readers
# -----------------------------------------------------------------------------

def _tokenize(line: str) -> List[str]:
    return re.split(r"\s+", line.strip())


def _is_number(text: str) -> bool:
    try:
        float(text.replace("D", "E").replace("d", "e"))
        return True
    except Exception:
        return False


def read_whitespace_flexible(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    """Flexible reader for VizieR/ASCII whitespace tables.

    It searches for a header line containing the requested column names and then
    extracts those columns from subsequent numeric rows. Extra trailing columns
    are ignored. Separator rows such as '--- --- ---' are skipped.
    Headerless whitespace files are handled by pandas with integer columns.
    """
    if getattr(args, "no_header", False):
        return pd.read_csv(
            path, sep=r"\s+", engine="python", comment=args.comment,
            header=None, skiprows=getattr(args, "skiprows", 0), on_bad_lines="skip"
        )
    lines = path.read_text(errors="replace").splitlines()[getattr(args, "skiprows", 0):]
    ignore_phases = bool(getattr(args, "ignore_phases", False) or getattr(args, "frequency_only", False))
    needed = [args.freq_col, args.amp_col] if ignore_phases else [args.freq_col, args.amp_col, args.phase_col]
    optional = [c for c in (args.phase_col, args.snr_col, args.label_col) if c and c not in needed]
    header_tokens: Optional[List[str]] = None
    header_i: Optional[int] = None

    # First try exact tokens in a whitespace header.
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if args.comment and stripped.startswith(args.comment):
            stripped_no_comment = stripped.lstrip(args.comment).strip()
        else:
            stripped_no_comment = stripped
        toks = _tokenize(stripped_no_comment)
        if all(col in toks for col in needed):
            header_tokens = toks
            header_i = i
            break

    if header_tokens is None:
        # Fall back to pandas fixed-width/whitespace reader with skipped comments.
        try:
            return pd.read_csv(path, sep=r"\s+", engine="python", comment=args.comment, on_bad_lines="skip")
        except Exception as exc:
            raise ValueError(
                f"Could not find a whitespace header containing columns {needed}. "
                f"Try inspecting `head -30 {path}` or give --separator comma/tab."
            ) from exc

    col_index = {col: header_tokens.index(col) for col in needed if col in header_tokens}
    for col in optional:
        if col in header_tokens:
            col_index[col] = header_tokens.index(col)

    records: List[Dict[str, Any]] = []
    max_needed_index = max(col_index.values())
    for line in lines[(header_i or 0) + 1:]:
        s = line.strip()
        if not s:
            continue
        if args.comment and s.startswith(args.comment):
            continue
        if set(s.replace(" ", "")) <= {"-", "="}:
            continue
        toks = _tokenize(s)
        if len(toks) <= max_needed_index:
            continue
        check_cols = [args.freq_col, args.amp_col]
        if not ignore_phases and args.phase_col in col_index:
            check_cols.append(args.phase_col)
        if not all(_is_number(toks[col_index[col]]) for col in check_cols):
            continue
        rec = {col: toks[idx] for col, idx in col_index.items() if idx < len(toks)}
        records.append(rec)

    if not records:
        raise ValueError(f"No numeric data rows found after header in {path}")
    return pd.DataFrame.from_records(records)




def _normalise_separator(separator: Optional[str]) -> Tuple[Optional[str], str]:
    """Return (sep, engine) for pandas.read_csv."""
    if separator is None or str(separator).strip() == "":
        return None, "python"
    sep = str(separator)
    if sep.lower() in {"tab", "\\t"}:
        return "\t", "python"
    if sep.lower() in {"space", "whitespace", "ws"}:
        return r"\s+", "python"
    return sep, "python"


def _is_int_like(x: Any) -> bool:
    try:
        int(str(x))
        return True
    except Exception:
        return False


def _column_by_name_or_index(
    df: pd.DataFrame,
    *,
    name_value: Optional[str],
    index_value: Optional[int],
    default_index_1based: Optional[int],
    no_header: bool,
    required: bool,
    label: str,
) -> Optional[pd.Series]:
    """Select a column by name or by 1-based index."""
    idx_1based = index_value
    if idx_1based is None and name_value is not None and _is_int_like(name_value):
        idx_1based = int(str(name_value))
    if idx_1based is None and no_header and default_index_1based is not None:
        idx_1based = default_index_1based
    if idx_1based is not None:
        idx0 = int(idx_1based) - 1
        if idx0 < 0 or idx0 >= df.shape[1]:
            raise ValueError(
                f"Column index for {label} is out of range: {idx_1based}. "
                f"The input table has {df.shape[1]} columns. Column indices are 1-based."
            )
        return df.iloc[:, idx0]
    if name_value and name_value in df.columns:
        return df[name_value]
    if required:
        raise ValueError(
            f"Required {label} column '{name_value}' not found. "
            f"Available columns: {list(df.columns)}. For headerless files use "
            f"--no-header and column numbers such as --{label}-col-index N, "
            f"or pass a number to --{label}-col."
        )
    return None

def read_frequency_table(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    sep = args.separator
    try:
        if sep in ("whitespace", "space", "ws"):
            raw = read_whitespace_flexible(path, args)
        elif getattr(args, "no_header", False):
            sep_value, engine = _normalise_separator(sep)
            raw = pd.read_csv(
                path, sep=sep_value, engine=engine, comment=args.comment,
                header=None, skiprows=getattr(args, "skiprows", 0), on_bad_lines="skip"
            )
        elif sep == "tab":
            raw = pd.read_csv(path, sep="\t", comment=args.comment, skiprows=getattr(args, "skiprows", 0), on_bad_lines="skip")
        elif sep == "comma":
            raw = pd.read_csv(path, sep=",", comment=args.comment, skiprows=getattr(args, "skiprows", 0), on_bad_lines="skip")
        elif sep:
            raw = pd.read_csv(path, sep=sep, comment=args.comment, skiprows=getattr(args, "skiprows", 0), on_bad_lines="skip")
        else:
            raw = pd.read_csv(path, sep=None, engine="python", comment=args.comment, skiprows=getattr(args, "skiprows", 0), on_bad_lines="skip")
    except Exception:
        # Last-resort fallback for auto mode.
        raw = read_whitespace_flexible(path, args)

    freq_series = _column_by_name_or_index(
        raw, name_value=args.freq_col, index_value=getattr(args, "freq_col_index", None),
        default_index_1based=1, no_header=getattr(args, "no_header", False), required=True, label="freq"
    )
    amp_series = _column_by_name_or_index(
        raw, name_value=args.amp_col, index_value=getattr(args, "amp_col_index", None),
        default_index_1based=2, no_header=getattr(args, "no_header", False), required=True, label="amp"
    )
    ignore_phases = bool(getattr(args, "ignore_phases", False) or getattr(args, "frequency_only", False))
    phase_series = _column_by_name_or_index(
        raw, name_value=args.phase_col, index_value=getattr(args, "phase_col_index", None),
        default_index_1based=3, no_header=getattr(args, "no_header", False), required=not ignore_phases, label="phase"
    )
    snr_series = _column_by_name_or_index(
        raw, name_value=args.snr_col, index_value=getattr(args, "snr_col_index", None),
        default_index_1based=None, no_header=getattr(args, "no_header", False), required=False, label="snr"
    )
    label_series = _column_by_name_or_index(
        raw, name_value=args.label_col, index_value=getattr(args, "label_col_index", None),
        default_index_1based=None, no_header=getattr(args, "no_header", False), required=False, label="label"
    )

    out = pd.DataFrame()
    out["input_index"] = np.arange(len(raw), dtype=int)
    out["frequency"] = pd.to_numeric(freq_series, errors="coerce")
    out["amplitude"] = pd.to_numeric(amp_series, errors="coerce")
    if ignore_phases or phase_series is None:
        out["phase_input"] = np.nan
    else:
        out["phase_input"] = pd.to_numeric(phase_series, errors="coerce")

    if snr_series is not None:
        out["snr"] = pd.to_numeric(snr_series, errors="coerce")
    else:
        out["snr"] = np.nan
    if label_series is not None:
        out["label"] = label_series.astype(str)
    else:
        out["label"] = ""

    if ignore_phases:
        phase_rad = np.full(len(out), np.nan, dtype=float)
    else:
        phase = out["phase_input"].to_numpy(float)
        if args.phase_unit == "cycles":
            phase_rad = TWOPI * phase
        elif args.phase_unit == "deg":
            phase_rad = np.deg2rad(phase)
        else:
            phase_rad = phase
        if args.phase_convention == "cosine":
            # Convert A cos(wt+phi_c) to equivalent A sin(wt+phi_s).
            phase_rad = phase_rad + np.pi / 2.0
    out["phase_rad"] = circular_wrap_rad(phase_rad)

    out = out.replace([np.inf, -np.inf], np.nan)
    required_subset = ["frequency", "amplitude"] if ignore_phases else ["frequency", "amplitude", "phase_rad"]
    out = out.dropna(subset=required_subset).copy()
    out = out[out["frequency"] > 0].copy()
    out = out[out["amplitude"] >= args.min_amp].copy()
    if args.min_snr is not None and np.isfinite(args.min_snr):
        out = out[(out["snr"].isna()) | (out["snr"] >= args.min_snr)].copy()
    if args.max_freq is not None:
        out = out[out["frequency"] <= args.max_freq].copy()

    if args.deduplicate:
        before = len(out)
        dedup_subset = ["frequency", "amplitude"] if ignore_phases else ["frequency", "amplitude", "phase_rad"]
        out = out.drop_duplicates(subset=dedup_subset, keep="first").copy()
        out.attrs["n_deduplicated"] = before - len(out)
    else:
        out.attrs["n_deduplicated"] = 0

    out = out.sort_values("frequency").reset_index(drop=True)
    out["row_id"] = np.arange(len(out), dtype=int)
    return out


# -----------------------------------------------------------------------------
# Matching helpers
# -----------------------------------------------------------------------------

def choose_tolerance(args: argparse.Namespace) -> float:
    if args.tol is not None:
        return float(args.tol)
    if args.baseline is not None and args.baseline > 0:
        return float(args.tol_factor) / float(args.baseline)
    return 1e-4


def find_best_match(df: pd.DataFrame, expected_freq: float, tol: float, exclude_indices: Optional[set[int]] = None) -> Optional[pd.Series]:
    if expected_freq <= 0 or len(df) == 0:
        return None
    exclude_indices = exclude_indices or set()
    tmp = df.loc[~df["input_index"].isin(exclude_indices)].copy()
    if len(tmp) == 0:
        return None
    tmp["abs_delta"] = np.abs(tmp["frequency"] - expected_freq)
    cand = tmp[tmp["abs_delta"] <= tol].copy()
    if len(cand) == 0:
        return None

    # In dense frequency lists there can be several peaks inside one formal
    # resolution element.  Choosing the nearest peak can then select a tiny
    # residual instead of the physically relevant component and can destroy the
    # amplitude/phase diagnostics.  The default "balanced" rule therefore
    # prefers high-amplitude peaks but still penalizes distance from the expected
    # frequency.  The old behaviour is available with --match-selection nearest.
    try:
        mode = getattr(args_global, "match_selection", "balanced")
        penalty = float(getattr(args_global, "match_distance_penalty", 0.15))
    except NameError:
        mode = "nearest"
        penalty = 0.15

    if mode == "nearest":
        cand = cand.sort_values(["abs_delta", "amplitude"], ascending=[True, False])
    elif mode == "strongest":
        cand = cand.sort_values(["amplitude", "abs_delta"], ascending=[False, True])
    else:
        amp_max = float(cand["amplitude"].max()) if len(cand) else 0.0
        if amp_max <= 0 or not np.isfinite(amp_max):
            cand["match_score"] = -cand["abs_delta"] / max(tol, 1e-12)
        else:
            cand["match_score"] = cand["amplitude"] / amp_max - penalty * (cand["abs_delta"] / max(tol, 1e-12))
        cand = cand.sort_values(["match_score", "amplitude", "abs_delta"], ascending=[False, False, True])
    return cand.iloc[0]


def make_match(df: pd.DataFrame, category: str, model: str, label: str, expected_freq: float, tol: float, notes: str = "") -> FreqMatch:
    row = find_best_match(df, expected_freq, tol)
    if row is None:
        return FreqMatch(category, model, label, float(expected_freq), None, None, None, None, None, False, notes)
    return FreqMatch(
        category=category,
        model=model,
        label=label,
        expected_frequency=float(expected_freq),
        observed_frequency=float(row["frequency"]),
        delta_frequency=float(row["frequency"] - expected_freq),
        input_index=int(row["input_index"]),
        amplitude=float(row["amplitude"]),
        phase_rad=float(row["phase_rad"]),
        matched=True,
        notes=notes,
    )


def match_to_df(matches: Sequence[FreqMatch]) -> pd.DataFrame:
    return pd.DataFrame([asdict(m) for m in matches]) if matches else pd.DataFrame()


def count_matched(matches: Sequence[FreqMatch], filt: Optional[Callable[[FreqMatch], bool]] = None) -> int:
    if filt is None:
        return sum(1 for m in matches if m.matched)
    return sum(1 for m in matches if m.matched and filt(m))


# -----------------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------------

def _f0_candidate_metrics(
    df: pd.DataFrame,
    row: pd.Series,
    nmax: int,
    tol: float,
) -> Dict[str, Any]:
    """Measure how well one observed frequency anchors a primary harmonic comb.

    A single large peak is not sufficient evidence for the primary frequency:
    in strongly modulated pulsators a harmonic side peak can be larger than the
    true fundamental.  The useful evidence is instead the *whole* sequence
    f, 2f, 3f, ... .  This helper records both its breadth and its strength.
    """
    f = float(row["frequency"])
    max_frequency = float(df["frequency"].max())
    n_tested = min(int(nmax), int(math.floor((max_frequency + tol) / f)))
    n_tested = max(1, n_tested)

    weights = np.asarray([1.0 / math.sqrt(k) for k in range(1, n_tested + 1)], dtype=float)
    matched_orders: List[int] = []
    matched_amplitudes: List[float] = []
    matched_input_indices: List[int] = []
    used_input_indices: set[int] = set()
    matched_weight = 0.0
    residual_weighted_sum = 0.0

    for k, weight in enumerate(weights, start=1):
        # One observed frequency may support at most one harmonic order for a
        # given f0 candidate.  This matters when f is comparable to or smaller
        # than the matching tolerance: overlapping k*f windows otherwise let
        # the same low-frequency peak masquerade as many consecutive
        # harmonics and dominate the automatic f0 ranking.
        available = df[~df["input_index"].isin(used_input_indices)]
        match = find_best_match(available, k * f, tol)
        if match is None:
            continue
        input_index = int(match["input_index"])
        used_input_indices.add(input_index)
        matched_orders.append(k)
        matched_input_indices.append(input_index)
        matched_amplitudes.append(max(float(match["amplitude"]), 0.0))
        matched_weight += float(weight)
        scaled_delta = abs(float(match["frequency"]) - k * f) / max(tol, 1e-15)
        residual_weighted_sum += float(weight) * max(0.0, 1.0 - scaled_delta)

    matched_set = set(matched_orders)
    contiguous = 0
    for k in range(1, n_tested + 1):
        if k not in matched_set:
            break
        contiguous = k

    total_weight = float(np.sum(weights))
    weighted_coverage = matched_weight / total_weight if total_weight > 0 else 0.0
    residual_quality = residual_weighted_sum / matched_weight if matched_weight > 0 else 0.0

    return {
        "frequency": f,
        "input_index": int(row["input_index"]),
        "fundamental_amplitude": float(row["amplitude"]),
        "n_harmonics_tested": n_tested,
        "n_harmonics_matched": len(matched_orders),
        "matched_orders": ",".join(str(k) for k in matched_orders),
        "matched_input_indices": ",".join(str(i) for i in matched_input_indices),
        "contiguous_harmonics": contiguous,
        "weighted_coverage": weighted_coverage,
        "total_harmonic_amplitude": float(np.sum(matched_amplitudes)),
        "residual_quality": residual_quality,
    }


def rank_f0_candidates(df: pd.DataFrame, args: argparse.Namespace, tol: float) -> pd.DataFrame:
    """Rank observed frequencies as possible primary-frequency anchors.

    The score deliberately combines independent pieces of evidence.  Coverage
    alone would favour a high-frequency candidate for which only one or two
    multiples can be tested; total amplitude alone would reproduce the old
    largest-peak failure.  Breadth and a contiguous low-order sequence prevent
    both pathologies.
    """
    cand = df.copy()
    if args.f0_min is not None:
        cand = cand[cand["frequency"] >= args.f0_min]
    if len(cand) == 0:
        raise ValueError("No frequency remains for automatic f0 selection. Check --f0-min or input table.")

    max_candidates = int(getattr(args, "f0_auto_max_candidates", 200) or 0)
    if max_candidates > 0 and len(cand) > max_candidates:
        cand = cand.nlargest(max_candidates, "amplitude")

    records = [
        _f0_candidate_metrics(df, row, int(args.nmax), tol)
        for _, row in cand.iterrows()
    ]
    ranking = pd.DataFrame(records)
    if len(ranking) == 0:
        raise ValueError("No frequency remains for automatic f0 selection.")

    max_matched = max(int(ranking["n_harmonics_matched"].max()), 1)
    max_contiguous = max(int(ranking["contiguous_harmonics"].max()), 1)
    max_harmonic_amplitude = max(float(ranking["total_harmonic_amplitude"].max()), 1e-15)

    ranking["breadth_score"] = ranking["n_harmonics_matched"] / max_matched
    ranking["contiguous_score"] = ranking["contiguous_harmonics"] / max_contiguous
    ranking["amplitude_support_score"] = ranking["total_harmonic_amplitude"] / max_harmonic_amplitude
    ranking["score_0_1"] = (
        0.30 * ranking["weighted_coverage"]
        + 0.25 * ranking["breadth_score"]
        + 0.15 * ranking["contiguous_score"]
        + 0.20 * ranking["amplitude_support_score"]
        + 0.10 * ranking["residual_quality"]
    ).clip(0.0, 1.0)

    min_harmonics = max(1, int(getattr(args, "f0_auto_min_harmonics", 2)))
    ranking["eligible"] = ranking["n_harmonics_matched"] >= min_harmonics
    ranking["selected"] = False
    ranking = ranking.sort_values(
        [
            "eligible",
            "score_0_1",
            "n_harmonics_matched",
            "contiguous_harmonics",
            "total_harmonic_amplitude",
            "frequency",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)

    # If no frequency has the requested harmonic support, retaining a clearly
    # labelled largest-amplitude fallback is safer than silently returning no f0.
    if not bool(ranking["eligible"].any()):
        strongest_idx = int(ranking["fundamental_amplitude"].idxmax())
        ranking.loc[strongest_idx, "selected"] = True
        ranking["selection_note"] = "fallback: no candidate met --f0-auto-min-harmonics"
        ranking = pd.concat(
            [ranking.loc[[strongest_idx]], ranking.drop(index=strongest_idx)],
            ignore_index=True,
        )
    else:
        ranking.loc[0, "selected"] = True
        ranking["selection_note"] = "harmonic-series ranking"

    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=int))
    return ranking


def determine_f0(
    df: pd.DataFrame,
    args: argparse.Namespace,
    tol: float,
) -> Tuple[float, int, pd.DataFrame]:
    if args.f0 is not None:
        row = find_best_match(df, args.f0, tol)
        idx = int(row["input_index"]) if row is not None else -1
        args.f0_selection_method_used = "manual --f0"
        return float(args.f0), idx, pd.DataFrame()
    cand = df.copy()
    if args.f0_min is not None:
        cand = cand[cand["frequency"] >= args.f0_min]
    if len(cand) == 0:
        raise ValueError("No frequency remains for automatic f0 selection. Check --f0-min or input table.")
    if getattr(args, "f0_auto_method", "harmonic-series") == "strongest":
        row = cand.sort_values("amplitude", ascending=False).iloc[0]
        args.f0_selection_method_used = "automatic strongest peak (legacy)"
        return float(row["frequency"]), int(row["input_index"]), pd.DataFrame()

    ranking = rank_f0_candidates(df, args, tol)
    selected = ranking[ranking["selected"]].iloc[0]
    note = str(selected.get("selection_note", "harmonic-series ranking"))
    args.f0_selection_method_used = f"automatic harmonic-series ({note})"
    return float(selected["frequency"]), int(selected["input_index"]), ranking


def detect_harmonics(df: pd.DataFrame, f0: float, nmax: int, tol: float) -> Tuple[List[FreqMatch], Dict[int, FreqMatch]]:
    matches: List[FreqMatch] = []
    hmap: Dict[int, FreqMatch] = {}
    for k in range(1, nmax + 1):
        m = make_match(df, "primary_harmonic", "primary", f"{k} f0", k * f0, tol)
        matches.append(m)
        if m.matched:
            hmap[k] = m
    return matches, hmap


def _cluster_representative_amplitude(rows: List[Tuple[float, int, int, pd.Series]]) -> float:
    vals = []
    for _fb_abs, _sign, _harmonic, row in rows:
        try:
            vals.append(float(row["amplitude"]))
        except Exception:
            pass
    return max(vals) if vals else 0.0


def _cluster_harmonic_support(rows: List[Tuple[float, int, int, pd.Series]]) -> int:
    return len({int(harmonic) for _fb_abs, _sign, harmonic, _row in rows})


def _cluster_weighted_centre(rows: List[Tuple[float, int, int, pd.Series]]) -> float:
    fbs = np.asarray([float(fb_abs) for fb_abs, _sign, _harmonic, _row in rows], dtype=float)
    amps = np.asarray([
        max(float(row.get("amplitude", 0.0)), 0.0)
        for _fb_abs, _sign, _harmonic, row in rows
    ], dtype=float)
    if len(fbs) == 0:
        return np.nan
    if np.sum(amps) > 0:
        return float(np.average(fbs, weights=amps))
    return float(np.mean(fbs))


def _collapse_offsets(
    offsets: List[Tuple[float, int, int, pd.Series]],
    tol: float,
    max_mult: int,
) -> List[Tuple[float, List[Tuple[float, int, int, pd.Series]]]]:
    """Group by absolute fB and collapse integer multiples.

    Older versions kept the smallest offset first.  That is good for a
    Blazhko frequency and its harmonics, but it fails for general quadratic
    coupling: a weak secondary--secondary difference near f0 can be smaller
    than the true close mode separation and can incorrectly absorb the real
    candidates.  We therefore keep the offsets supported at the largest number
    of primary harmonics first, use peak amplitude only as a secondary key, and
    collapse weaker integer multiples only into already-kept better-supported
    systems.
    """
    offsets = sorted(offsets, key=lambda x: x[0])
    clusters: List[Tuple[float, List[Tuple[float, int, int, pd.Series]]]] = []
    for fb_abs, sign, harmonic, row in offsets:
        placed = False
        for i, (center, rows) in enumerate(clusters):
            if abs(fb_abs - center) <= tol:
                rows.append((fb_abs, sign, harmonic, row))
                clusters[i] = (_cluster_weighted_centre(rows), rows)
                placed = True
                break
        if not placed:
            clusters.append((fb_abs, [(fb_abs, sign, harmonic, row)]))

    if not clusters or max_mult <= 1:
        return clusters

    # Prefer repeatedly supported offsets. This prevents a single strong cross
    # term from becoming the base frequency of a whole candidate system.
    clusters_by_strength = sorted(
        clusters,
        key=lambda x: (-_cluster_harmonic_support(x[1]), -_cluster_representative_amplitude(x[1]), x[0]),
    )
    kept: List[Tuple[float, List[Tuple[float, int, int, pd.Series]]]] = []
    for fb_abs, rows in clusters_by_strength:
        is_multiple = False
        amp = _cluster_representative_amplitude(rows)
        support = _cluster_harmonic_support(rows)
        for base, base_rows in kept:
            base_amp = _cluster_representative_amplitude(base_rows)
            base_support = _cluster_harmonic_support(base_rows)
            for m in range(2, max_mult + 1):
                if abs(fb_abs - m * base) <= max(2.0 * tol, 0.02 * base):
                    # Collapse only if the present candidate is not stronger
                    # than the proposed base.  This keeps true high-amplitude
                    # close modes, while still collapsing fm, 2fm, 3fm in
                    # ordinary modulation spectra.
                    if amp <= 1.25 * base_amp and support <= base_support:
                        is_multiple = True
                    break
            if is_multiple:
                break
        if not is_multiple:
            kept.append((fb_abs, rows))

    return sorted(kept, key=lambda x: x[0])


def assign_candidate_families(
    candidates: Sequence[Candidate],
    tol: float,
    max_order: int,
    tol_factor: float = 0.5,
) -> List[Candidate]:
    """Annotate harmonically related close-frequency offsets as one family.

    The candidates are deliberately retained individually: an integer ratio
    between two close modes is not, by itself, proof that one is a modulation
    harmonic.  Family membership records the degeneracy and supplies a single
    fundamental modulation spacing for family-aware diagnostics.
    """
    cands = list(candidates)
    if not cands:
        return cands

    relation_tol = max(float(tol_factor) * tol, 0.0)
    components: List[List[int]] = []

    def harmonic_order(root: Candidate, member: Candidate) -> Optional[int]:
        small = float(root.fB_abs)
        large = float(member.fB_abs)
        if small <= 0 or large < small:
            return None
        order = int(round(large / small))
        if order < 1 or order > max_order:
            return None
        allowed = max(relation_tol, 0.02 * small)
        return order if abs(large - order * small) <= allowed else None

    # Greedy assignment in increasing spacing keeps every family anchored to
    # an actually detected fundamental and avoids transitive chains whose end
    # points are not harmonically related within the configured maximum order.
    for idx in sorted(range(len(cands)), key=lambda i: cands[i].fB_abs):
        placed = False
        for component in components:
            root_idx = component[0]
            if harmonic_order(cands[root_idx], cands[idx]) is not None:
                component.append(idx)
                placed = True
                break
        if not placed:
            components.append([idx])

    for family_id, component in enumerate(components, start=1):
        root_idx = component[0]
        root = cands[root_idx]
        fundamental = float(root.fB_abs)
        orders: Dict[int, int] = {}
        for idx in component:
            order = harmonic_order(root, cands[idx]) or 1
            orders[idx] = order
        max_family_order = max(orders.values())
        member_ids = ",".join(str(cands[idx].candidate_id) for idx in component)
        member_orders = ",".join(
            f"c{cands[idx].candidate_id}:{orders[idx]}" for idx in component
        )
        for idx in component:
            cands[idx].family_id = family_id
            cands[idx].family_fundamental = fundamental
            cands[idx].family_order = orders[idx]
            cands[idx].family_max_order = max_family_order
            cands[idx].family_representative_id = root.candidate_id
            cands[idx].family_member_ids = member_ids
            cands[idx].family_member_orders = member_orders
            cands[idx].family_representative = idx == root_idx
    return cands


def modulation_lmax_for_candidate(cand: Candidate, base_lmax: int) -> int:
    """Return the side-order range needed to cover a modulation family."""
    if cand.family_representative and cand.family_max_order > 1:
        return max(base_lmax, base_lmax * cand.family_max_order)
    return base_lmax


def detect_candidates(
    df: pd.DataFrame,
    f0: float,
    hmap: Dict[int, FreqMatch],
    args: argparse.Namespace,
    tol: float,
) -> List[Candidate]:
    """Detect candidate spacings around every observed primary harmonic.

    A candidate need not have a measurable side peak next to ``f0``.  Repeated
    offsets around higher ``k f0`` terms are clustered into the same spacing,
    which is essential for weak secondary modulations such as the one in
    V366 Lyr.  The actual ``f'=f0+/-delta`` fields remain empty when no such
    component is observed, so a grid-only modulation candidate cannot
    accidentally become a quadratic-coupling secondary oscillator.
    """
    offsets: List[Tuple[float, int, int, pd.Series]] = []

    # Manual signed offsets are added without requiring an observed side peak.
    manual_candidates: List[Candidate] = []
    for fb in parse_float_list(args.fb):
        if abs(fb) <= args.min_fb:
            continue
        sign = 1 if fb > 0 else -1
        row = find_best_match(df, f0 + fb, tol)
        c = Candidate(
            candidate_id=0,
            fB_abs=abs(fb),
            signs="manual_right" if sign > 0 else "manual_left",
            right_frequency=(float(row["frequency"]) if row is not None and sign > 0 else None),
            left_frequency=(float(row["frequency"]) if row is not None and sign < 0 else None),
            right_input_index=(int(row["input_index"]) if row is not None and sign > 0 else None),
            left_input_index=(int(row["input_index"]) if row is not None and sign < 0 else None),
            right_amplitude=(float(row["amplitude"]) if row is not None and sign > 0 else None),
            left_amplitude=(float(row["amplitude"]) if row is not None and sign < 0 else None),
            right_phase_rad=(float(row["phase_rad"]) if row is not None and sign > 0 else None),
            left_phase_rad=(float(row["phase_rad"]) if row is not None and sign < 0 else None),
            source="manual",
            discovery_harmonics="1" if row is not None else "",
            discovery_harmonic_count=1 if row is not None else 0,
            discovery_peak_count=1 if row is not None else 0,
            discovery_total_amplitude=float(row["amplitude"]) if row is not None else 0.0,
            discovery_max_amplitude=float(row["amplitude"]) if row is not None else 0.0,
        )
        manual_candidates.append(c)

    if not args.no_auto_fb:
        available_orders = set(int(k) for k in hmap)
        for _, row in df.iterrows():
            freq = float(row["frequency"])
            if not np.isfinite(freq) or freq <= 0:
                continue
            harmonic = int(round(freq / f0)) if f0 > 0 else 0
            if harmonic not in available_orders:
                continue
            primary = hmap[harmonic]
            if primary.input_index is not None and int(row["input_index"]) == int(primary.input_index):
                continue
            fb = freq - harmonic * f0
            if abs(fb) < args.min_fb or abs(fb) > args.side_window:
                continue
            sign = 1 if fb > 0 else -1
            offsets.append((abs(fb), sign, harmonic, row))

    clusters = _collapse_offsets(offsets, tol=args.fb_cluster_tol_factor * tol, max_mult=(1 if args.no_collapse_fb_multiples else args.fb_collapse_max_multiple))
    candidates: List[Candidate] = []
    for fb_abs, rows in clusters:
        right_rows = [r for _offset, sign, harmonic, r in rows if sign > 0 and harmonic == 1]
        left_rows = [r for _offset, sign, harmonic, r in rows if sign < 0 and harmonic == 1]
        right = sorted(right_rows, key=lambda r: float(r["amplitude"]), reverse=True)[0] if right_rows else None
        left = sorted(left_rows, key=lambda r: float(r["amplitude"]), reverse=True)[0] if left_rows else None
        all_signs = {sign for _offset, sign, _harmonic, _row in rows}
        signs = "both" if all_signs == {-1, 1} else ("right" if 1 in all_signs else "left")
        harmonic_orders = sorted({int(harmonic) for _offset, _sign, harmonic, _row in rows})
        amplitudes = [max(float(r.get("amplitude", 0.0)), 0.0) for _offset, _sign, _harmonic, r in rows]
        rep_fb_abs = float(_cluster_weighted_centre(rows))
        candidates.append(Candidate(
            candidate_id=0,
            fB_abs=rep_fb_abs,
            signs=signs,
            right_frequency=float(right["frequency"]) if right is not None else None,
            left_frequency=float(left["frequency"]) if left is not None else None,
            right_input_index=int(right["input_index"]) if right is not None else None,
            left_input_index=int(left["input_index"]) if left is not None else None,
            right_amplitude=float(right["amplitude"]) if right is not None else None,
            left_amplitude=float(left["amplitude"]) if left is not None else None,
            right_phase_rad=float(right["phase_rad"]) if right is not None else None,
            left_phase_rad=float(left["phase_rad"]) if left is not None else None,
            source="auto_all_harmonics",
            discovery_harmonics=",".join(str(x) for x in harmonic_orders),
            discovery_harmonic_count=len(harmonic_orders),
            discovery_peak_count=len(rows),
            discovery_total_amplitude=float(sum(amplitudes)),
            discovery_max_amplitude=float(max(amplitudes) if amplitudes else 0.0),
        ))

    # Merge manual and auto candidates with the same fB.
    all_cands = manual_candidates + candidates
    merged: List[Candidate] = []
    for c in sorted(all_cands, key=lambda x: x.fB_abs):
        same = None
        for m in merged:
            if abs(c.fB_abs - m.fB_abs) <= args.fb_cluster_tol_factor * tol:
                same = m
                break
        if same is None:
            merged.append(c)
        else:
            # Fill missing sides.
            for attr in ["right_frequency", "right_input_index", "right_amplitude", "right_phase_rad", "left_frequency", "left_input_index", "left_amplitude", "left_phase_rad"]:
                if getattr(same, attr) is None and getattr(c, attr) is not None:
                    setattr(same, attr, getattr(c, attr))
            if same.right_frequency is not None and same.left_frequency is not None:
                same.signs = "both"
            same.source = "manual+auto" if same.source != c.source else same.source
            harmonics = sorted({
                int(x)
                for text in (same.discovery_harmonics, c.discovery_harmonics)
                for x in str(text).split(",") if str(x).strip()
            })
            same.discovery_harmonics = ",".join(str(x) for x in harmonics)
            same.discovery_harmonic_count = len(harmonics)
            same.discovery_peak_count = max(same.discovery_peak_count, c.discovery_peak_count)
            same.discovery_total_amplitude = max(same.discovery_total_amplitude, c.discovery_total_amplitude)
            same.discovery_max_amplitude = max(same.discovery_max_amplitude, c.discovery_max_amplitude)

    # A spacing seen only once at a high harmonic is too easily produced by an
    # unrelated mode or alias.  Preserve the legacy f0-side candidates, but
    # require repeated harmonic support when the f0 side itself is absent.
    min_support = max(1, int(getattr(args, "fb_min_harmonic_support", 2) or 2))
    merged = [
        c for c in merged
        if c.source.startswith("manual")
        or c.right_frequency is not None
        or c.left_frequency is not None
        or c.discovery_harmonic_count >= min_support
    ]

    # Discard very weak automatic close peaks as candidate generators.  They
    # remain in the frequency list and can still be matched as combination
    # terms, but they should not start a whole modulation/beating system.
    auto_amps = [c.representative_amplitude or 0.0 for c in merged if not c.source.startswith("manual")]
    max_auto_amp = max(auto_amps) if auto_amps else 0.0
    if max_auto_amp > 0 and args.min_candidate_rel_amp > 0:
        merged = [
            c for c in merged
            if c.source.startswith("manual")
            or c.discovery_harmonic_count >= max(3, min_support)
            or (c.representative_amplitude or 0.0) >= args.min_candidate_rel_amp * max_auto_amp
        ]

    # Repeated occurrence at several harmonics is stronger spacing evidence
    # than one very high-amplitude isolated side peak.  Manual candidates retain
    # priority, then support count, peak count, and amplitude determine rank.
    merged = sorted(
        merged,
        key=lambda c: (
            0 if c.source.startswith("manual") else 1,
            -int(c.discovery_harmonic_count),
            -int(c.discovery_peak_count),
            -(c.representative_amplitude or 0.0),
            c.fB_abs,
        ),
    )[: args.max_candidates]
    for i, c in enumerate(merged, start=1):
        c.candidate_id = i
    return assign_candidate_families(
        merged,
        tol=tol,
        max_order=max(1, int(args.fb_collapse_max_multiple)),
        tol_factor=float(getattr(args, "fb_family_tol_factor", args.fb_cluster_tol_factor)),
    )


def detect_low_terms(df: pd.DataFrame, cand: Candidate, lmax: int, tol: float) -> List[FreqMatch]:
    out: List[FreqMatch] = []
    for l in range(1, lmax + 1):
        out.append(make_match(df, "low_frequency", f"low_c{cand.candidate_id}", f"{l} fB_c{cand.candidate_id}", l * cand.fB_abs, tol))
    return out


def detect_linear_sequences(df: pd.DataFrame, f0: float, cand: Candidate, lmax: int, tol: float) -> Tuple[List[FreqMatch], Dict[Tuple[int, int], FreqMatch]]:
    out: List[FreqMatch] = []
    smap: Dict[Tuple[int, int], FreqMatch] = {}
    signs = []
    if cand.right_frequency is not None or "manual_right" in cand.signs or cand.signs == "manual":
        signs.append(1)
    if cand.left_frequency is not None or "manual_left" in cand.signs:
        signs.append(-1)
    if not signs:
        return out, smap
    for sign in signs:
        side = "plus" if sign > 0 else "minus"
        fprime = f0 + sign * cand.fB_abs
        if fprime <= 0:
            continue
        for l in range(1, lmax + 1):
            expected = l * fprime
            m = make_match(df, "secondary_harmonic_sequence", f"linear_sequence_c{cand.candidate_id}", f"{l} f'_{side}_c{cand.candidate_id}", expected, tol, notes=f"f'={fprime:.10g}")
            out.append(m)
            if m.matched:
                smap[(sign, l)] = m
    return out, smap


def find_secondary_modulation_grid_overlaps(
    cand: Candidate,
    candidates: Sequence[Candidate],
    secondary_map: Dict[Tuple[int, int], FreqMatch],
    base_modulation_lmax: int,
    tol: float,
) -> Dict[Tuple[int, int], str]:
    """Find higher secondary harmonics degenerate with a modulation grid.

    For example, if delta_2 ~= delta_1/2, then
    Every own-secondary term is algebraically identical to a point on the
    candidate's own modulation grid,

        l(f0+s delta) = l f0 + s l delta.

    It therefore cannot be counted as independent evidence for a non-sinusoidal
    secondary oscillator.  The same exclusion is applied when the term also
    falls on another candidate's modulation grid.  Fundamental (l=1) secondary
    peaks remain available as the one-sided f' hypotheses.
    """
    overlaps: Dict[Tuple[int, int], str] = {}
    for (secondary_sign, order), match in secondary_map.items():
        if order < 2 or not match.matched:
            continue
        secondary_offset = secondary_sign * order * cand.fB_abs
        reasons: List[str] = [
            f"c{cand.candidate_id} own modulation grid n={order},L={secondary_sign * order:+d} (algebraic identity)"
        ]
        for other in candidates:
            if other.candidate_id == cand.candidate_id:
                continue
            other_lmax = modulation_lmax_for_candidate(other, base_modulation_lmax)
            for grid_order in range(1, other_lmax + 1):
                for grid_sign in (1, -1):
                    grid_offset = grid_sign * grid_order * other.fB_abs
                    allowed = max(tol, 0.02 * min(abs(secondary_offset), abs(grid_offset)))
                    if abs(secondary_offset - grid_offset) <= allowed:
                        reasons.append(
                            f"c{other.candidate_id} modulation grid L={grid_sign * grid_order:+d}"
                        )
        if reasons:
            reason = "; ".join(sorted(set(reasons)))
            overlaps[(secondary_sign, order)] = reason
            suffix = f"excluded as independent secondary harmonic: {reason}"
            match.notes = f"{match.notes}; {suffix}" if match.notes else suffix
    return overlaps


def detect_modulation_grid(df: pd.DataFrame, f0: float, cand: Candidate, nmax: int, lmax: int, tol: float) -> List[FreqMatch]:
    out: List[FreqMatch] = []
    fm = cand.fB_abs
    for n in range(1, nmax + 1):
        for L in range(1, lmax + 1):
            for sign in (1, -1):
                expected = n * f0 + sign * L * fm
                if expected <= 0:
                    continue
                label = f"{n} f0 {'+' if sign > 0 else '-'} {L} fm"
                out.append(make_match(df, "modulation_multiplet", f"modulation_c{cand.candidate_id}", label, expected, tol, notes=f"n={n};L={sign*L}"))
    return out


def detect_coupling_grid(df: pd.DataFrame, f0: float, cand: Candidate, nmax: int, lmax: int, tol: float) -> List[FreqMatch]:
    out: List[FreqMatch] = []
    fb = cand.fB_abs
    for n in range(0, nmax + lmax + 1):
        for L in range(-lmax, lmax + 1):
            if L == 0:
                continue
            expected = n * f0 + L * fb
            if expected <= 0:
                continue
            label = f"{n} f0 {L:+d} fB_c{cand.candidate_id}"
            notes = f"parent_k=n-L={n-L}; L={L}"
            out.append(make_match(df, "quadratic_grid", f"quadratic_c{cand.candidate_id}", label, expected, tol, notes=notes))
    return out


def _candidate_signs(c: Candidate) -> List[int]:
    signs: List[int] = []
    manual_right = "manual_right" in c.signs or c.signs == "manual"
    manual_left = "manual_left" in c.signs
    if c.right_frequency is not None or manual_right:
        signs.append(1)
    if c.left_frequency is not None or manual_left:
        signs.append(-1)
    return signs


def detect_secondary_secondary_quadratic_terms(
    df: pd.DataFrame,
    f0: float,
    candidates: Sequence[Candidate],
    lmax: int,
    tol: float,
) -> List[FreqMatch]:
    """Expected terms from direct quadratic products of secondary oscillations.

    For two secondary oscillations with frequencies f_i and f_j, quadratic
    products generate f_i+f_j and |f_i-f_j| (and the analogous terms for
    their harmonics).  These terms are absent in the independent-coupling
    model, where each secondary interacts only with the primary.
    """
    out: List[FreqMatch] = []
    cands = list(candidates)
    for ia in range(len(cands)):
        ci = cands[ia]
        for ja in range(ia + 1, len(cands)):
            cj = cands[ja]
            for si in _candidate_signs(ci):
                fi = f0 + si * ci.fB_abs
                if fi <= 0:
                    continue
                for sj in _candidate_signs(cj):
                    fj = f0 + sj * cj.fB_abs
                    if fj <= 0:
                        continue
                    for li in range(1, lmax + 1):
                        for lj in range(1, lmax + 1):
                            fsum = li * fi + lj * fj
                            fdiff = abs(li * fi - lj * fj)
                            labbase = f"c{ci.candidate_id},c{cj.candidate_id}; li={li}, lj={lj}"
                            if fdiff > 0:
                                out.append(make_match(
                                    df,
                                    "secondary_secondary_quadratic",
                                    "general_quadratic_secondary_secondary",
                                    f"|{li}f'_c{ci.candidate_id}-{lj}f'_c{cj.candidate_id}|",
                                    fdiff,
                                    tol,
                                    notes=labbase + "; difference",
                                ))
                            out.append(make_match(
                                df,
                                "secondary_secondary_quadratic",
                                "general_quadratic_secondary_secondary",
                                f"{li}f'_c{ci.candidate_id}+{lj}f'_c{cj.candidate_id}",
                                fsum,
                                tol,
                                notes=labbase + "; sum",
                            ))
    return out


def secondary_secondary_summary(matches: Sequence[FreqMatch], f0: float) -> Dict[str, Any]:
    matched = [m for m in matches if m.matched]
    low = [m for m in matched if (m.observed_frequency is not None and m.observed_frequency < 0.5 * f0)]
    near_harm = []
    for m in matched:
        if m.observed_frequency is None:
            continue
        n = int(round(m.observed_frequency / f0)) if f0 > 0 else 0
        if n >= 1 and abs(m.observed_frequency - n * f0) > 1e-12:
            near_harm.append(m)
    return {
        "secondary_secondary_matches": len(matched),
        "secondary_secondary_tested": len(matches),
        "secondary_secondary_low_matches": len(low),
        "secondary_secondary_near_harmonic_matches": len(near_harm),
        "secondary_secondary_coverage": len(matched) / max(1, len(matches)),
    }


def parse_n_L_from_label(label: str) -> Optional[Tuple[int, int]]:
    m = re.match(r"\s*(\d+)\s+f0\s+([+-]\d+)\s+fB", label)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"\s*(\d+)\s+f0\s+([+-])\s+(\d+)\s+fm", label)
    if m:
        sign = 1 if m.group(2) == "+" else -1
        return int(m.group(1)), sign * int(m.group(3))
    return None


# -----------------------------------------------------------------------------
# Diagnostics and scoring
# -----------------------------------------------------------------------------



def complex_coeff(m: FreqMatch) -> complex:
    """Complex observable coefficient up to a common convention factor.

    The absolute sine/cosine convention introduces a common phase offset for all
    positive-frequency components. Ratios of components are therefore usable as
    long as the input phases share the same epoch and convention.
    """
    if m.amplitude is None or m.phase_rad is None:
        return np.nan + 1j * np.nan
    return float(m.amplitude) * np.exp(1j * float(m.phase_rad))


def _complex_scatter(y: np.ndarray, yfit: np.ndarray) -> float:
    y = np.asarray(y, dtype=complex)
    yfit = np.asarray(yfit, dtype=complex)
    good = np.isfinite(y.real) & np.isfinite(y.imag) & np.isfinite(yfit.real) & np.isfinite(yfit.imag)
    if good.sum() == 0:
        return np.nan
    yy = y[good]
    rr = yy - yfit[good]
    denom = float(np.sqrt(np.mean(np.abs(yy) ** 2)))
    if denom <= 0 or not np.isfinite(denom):
        denom = 1.0
    return float(np.sqrt(np.mean(np.abs(rr) ** 2)) / denom)


def _finite_complex(z: complex) -> bool:
    return bool(np.isfinite(z.real) and np.isfinite(z.imag))


def _complex_information_criteria(rss: float, n_complex: int, k_real: int) -> Tuple[float, float]:
    """Return AICc and BIC for a complex least-squares fit.

    Each complex coefficient contributes two real observations.  The residual
    sum of squares is evaluated in that common real-imaginary observation
    space, so models with different parameter counts can be compared without
    changing the data or residual normalization.
    """
    n_obs = 2 * int(n_complex)
    if n_obs <= 0 or k_real < 0 or not np.isfinite(rss):
        return np.nan, np.nan
    rss_safe = max(float(rss), np.finfo(float).tiny)
    base = n_obs * math.log(rss_safe / n_obs)
    aic = base + 2.0 * k_real
    aicc = (
        aic + (2.0 * k_real * (k_real + 1)) / (n_obs - k_real - 1)
        if n_obs > k_real + 1
        else np.nan
    )
    bic = base + k_real * math.log(n_obs)
    return float(aicc) if np.isfinite(aicc) else np.nan, float(bic)


def _physical_selection_class(
    n_points: int,
    min_points: int,
    modulation_scatter: float,
    coupling_scatter: float,
    delta_aicc: float,
    delta_bic: float,
    good_limit: float,
    delta_min: float,
) -> Dict[str, Any]:
    """Classify a common-data modulation--coupling comparison.

    ``delta IC`` is defined as IC(coupling)-IC(modulation), so positive
    values favour modulation.  A direction is assigned only when AICc and BIC
    both reach the configured absolute threshold and agree in sign.  If their
    preferred direction nevertheless has the larger raw common-data scatter,
    the lower-scatter direction overrides the parameter-penalized IC result
    and is capped at ``lean``.  Otherwise the winning model is ``strong`` when
    its absolute common-fit scatter also passes the shared fit threshold, and
    ``lean`` when it does not.  Conflicting, insignificant, or unavailable
    criteria remain direction-free ``ambiguous`` results.
    """
    mod_good = bool(n_points >= min_points and np.isfinite(modulation_scatter)
                    and modulation_scatter <= good_limit)
    coup_good = bool(n_points >= min_points and np.isfinite(coupling_scatter)
                     and coupling_scatter <= good_limit)
    finite_ic = bool(np.isfinite(delta_aicc) and np.isfinite(delta_bic))
    min_abs_delta = (
        min(abs(float(delta_aicc)), abs(float(delta_bic)))
        if finite_ic else np.nan
    )
    finite_scatters = bool(
        np.isfinite(modulation_scatter) and np.isfinite(coupling_scatter)
    )
    scatter_eps = 1e-12
    raw_scatter_direction = "none"
    if finite_scatters:
        if float(modulation_scatter) + scatter_eps < float(coupling_scatter):
            raw_scatter_direction = "modulation"
        elif float(coupling_scatter) + scatter_eps < float(modulation_scatter):
            raw_scatter_direction = "coupling"

    ic_scatter_agree = False

    if n_points < min_points or not finite_ic:
        selection_class = "ambiguous"
        preferred_model = "none"
        selection_strength = "ambiguous"
        decision_detail = "ambiguous_insufficient_complex_data"
        reason = (
            f"common points={n_points} < {min_points} or one of AICc/BIC is unavailable"
        )
    elif delta_aicc >= delta_min and delta_bic >= delta_min:
        if raw_scatter_direction == "coupling":
            preferred_model = "coupling"
            selection_strength = "lean"
            selection_class = "coupling_lean"
            decision_detail = "coupling_lean_scatter_override"
            reason = (
                "AICc and BIC favour modulation after parameter penalization, "
                f"but raw scatter favours coupling ({float(coupling_scatter):.3g} < "
                f"{float(modulation_scatter):.3g}); the direction follows raw scatter "
                "and is capped at COUP-LEAN"
            )
        else:
            preferred_model = "modulation"
            selection_strength = "strong" if mod_good else "lean"
            selection_class = f"modulation_{selection_strength}"
            decision_detail = selection_class
            ic_scatter_agree = True
            if mod_good:
                reason = (
                    f"AICc and BIC favour modulation by at least {delta_min:.3g}, "
                    f"and the modulation scatter passes {good_limit:.3g}"
                )
            else:
                reason = (
                    f"AICc and BIC favour modulation by at least {delta_min:.3g}, "
                    f"but its absolute scatter exceeds {good_limit:.3g}"
                )
    elif delta_aicc <= -delta_min and delta_bic <= -delta_min:
        if raw_scatter_direction == "modulation":
            preferred_model = "modulation"
            selection_strength = "lean"
            selection_class = "modulation_lean"
            decision_detail = "modulation_lean_scatter_override"
            reason = (
                "AICc and BIC favour coupling after parameter penalization, "
                f"but raw scatter favours modulation ({float(modulation_scatter):.3g} < "
                f"{float(coupling_scatter):.3g}); the direction follows raw scatter "
                "and is capped at MOD-LEAN"
            )
        else:
            preferred_model = "coupling"
            selection_strength = "strong" if coup_good else "lean"
            selection_class = f"coupling_{selection_strength}"
            decision_detail = selection_class
            ic_scatter_agree = True
            if coup_good:
                reason = (
                    f"AICc and BIC favour coupling by at least {delta_min:.3g}, "
                    f"and the coupling scatter passes {good_limit:.3g}"
                )
            else:
                reason = (
                    f"AICc and BIC favour coupling by at least {delta_min:.3g}, "
                    f"but its absolute scatter exceeds {good_limit:.3g}"
                )
    else:
        selection_class = "ambiguous"
        preferred_model = "none"
        selection_strength = "ambiguous"
        decision_detail = "ambiguous_insignificant_or_conflicting_ic"
        reason = (
            f"AICc/BIC difference is < {delta_min:.3g} or the criteria disagree"
        )

    tier = {
        "modulation_strong": 4,
        "coupling_strong": 4,
        "modulation_lean": 3,
        "coupling_lean": 3,
        "ambiguous": 2 if n_points >= min_points and finite_ic else 1,
    }[selection_class]
    physical_interpretation = {
        "modulation_strong": "simple_low_order_periodic_modulation_compatible",
        "modulation_lean": "complex_or_nonstationary_Blazhko_like_variability",
        "coupling_lean": "coupling_relatively_preferred_but_absolute_fit_poor",
        "coupling_strong": "quadratic_coupling_relation_compatible",
        "ambiguous": "no_reliable_modulation_coupling_direction",
    }[selection_class]
    if decision_detail == "modulation_lean_scatter_override":
        physical_interpretation = "modulation_preferred_by_lower_raw_scatter_despite_ic_penalty"
    elif decision_detail == "coupling_lean_scatter_override":
        physical_interpretation = "coupling_preferred_by_lower_raw_scatter_despite_ic_penalty"
    return {
        "modulation_fit_good": mod_good,
        "coupling_fit_good": coup_good,
        "preferred_physical_model": preferred_model,
        "selection_strength": selection_strength,
        "selection_class": selection_class,
        "selection_code": physical_selection_code(selection_class),
        "selection_tier": tier,
        "physical_interpretation": physical_interpretation,
        "ic_consensus": ic_scatter_agree,
        "ic_min_abs_delta": min_abs_delta,
        "model_selection_decision": decision_detail,
        "model_selection_reason": reason,
    }


def _clip01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _ranked_physical_score(
    selection_class: str,
    preferred_scatter: Any,
    best_scatter: Any,
    delta_aicc: Any,
    delta_bic: Any,
    coverage: Any,
    n_points: int,
    cand: Candidate,
    args: argparse.Namespace,
) -> float:
    """Place strong, leaning, and ambiguous decisions in disjoint score bands.

    The bands make the headline ordering explicit:

    - strong physical preference: 0.75--1.00;
    - uncertain but directional preference (including BL-MODLIKE): 0.55--0.74;
    - direction-free ambiguity: 0.50--0.549.

    Within a band, absolute fit quality is most important, followed by the
    weaker of the AICc/BIC differences, multiplet coverage, and discovery
    support.  Thus formerly identical 0.50 AMBIG rows become meaningfully
    ordered without allowing a poor absolute fit to masquerade as a strong
    physical identification.
    """
    good = max(1e-12, float(getattr(args, "physical_fit_scatter_good", 0.35)))
    bad = max(good + 1e-12, float(getattr(args, "physical_fit_scatter_bad", 0.75)))
    delta_min = max(1e-12, float(getattr(args, "model_selection_delta_ic", 2.0)))
    cov = _clip01(coverage)
    point_support = _clip01(float(n_points) / max(1, int(getattr(args, "physical_fit_min_points", 8))))
    discovery_support = _clip01(float(cand.discovery_harmonic_count) / 5.0)
    representative = 1.0 if cand.family_representative else 0.0

    finite_deltas = [abs(float(v)) for v in (delta_aicc, delta_bic) if np.isfinite(v)]
    ic_support = (
        _clip01(min(finite_deltas) / (5.0 * delta_min))
        if len(finite_deltas) == 2 else 0.0
    )

    if selection_class.endswith("_strong"):
        fit_excellence = _clip01((good - float(preferred_scatter)) / good) if np.isfinite(preferred_scatter) else 0.0
        quality = (
            0.55 * fit_excellence
            + 0.25 * ic_support
            + 0.15 * cov
            + 0.05 * discovery_support
        )
        return 0.75 + 0.25 * _clip01(quality)

    if selection_class.endswith("_lean") or selection_class == "blazhko_modlike":
        fit_nearness = _clip01((bad - float(preferred_scatter)) / (bad - good)) if np.isfinite(preferred_scatter) else 0.0
        quality = (
            0.45 * fit_nearness
            + 0.30 * ic_support
            + 0.15 * cov
            + 0.05 * discovery_support
            + 0.05 * representative
        )
        return 0.55 + 0.19 * _clip01(quality)

    best_fit_support = _clip01((bad - float(best_scatter)) / bad) if np.isfinite(best_scatter) else 0.0
    quality = (
        0.45 * best_fit_support
        + 0.20 * point_support
        + 0.20 * cov
        + 0.10 * discovery_support
        + 0.05 * representative
    )
    return 0.50 + 0.049 * _clip01(quality)


def common_complex_model_comparison(
    cand: Candidate,
    hmap: Dict[int, FreqMatch],
    mod_matches: Sequence[FreqMatch],
    secondary_map: Dict[Tuple[int, int], FreqMatch],
    origin_sign: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Compare modulation and coupling on identical complex side peaks.

    The shared observable is ``y = C_side/C_n`` for every retained side peak.
    The modulation model fits ``y = a_L + n b_L``.  The coupling model fits
    ``y = q_L x`` with

        x = C_parent C_secondary/C_n                 (sum),
        x = C_parent conj(C_secondary)/C_n           (difference).

    Thus both models use exactly the same complex observations, the same
    normalization, and the same residual scale.  Separate complex parameters
    are fitted for each signed side order L.  AICc and BIC account for the
    resulting parameter-count difference (four real parameters per modulation
    group, two per coupling group).

    Higher ``l f'`` components that coincide algebraically with the multiplet
    grid remain usable here as measured complex predictors.  Their mere
    presence is handled elsewhere and is not counted as independent coupling
    evidence.
    """
    origin_name = "right" if origin_sign > 0 else "left"
    min_group = max(3, int(getattr(args, "common_fit_min_per_group", 3) or 3))
    min_points = max(1, int(getattr(args, "physical_fit_min_points", 8) or 8))
    good_limit = float(getattr(args, "physical_fit_scatter_good", 0.35))
    delta_min = float(getattr(args, "model_selection_delta_ic", 2.0))
    phase_disabled = bool(getattr(args, "ignore_phases", False) or getattr(args, "frequency_only", False))

    empty: Dict[str, Any] = {
        "candidate_id": cand.candidate_id,
        "secondary_origin": origin_name,
        "secondary_origin_sign": origin_sign,
        "common_fit_points": 0,
        "common_real_observations": 0,
        "common_fit_groups": 0,
        "common_fit_orders": "",
        "common_normalization": "C_side/C_n",
        "modulation_parameter_count": 0,
        "coupling_parameter_count": 0,
        "modulation_common_scatter": np.nan,
        "coupling_common_scatter": np.nan,
        "modulation_aicc": np.nan,
        "coupling_aicc": np.nan,
        "delta_aicc_coupling_minus_modulation": np.nan,
        "modulation_bic": np.nan,
        "coupling_bic": np.nan,
        "delta_bic_coupling_minus_modulation": np.nan,
        "modulation_fit_good": False,
        "coupling_fit_good": False,
        "preferred_physical_model": "none",
        "selection_strength": "ambiguous",
        "selection_class": "ambiguous",
        "selection_code": "AMBIG",
        "selection_tier": 1,
        "physical_interpretation": "no_reliable_modulation_coupling_direction",
        "blazhko_modlike_basis": "",
        "ic_consensus": False,
        "ic_min_abs_delta": np.nan,
        "model_selection_decision": "ambiguous_insufficient_complex_data",
        "model_selection_reason": "phases disabled" if phase_disabled else "insufficient common complex side peaks",
    }
    if phase_disabled:
        return empty

    # One observed peak may fall inside more than one tolerance window when fB
    # is unresolved.  Retain it only once, at the closest expected position.
    by_input: Dict[int, Dict[str, Any]] = {}
    for match in mod_matches:
        if not match.matched or match.input_index is None:
            continue
        parsed = parse_n_L_from_label(match.label)
        if parsed is None:
            continue
        n, L = parsed
        l_abs = abs(L)
        parent_n = hmap.get(n)
        secondary = secondary_map.get((origin_sign, l_abs))
        operation_sign = (1 if L > 0 else -1) * origin_sign
        parent_k = n - operation_sign * l_abs
        coupling_parent = hmap.get(parent_k)
        if (
            n <= 0
            or parent_n is None
            or not parent_n.matched
            or secondary is None
            or not secondary.matched
            or coupling_parent is None
            or not coupling_parent.matched
        ):
            continue
        Cn = complex_coeff(parent_n)
        Cs = complex_coeff(match)
        Cp = complex_coeff(coupling_parent)
        Csec = complex_coeff(secondary)
        if not all(_finite_complex(z) for z in (Cn, Cs, Cp, Csec)) or abs(Cn) <= 0:
            continue
        y = Cs / Cn
        sec_factor = Csec if operation_sign > 0 else np.conj(Csec)
        x = Cp * sec_factor / Cn
        if not _finite_complex(y) or not _finite_complex(x) or abs(x) <= 0:
            continue
        record = {
            "input_index": int(match.input_index),
            "n": int(n),
            "L": int(L),
            "y": y,
            "x": x,
            "abs_delta": abs(float(match.delta_frequency or 0.0)),
        }
        previous = by_input.get(int(match.input_index))
        if previous is None or record["abs_delta"] < previous["abs_delta"]:
            by_input[int(match.input_index)] = record

    if not by_input:
        return empty
    common_df = pd.DataFrame(list(by_input.values()))
    retained_groups = [
        int(L) for L, grp in common_df.groupby("L") if len(grp) >= min_group
    ]
    common_df = common_df[common_df["L"].isin(retained_groups)].copy()
    if len(common_df) == 0:
        return empty

    rss_mod = 0.0
    rss_coup = 0.0
    y_power = 0.0
    groups_used = 0
    for L, grp in common_df.groupby("L"):
        n_values = grp["n"].to_numpy(float)
        y_values = np.asarray(grp["y"].to_list(), dtype=complex)
        x_values = np.asarray(grp["x"].to_list(), dtype=complex)

        Xmod = np.column_stack([np.ones_like(n_values), n_values]).astype(complex)
        beta_mod, *_ = np.linalg.lstsq(Xmod, y_values, rcond=None)
        y_mod = Xmod @ beta_mod

        denom = np.vdot(x_values, x_values)
        if not _finite_complex(complex(denom)) or abs(denom) <= 0:
            continue
        q_coup = np.vdot(x_values, y_values) / denom
        y_coup = q_coup * x_values

        rss_mod += float(np.sum(np.abs(y_values - y_mod) ** 2))
        rss_coup += float(np.sum(np.abs(y_values - y_coup) ** 2))
        y_power += float(np.sum(np.abs(y_values) ** 2))
        groups_used += 1

    n_points = int(len(common_df))
    if groups_used == 0 or y_power <= 0:
        return empty
    mod_scatter = math.sqrt(rss_mod / y_power)
    coup_scatter = math.sqrt(rss_coup / y_power)
    k_mod_real = 4 * groups_used
    k_coup_real = 2 * groups_used
    mod_aicc, mod_bic = _complex_information_criteria(rss_mod, n_points, k_mod_real)
    coup_aicc, coup_bic = _complex_information_criteria(rss_coup, n_points, k_coup_real)
    delta_aicc = coup_aicc - mod_aicc if np.isfinite(coup_aicc) and np.isfinite(mod_aicc) else np.nan
    delta_bic = coup_bic - mod_bic if np.isfinite(coup_bic) and np.isfinite(mod_bic) else np.nan
    selection = _physical_selection_class(
        n_points,
        min_points,
        mod_scatter,
        coup_scatter,
        delta_aicc,
        delta_bic,
        good_limit,
        delta_min,
    )

    return {
        **empty,
        "common_fit_points": n_points,
        "common_real_observations": 2 * n_points,
        "common_fit_groups": groups_used,
        "common_fit_orders": ",".join(str(x) for x in sorted(retained_groups)),
        "modulation_parameter_count": k_mod_real,
        "coupling_parameter_count": k_coup_real,
        "modulation_common_scatter": float(mod_scatter),
        "coupling_common_scatter": float(coup_scatter),
        "modulation_aicc": mod_aicc,
        "coupling_aicc": coup_aicc,
        "delta_aicc_coupling_minus_modulation": delta_aicc,
        "modulation_bic": mod_bic,
        "coupling_bic": coup_bic,
        "delta_bic_coupling_minus_modulation": delta_bic,
        **selection,
    }


def _common_anchor_direction(comp: Dict[str, Any]) -> str:
    """Return the physical direction of one fixed-f' common comparison."""
    selection_class = str(comp.get("selection_class", "ambiguous"))
    if selection_class.startswith("modulation_"):
        return "modulation"
    if selection_class.startswith("coupling_"):
        return "coupling"
    return "ambiguous"


def _common_anchor_summary(comp: Dict[str, Any]) -> str:
    origin = str(comp.get("secondary_origin", "unknown"))
    code = str(comp.get("selection_code", "AMBIG") or "AMBIG")
    return (
        f"{origin}={code} "
        f"(N={int(comp.get('common_fit_points', 0) or 0)}, "
        f"s_mod={format_float(comp.get('modulation_common_scatter'), 3)}, "
        f"s_coup={format_float(comp.get('coupling_common_scatter'), 3)}, "
        f"deltaAICc={format_float(comp.get('delta_aicc_coupling_minus_modulation'), 4)}, "
        f"deltaBIC={format_float(comp.get('delta_bic_coupling_minus_modulation'), 4)})"
    )


def consensus_common_model_comparison(
    common_comparisons: Sequence[Dict[str, Any]],
    default: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine right/left fixed-f' comparisons without cherry-picking.

    The two coupling anchors are alternative physical hypotheses: in a true
    one-sided coupling case only one observed side peak is the independent
    secondary, while the opposite peak may itself be a coupling product.
    Consequently a modulation--coupling disagreement remains direction-free.
    An ambiguous anchor, however, is absence of a directional decision rather
    than evidence for the opposite model: when the other usable anchor favours
    modulation, retain that direction but cap it at MOD-LEAN; when it favours
    coupling, retain the supported one-sided coupling hypothesis.  With only
    one usable anchor, that comparison is retained unchanged.
    """
    usable = [
        dict(comp) for comp in common_comparisons
        if int(comp.get("common_fit_points", 0) or 0) > 0
    ]
    by_origin = {str(comp.get("secondary_origin", "unknown")): comp for comp in usable}
    anchor_summary = "; ".join(
        _common_anchor_summary(by_origin[origin])
        for origin in ("right", "left")
        if origin in by_origin
    )

    anchor_fields: Dict[str, Any] = {
        "usable_anchor_count": len(usable),
        "anchor_consensus_status": "none",
        "anchor_consensus_reason": "no usable right/left f' common comparison",
        "anchor_comparison_summary": anchor_summary,
    }
    for origin in ("right", "left"):
        comp = by_origin.get(origin)
        anchor_fields.update({
            f"{origin}_anchor_selection_code": (
                str(comp.get("selection_code", "AMBIG")) if comp is not None else "--"
            ),
            f"{origin}_anchor_common_fit_points": (
                int(comp.get("common_fit_points", 0) or 0) if comp is not None else 0
            ),
            f"{origin}_anchor_modulation_scatter": (
                comp.get("modulation_common_scatter", np.nan) if comp is not None else np.nan
            ),
            f"{origin}_anchor_coupling_scatter": (
                comp.get("coupling_common_scatter", np.nan) if comp is not None else np.nan
            ),
            f"{origin}_anchor_delta_aicc": (
                comp.get("delta_aicc_coupling_minus_modulation", np.nan) if comp is not None else np.nan
            ),
            f"{origin}_anchor_delta_bic": (
                comp.get("delta_bic_coupling_minus_modulation", np.nan) if comp is not None else np.nan
            ),
        })

    if not usable:
        return {**default, **anchor_fields}

    if len(usable) == 1:
        chosen = dict(usable[0])
        origin = str(chosen.get("secondary_origin", "unknown"))
        chosen.update({
            **anchor_fields,
            "anchor_consensus_status": f"single-{origin}",
            "anchor_consensus_reason": (
                f"only the {origin} f' anchor has enough common complex data"
            ),
        })
        return chosen

    modulation = [comp for comp in usable if _common_anchor_direction(comp) == "modulation"]
    coupling = [comp for comp in usable if _common_anchor_direction(comp) == "coupling"]
    ambiguous = [comp for comp in usable if _common_anchor_direction(comp) == "ambiguous"]

    def ic_strength(comp: Dict[str, Any]) -> float:
        value = comp.get("ic_min_abs_delta", np.nan)
        return float(value) if np.isfinite(value) else -np.inf

    def preferred_scatter(comp: Dict[str, Any], direction: str) -> float:
        key = f"{direction}_common_scatter"
        value = comp.get(key, np.nan)
        return float(value) if np.isfinite(value) else np.inf

    # Modulation must defeat both possible one-sided coupling hypotheses.  Use
    # the weaker of the agreeing anchors so its strength and score remain
    # conservative.
    if len(modulation) == len(usable):
        chosen = min(
            modulation,
            key=lambda comp: (
                int(comp.get("selection_tier", 0) or 0),
                ic_strength(comp),
                -preferred_scatter(comp, "modulation"),
                int(comp.get("common_fit_points", 0) or 0),
            ),
        )
        chosen = dict(chosen)
        chosen.update({
            **anchor_fields,
            "anchor_consensus_status": "agree-modulation",
            "anchor_consensus_reason": (
                "both usable right/left f' anchors favour modulation; the weaker anchor sets the reported strength"
            ),
        })
        return chosen

    # An ambiguous alternative is direction-free rather than evidence against
    # modulation.  Follow the informative anchor, but do not call the
    # two-anchor result MOD-STRONG when only one side supplies a directional IC
    # comparison.  This is especially appropriate for modulation, whose
    # multiplet pattern is intrinsically two-sided.
    if modulation and ambiguous and not coupling:
        chosen = max(
            modulation,
            key=lambda comp: (
                int(comp.get("selection_tier", 0) or 0),
                ic_strength(comp),
                -preferred_scatter(comp, "modulation"),
                int(comp.get("common_fit_points", 0) or 0),
            ),
        )
        chosen = dict(chosen)
        origin = str(chosen.get("secondary_origin", "unknown"))
        anchor_reason = (
            f"the {origin} f' anchor favours modulation while the other usable "
            "anchor remains direction-free; the anchor-level modulation preference is limited to LEAN"
        )
        source_decision = str(chosen.get("model_selection_decision", ""))
        source_reason = str(chosen.get("model_selection_reason", ""))
        chosen.update({
            **anchor_fields,
            "preferred_physical_model": "modulation",
            "selection_strength": "lean",
            "selection_class": "modulation_lean",
            "selection_code": "MOD-LEAN",
            "selection_tier": 3,
            "physical_interpretation": "modulation_direction_supported_but_anchor_support_incomplete",
            "ic_consensus": False,
            "model_selection_decision": (
                source_decision
                if source_decision.endswith("_scatter_override")
                else "modulation_lean_anchor_support"
            ),
            "model_selection_reason": source_reason,
            "anchor_consensus_status": f"modulation-{origin}-other-ambiguous",
            "anchor_consensus_reason": anchor_reason,
        })
        return chosen

    # Coupling is a union of the right- and left-secondary hypotheses.  One
    # supported anchor is sufficient only when no usable alternative favours
    # modulation.  This preserves a genuine one-sided coupling case while a
    # modulation/coupling split stays AMBIG.
    if coupling and not modulation:
        chosen = max(
            coupling,
            key=lambda comp: (
                int(comp.get("selection_tier", 0) or 0),
                ic_strength(comp),
                -preferred_scatter(comp, "coupling"),
                int(comp.get("common_fit_points", 0) or 0),
            ),
        )
        chosen = dict(chosen)
        origin = str(chosen.get("secondary_origin", "unknown"))
        status = "agree-coupling" if len(coupling) == len(usable) else f"coupling-{origin}-other-ambiguous"
        reason = (
            "both usable right/left f' anchors favour coupling"
            if len(coupling) == len(usable)
            else f"the {origin} f' anchor favours coupling and the other usable anchor does not favour modulation"
        )
        chosen.update({
            **anchor_fields,
            "anchor_consensus_status": status,
            "anchor_consensus_reason": reason,
        })
        return chosen

    # No directional consensus.  Prefer an actually ambiguous anchor as the
    # representative numeric row; clear the scalar IC deltas so the table does
    # not misrepresent one side as the two-anchor result.  The complete
    # right/left values remain in dedicated CSV fields and in the diagnostic.
    representative_pool = ambiguous if ambiguous else usable
    chosen = max(
        representative_pool,
        key=lambda comp: (
            int(comp.get("common_fit_points", 0) or 0),
            -min(
                float(comp.get("modulation_common_scatter", np.inf)),
                float(comp.get("coupling_common_scatter", np.inf)),
            ),
        ),
    )
    chosen = dict(chosen)
    directions = "/".join(_common_anchor_direction(comp) for comp in usable)
    if modulation and coupling:
        status = "conflict-modulation-coupling"
        reason = "right/left f' anchors favour different physical models"
    else:
        status = "agree-ambiguous"
        reason = "both usable right/left f' anchors remain direction-free"
    chosen.update({
        **anchor_fields,
        "preferred_physical_model": "none",
        "selection_strength": "ambiguous",
        "selection_class": "ambiguous",
        "selection_code": "AMBIG",
        "selection_tier": 2,
        "physical_interpretation": "no_reliable_modulation_coupling_direction",
        "ic_consensus": False,
        "ic_min_abs_delta": np.nan,
        "model_selection_decision": "ambiguous_anchor_consensus",
        "model_selection_reason": f"{reason}; anchor directions={directions}",
        "delta_aicc_coupling_minus_modulation": np.nan,
        "delta_bic_coupling_minus_modulation": np.nan,
        "anchor_consensus_status": status,
        "anchor_consensus_reason": reason,
    })
    return chosen


def apply_side_order_envelope_evidence(
    common_selection: Dict[str, Any],
    modulation_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine the two higher-side-order diagnostics with the common fit.

    Complete same-harmonic |L|=1,2,3 tracks have priority and may set a LEAN
    physical direction even when no usable common modulation/coupling fit is
    available.  Only when that cross-L test is non-directional may the separate
    L=+/-2,+/-3 amplitude runs along harmonic order n supply the direction.
    The n-sequence diagnostic has only LEAN authority and still requires a
    usable common fit: it may resolve AMBIG or reverse LEAN, but a conflict with
    a STRONG common-fit result only downgrades that original direction to LEAN.
    Agreement never promotes a weaker result to STRONG.
    """
    side_order_fields = {
        key: value for key, value in modulation_summary.items()
        if key.startswith("side_order_")
    }
    result = {**common_selection, **side_order_fields}
    envelope_evidence = str(
        modulation_summary.get("side_order_envelope_evidence", "insufficient")
    )
    envelope_reason = str(
        modulation_summary.get("side_order_envelope_reason", "")
    )
    sequence_evidence = str(
        modulation_summary.get("side_order_sequence_evidence", "insufficient")
    )
    sequence_reason = str(
        modulation_summary.get("side_order_sequence_reason", "")
    )
    current_direction = str(common_selection.get("preferred_physical_model", "none"))
    common_points = int(common_selection.get("common_fit_points", 0) or 0)

    directional = {"modulation", "coupling"}
    if envelope_evidence in directional:
        evidence = envelope_evidence
        evidence_reason = envelope_reason
        evidence_source = "complete_tracks"
        if sequence_evidence == envelope_evidence:
            result["side_order_sequence_action"] = "subordinate_agrees"
        elif sequence_evidence in directional:
            result["side_order_sequence_action"] = "subordinate_conflict_complete_tracks_priority"
            evidence_reason += (
                f"; the lower-priority harmonic-order sequences favour "
                f"{sequence_evidence}, but complete tracks retain priority"
            )
        else:
            result["side_order_sequence_action"] = "not_directional"
    elif sequence_evidence in directional:
        evidence = sequence_evidence
        evidence_reason = sequence_reason
        evidence_source = "harmonic_order_sequences"
        result["side_order_envelope_action"] = "not_directional"
    else:
        result.update({
            "side_order_selected_evidence": "none",
            "side_order_evidence_source": "none",
            "side_order_evidence_action": "not_directional",
            "side_order_envelope_action": "not_directional",
            "side_order_sequence_action": "not_directional",
        })
        return result

    result["side_order_selected_evidence"] = evidence
    result["side_order_evidence_source"] = evidence_source
    if common_points <= 0 and evidence_source != "complete_tracks":
        # The lower-priority harmonic-order sequences remain supplementary:
        # without a common physical-model comparison they cannot become a
        # standalone classification.  Complete tracks are handled below and
        # are allowed to supply an independent LEAN direction.
        action = "not_applied_no_common_fit"
        result["side_order_evidence_action"] = action
        result["side_order_sequence_action"] = action
        return result

    if current_direction == evidence:
        action = "agrees"
        result["side_order_evidence_action"] = action
        if evidence_source == "complete_tracks":
            result["side_order_envelope_action"] = action
        else:
            result["side_order_sequence_action"] = action
        prior_reason = str(result.get("model_selection_reason", ""))
        evidence_name = (
            "complete-track |L| envelope"
            if evidence_source == "complete_tracks"
            else "higher-side-order harmonic-order sequence"
        )
        result["model_selection_reason"] = (
            f"{prior_reason}; independent {evidence_name} agrees: {evidence_reason}"
            if prior_reason else f"independent {evidence_name} agrees: {evidence_reason}"
        )
        return result

    # The separate L=2/3 sequences are deliberately only a LEAN-level test.
    # They can expose tension in an otherwise strong common fit, but cannot
    # reverse a STRONG physical direction on their own.  Retain that direction
    # and lower only its confidence.  The complete-track diagnostic is not
    # subject to this guard because it is the higher-priority strong evidence.
    current_strength = str(common_selection.get("selection_strength", "ambiguous"))
    if (
        evidence_source == "harmonic_order_sequences"
        and current_direction in directional
        and current_strength == "strong"
    ):
        action = "conflicts_with_strong_downgrades"
        selection_class = f"{current_direction}_lean"
        retained_name = current_direction
        result.update({
            "preferred_physical_model": current_direction,
            "selection_strength": "lean",
            "selection_class": selection_class,
            "selection_code": physical_selection_code(selection_class),
            "selection_tier": 3,
            "physical_interpretation": (
                f"{retained_name}_retained_at_lean_under_harmonic_sequence_conflict"
            ),
            "model_selection_decision": (
                f"{retained_name}_lean_side_order_harmonic_sequence_conflict"
            ),
            "model_selection_reason": (
                f"the common-fit/anchor result was "
                f"{common_selection.get('selection_code', retained_name.upper())}, "
                f"but {evidence_reason}; the individual L=2/3 harmonic-order "
                f"amplitude run has only LEAN authority, so it cannot reverse "
                f"a STRONG result: the {retained_name} direction is retained "
                "and its confidence is reduced to LEAN"
            ),
            "side_order_evidence_action": action,
            "side_order_sequence_action": action,
        })
        return result

    source_code = str(common_selection.get("selection_code", "AMBIG") or "AMBIG")
    if common_points <= 0 and evidence_source == "complete_tracks":
        action = "resolves_without_common_fit"
    else:
        action = "resolves_ambiguous" if current_direction == "none" else "overrides_common_direction"
    selection_class = f"{evidence}_lean"
    interpretation = (
        "modulation_preferred_by_higher_side_order_amplitude_run"
        if evidence == "modulation"
        else "coupling_preferred_by_nonmonotonic_higher_side_order_amplitude_run"
    )
    decision_suffix = (
        "side_order_envelope"
        if evidence_source == "complete_tracks"
        else "side_order_harmonic_sequence"
    )
    pattern_name = (
        "cross-|L| complete-track amplitude pattern"
        if evidence_source == "complete_tracks"
        else "individual L=2/3 harmonic-order amplitude run"
    )
    result.update({
        "preferred_physical_model": evidence,
        "selection_strength": "lean",
        "selection_class": selection_class,
        "selection_code": physical_selection_code(selection_class),
        "selection_tier": 3,
        "physical_interpretation": interpretation,
        "ic_consensus": False,
        "model_selection_decision": f"{evidence}_lean_{decision_suffix}",
        "model_selection_reason": (
            (
                f"no usable common modulation/coupling fit was available, but "
                f"{evidence_reason}; the {pattern_name} independently sets the "
                "physical direction and is conservatively capped at LEAN"
            )
            if action == "resolves_without_common_fit"
            else (
                f"the common-fit/anchor result was {source_code}, but {evidence_reason}; "
                f"the {pattern_name} sets the physical direction and "
                "is conservatively capped at LEAN"
            )
        ),
        "side_order_evidence_action": action,
    })
    if evidence_source == "complete_tracks":
        result["side_order_envelope_action"] = action
    else:
        result["side_order_sequence_action"] = action
    return result


def _symmetric_fit_score(scatter: Any, args: argparse.Namespace) -> float:
    """Map the common scatter to one score scale shared by both models."""
    if not np.isfinite(scatter):
        return 0.0
    good = float(getattr(args, "physical_fit_scatter_good", 0.35))
    bad = max(good + 1e-12, float(getattr(args, "physical_fit_scatter_bad", 0.75)))
    threshold = float(getattr(args, "report_score_min", 0.5))
    value = max(0.0, float(scatter))
    if value <= good:
        return min(1.0, threshold + (1.0 - threshold) * (1.0 - value / max(good, 1e-12)))
    return max(0.0, threshold * (bad - value) / (bad - good))


def complex_modulation_fit_stats(hmap: Dict[int, FreqMatch], mod_matches: Sequence[FreqMatch]) -> Dict[str, Any]:
    """Fit simple complex AM/PM sideband relations.

    For small-to-moderate modulation, the complex sideband coefficient around
    the nth harmonic can be written phenomenologically as

        C_{n,L}/C_n ~= a_L + n b_L,

    where a_L is AM-like and b_L is PM/FM-like.  This is not a full B11
    decomposition, but it is a much stronger test than frequency positions
    alone.  Pure AM corresponds to a constant in n; pure PM/FM corresponds
    approximately to a term proportional to n.
    """
    rows: List[Dict[str, Any]] = []
    for m in mod_matches:
        if not m.matched:
            continue
        parsed = parse_n_L_from_label(m.label)
        if parsed is None:
            continue
        n, L = parsed
        if n <= 0 or n not in hmap or not hmap[n].matched:
            continue
        parent = hmap[n]
        if parent.amplitude is None or parent.amplitude <= 0 or m.amplitude is None:
            continue
        Cp = complex_coeff(parent)
        Cs = complex_coeff(m)
        if not np.isfinite(Cp.real) or abs(Cp) <= 0:
            continue
        R = Cs / Cp
        rows.append({"n": int(n), "L": int(L), "R": R})

    if not rows:
        return {
            "modfit_points": 0,
            "modfit_groups": 0,
            "combined_modulation_complex_scatter": np.nan,
            "pure_am_complex_scatter": np.nan,
            "pure_pm_complex_scatter": np.nan,
            "modfit_orders": "",
        }

    scat_combined: List[float] = []
    scat_am: List[float] = []
    scat_pm: List[float] = []
    groups_used = 0
    orders_used: List[int] = []
    df = pd.DataFrame(rows)
    for L, grp in df.groupby("L"):
        if len(grp) < 3:
            continue
        n = grp["n"].to_numpy(float)
        y = np.asarray(grp["R"].to_list(), dtype=complex)
        # Combined AM+PM/PM-like fit: y = a + b n.
        X = np.column_stack([np.ones_like(n), n])
        beta, *_ = np.linalg.lstsq(X.astype(complex), y, rcond=None)
        y_comb = X @ beta
        # Pure AM-like fit: y = const.
        a = np.mean(y)
        y_am = np.full_like(y, a, dtype=complex)
        # Pure PM/FM-like fit: y = b n, through the origin.
        den = np.sum(n * n)
        b = np.sum(n * y) / den if den > 0 else 0j
        y_pm = b * n
        scat_combined.append(_complex_scatter(y, y_comb))
        scat_am.append(_complex_scatter(y, y_am))
        scat_pm.append(_complex_scatter(y, y_pm))
        groups_used += 1
        orders_used.append(int(L))

    if groups_used == 0:
        return {
            "modfit_points": len(rows),
            "modfit_groups": 0,
            "combined_modulation_complex_scatter": np.nan,
            "pure_am_complex_scatter": np.nan,
            "pure_pm_complex_scatter": np.nan,
            "modfit_orders": "",
        }
    return {
        "modfit_points": int(len(rows)),
        "modfit_groups": int(groups_used),
        "combined_modulation_complex_scatter": float(np.nanmedian(scat_combined)),
        "pure_am_complex_scatter": float(np.nanmedian(scat_am)),
        "pure_pm_complex_scatter": float(np.nanmedian(scat_pm)),
        "modfit_orders": ",".join(str(x) for x in sorted(set(abs(o) for o in orders_used))),
    }


def side_order_envelope_stats(
    mod_matches: Sequence[FreqMatch],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Measure whether the observed side-peak envelope decays with |L|.

    This diagnostic is deliberately orthogonal to the fits along primary
    harmonic order n.  It compares A(n,2)/A(n,1) and A(n,3)/A(n,2) only within
    the same harmonic n and on the same frequency-grid side.  Consequently the
    result is not biased merely because different primary harmonics are
    available at different side orders.

    A usable track contains all of |L|=1,2,3.  Robust (log-median) adjacent
    ratios are calculated across tracks.  A consistently declining envelope
    is modulation-like; a strong higher-order rise is coupling-like.  The
    result is intentionally only a LEAN-level discriminator when it changes a
    physical classification.
    """
    min_tracks = max(
        2, int(getattr(args, "side_order_envelope_min_tracks", 2) or 2)
    )
    rise_tolerance = max(
        1.0, float(getattr(args, "side_order_envelope_rise_tolerance", 1.15))
    )
    reversal_ratio = max(
        rise_tolerance,
        float(getattr(args, "side_order_envelope_reversal_ratio", 1.50)),
    )
    empty: Dict[str, Any] = {
        "side_order_envelope_evidence": "insufficient",
        "side_order_envelope_track_count": 0,
        "side_order_envelope_harmonic_count": 0,
        "side_order_envelope_right_tracks": 0,
        "side_order_envelope_left_tracks": 0,
        "side_order_l2_l1_median_ratio": np.nan,
        "side_order_l3_l2_median_ratio": np.nan,
        "side_order_monotonic_track_fraction": np.nan,
        "side_order_reversal_track_fraction": np.nan,
        "side_order_envelope_action": "not_applied",
        "side_order_envelope_reason": (
            "fewer than the required complete same-harmonic |L|=1,2,3 tracks"
        ),
    }

    # One observed peak can match more than one L window when fB is poorly
    # resolved.  Keep it only at the closest expected grid position before
    # constructing the L tracks.
    by_input: Dict[int, Dict[str, Any]] = {}
    for match in mod_matches:
        if (
            not match.matched
            or match.input_index is None
            or match.amplitude is None
            or not np.isfinite(match.amplitude)
            or float(match.amplitude) <= 0
        ):
            continue
        parsed = parse_n_L_from_label(match.label)
        if parsed is None:
            continue
        n, signed_L = parsed
        order = abs(int(signed_L))
        if n <= 0 or order not in (1, 2, 3):
            continue
        record = {
            "input_index": int(match.input_index),
            "n": int(n),
            "side": 1 if signed_L > 0 else -1,
            "order": order,
            "amplitude": float(match.amplitude),
            "abs_delta": abs(float(match.delta_frequency or 0.0)),
        }
        previous = by_input.get(int(match.input_index))
        if previous is None or record["abs_delta"] < previous["abs_delta"]:
            by_input[int(match.input_index)] = record

    tracks: Dict[Tuple[int, int], Dict[int, float]] = {}
    for record in by_input.values():
        key = (int(record["n"]), int(record["side"]))
        tracks.setdefault(key, {})[int(record["order"])] = float(record["amplitude"])

    complete: List[Dict[str, Any]] = []
    for (n, side), amplitudes in tracks.items():
        if not all(order in amplitudes and amplitudes[order] > 0 for order in (1, 2, 3)):
            continue
        ratio_21 = amplitudes[2] / amplitudes[1]
        ratio_32 = amplitudes[3] / amplitudes[2]
        if not (np.isfinite(ratio_21) and np.isfinite(ratio_32)):
            continue
        complete.append({
            "n": n,
            "side": side,
            "ratio_21": float(ratio_21),
            "ratio_32": float(ratio_32),
        })

    if not complete:
        return empty

    ratio_21 = np.asarray([row["ratio_21"] for row in complete], dtype=float)
    ratio_32 = np.asarray([row["ratio_32"] for row in complete], dtype=float)
    median_21 = float(np.exp(np.median(np.log(ratio_21))))
    median_32 = float(np.exp(np.median(np.log(ratio_32))))
    monotonic = (ratio_21 <= rise_tolerance) & (ratio_32 <= rise_tolerance)
    reversal = (ratio_21 >= reversal_ratio) | (ratio_32 >= reversal_ratio)
    monotonic_fraction = float(np.mean(monotonic))
    reversal_fraction = float(np.mean(reversal))
    harmonic_count = len({int(row["n"]) for row in complete})
    track_count = len(complete)

    result = {
        **empty,
        "side_order_envelope_track_count": track_count,
        "side_order_envelope_harmonic_count": harmonic_count,
        "side_order_envelope_right_tracks": sum(int(row["side"]) > 0 for row in complete),
        "side_order_envelope_left_tracks": sum(int(row["side"]) < 0 for row in complete),
        "side_order_l2_l1_median_ratio": median_21,
        "side_order_l3_l2_median_ratio": median_32,
        "side_order_monotonic_track_fraction": monotonic_fraction,
        "side_order_reversal_track_fraction": reversal_fraction,
    }
    if track_count < min_tracks or harmonic_count < 2:
        result["side_order_envelope_reason"] = (
            f"only {track_count} complete |L|=1,2,3 track(s) across "
            f"{harmonic_count} harmonic(s); require at least {min_tracks} tracks "
            "and two distinct harmonics"
        )
        return result

    # Three quarters of individual tracks must decline as well as both robust
    # step ratios.  This prevents two compensating outliers from producing an
    # apparently smooth median envelope.
    if (
        median_21 <= rise_tolerance
        and median_32 <= rise_tolerance
        and monotonic_fraction >= 0.75
    ):
        result.update({
            "side_order_envelope_evidence": "modulation",
            "side_order_envelope_reason": (
                f"higher side orders decline: median A2/A1={median_21:.3g}, "
                f"A3/A2={median_32:.3g}, and {monotonic_fraction:.0%} of "
                f"{track_count} complete tracks are monotonic within the "
                f"{rise_tolerance:.3g} rise tolerance"
            ),
        })
    elif (
        (median_21 >= reversal_ratio or median_32 >= reversal_ratio)
        and reversal_fraction >= 0.50
    ):
        result.update({
            "side_order_envelope_evidence": "coupling",
            "side_order_envelope_reason": (
                f"higher side-order envelope reverses strongly: median "
                f"A2/A1={median_21:.3g}, A3/A2={median_32:.3g}, and "
                f"{reversal_fraction:.0%} of {track_count} complete tracks "
                f"contain a rise by at least {reversal_ratio:.3g}"
            ),
        })
    else:
        result.update({
            "side_order_envelope_evidence": "ambiguous",
            "side_order_envelope_reason": (
                f"mixed side-order envelope: median A2/A1={median_21:.3g}, "
                f"A3/A2={median_32:.3g}, monotonic-track fraction="
                f"{monotonic_fraction:.0%}, strong-reversal fraction="
                f"{reversal_fraction:.0%}"
            ),
        })
    return result


def side_order_harmonic_sequence_stats(
    mod_matches: Sequence[FreqMatch],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Test the amplitude run of each signed L=2 and L=3 series along n.

    Unlike :func:`side_order_envelope_stats`, this diagnostic does not require
    L=1,2,3 to occur at the same harmonic.  It follows the raw measured
    amplitudes A(n,L) separately for L=+2,-2,+3,-3.  Ratios between consecutive
    *observed* points are converted to a per-harmonic factor, so a gap in n does
    not receive the same weight as a one-order step.

    A steadily declining run is modulation-like.  A series is coupling-like
    only when it contains both a strong fall and a strong rise with an actual
    direction change; a merely increasing or noisy sequence remains ambiguous.
    This is a secondary discriminator and can change a physical decision only
    at LEAN strength.
    """
    min_points = max(
        4, int(getattr(args, "side_order_sequence_min_points", 4) or 4)
    )
    rise_tolerance = max(
        1.0, float(getattr(args, "side_order_envelope_rise_tolerance", 1.15))
    )
    reversal_ratio = max(
        rise_tolerance,
        float(getattr(args, "side_order_envelope_reversal_ratio", 1.50)),
    )
    monotonic_min = _clip01(
        getattr(args, "side_order_sequence_monotonic_fraction", 0.75)
    )

    # As in the complete-track diagnostic, prevent one poorly resolved input
    # peak from appearing in more than one expected L series.
    by_input: Dict[int, Dict[str, Any]] = {}
    for match in mod_matches:
        if (
            not match.matched
            or match.input_index is None
            or match.amplitude is None
            or not np.isfinite(match.amplitude)
            or float(match.amplitude) <= 0
        ):
            continue
        parsed = parse_n_L_from_label(match.label)
        if parsed is None:
            continue
        n, signed_L = parsed
        if n <= 0 or abs(int(signed_L)) not in (2, 3):
            continue
        record = {
            "input_index": int(match.input_index),
            "n": int(n),
            "L": int(signed_L),
            "amplitude": float(match.amplitude),
            "abs_delta": abs(float(match.delta_frequency or 0.0)),
        }
        previous = by_input.get(int(match.input_index))
        if previous is None or record["abs_delta"] < previous["abs_delta"]:
            by_input[int(match.input_index)] = record

    by_order: Dict[int, Dict[int, Dict[str, Any]]] = {
        signed_L: {} for signed_L in (2, -2, 3, -3)
    }
    for record in by_input.values():
        signed_L = int(record["L"])
        n = int(record["n"])
        previous = by_order[signed_L].get(n)
        if previous is None or record["abs_delta"] < previous["abs_delta"]:
            by_order[signed_L][n] = record

    series_rows: List[Dict[str, Any]] = []
    for signed_L in (2, -2, 3, -3):
        records = [by_order[signed_L][n] for n in sorted(by_order[signed_L])]
        n_values = np.asarray([row["n"] for row in records], dtype=float)
        amplitudes = np.asarray([row["amplitude"] for row in records], dtype=float)
        n_points = int(len(records))
        side_name = "right" if signed_L > 0 else "left"
        label = f"L={signed_L:+d}"
        row: Dict[str, Any] = {
            "signed_L": signed_L,
            "side": side_name,
            "series_label": label,
            "points": n_points,
            "n_min": int(n_values.min()) if n_points else 0,
            "n_max": int(n_values.max()) if n_points else 0,
            "harmonic_orders": ",".join(str(int(n)) for n in n_values),
            "step_count": max(0, n_points - 1),
            "median_amplitude_ratio_per_harmonic": np.nan,
            "log_amplitude_slope_per_harmonic": np.nan,
            "monotonic_step_fraction": np.nan,
            "strong_rise_step_fraction": np.nan,
            "strong_fall_step_fraction": np.nan,
            "direction_change_count": 0,
            "evidence": "insufficient",
            "reason": f"{label} has {n_points} matched point(s); require {min_points}",
        }
        if n_points < 2:
            series_rows.append(row)
            continue

        delta_n = np.diff(n_values)
        raw_ratios = amplitudes[1:] / amplitudes[:-1]
        per_harmonic_ratios = np.power(raw_ratios, 1.0 / delta_n)
        log_amplitudes = np.log(amplitudes)
        slope = float(np.polyfit(n_values, log_amplitudes, 1)[0])
        median_ratio = float(np.exp(np.median(np.log(per_harmonic_ratios))))
        monotonic_steps = per_harmonic_ratios <= rise_tolerance
        strong_rises = per_harmonic_ratios >= reversal_ratio
        strong_falls = per_harmonic_ratios <= (1.0 / reversal_ratio)
        monotonic_fraction = float(np.mean(monotonic_steps))
        strong_rise_fraction = float(np.mean(strong_rises))
        strong_fall_fraction = float(np.mean(strong_falls))

        directions: List[int] = []
        for ratio in per_harmonic_ratios:
            if ratio > rise_tolerance:
                directions.append(1)
            elif ratio < (1.0 / rise_tolerance):
                directions.append(-1)
        direction_changes = sum(
            directions[i] != directions[i - 1]
            for i in range(1, len(directions))
        )
        row.update({
            "median_amplitude_ratio_per_harmonic": median_ratio,
            "log_amplitude_slope_per_harmonic": slope,
            "monotonic_step_fraction": monotonic_fraction,
            "strong_rise_step_fraction": strong_rise_fraction,
            "strong_fall_step_fraction": strong_fall_fraction,
            "direction_change_count": int(direction_changes),
        })

        if n_points < min_points:
            row["reason"] = (
                f"{label} has {n_points} matched points over n="
                f"{row['n_min']}..{row['n_max']}; require {min_points}"
            )
        elif (
            slope < 0
            and monotonic_fraction >= monotonic_min
            and not bool(np.any(strong_rises))
        ):
            row.update({
                "evidence": "modulation",
                "reason": (
                    f"{label} declines with n: median per-harmonic amplitude "
                    f"ratio={median_ratio:.3g}, log-slope={slope:.3g}, and "
                    f"{monotonic_fraction:.0%} of {n_points - 1} observed "
                    "steps do not rise significantly"
                ),
            })
        elif (
            bool(np.any(strong_rises))
            and bool(np.any(strong_falls))
            and direction_changes >= 1
        ):
            row.update({
                "evidence": "coupling",
                "reason": (
                    f"{label} is strongly non-monotonic along n: "
                    f"{strong_rise_fraction:.0%} strong-rise and "
                    f"{strong_fall_fraction:.0%} strong-fall steps, with "
                    f"{direction_changes} direction change(s)"
                ),
            })
        else:
            row.update({
                "evidence": "ambiguous",
                "reason": (
                    f"{label} has a mixed but non-directional run: median "
                    f"per-harmonic ratio={median_ratio:.3g}, log-slope="
                    f"{slope:.3g}, monotonic-step fraction="
                    f"{monotonic_fraction:.0%}, direction changes="
                    f"{direction_changes}"
                ),
            })
        series_rows.append(row)

    modulation_series = [row for row in series_rows if row["evidence"] == "modulation"]
    coupling_series = [row for row in series_rows if row["evidence"] == "coupling"]
    eligible_series = [row for row in series_rows if int(row["points"]) >= min_points]
    if modulation_series and coupling_series:
        evidence = "ambiguous"
        reason = (
            "higher-side-order n-series conflict: modulation-like "
            + ",".join(row["series_label"] for row in modulation_series)
            + "; coupling-like "
            + ",".join(row["series_label"] for row in coupling_series)
        )
    elif modulation_series:
        evidence = "modulation"
        reason = (
            "all directional higher-side-order n-series are modulation-like: "
            + ",".join(row["series_label"] for row in modulation_series)
        )
    elif coupling_series:
        evidence = "coupling"
        reason = (
            "all directional higher-side-order n-series are coupling-like: "
            + ",".join(row["series_label"] for row in coupling_series)
        )
    elif eligible_series:
        evidence = "ambiguous"
        reason = (
            f"{len(eligible_series)} sufficiently sampled higher-side-order "
            "n-series remain individually ambiguous"
        )
    else:
        evidence = "insufficient"
        reason = (
            f"no signed L=2 or L=3 series has the required {min_points} "
            "matched amplitudes"
        )

    compact_summary = "; ".join(
        f"{row['series_label']}:{row['evidence']}(N={row['points']},"
        f"r={format_float(row['median_amplitude_ratio_per_harmonic'],3)},"
        f"turns={row['direction_change_count']})"
        for row in series_rows
    )
    return {
        "side_order_sequence_evidence": evidence,
        "side_order_sequence_eligible_series": len(eligible_series),
        "side_order_sequence_informative_series": len(modulation_series) + len(coupling_series),
        "side_order_sequence_modulation_series": len(modulation_series),
        "side_order_sequence_coupling_series": len(coupling_series),
        "side_order_sequence_action": "not_applied",
        "side_order_sequence_reason": reason,
        "side_order_sequence_summary": compact_summary,
        "side_order_sequence_rows": series_rows,
    }


def _modulation_coverage_for_orders(
    hmap: Dict[int, FreqMatch],
    mod_matches: Sequence[FreqMatch],
    side_orders: Iterable[int],
) -> Dict[str, Any]:
    """Coverage over observed primary harmonics and model-required side orders."""
    wanted = {abs(int(x)) for x in side_orders if int(x) != 0}
    available_harmonics = set(int(k) for k in hmap)
    selected: List[FreqMatch] = []
    for match in mod_matches:
        parsed = parse_n_L_from_label(match.label)
        if parsed is None:
            continue
        n, L = parsed
        if n in available_harmonics and abs(L) in wanted:
            selected.append(match)
    tested = len(selected)
    found = count_matched(selected)
    return {
        "tested_positions": tested,
        "frequency_matches": found,
        "coverage": found / tested if tested else 0.0,
    }


def modulation_stats(
    cand: Candidate,
    hmap: Dict[int, FreqMatch],
    mod_matches: Sequence[FreqMatch],
    low_matches: Sequence[FreqMatch],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    raw_tested = len(mod_matches)
    raw_found = count_matched(mod_matches)
    low_orders = []
    for m in low_matches:
        if m.matched:
            mm = re.match(r"\s*(\d+)\s+fB", m.label)
            if mm:
                low_orders.append(int(mm.group(1)))
    found_orders = set()
    found_side_orders_by_n: Dict[int, List[int]] = {}
    pair_asym: List[float] = []
    am_ratios: List[float] = []
    pm_ratios: List[float] = []
    plusminus: Dict[Tuple[int, int], Dict[int, FreqMatch]] = {}
    for m in mod_matches:
        nl = parse_n_L_from_label(m.label)
        if nl is None:
            continue
        n, Lsigned = nl
        if n not in hmap:
            continue
        L = abs(Lsigned)
        if m.matched:
            found_orders.add(L)
            found_side_orders_by_n.setdefault(n, []).append(Lsigned)
        plusminus.setdefault((n, L), {})[1 if Lsigned > 0 else -1] = m
    for (n, L), pair in plusminus.items():
        if L != 1:
            continue
        p = pair.get(1)
        q = pair.get(-1)
        if p and q and p.matched and q.matched and p.amplitude is not None and q.amplitude is not None:
            denom = p.amplitude + q.amplitude
            if denom > 0:
                pair_asym.append(abs(p.amplitude - q.amplitude) / denom)
                if n in hmap and hmap[n].amplitude and hmap[n].amplitude > 0:
                    mean_side = 0.5 * (p.amplitude + q.amplitude)
                    r = mean_side / hmap[n].amplitude
                    am_ratios.append(r)
                    pm_ratios.append(r / n if n > 0 else np.nan)
    _, am_scatter, nam = robust_log_scatter(am_ratios)
    _, pm_scatter, npm = robust_log_scatter(pm_ratios)
    modfit = complex_modulation_fit_stats(hmap, mod_matches)
    side_order_envelope = side_order_envelope_stats(mod_matches, args)
    side_order_sequences = side_order_harmonic_sequence_stats(mod_matches, args)
    l1_matches = [
        m for m in mod_matches
        if (parse_n_L_from_label(m.label) is not None and abs(parse_n_L_from_label(m.label)[1]) == 1)
    ]
    l1_modfit = complex_modulation_fit_stats(hmap, l1_matches)
    l1_coverage = _modulation_coverage_for_orders(hmap, mod_matches, [1])
    max_found_order = max(found_orders) if found_orders else 1
    structured_orders = list(range(1, max_found_order + 1))
    structured_coverage = _modulation_coverage_for_orders(hmap, mod_matches, structured_orders)

    fit_orders = [int(x) for x in str(modfit.get("modfit_orders", "")).split(",") if str(x).strip()]
    fit_coverage = _modulation_coverage_for_orders(hmap, mod_matches, fit_orders or [1])

    # The general frequency-pattern row uses whichever explicitly tested
    # modulation submodel is better supported: a sinusoidal L=1 grid or the
    # contiguous non-sinusoidal grid up to the largest detected side order.
    # It never pays a denominator penalty for absent primary harmonics or for
    # higher side orders that the selected model does not predict.
    if structured_coverage["coverage"] > l1_coverage["coverage"] + 1e-12:
        selected_coverage = structured_coverage
        coverage_basis = f"non-sinusoidal L=1..{max_found_order}"
    else:
        selected_coverage = l1_coverage
        coverage_basis = "sinusoidal L=1"
    return {
        "tested_positions": selected_coverage["tested_positions"],
        "frequency_matches": selected_coverage["frequency_matches"],
        "coverage": selected_coverage["coverage"],
        "coverage_basis": coverage_basis,
        "raw_tested_positions": raw_tested,
        "raw_frequency_matches": raw_found,
        "raw_coverage": raw_found / raw_tested if raw_tested else 0.0,
        "available_primary_harmonics": ",".join(str(x) for x in sorted(hmap)),
        "available_primary_harmonic_count": len(hmap),
        "l1_tested_positions": l1_coverage["tested_positions"],
        "l1_frequency_matches": l1_coverage["frequency_matches"],
        "l1_coverage": l1_coverage["coverage"],
        "structured_tested_positions": structured_coverage["tested_positions"],
        "structured_frequency_matches": structured_coverage["frequency_matches"],
        "structured_coverage": structured_coverage["coverage"],
        "structured_side_orders": ",".join(str(x) for x in structured_orders),
        "fit_tested_positions": fit_coverage["tested_positions"],
        "fit_frequency_matches": fit_coverage["frequency_matches"],
        "fit_coverage": fit_coverage["coverage"],
        "low_frequency_matches": len(low_orders),
        "low_orders_found": ",".join(str(x) for x in sorted(set(low_orders))),
        "max_side_order_found": max(found_orders) if found_orders else 0,
        "side_orders_found": ",".join(str(x) for x in sorted(found_orders)),
        "n_symmetric_pairs_l1": len(pair_asym),
        "median_abs_side_asymmetry_l1": float(np.median(pair_asym)) if pair_asym else np.nan,
        "am_ratio_log_scatter_l1": am_scatter,
        "pm_ratio_log_scatter_l1_over_n": pm_scatter,
        "n_am_ratio_points": nam,
        "n_pm_ratio_points": npm,
        **side_order_envelope,
        **side_order_sequences,
        **modfit,
        "l1_modfit_points": l1_modfit.get("modfit_points", 0),
        "l1_modfit_groups": l1_modfit.get("modfit_groups", 0),
        "l1_combined_modulation_complex_scatter": l1_modfit.get("combined_modulation_complex_scatter", np.nan),
    }


def phase_and_amplitude_tests(
    cand: Candidate,
    hmap: Dict[int, FreqMatch],
    secondary_map: Dict[Tuple[int, int], FreqMatch],
    coupling_matches: Sequence[FreqMatch],
    secondary_exclusions: Optional[Dict[Tuple[int, int], str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Test right- and left-secondary coupling as separate hypotheses.

    A hypothesis is anchored to one observed secondary frequency
    f'=f0+s*delta (s=+1 or -1).  Both sides of every multiplet are then
    predicted from that *same* f' through sums and differences with primary
    harmonics.  No opposite-side peak is allowed to serve as a second
    secondary oscillator inside the same test.
    """
    exclusions = secondary_exclusions or {}
    rows: List[Dict[str, Any]] = []
    origin_signs = [
        sign for sign in (1, -1)
        if (sign, 1) in secondary_map and secondary_map[(sign, 1)].matched
    ]
    relation_test_counts: Dict[int, int] = {}
    relation_test_counts_l1: Dict[int, int] = {}
    for origin_sign in origin_signs:
        origin_name = "right" if origin_sign > 0 else "left"
        tested_for_origin = 0
        tested_l1_for_origin = 0
        for m in coupling_matches:
            parsed = parse_n_L_from_label(m.label)
            if parsed is None:
                continue
            n, L = parsed
            l_abs = abs(L)
            sec_key = (origin_sign, l_abs)
            sec = secondary_map.get(sec_key)
            if sec is None or not sec.matched:
                continue

            # If f'=f0+s*delta, the operation sign (+ sum, - difference) is
            # op=sign(L)*s and the necessary primary parent is k=n-op*l.
            operation_sign = (1 if L > 0 else -1) * origin_sign
            parent_k = n - operation_sign * l_abs
            if parent_k <= 0 or parent_k not in hmap:
                continue
            tested_for_origin += 1
            if l_abs == 1:
                tested_l1_for_origin += 1
            if not m.matched:
                continue
            parent = hmap[parent_k]
            if parent.amplitude is None or sec.amplitude is None or parent.amplitude <= 0 or sec.amplitude <= 0:
                continue
            ratio = m.amplitude / (parent.amplitude * sec.amplitude) if m.amplitude is not None else np.nan
            if m.phase_rad is None or parent.phase_rad is None or sec.phase_rad is None:
                phi_comb = np.nan
            else:
                phi_comb = float(circular_wrap_rad(
                    m.phase_rad - parent.phase_rad - operation_sign * sec.phase_rad
                ))
            rows.append({
                "candidate_id": cand.candidate_id,
                "secondary_origin": origin_name,
                "secondary_origin_sign": origin_sign,
                "n": n,
                "L": L,
                "parent_k": parent_k,
                "coupling_operation_sign": operation_sign,
                "predicted_counterpart": bool(n == 1 and L == -origin_sign),
                "side_frequency": m.observed_frequency,
                "side_amplitude": m.amplitude,
                "parent_amplitude": parent.amplitude,
                "secondary_harmonic_l": l_abs,
                "secondary_side_sign": origin_sign,
                "secondary_frequency": sec.observed_frequency,
                "secondary_amplitude": sec.amplitude,
                "secondary_is_grid_degenerate": bool(sec_key in exclusions),
                "secondary_exclusion_reason": exclusions.get(sec_key, ""),
                "amplitude_ratio_side_over_parent_secondary": ratio,
                "combination_phase_rad": phi_comb,
                "combination_phase_cycles": phi_comb / TWOPI if np.isfinite(phi_comb) else np.nan,
                "match_label": m.label,
            })
        relation_test_counts[origin_sign] = tested_for_origin
        relation_test_counts_l1[origin_sign] = tested_l1_for_origin

    df = pd.DataFrame(rows)
    hypotheses: List[Dict[str, Any]] = []
    for origin_sign in origin_signs:
        origin_name = "right" if origin_sign > 0 else "left"
        sub = df[df["secondary_origin_sign"].eq(origin_sign)].copy() if len(df) else pd.DataFrame()
        summary = summarize_coupling_phase_subset(sub)
        l1_summary = summarize_coupling_phase_subset(
            sub[sub["L"].abs().eq(1)].copy()
        ) if len(sub) else summarize_coupling_phase_subset(pd.DataFrame())
        side_orders = sorted({abs(int(v)) for v in sub["L"].to_numpy()}) if len(sub) else []
        excluded_orders = sorted({
            order for (sign, order) in exclusions if sign == origin_sign
        })
        hypotheses.append({
            "candidate_id": cand.candidate_id,
            "secondary_origin": origin_name,
            "secondary_origin_sign": origin_sign,
            "side_orders_found": ",".join(str(x) for x in side_orders),
            "max_side_order_found": max(side_orders) if side_orders else 0,
            "counterpart_predicted_and_matched": bool(
                len(sub) and sub["predicted_counterpart"].any()
            ),
            "excluded_secondary_harmonic_orders": ",".join(str(x) for x in excluded_orders),
            "excluded_secondary_harmonic_count": len(excluded_orders),
            "fit_secondary_harmonic_orders": ",".join(str(x) for x in side_orders),
            "n_phase_points": summary["n_phase_points"],
            "combination_phase_scatter_rad": summary["combination_phase_scatter_rad"],
            "log10_amp_ratio_scatter": summary["log10_amp_ratio_scatter"],
            "coupling_complex_scatter": summary["coupling_complex_scatter"],
            "coupling_complex_points": summary["coupling_complex_points"],
            "n_amplitude_ratios": int(len(sub)),
            "relation_matches": int(len(sub)),
            "relation_matches_l1": int(sub["L"].abs().eq(1).sum()) if len(sub) else 0,
            "relation_tested": int(relation_test_counts.get(origin_sign, 0)),
            "relation_tested_l1": int(relation_test_counts_l1.get(origin_sign, 0)),
            "l1_phase_points": l1_summary["n_phase_points"],
            "l1_phase_scatter_rad": l1_summary["combination_phase_scatter_rad"],
            "l1_amp_ratio_log_scatter": l1_summary["log10_amp_ratio_scatter"],
            "l1_coupling_complex_scatter": l1_summary["coupling_complex_scatter"],
            "l1_coupling_complex_points": l1_summary["coupling_complex_points"],
        })
    return df, {"candidate_id": cand.candidate_id, "hypotheses": hypotheses}


def coupling_stats(
    cand: Candidate,
    coupling_matches: Sequence[FreqMatch],
    low_matches: Sequence[FreqMatch],
    secondary_map: Dict[Tuple[int, int], FreqMatch],
    phase_summary: Dict[str, Any],
    secondary_exclusions: Optional[Dict[Tuple[int, int], str]] = None,
) -> Dict[str, Any]:
    exclusions = secondary_exclusions or {}
    found = count_matched(coupling_matches)
    tested = len(coupling_matches)
    low_orders: List[int] = []
    for m in low_matches:
        if m.matched:
            mm = re.match(r"\s*(\d+)\s+fB", m.label)
            if mm:
                low_orders.append(int(mm.group(1)))

    hypothesis_stats: List[Dict[str, Any]] = []
    rel_min = float(getattr(args_global, "secondary_harmonic_rel_amp_min", 0.20))
    for phase_hyp in phase_summary.get("hypotheses", []):
        origin_sign = int(phase_hyp["secondary_origin_sign"])
        secondary_orders = sorted({
            order for (sign, order), match in secondary_map.items()
            if sign == origin_sign and match.matched and (sign, order) not in exclusions
        })
        excluded_orders = sorted({
            order for (sign, order) in exclusions if sign == origin_sign
        })
        fundamental = secondary_map.get((origin_sign, 1))
        fundamental_amp = float(fundamental.amplitude) if fundamental is not None and fundamental.amplitude is not None else 0.0
        significant_orders: List[int] = []
        for order in secondary_orders:
            match = secondary_map.get((origin_sign, order))
            amp = float(match.amplitude) if match is not None and match.amplitude is not None else 0.0
            if order == 1 or (fundamental_amp > 0 and amp >= rel_min * fundamental_amp):
                significant_orders.append(order)

        side_orders = [int(x) for x in str(phase_hyp.get("side_orders_found", "")).split(",") if x]
        relation_matches = int(phase_hyp.get("relation_matches", 0) or 0)
        # Relation rows are already restricted to a single f' hypothesis.
        l1_matches = int(phase_hyp.get("relation_matches_l1", 0) or 0)
        relation_tested = int(phase_hyp.get("relation_tested", 0) or 0)
        relation_tested_l1 = int(phase_hyp.get("relation_tested_l1", 0) or 0)
        high_matches = max(0, relation_matches - l1_matches)
        hyp = {
            **phase_hyp,
            "tested_positions": relation_tested,
            "frequency_matches": relation_matches,
            "coverage": relation_matches / relation_tested if relation_tested else 0.0,
            "low_frequency_matches": len(low_orders),
            "low_orders_found": ",".join(str(x) for x in sorted(set(low_orders))),
            "coupling_L1_matches": l1_matches,
            "coupling_L1_tested": relation_tested_l1,
            "coupling_highL_matches": high_matches,
            "secondary_harmonic_orders_found": ",".join(str(x) for x in secondary_orders),
            "significant_secondary_harmonic_orders_found": ",".join(str(x) for x in significant_orders),
            "excluded_secondary_harmonic_orders": ",".join(str(x) for x in excluded_orders),
            "n_secondary_harmonic_orders": len(secondary_orders),
            "n_significant_secondary_harmonic_orders": len(significant_orders),
            "max_secondary_harmonic_order": max(secondary_orders) if secondary_orders else 0,
            "max_significant_secondary_harmonic_order": max(significant_orders) if significant_orders else 0,
            "higher_order_coupling_required": max(
                0,
                (max(side_orders) if side_orders else 0) - (max(significant_orders) if significant_orders else 0),
            ),
            "phase_test_points": int(phase_hyp.get("n_phase_points", 0) or 0),
            "phase_scatter_rad": phase_hyp.get("combination_phase_scatter_rad", np.nan),
            "amp_ratio_log_scatter": phase_hyp.get("log10_amp_ratio_scatter", np.nan),
            "l1_phase_test_points": int(phase_hyp.get("l1_phase_points", 0) or 0),
            "amplitude_test_points": int(phase_hyp.get("n_amplitude_ratios", 0) or 0),
        }
        hypothesis_stats.append(hyp)

    return {
        "candidate_id": cand.candidate_id,
        "tested_positions": tested,
        "frequency_matches": found,
        "coverage": found / tested if tested else 0.0,
        "low_frequency_matches": len(low_orders),
        "low_orders_found": ",".join(str(x) for x in sorted(set(low_orders))),
        "hypotheses": hypothesis_stats,
        "secondary_grid_overlap_exclusions": len(exclusions),
    }


def score_from_coverage(coverage: float, base: float = 0.0) -> float:
    return max(0.0, min(1.0, base + coverage))


def find_close_harmonic_side_peaks(
    df: pd.DataFrame,
    f0: float,
    nmax: int,
    tol: float,
    side_window: float,
) -> List[Dict[str, Any]]:
    """Return all significant side-peak-like frequencies near primary harmonics.

    Linear beating with a single close-frequency component is only a plausible
    global interpretation when the peak is isolated.  If the same frequency
    list contains side-peak-like structure around the primary harmonics, linear
    beating is retained only as a local compatibility check and is capped below
    the report threshold by default.
    """
    rows: List[Dict[str, Any]] = []
    if f0 <= 0 or nmax <= 0 or side_window <= 0:
        return rows
    for _, r in df.iterrows():
        freq = float(r["frequency"])
        if not np.isfinite(freq) or freq <= 0:
            continue
        n = int(round(freq / f0))
        if not (1 <= n <= nmax):
            continue
        delta = freq - n * f0
        if abs(delta) <= tol:
            continue
        if abs(delta) <= side_window:
            rows.append({
                "near_harmonic": n,
                "frequency": freq,
                "offset": delta,
                "amplitude": float(r.get("amplitude", np.nan)),
                "input_index": int(r.get("input_index", -1)),
            })
    return rows


def score_models_for_candidate(
    cand: Candidate,
    hmap: Dict[int, FreqMatch],
    linear_matches: Sequence[FreqMatch],
    modulation_matches: Sequence[FreqMatch],
    coupling_matches: Sequence[FreqMatch],
    low_matches: Sequence[FreqMatch],
    secondary_map: Dict[Tuple[int, int], FreqMatch],
    modstat: Dict[str, Any],
    coupstat: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    low_found = int(modstat["low_frequency_matches"])
    mod_found = int(modstat["frequency_matches"])
    mod_tested = int(modstat["tested_positions"])
    mod_cov = float(modstat["coverage"])
    mod_coverage_basis = str(modstat.get("coverage_basis", "model-dependent"))
    mod_l1_found = int(modstat.get("l1_frequency_matches", mod_found) or 0)
    mod_l1_tested = int(modstat.get("l1_tested_positions", mod_tested) or 0)
    mod_l1_cov = float(modstat.get("l1_coverage", mod_cov) or 0.0)
    mod_struct_found = int(modstat.get("structured_frequency_matches", mod_found) or 0)
    mod_struct_tested = int(modstat.get("structured_tested_positions", mod_tested) or 0)
    mod_struct_cov = float(modstat.get("structured_coverage", mod_cov) or 0.0)
    mod_fit_found = int(modstat.get("fit_frequency_matches", mod_found) or 0)
    mod_fit_tested = int(modstat.get("fit_tested_positions", mod_tested) or 0)
    mod_fit_cov = float(modstat.get("fit_coverage", mod_cov) or 0.0)
    max_side = int(modstat["max_side_order_found"])
    asym = modstat.get("median_abs_side_asymmetry_l1", np.nan)
    am_scatter = modstat.get("am_ratio_log_scatter_l1", np.nan)
    pm_scatter = modstat.get("pm_ratio_log_scatter_l1_over_n", np.nan)
    n_pairs = int(modstat.get("n_symmetric_pairs_l1", 0) or 0)
    modfit_points = int(modstat.get("modfit_points", 0) or 0)
    modfit_groups = int(modstat.get("modfit_groups", 0) or 0)
    combined_modfit_scatter = modstat.get("combined_modulation_complex_scatter", np.nan)
    l1_modfit_points = int(modstat.get("l1_modfit_points", 0) or 0)
    l1_combined_modfit_scatter = modstat.get("l1_combined_modulation_complex_scatter", np.nan)
    pure_am_complex_scatter = modstat.get("pure_am_complex_scatter", np.nan)
    pure_pm_complex_scatter = modstat.get("pure_pm_complex_scatter", np.nan)
    modfit_good = (modfit_points >= args_global.modfit_min_points and np.isfinite(combined_modfit_scatter)
                   and combined_modfit_scatter <= args_global.modfit_scatter_good)
    modfit_bad = (modfit_points >= args_global.modfit_min_points and np.isfinite(combined_modfit_scatter)
                  and combined_modfit_scatter > args_global.modfit_scatter_bad)
    phase_disabled = bool(getattr(args_global, "ignore_phases", False) or getattr(args_global, "frequency_only", False))
    common_comparisons = [
        common_complex_model_comparison(
            cand, hmap, modulation_matches, secondary_map, origin_sign, args_global
        )
        for origin_sign in (1, -1)
        if (origin_sign, 1) in secondary_map and secondary_map[(origin_sign, 1)].matched
    ]

    def cap_unverified_coupling(score: float, n_relation_points: int, diagnostic_name: str) -> Tuple[float, bool, str]:
        """Cap coupling-like physical interpretations when only the grid is checked.

        A small number of phase points can give an artificially tiny scatter
        (even exactly zero for one point). In normal phase-aware mode, such
        rows remain in the full CSV as grid compatibility but are not promoted
        as coupling evidence in the Markdown report.
        """
        if phase_disabled:
            return score, False, ""
        min_pts = int(getattr(args_global, "phase_min_points", 5) or 5)
        cap = float(getattr(args_global, "coupling_unverified_cap", 0.49) or 0.49)
        if int(n_relation_points or 0) < min_pts:
            note = (
                f"; relation evidence insufficient for {diagnostic_name}: "
                f"only {int(n_relation_points or 0)} phase point(s) < {min_pts}; "
                f"score capped at {cap:.2f}"
            )
            return min(float(score), cap), True, note
        return score, False, ""

    lin_found = count_matched(linear_matches)
    lin_higher = count_matched(linear_matches, lambda m: not m.label.strip().startswith("1 "))
    # model-related negative evidence for linear beating: low terms and broad equidistant/coupling grid.
    broad_grid_evidence = max(0, mod_found - lin_found)

    # Global gate for linear beating.  A single close-frequency beating
    # explanation is only meaningful when the star contains one isolated close
    # side peak.  If side peaks are present around the harmonics, linear beating
    # may still describe an individual local peak, but it should not be promoted
    # as a global solution for the frequency list.
    linear_global_side_peaks = int(getattr(args_global, "linear_beating_global_side_peak_count", 0) or 0)
    linear_max_side_peaks = int(getattr(args_global, "linear_beating_max_side_peaks", 1) or 1)
    linear_global_gate_enabled = not bool(getattr(args_global, "no_linear_beating_global_gate", False))
    linear_global_gated = linear_global_gate_enabled and linear_global_side_peaks > linear_max_side_peaks
    linear_gate_cap = float(getattr(args_global, "linear_beating_gated_cap", 0.49) or 0.49)
    if linear_global_gated:
        linear_gate_note = (
            f"; global linear-beating gate: {linear_global_side_peaks} side-peak-like "
            f"frequencies near primary harmonics > allowed {linear_max_side_peaks}; "
            f"score capped at {linear_gate_cap:.2f}"
        )
        linear_gate_comment = (
            " The frequency list contains multiplet-like side structure around "
            "the primary harmonics, so linear beating is retained only as a local "
            "close-peak compatibility check and is not promoted as a global solution."
        )
    else:
        linear_gate_note = ""
        linear_gate_comment = ""

    # Linear sinusoidal secondary.
    lin_s_score = 0.0
    if cand.representative_frequency is not None:
        lin_s_score += 0.45
    if lin_higher == 0:
        lin_s_score += 0.20
    if low_found == 0:
        lin_s_score += 0.20
    if broad_grid_evidence <= 1:
        lin_s_score += 0.15
    lin_s_negative = low_found + max(0, broad_grid_evidence)
    if lin_s_negative > 5:
        lin_s_score *= 0.25
    if linear_global_gated:
        lin_s_score = min(lin_s_score, linear_gate_cap)
        lin_s_negative += max(0, linear_global_side_peaks - linear_max_side_peaks)

    if cand.representative_frequency is None:
       lin_s_score = 0.0

    rows.append({
        "candidate_id": cand.candidate_id, "fB_abs": cand.fB_abs,
        "model_group": "linear_beating", "model": "linear_beating_sinusoidal_secondary",
        "score_0_1": min(1.0, lin_s_score), "frequency_matches": 1 if cand.representative_frequency is not None else 0,
        "tested_positions": 1, "coverage": 1.0 if cand.representative_frequency is not None else 0.0,
        "low_frequency_matches": low_found, "negative_evidence": lin_s_negative,
        "amplitude_test_points": 0, "phase_test_points": 0, "phase_scatter_rad": np.nan, "amp_ratio_log_scatter": np.nan,
        "diagnostic": "single close frequency only" + linear_gate_note,
        "comment": "Only f' belongs to this model. Harmonics of f', low fB terms, or a broad equidistant grid tied to the same fB disfavor it as a complete explanation." + linear_gate_comment,
    })


    # Linear non-sinusoidal secondary.
    lin_ns_score = 0.0
    if lin_found >= 2:
        lin_ns_score += 0.45
    elif lin_found == 1:
        lin_ns_score += 0.20
    if low_found == 0:
        lin_ns_score += 0.25
    if broad_grid_evidence <= max(1, lin_found):
        lin_ns_score += 0.20
    if lin_found >= 3:
        lin_ns_score += 0.10
    lin_ns_negative = low_found + max(0, broad_grid_evidence - lin_found)
    if lin_ns_negative > 5:
        lin_ns_score *= 0.35
    if linear_global_gated:
        lin_ns_score = min(lin_ns_score, linear_gate_cap)
        lin_ns_negative += max(0, linear_global_side_peaks - linear_max_side_peaks)
    rows.append({
        "candidate_id": cand.candidate_id, "fB_abs": cand.fB_abs,
        "model_group": "linear_beating", "model": "linear_beating_nonsinusoidal_secondary",
        "score_0_1": min(1.0, lin_ns_score), "frequency_matches": lin_found,
        "tested_positions": len(linear_matches), "coverage": lin_found / len(linear_matches) if linear_matches else 0.0,
        "low_frequency_matches": low_found, "negative_evidence": lin_ns_negative,
        "amplitude_test_points": 0, "phase_test_points": 0, "phase_scatter_rad": np.nan, "amp_ratio_log_scatter": np.nan,
        "diagnostic": "l f' harmonic sequence" + linear_gate_note,
        "comment": "Looks for l f' = l f0 + l fB and requires no candidate-related low-frequency or broad modulation/coupling grid." + linear_gate_comment,
    })

    # The equidistant multiplet grid is shared exactly by periodic modulation
    # and quadratic coupling.  It is reported as a neutral pattern diagnostic
    # and is never eligible for physical-model ranking.
    multiplet_grid_score = 0.15 + 0.85 * min(1.0, mod_cov)
    if mod_found < 2:
        multiplet_grid_score *= 0.4
    rows.append({
        "candidate_id": cand.candidate_id, "fB_abs": cand.fB_abs,
        "model_group": "shared_pattern", "model": "equidistant_multiplet_frequency_grid",
        "score_0_1": min(1.0, multiplet_grid_score), "frequency_matches": mod_found,
        "tested_positions": mod_tested, "coverage": mod_cov,
        "low_frequency_matches": low_found, "negative_evidence": 0,
        "amplitude_test_points": int(modstat.get("n_am_ratio_points", 0) or 0), "phase_test_points": 0,
        "phase_scatter_rad": np.nan, "amp_ratio_log_scatter": np.nan,
        "physical_ranking_eligible": False,
        "model_selection_decision": "shared_frequency_pattern_only",
        "diagnostic": f"equidistant grid; coverage basis={mod_coverage_basis}; side orders {modstat.get('side_orders_found','')}; low orders {modstat.get('low_orders_found','')}; modulation family {cand.family_member_orders or f'c{cand.candidate_id}:1'}",
        "comment": "Neutral frequency-pattern test for nf0 ± L fB. The same positions follow from modulation and from quadratic coupling, so this row cannot select either physical model.",
    })

    # Non-sinusoidal modulation / modulation with higher side orders.
    nonsin_evidence = 0.0
    if max_side >= 2:
        nonsin_evidence += 0.35
    if low_found >= 2:
        nonsin_evidence += 0.35
    elif low_found == 1:
        nonsin_evidence += 0.12
    if mod_struct_cov > 0.45:
        nonsin_evidence += 0.20
    if max_side >= 3 or low_found >= 3:
        nonsin_evidence += 0.10
    family_orders = cand.family_member_orders if cand.family_representative else ""
    if cand.family_representative and cand.family_max_order > 1:
        # A resolved subharmonic/harmonic spacing family is direct evidence that
        # a single sinusoidal modulation function is insufficient.
        nonsin_evidence += 0.20
    rows.append({
        "candidate_id": cand.candidate_id, "fB_abs": cand.fB_abs,
        "model_group": "modulation", "model": "non_sinusoidal_modulation",
        "score_0_1": min(1.0, nonsin_evidence), "frequency_matches": mod_struct_found,
        "tested_positions": mod_struct_tested, "coverage": mod_struct_cov,
        "low_frequency_matches": low_found, "negative_evidence": 0,
        "amplitude_test_points": int(modstat.get("n_am_ratio_points", 0) or 0), "phase_test_points": 0,
        "phase_scatter_rad": np.nan, "amp_ratio_log_scatter": np.nan,
        "diagnostic": f"higher modulation orders: side L={modstat.get('side_orders_found','')}, low L={modstat.get('low_orders_found','')}; family orders={family_orders or '--'}",
        "comment": "Higher side orders, harmonics/subharmonics of fm, or a harmonically related candidate family indicate a non-sinusoidal modulation function or more complex modulation than a single sinusoid.",
    })

    # Pure sinusoidal AM and PM/FM heuristics.
    pure_am_score = 0.0
    if mod_l1_cov > 0.25 and max_side <= 1 and n_pairs >= 2 and np.isfinite(asym) and asym < args_global.am_symmetry_limit:
        pure_am_score += 0.45
        if np.isfinite(am_scatter) and am_scatter < args_global.am_scatter_limit:
            pure_am_score += 0.25
        if modfit_points >= args_global.modfit_min_points and np.isfinite(pure_am_complex_scatter) and pure_am_complex_scatter <= args_global.modfit_scatter_good:
            pure_am_score += 0.20
        if low_found <= 1:
            pure_am_score += 0.15
    rows.append({
        "candidate_id": cand.candidate_id, "fB_abs": cand.fB_abs,
        "model_group": "modulation", "model": "AM_like_sideband_scaling_indicator",
        "score_0_1": min(1.0, pure_am_score), "frequency_matches": mod_l1_found,
        "tested_positions": mod_l1_tested, "coverage": mod_l1_cov,
        "low_frequency_matches": low_found, "negative_evidence": int(max_side > 1) + int(np.isfinite(asym) and asym >= args_global.am_symmetry_limit),
        "amplitude_test_points": int(modstat.get("n_am_ratio_points", 0) or 0), "phase_test_points": 0,
        "phase_scatter_rad": np.nan, "amp_ratio_log_scatter": am_scatter,
        "diagnostic": f"pair asym={format_float(asym,3)}, AM-ratio scatter={format_float(am_scatter,3)} dex",
        "comment": "In a modulation interpretation, this flag indicates AM-like sideband scaling. It is a diagnostic qualifier, not evidence for purely amplitude modulation by itself.",
    })

    pure_pm_score = 0.0
    if mod_l1_cov > 0.25 and n_pairs >= 2 and np.isfinite(asym) and asym < args_global.am_symmetry_limit:
        pure_pm_score += 0.35
        if np.isfinite(pm_scatter) and pm_scatter < args_global.am_scatter_limit:
            pure_pm_score += 0.25
        if modfit_points >= args_global.modfit_min_points and np.isfinite(pure_pm_complex_scatter) and pure_pm_complex_scatter <= args_global.modfit_scatter_good:
            pure_pm_score += 0.25
        if low_found == 0:
            pure_pm_score += 0.10
        if max_side >= 2:
            pure_pm_score += 0.10
    rows.append({
        "candidate_id": cand.candidate_id, "fB_abs": cand.fB_abs,
        "model_group": "modulation", "model": "FM_PM_like_sideband_scaling_indicator",
        "score_0_1": min(1.0, pure_pm_score), "frequency_matches": mod_l1_found,
        "tested_positions": mod_l1_tested, "coverage": mod_l1_cov,
        "low_frequency_matches": low_found, "negative_evidence": int(np.isfinite(asym) and asym >= args_global.am_symmetry_limit),
        "amplitude_test_points": int(modstat.get("n_pm_ratio_points", 0) or 0), "phase_test_points": 0,
        "phase_scatter_rad": np.nan, "amp_ratio_log_scatter": pm_scatter,
        "diagnostic": f"pair asym={format_float(asym,3)}, PM-ratio scatter={format_float(pm_scatter,3)} dex",
        "comment": "In a modulation interpretation, this flag indicates FM/PM-like sideband scaling. It does not exclude simultaneous amplitude modulation; light-curve amplitude changes should be used as external evidence.",
    })

    combined_score = 0.0
    if mod_l1_cov > 0.45:
        combined_score += 0.45
    if n_pairs >= 2 and np.isfinite(asym) and asym >= args_global.combined_asymmetry_limit:
        combined_score += 0.35
    if max_side >= 2 or low_found >= 2:
        combined_score += 0.15
    if mod_l1_cov > 0.7:
        combined_score += 0.05
    rows.append({
        "candidate_id": cand.candidate_id, "fB_abs": cand.fB_abs,
        "model_group": "modulation", "model": "combined_AM_FM_PM_indicator",
        "score_0_1": min(1.0, combined_score), "frequency_matches": mod_l1_found,
        "tested_positions": mod_l1_tested, "coverage": mod_l1_cov,
        "low_frequency_matches": low_found, "negative_evidence": 0,
        "amplitude_test_points": n_pairs, "phase_test_points": 0,
        "phase_scatter_rad": np.nan, "amp_ratio_log_scatter": np.nan,
        "diagnostic": f"median |A+−A−|/(A++A−) for L=1 = {format_float(asym,3)}",
        "comment": "In the modulation interpretation, persistent side-pair asymmetry is expected when AM and FM/PM components coexist. This is a qualifier of the modulation solution, not a separate model.",
    })


    # General combined AM+FM/PM modulation from complex normalized sideband fit.
    general_mod_score = 0.0
    if mod_fit_cov > 0.45:
        general_mod_score += 0.35
    if modfit_good:
        general_mod_score += 0.45
    elif modfit_points >= args_global.modfit_min_points and np.isfinite(combined_modfit_scatter):
        # Some evidence, but not a clean low-dimensional AM+PM fit.
        general_mod_score += max(0.0, 0.30 * (1.0 - combined_modfit_scatter / max(args_global.modfit_scatter_bad, 1e-6)))
    if max_side >= 2 or low_found >= 2:
        general_mod_score += 0.10
    if n_pairs >= 2 and np.isfinite(asym) and asym >= args_global.combined_asymmetry_limit:
        general_mod_score += 0.10
    if phase_disabled:
        general_mod_diagnostic = (
            f"phase-independent modulation indicators only; side orders {modstat.get('side_orders_found','')}; "
            f"L=1 side-pair asymmetry={format_float(asym,3)}; AM scatter={format_float(am_scatter,3)}; PM scatter={format_float(pm_scatter,3)}"
        )
        general_mod_comment = "Frequency-only/ignore-phases mode: complex AM+FM/PM sideband fitting is disabled. This row uses only the frequency grid and real-amplitude sideband diagnostics."
        general_mod_points = int(modstat.get("n_am_ratio_points", 0) or 0)
        general_mod_scatter = am_scatter if np.isfinite(am_scatter) else pm_scatter
    else:
        general_mod_diagnostic = f"complex sideband fit C(n,L)/C(n,0) ~= a_L + n b_L; scatter={format_float(combined_modfit_scatter,3)}"
        general_mod_comment = "General combined AM+FM/PM test. For each side order L it fits the complex normalized sidebands as an AM-like constant plus a PM/FM-like term proportional to harmonic order n. A poor fit disfavors this simple low-order modulation approximation, but not necessarily a Blazhko-type almost-periodic interpretation."
        general_mod_points = modfit_points
        general_mod_scatter = combined_modfit_scatter
    rows.append({
        "candidate_id": cand.candidate_id, "fB_abs": cand.fB_abs,
        "model_group": "modulation", "model": "general_combined_AM_FM_PM_modulation",
        "score_0_1": min(1.0, general_mod_score), "frequency_matches": mod_fit_found,
        "tested_positions": mod_fit_tested, "coverage": mod_fit_cov,
        "low_frequency_matches": low_found, "negative_evidence": int(modfit_bad),
        "amplitude_test_points": general_mod_points, "phase_test_points": 0,
        "phase_scatter_rad": np.nan, "amp_ratio_log_scatter": general_mod_scatter,
        "diagnostic": general_mod_diagnostic,
        "comment": general_mod_comment,
    })

    # Coupling submodels.  Right- and left-side f' hypotheses are scored
    # separately.  Each one must predict both grid sides from the same f'.
    coup_grid_found = int(coupstat["frequency_matches"])
    coup_grid_tested = int(coupstat["tested_positions"])
    coup_grid_cov = float(coupstat["coverage"])
    coupling_hypotheses = list(coupstat.get("hypotheses", []))

    complex_bad_limit = float(getattr(args_global, "coupling_complex_scatter_bad", 0.75))
    complex_bad_cap = float(getattr(args_global, "coupling_complex_bad_cap", 0.49))
    complex_min_points = int(getattr(args_global, "coupling_complex_min_points", args_global.phase_min_points) or args_global.phase_min_points)

    def apply_complex_bad_cap(score: float, scatter: Any, points: int, label: str) -> Tuple[float, bool, str]:
        if phase_disabled or points < complex_min_points or not np.isfinite(scatter):
            return score, False, ""
        if float(scatter) <= complex_bad_limit:
            return score, False, ""
        note = (
            f"; {label} complex-ratio scatter={float(scatter):.3g} > "
            f"{complex_bad_limit:.3g}; physical coupling score capped at {complex_bad_cap:.2f}"
        )
        return min(float(score), complex_bad_cap), True, note

    def record_legacy_fit_scatter_difference(
        score: float,
        coupling_scatter: Any,
        coupling_points: int,
        modulation_scatter: Any,
        modulation_points: int,
        label: str,
    ) -> Tuple[float, bool, str, float]:
        """Record old model-specific scatters without using them for selection.

        These quantities have different normalizations and parameter counts.
        They remain useful as subdiagnostics, but only the common-data AICc/BIC
        comparison below is allowed to select a physical model.
        """
        improvement = (
            float(modulation_scatter) - float(coupling_scatter)
            if np.isfinite(modulation_scatter) and np.isfinite(coupling_scatter)
            else np.nan
        )
        note = (
            f"; legacy {label} scatters recorded only as separate subdiagnostics "
            f"(coupling={format_float(coupling_scatter,3)}, modulation={format_float(modulation_scatter,3)}, "
            f"points={coupling_points}/{modulation_points}); not used for physical selection"
        )
        return score, False, note, improvement

    for hyp in coupling_hypotheses:
        origin = str(hyp.get("secondary_origin", "unknown"))
        suffix = f"_{origin}_fprime"
        coup_found = int(hyp.get("frequency_matches", 0) or 0)
        coup_tested = int(hyp.get("tested_positions", 0) or 0)
        coup_cov = float(hyp.get("coverage", 0.0) or 0.0)
        coup_L1 = int(hyp.get("coupling_L1_matches", 0) or 0)
        coup_L1_tested = int(hyp.get("coupling_L1_tested", 0) or 0)
        coup_highL = int(hyp.get("coupling_highL_matches", 0) or 0)
        counterpart_matched = bool(hyp.get("counterpart_predicted_and_matched", False))
        excluded_orders = str(hyp.get("excluded_secondary_harmonic_orders", ""))

        nphase = int(hyp.get("phase_test_points", 0) or 0)
        ph_scatter = hyp.get("phase_scatter_rad", np.nan)
        amp_scatter = hyp.get("amp_ratio_log_scatter", np.nan)
        coup_complex_scatter = hyp.get("coupling_complex_scatter", np.nan)
        coup_complex_points = int(hyp.get("coupling_complex_points", 0) or 0)
        phase_good = nphase >= args_global.phase_min_points and np.isfinite(ph_scatter) and ph_scatter <= args_global.phase_scatter_good
        phase_bad = nphase >= args_global.phase_min_points and np.isfinite(ph_scatter) and ph_scatter > args_global.phase_scatter_bad

        nphase_l1 = int(hyp.get("l1_phase_test_points", 0) or 0)
        ph_scatter_l1 = hyp.get("l1_phase_scatter_rad", np.nan)
        amp_scatter_l1 = hyp.get("l1_amp_ratio_log_scatter", np.nan)
        coup_complex_scatter_l1 = hyp.get("l1_coupling_complex_scatter", np.nan)
        coup_complex_points_l1 = int(hyp.get("l1_coupling_complex_points", 0) or 0)
        phase_good_l1 = nphase_l1 >= args_global.phase_min_points and np.isfinite(ph_scatter_l1) and ph_scatter_l1 <= args_global.phase_scatter_good
        phase_bad_l1 = nphase_l1 >= args_global.phase_min_points and np.isfinite(ph_scatter_l1) and ph_scatter_l1 > args_global.phase_scatter_bad

        significant_secondary_orders_count = int(hyp.get("n_significant_secondary_harmonic_orders", 0) or 0)
        higher_significant_secondary_present = significant_secondary_orders_count >= 2

        # Simple sinusoidal-secondary quadratic coupling.
        qsin_score = 0.0
        if coup_L1 >= 2:
            qsin_score += 0.35
        if low_found >= 1:
            qsin_score += 0.20
        if not higher_significant_secondary_present:
            qsin_score += 0.15
        if counterpart_matched:
            qsin_score += 0.10
        if phase_good_l1:
            qsin_score += 0.25
        elif nphase_l1 >= args_global.phase_min_points and not phase_bad_l1:
            qsin_score += 0.10
        if np.isfinite(amp_scatter_l1) and amp_scatter_l1 < args_global.coupling_amp_scatter_good:
            qsin_score += 0.05
        if phase_bad_l1:
            qsin_score *= 0.35
        if higher_significant_secondary_present:
            qsin_score = 0.0
        qsin_score, qsin_unverified, qsin_unverified_note = cap_unverified_coupling(
            qsin_score, nphase_l1, f"simple quadratic coupling ({origin} f')"
        )
        counterpart_note = ""
        counterpart_failed = False
        if not counterpart_matched:
            counterpart_failed = True
            qsin_score = min(qsin_score, float(getattr(args_global, "coupling_unverified_cap", 0.49)))
            counterpart_note = "; opposite first-order side not predicted/matched from this single f'"
        qsin_score, qsin_complex_bad, qsin_complex_note = apply_complex_bad_cap(
            qsin_score, coup_complex_scatter_l1, coup_complex_points_l1, "L=1"
        )
        qsin_score, qsin_fit_failed, qsin_fit_note, qsin_fit_improvement = record_legacy_fit_scatter_difference(
            qsin_score,
            coup_complex_scatter_l1,
            coup_complex_points_l1,
            l1_combined_modfit_scatter,
            l1_modfit_points,
            "L=1",
        )
        qsin_diagnostic = (
            f"single {origin}-side f'; core L=1 relations; counterpart matched={counterpart_matched}; "
            f"L1 phase scatter={format_float(ph_scatter_l1,3)}, "
            f"L1 complex-ratio scatter={format_float(coup_complex_scatter_l1,3)}"
        )
        if phase_disabled:
            qsin_diagnostic = (
                f"single {origin}-side f'; core L=1 relations; counterpart matched={counterpart_matched}; "
                f"phase diagnostics disabled; L1 amplitude-ratio scatter={format_float(amp_scatter_l1,3)} dex"
            )
        qsin_diagnostic += qsin_unverified_note + counterpart_note + qsin_complex_note + qsin_fit_note
        if excluded_orders:
            qsin_diagnostic += f"; higher secondary orders excluded as modulation-grid-degenerate: {excluded_orders}"
        qsin_comment = (
            "Both multiplet sides are predicted from one fixed secondary frequency; "
            "the opposite side is not reused as another secondary oscillator."
        )
        if higher_significant_secondary_present:
            qsin_diagnostic += f"; not applicable because independent significant secondary harmonics remain: {hyp.get('significant_secondary_harmonic_orders_found','')}"
            qsin_comment += " Independent higher secondary harmonics make the sinusoidal-secondary submodel inapplicable."
        rows.append({
            "candidate_id": cand.candidate_id, "fB_abs": cand.fB_abs,
            "coupling_origin": origin,
            "model_group": "coupling", "model": f"quadratic_coupling_sinusoidal_secondary{suffix}",
            "score_0_1": min(1.0, qsin_score), "frequency_matches": coup_L1,
            "tested_positions": coup_L1_tested,
            "coverage": coup_L1 / coup_L1_tested if coup_L1_tested else 0.0,
            "low_frequency_matches": low_found,
            "negative_evidence": max(0, significant_secondary_orders_count - 1) + int(phase_bad_l1) + int(qsin_unverified) + int(counterpart_failed) + int(qsin_complex_bad) + int(qsin_fit_failed),
            "amplitude_test_points": int(hyp.get("relation_matches_l1", 0) or 0),
            "phase_test_points": 0 if phase_disabled else nphase_l1,
            "phase_scatter_rad": np.nan if phase_disabled else ph_scatter_l1,
            "amp_ratio_log_scatter": amp_scatter_l1,
            "modulation_fit_scatter": l1_combined_modfit_scatter,
            "coupling_fit_scatter": coup_complex_scatter_l1,
            "fit_scatter_improvement": qsin_fit_improvement,
            "fit_comparison_passed": not qsin_fit_failed,
            "diagnostic": qsin_diagnostic,
            "comment": qsin_comment,
        })

        # Non-sinusoidal-secondary quadratic coupling.  Only higher secondary
        # harmonics that are not another modulation grid may support this row.
        nsec_orders = int(hyp.get("n_significant_secondary_harmonic_orders", hyp.get("n_secondary_harmonic_orders", 0)) or 0)
        higher_secondary_present = nsec_orders >= 2
        qnon_score = 0.0
        if higher_secondary_present:
            if coup_found >= 4:
                qnon_score += 0.25
            qnon_score += 0.25
            if coup_highL >= 1:
                qnon_score += 0.15
            if low_found >= 1:
                qnon_score += 0.10
            if counterpart_matched:
                qnon_score += 0.10
            if phase_good:
                qnon_score += 0.25
            elif nphase >= args_global.phase_min_points and not phase_bad:
                qnon_score += 0.10
            if np.isfinite(amp_scatter) and amp_scatter < args_global.coupling_amp_scatter_good:
                qnon_score += 0.10
            if np.isfinite(coup_complex_scatter) and coup_complex_scatter < args_global.coupling_complex_scatter_good:
                qnon_score += 0.05
            if phase_bad:
                qnon_score *= 0.40
        qnon_score, qnon_unverified, qnon_unverified_note = cap_unverified_coupling(
            qnon_score, nphase, f"non-sinusoidal quadratic coupling ({origin} f')"
        )
        qnon_counterpart_failed = higher_secondary_present and not counterpart_matched
        if qnon_counterpart_failed:
            qnon_score = min(qnon_score, float(getattr(args_global, "coupling_unverified_cap", 0.49)))
        qnon_score, qnon_complex_bad, qnon_complex_note = apply_complex_bad_cap(
            qnon_score, coup_complex_scatter, coup_complex_points, "all-L"
        )
        qnon_score, qnon_fit_failed, qnon_fit_note, qnon_fit_improvement = record_legacy_fit_scatter_difference(
            qnon_score,
            coup_complex_scatter,
            coup_complex_points,
            combined_modfit_scatter,
            modfit_points,
            "all-L",
        )
        qnon_diagnostic = (
            f"single {origin}-side f'; secondary harmonic orders {hyp.get('secondary_harmonic_orders_found','')} "
            f"(significant: {hyp.get('significant_secondary_harmonic_orders_found','')}); "
            f"excluded modulation-grid-degenerate orders {excluded_orders or '--'}; "
            f"side orders {hyp.get('side_orders_found','')}; counterpart matched={counterpart_matched}; "
            f"complex-ratio scatter={format_float(coup_complex_scatter,3)}"
        )
        if phase_disabled:
            qnon_diagnostic += f"; phase diagnostics disabled; amplitude-ratio scatter={format_float(amp_scatter,3)} dex"
        qnon_diagnostic += qnon_unverified_note + qnon_complex_note + qnon_fit_note
        if qnon_counterpart_failed:
            qnon_diagnostic += "; opposite first-order side not predicted/matched from this single f'"
        rows.append({
            "candidate_id": cand.candidate_id, "fB_abs": cand.fB_abs,
            "coupling_origin": origin,
            "model_group": "coupling", "model": f"quadratic_coupling_nonsinusoidal_secondary{suffix}",
            "score_0_1": min(1.0, qnon_score),
            "frequency_matches": coup_found if higher_secondary_present else 0,
            "tested_positions": coup_tested,
            "coverage": coup_cov if higher_secondary_present else 0.0,
            "low_frequency_matches": low_found,
            "negative_evidence": int(not higher_secondary_present) + int(phase_bad) + int(qnon_unverified) + int(qnon_counterpart_failed) + int(qnon_complex_bad) + int(qnon_fit_failed),
            "amplitude_test_points": int(hyp.get("amplitude_test_points", 0) or 0),
            "phase_test_points": 0 if phase_disabled else nphase,
            "phase_scatter_rad": np.nan if phase_disabled else ph_scatter,
            "amp_ratio_log_scatter": amp_scatter,
            "modulation_fit_scatter": combined_modfit_scatter,
            "coupling_fit_scatter": coup_complex_scatter,
            "fit_scatter_improvement": qnon_fit_improvement,
            "fit_comparison_passed": not qnon_fit_failed,
            "diagnostic": qnon_diagnostic,
            "comment": "Requires independent harmonics of the same one-sided f' plus compatible coupling relations on both grid sides; harmonics coincident with another modulation grid are not independent evidence.",
        })

        # General nonlinear coupling is also tied to the same one-sided f'.
        higher_required = int(hyp.get("higher_order_coupling_required", 0) or 0)
        general_coup_score = 0.10 + 0.35 * min(1.0, coup_cov)
        if low_found >= 1:
            general_coup_score += 0.10
        if higher_required > 0:
            general_coup_score += 0.15
        if phase_good:
            general_coup_score += 0.10
        if modfit_good:
            general_coup_score *= 0.60
        elif modfit_bad and coup_cov > 0.45:
            general_coup_score += 0.15
        general_coup_score, general_unverified, general_unverified_note = cap_unverified_coupling(
            general_coup_score, nphase, f"general nonlinear coupling ({origin} f')"
        )
        general_coup_score, general_complex_bad, general_complex_note = apply_complex_bad_cap(
            general_coup_score, coup_complex_scatter, coup_complex_points, "all-L"
        )
        general_coup_score, general_fit_failed, general_fit_note, general_fit_improvement = record_legacy_fit_scatter_difference(
            general_coup_score,
            coup_complex_scatter,
            coup_complex_points,
            combined_modfit_scatter,
            modfit_points,
            "all-L",
        )
        rows.append({
            "candidate_id": cand.candidate_id, "fB_abs": cand.fB_abs,
            "coupling_origin": origin,
            "model_group": "coupling", "model": f"general_nonlinear_coupling{suffix}",
            "score_0_1": min(1.0, general_coup_score), "frequency_matches": coup_found,
            "tested_positions": coup_tested, "coverage": coup_cov,
            "low_frequency_matches": low_found,
            "negative_evidence": int(modfit_good) + int(general_unverified) + int(general_complex_bad) + int(general_fit_failed),
            "amplitude_test_points": int(hyp.get("amplitude_test_points", 0) or 0),
            "phase_test_points": 0 if phase_disabled else nphase,
            "phase_scatter_rad": np.nan if phase_disabled else ph_scatter,
            "amp_ratio_log_scatter": amp_scatter,
            "modulation_fit_scatter": combined_modfit_scatter,
            "coupling_fit_scatter": coup_complex_scatter,
            "fit_scatter_improvement": general_fit_improvement,
            "fit_comparison_passed": not general_fit_failed,
            "diagnostic": f"single {origin}-side f'; higher-order required={higher_required}; modulation-fit scatter={format_float(combined_modfit_scatter,3)}; coupling complex scatter={format_float(coup_complex_scatter,3)}" + general_unverified_note + general_complex_note + general_fit_note,
            "comment": "General nonlinear coupling test anchored to one secondary frequency; both frequency-grid sides are predicted from that same f'.",
        })

    # Symmetric physical-model selection.  All legacy modulation/coupling
    # subdiagnostics remain in the full CSV.  The common-data AICc/BIC
    # comparison establishes the baseline physical row; the independent
    # higher-side-order amplitude diagnostics may subsequently redirect it,
    # conservatively, at LEAN strength.
    comparison_cap = max(0.0, min(
        float(getattr(args_global, "report_score_min", 0.5)) - 0.01,
        float(getattr(args_global, "physical_model_nonwinner_cap", 0.49)),
    ))
    for row in rows:
        if row.get("model_group") in {"modulation", "coupling"}:
            row["physical_ranking_eligible"] = False
            if row.get("model") not in FEATURE_FLAG_MODELS:
                row["score_0_1"] = min(float(row.get("score_0_1", 0.0) or 0.0), comparison_cap)

    usable_comparisons = [
        comp for comp in common_comparisons
        if int(comp.get("common_fit_points", 0) or 0) > 0
    ]
    default_common = {
        "candidate_id": cand.candidate_id,
        "secondary_origin": "none",
        "common_fit_points": 0,
        "common_real_observations": 0,
        "common_fit_groups": 0,
        "common_fit_orders": "",
        "common_normalization": "C_side/C_n",
        "modulation_parameter_count": 0,
        "coupling_parameter_count": 0,
        "modulation_common_scatter": np.nan,
        "coupling_common_scatter": np.nan,
        "modulation_aicc": np.nan,
        "coupling_aicc": np.nan,
        "delta_aicc_coupling_minus_modulation": np.nan,
        "modulation_bic": np.nan,
        "coupling_bic": np.nan,
        "delta_bic_coupling_minus_modulation": np.nan,
        "modulation_fit_good": False,
        "coupling_fit_good": False,
        "preferred_physical_model": "none",
        "selection_strength": "ambiguous",
        "selection_class": "ambiguous",
        "selection_code": "AMBIG",
        "selection_tier": 1,
        "physical_interpretation": "no_reliable_modulation_coupling_direction",
        "blazhko_modlike_basis": "",
        "ic_consensus": False,
        "ic_min_abs_delta": np.nan,
        "model_selection_decision": "ambiguous_insufficient_complex_data",
        "model_selection_reason": "no right/left f' anchor has enough common complex data for a usable comparison",
    }
    best_common = consensus_common_model_comparison(common_comparisons, default_common)
    best_common = apply_side_order_envelope_evidence(best_common, modstat)

    # A poor low-order stationary modulation fit does not by itself reject a
    # Blazhko-type phenomenon.  When the equidistant multiplet is well
    # populated and the global linear-beating gate is active, use the more
    # physical BL-MODLIKE residual class if (a) IC already favours modulation,
    # or (b) the coupling fit is absolutely bad even when its smaller parameter
    # count wins relatively.  The latter is the important exclusion case:
    # neither simple stationary modulation nor coupling works, but the dense
    # Blazhko-like multiplet remains.  This label is deliberately not assigned
    # when beating is still viable, the multiplet support is weak, or coupling
    # has not crossed the shared bad-fit threshold.
    blazhko_min_coverage = float(getattr(args_global, "blazhko_modlike_min_coverage", 0.5))
    blazhko_min_matches = int(getattr(args_global, "blazhko_modlike_min_matches", 4) or 4)
    common_mod_scatter = best_common.get("modulation_common_scatter", np.nan)
    common_coup_scatter = best_common.get("coupling_common_scatter", np.nan)
    good_scatter_limit = float(getattr(args_global, "physical_fit_scatter_good", 0.35))
    bad_scatter_limit = float(getattr(args_global, "physical_fit_scatter_bad", 0.75))
    usable_mod_scatters = [
        float(comp.get("modulation_common_scatter"))
        for comp in usable_comparisons
        if np.isfinite(comp.get("modulation_common_scatter", np.nan))
    ]
    usable_coup_scatters = [
        float(comp.get("coupling_common_scatter"))
        for comp in usable_comparisons
        if np.isfinite(comp.get("coupling_common_scatter", np.nan))
    ]
    simple_modulation_poor = bool(
        usable_mod_scatters
        and len(usable_mod_scatters) == len(usable_comparisons)
        and all(value > good_scatter_limit for value in usable_mod_scatters)
    )
    coupling_absolutely_bad = bool(
        usable_coup_scatters
        and len(usable_coup_scatters) == len(usable_comparisons)
        and all(value > bad_scatter_limit for value in usable_coup_scatters)
    )
    relative_modulation_preference = bool(
        str(best_common.get("selection_class", "")) == "modulation_lean"
        and best_common.get("ic_consensus", False)
    )
    # BL-MODLIKE is a modulation-side residual class.  It must never overwrite
    # a coupling direction already selected by either higher-side-order test.
    # It may still arise from a direction-free common result when coupling is
    # absolutely bad; therefore no pre-existing modulation direction is
    # required here.  The explicit side-order guard also covers a LEAN-only
    # n-sequence conflict that has downgraded, but deliberately not reversed, a
    # STRONG modulation result.
    side_order_coupling_selected = bool(
        str(best_common.get("side_order_selected_evidence", "none")) == "coupling"
        and str(best_common.get("side_order_evidence_action", "not_directional"))
        not in {"not_directional", "not_applied_no_common_fit"}
    )
    if (
        simple_modulation_poor
        and (relative_modulation_preference or coupling_absolutely_bad)
        and linear_global_gated
        and mod_cov >= blazhko_min_coverage
        and mod_found >= blazhko_min_matches
        and not side_order_coupling_selected
    ):
        best_common = dict(best_common)
        blazhko_basis = (
            "relative_modulation_preference"
            if relative_modulation_preference
            else "beating_and_coupling_excluded"
        )
        best_common.update({
            "preferred_physical_model": "modulation",
            "selection_strength": "lean",
            "selection_class": "blazhko_modlike",
            "selection_code": "BL-MODLIKE",
            "selection_tier": 3,
            "physical_interpretation": "complex_or_nonstationary_Blazhko_like_variability",
            "model_selection_decision": "blazhko_modlike",
            "blazhko_modlike_basis": blazhko_basis,
            "model_selection_reason": (
                f"linear beating is globally gated by {linear_global_side_peaks} side peaks; "
                f"multiplet coverage {mod_cov:.3f} passes {blazhko_min_coverage:.3f}; "
                f"simple stationary modulation scatter {float(common_mod_scatter):.3g} exceeds "
                f"{good_scatter_limit:.3g}; "
                + (
                    "AICc/BIC significantly disfavour coupling"
                    if relative_modulation_preference
                    else (
                        f"all {len(usable_coup_scatters)} usable right/left coupling-anchor scatters "
                        f"exceed the bad-fit threshold {bad_scatter_limit:.3g}"
                    )
                )
            ),
        })

    common_fields = {
        "common_fit_points": int(best_common.get("common_fit_points", 0) or 0),
        "common_real_observations": int(best_common.get("common_real_observations", 0) or 0),
        "common_fit_groups": int(best_common.get("common_fit_groups", 0) or 0),
        "common_fit_orders": best_common.get("common_fit_orders", ""),
        "common_normalization": best_common.get("common_normalization", "C_side/C_n"),
        "modulation_parameter_count": int(best_common.get("modulation_parameter_count", 0) or 0),
        "coupling_parameter_count": int(best_common.get("coupling_parameter_count", 0) or 0),
        "modulation_common_scatter": best_common.get("modulation_common_scatter", np.nan),
        "coupling_common_scatter": best_common.get("coupling_common_scatter", np.nan),
        "modulation_aicc": best_common.get("modulation_aicc", np.nan),
        "coupling_aicc": best_common.get("coupling_aicc", np.nan),
        "delta_aicc_coupling_minus_modulation": best_common.get("delta_aicc_coupling_minus_modulation", np.nan),
        "modulation_bic": best_common.get("modulation_bic", np.nan),
        "coupling_bic": best_common.get("coupling_bic", np.nan),
        "delta_bic_coupling_minus_modulation": best_common.get("delta_bic_coupling_minus_modulation", np.nan),
        "preferred_physical_model": best_common.get("preferred_physical_model", "none"),
        "selection_strength": best_common.get("selection_strength", "ambiguous"),
        "selection_class": best_common.get("selection_class", "ambiguous"),
        "selection_code": best_common.get("selection_code", "AMBIG"),
        "selection_tier": int(best_common.get("selection_tier", 1) or 1),
        "physical_interpretation": best_common.get("physical_interpretation", "no_reliable_modulation_coupling_direction"),
        "blazhko_modlike_basis": best_common.get("blazhko_modlike_basis", ""),
        "ic_consensus": bool(best_common.get("ic_consensus", False)),
        "ic_min_abs_delta": best_common.get("ic_min_abs_delta", np.nan),
        "model_selection_decision": best_common.get("model_selection_decision", "ambiguous_insufficient_complex_data"),
        "model_selection_reason": best_common.get("model_selection_reason", ""),
        "comparison_secondary_origin": best_common.get("secondary_origin", "none"),
        "usable_anchor_count": int(best_common.get("usable_anchor_count", 0) or 0),
        "anchor_consensus_status": best_common.get("anchor_consensus_status", "none"),
        "anchor_consensus_reason": best_common.get("anchor_consensus_reason", ""),
        "anchor_comparison_summary": best_common.get("anchor_comparison_summary", ""),
        "right_anchor_selection_code": best_common.get("right_anchor_selection_code", "--"),
        "right_anchor_common_fit_points": int(best_common.get("right_anchor_common_fit_points", 0) or 0),
        "right_anchor_modulation_scatter": best_common.get("right_anchor_modulation_scatter", np.nan),
        "right_anchor_coupling_scatter": best_common.get("right_anchor_coupling_scatter", np.nan),
        "right_anchor_delta_aicc": best_common.get("right_anchor_delta_aicc", np.nan),
        "right_anchor_delta_bic": best_common.get("right_anchor_delta_bic", np.nan),
        "left_anchor_selection_code": best_common.get("left_anchor_selection_code", "--"),
        "left_anchor_common_fit_points": int(best_common.get("left_anchor_common_fit_points", 0) or 0),
        "left_anchor_modulation_scatter": best_common.get("left_anchor_modulation_scatter", np.nan),
        "left_anchor_coupling_scatter": best_common.get("left_anchor_coupling_scatter", np.nan),
        "left_anchor_delta_aicc": best_common.get("left_anchor_delta_aicc", np.nan),
        "left_anchor_delta_bic": best_common.get("left_anchor_delta_bic", np.nan),
        "side_order_envelope_evidence": best_common.get("side_order_envelope_evidence", "insufficient"),
        "side_order_envelope_track_count": int(best_common.get("side_order_envelope_track_count", 0) or 0),
        "side_order_envelope_harmonic_count": int(best_common.get("side_order_envelope_harmonic_count", 0) or 0),
        "side_order_envelope_right_tracks": int(best_common.get("side_order_envelope_right_tracks", 0) or 0),
        "side_order_envelope_left_tracks": int(best_common.get("side_order_envelope_left_tracks", 0) or 0),
        "side_order_l2_l1_median_ratio": best_common.get("side_order_l2_l1_median_ratio", np.nan),
        "side_order_l3_l2_median_ratio": best_common.get("side_order_l3_l2_median_ratio", np.nan),
        "side_order_monotonic_track_fraction": best_common.get("side_order_monotonic_track_fraction", np.nan),
        "side_order_reversal_track_fraction": best_common.get("side_order_reversal_track_fraction", np.nan),
        "side_order_envelope_action": best_common.get("side_order_envelope_action", "not_applied"),
        "side_order_envelope_reason": best_common.get("side_order_envelope_reason", ""),
        "side_order_sequence_evidence": best_common.get("side_order_sequence_evidence", "insufficient"),
        "side_order_sequence_eligible_series": int(best_common.get("side_order_sequence_eligible_series", 0) or 0),
        "side_order_sequence_informative_series": int(best_common.get("side_order_sequence_informative_series", 0) or 0),
        "side_order_sequence_modulation_series": int(best_common.get("side_order_sequence_modulation_series", 0) or 0),
        "side_order_sequence_coupling_series": int(best_common.get("side_order_sequence_coupling_series", 0) or 0),
        "side_order_sequence_action": best_common.get("side_order_sequence_action", "not_applied"),
        "side_order_sequence_reason": best_common.get("side_order_sequence_reason", ""),
        "side_order_sequence_summary": best_common.get("side_order_sequence_summary", ""),
        "side_order_selected_evidence": best_common.get("side_order_selected_evidence", "none"),
        "side_order_evidence_source": best_common.get("side_order_evidence_source", "none"),
        "side_order_evidence_action": best_common.get("side_order_evidence_action", "not_directional"),
    }
    mod_row = next((row for row in rows if row.get("model") == "general_combined_AM_FM_PM_modulation"), None)
    chosen_origin = str(best_common.get("secondary_origin", "none"))
    coup_model = f"general_nonlinear_coupling_{chosen_origin}_fprime"
    coup_row = next((row for row in rows if row.get("model") == coup_model), None)
    decision = str(best_common.get("model_selection_decision", "ambiguous_insufficient_complex_data"))
    selection_class = str(best_common.get("selection_class", "ambiguous"))
    preferred_model = str(best_common.get("preferred_physical_model", "none"))

    # A complete-track envelope may now select COUP-LEAN without a usable f'
    # common fit.  In that case there is no anchor-specific coupling row to
    # carry the physical headline, so create an auditable side-order row.  This
    # is not a new classification category; it is the ordinary COUP-LEAN class
    # with its evidence source made explicit.
    if (
        preferred_model == "coupling"
        and coup_row is None
        and str(best_common.get("side_order_evidence_source", "none")) == "complete_tracks"
        and str(best_common.get("side_order_evidence_action", "")) == "resolves_without_common_fit"
    ):
        coup_row = {
            "candidate_id": cand.candidate_id,
            "fB_abs": cand.fB_abs,
            "coupling_origin": "side_order_envelope",
            "model_group": "coupling",
            "model": "general_nonlinear_coupling_side_order_envelope",
            "score_0_1": comparison_cap,
            "frequency_matches": mod_found,
            "tested_positions": mod_tested,
            "coverage": mod_cov,
            "low_frequency_matches": low_found,
            "negative_evidence": 0,
            "amplitude_test_points": 3 * int(common_fields["side_order_envelope_track_count"]),
            "phase_test_points": 0,
            "phase_scatter_rad": np.nan,
            "amp_ratio_log_scatter": np.nan,
            "common_fit_scatter": np.nan,
            "model_aicc": np.nan,
            "model_bic": np.nan,
            "fit_quality_passed": False,
            "physical_ranking_eligible": False,
            **common_fields,
            "diagnostic": "complete-track side-order coupling preference without a usable common fit",
            "comment": (
                "LEAN coupling preference supplied independently by the complete "
                "same-harmonic |L|=1,2,3 amplitude tracks; no common-fit or "
                "information-criterion strength is claimed."
            ),
        }
        rows.append(coup_row)
    anchor_diagnostic = (
        f"anchor consensus={common_fields['anchor_consensus_status']} "
        f"({common_fields['anchor_consensus_reason']}); "
        f"{common_fields['anchor_comparison_summary']}"
    )
    side_order_diagnostic = (
        f"complete-track |L| envelope={common_fields['side_order_envelope_evidence']} "
        f"(tracks={common_fields['side_order_envelope_track_count']}, "
        f"A2/A1={format_float(common_fields['side_order_l2_l1_median_ratio'],3)}, "
        f"A3/A2={format_float(common_fields['side_order_l3_l2_median_ratio'],3)}); "
        f"L=2/3 n-sequences={common_fields['side_order_sequence_evidence']} "
        f"(eligible={common_fields['side_order_sequence_eligible_series']}, "
        f"informative={common_fields['side_order_sequence_informative_series']}); "
        f"selected side-order evidence={common_fields['side_order_selected_evidence']} "
        f"from {common_fields['side_order_evidence_source']} "
        f"(action={common_fields['side_order_evidence_action']})"
    )

    if mod_row is not None:
        mod_row.update(common_fields)
        mod_row["common_fit_scatter"] = best_common.get("modulation_common_scatter", np.nan)
        mod_row["model_aicc"] = best_common.get("modulation_aicc", np.nan)
        mod_row["model_bic"] = best_common.get("modulation_bic", np.nan)
        mod_row["fit_quality_passed"] = bool(best_common.get("modulation_fit_good", False))
        mod_row["diagnostic"] = (
            f"common C_side/C_n fit on {common_fields['common_fit_points']} identical complex side peaks; "
            f"k_mod/k_coup={common_fields['modulation_parameter_count']}/{common_fields['coupling_parameter_count']}; "
            f"scatter={format_float(mod_row['common_fit_scatter'],3)}; "
            f"delta AICc(coupling-modulation)={format_float(common_fields['delta_aicc_coupling_minus_modulation'],4)}; "
            f"delta BIC={format_float(common_fields['delta_bic_coupling_minus_modulation'],4)}; "
            f"decision={decision}; {side_order_diagnostic}; {anchor_diagnostic}"
        )
        if selection_class == "blazhko_modlike":
            mod_row["comment"] = (
                "Residual Blazhko-like interpretation after the simple stationary modulation and coupling relations are tested on the same normalized complex side peaks and linear beating is globally gated. BL-MODLIKE does not mean literal external modulation."
            )
        elif str(best_common.get("side_order_evidence_action", "")) == "resolves_without_common_fit":
            mod_row["comment"] = (
                "LEAN modulation preference supplied independently by the complete "
                "same-harmonic |L|=1,2,3 amplitude tracks; no common-fit or "
                "information-criterion strength is claimed."
            )
        else:
            mod_row["comment"] = "Physical modulation row evaluated on the same normalized complex side peaks and residual scale as coupling; AICc and BIC penalize its larger parameter count."

    for comp in common_comparisons:
        origin = str(comp.get("secondary_origin", "none"))
        row = next((r for r in rows if r.get("model") == f"general_nonlinear_coupling_{origin}_fprime"), None)
        if row is None:
            continue
        row.update({
            "common_fit_points": int(comp.get("common_fit_points", 0) or 0),
            "common_real_observations": int(comp.get("common_real_observations", 0) or 0),
            "common_fit_groups": int(comp.get("common_fit_groups", 0) or 0),
            "common_fit_orders": comp.get("common_fit_orders", ""),
            "common_normalization": comp.get("common_normalization", "C_side/C_n"),
            "modulation_parameter_count": int(comp.get("modulation_parameter_count", 0) or 0),
            "coupling_parameter_count": int(comp.get("coupling_parameter_count", 0) or 0),
            "modulation_common_scatter": comp.get("modulation_common_scatter", np.nan),
            "coupling_common_scatter": comp.get("coupling_common_scatter", np.nan),
            "common_fit_scatter": comp.get("coupling_common_scatter", np.nan),
            "modulation_aicc": comp.get("modulation_aicc", np.nan),
            "coupling_aicc": comp.get("coupling_aicc", np.nan),
            "model_aicc": comp.get("coupling_aicc", np.nan),
            "delta_aicc_coupling_minus_modulation": comp.get("delta_aicc_coupling_minus_modulation", np.nan),
            "modulation_bic": comp.get("modulation_bic", np.nan),
            "coupling_bic": comp.get("coupling_bic", np.nan),
            "model_bic": comp.get("coupling_bic", np.nan),
            "delta_bic_coupling_minus_modulation": comp.get("delta_bic_coupling_minus_modulation", np.nan),
            "fit_quality_passed": bool(comp.get("coupling_fit_good", False)),
            "preferred_physical_model": comp.get("preferred_physical_model", "none"),
            "selection_strength": comp.get("selection_strength", "ambiguous"),
            "selection_class": comp.get("selection_class", "ambiguous"),
            "selection_code": comp.get("selection_code", "AMBIG"),
            "selection_tier": int(comp.get("selection_tier", 1) or 1),
            "physical_interpretation": comp.get("physical_interpretation", "no_reliable_modulation_coupling_direction"),
            "blazhko_modlike_basis": comp.get("blazhko_modlike_basis", ""),
            "ic_consensus": bool(comp.get("ic_consensus", False)),
            "ic_min_abs_delta": comp.get("ic_min_abs_delta", np.nan),
            "model_selection_decision": comp.get("model_selection_decision", "ambiguous_insufficient_complex_data"),
            "model_selection_reason": comp.get("model_selection_reason", ""),
            "comparison_secondary_origin": origin,
        })
        row["diagnostic"] = (
            f"common C_side/C_n fit on {int(comp.get('common_fit_points',0) or 0)} identical complex side peaks; "
            f"k_mod/k_coup={int(comp.get('modulation_parameter_count',0) or 0)}/{int(comp.get('coupling_parameter_count',0) or 0)}; "
            f"scatter={format_float(comp.get('coupling_common_scatter'),3)}; "
            f"delta AICc(coupling-modulation)={format_float(comp.get('delta_aicc_coupling_minus_modulation'),4)}; "
            f"delta BIC={format_float(comp.get('delta_bic_coupling_minus_modulation'),4)}; "
            f"decision={comp.get('model_selection_decision','')}"
        )
        row["comment"] = "Physical coupling row evaluated on the same normalized complex side peaks and residual scale as modulation. Grid-degenerate l f' measurements enter this amplitude-phase fit but add no independent presence score."

    if preferred_model == "coupling" and coup_row is not None:
        # Preserve the chosen anchor's own fit quantities while attaching the
        # two-anchor decision and its right/left audit trail to the eligible
        # coupling row.
        consensus_only_fields = {
            key: value for key, value in common_fields.items()
            if key.startswith("anchor_")
            or key.startswith("right_anchor_")
            or key.startswith("left_anchor_")
            or key.startswith("side_order_")
            or key == "usable_anchor_count"
        }
        coup_row.update(consensus_only_fields)
        coup_row.update({
            "preferred_physical_model": preferred_model,
            "selection_strength": best_common.get("selection_strength", "ambiguous"),
            "selection_class": selection_class,
            "selection_code": best_common.get("selection_code", "AMBIG"),
            "selection_tier": int(best_common.get("selection_tier", 1) or 1),
            "physical_interpretation": best_common.get("physical_interpretation", "no_reliable_modulation_coupling_direction"),
            "ic_consensus": bool(best_common.get("ic_consensus", False)),
            "model_selection_decision": decision,
            "model_selection_reason": best_common.get("model_selection_reason", ""),
        })
        coup_row["diagnostic"] = (
            str(coup_row.get("diagnostic", ""))
            + f"; final decision={decision}; {side_order_diagnostic}; {anchor_diagnostic}"
        )

    finite_common_scatters = [
        float(value)
        for value in (
            best_common.get("modulation_common_scatter", np.nan),
            best_common.get("coupling_common_scatter", np.nan),
        )
        if np.isfinite(value)
    ]
    best_scatter = min(finite_common_scatters) if finite_common_scatters else np.nan
    score_delta_aicc = best_common.get("delta_aicc_coupling_minus_modulation", np.nan)
    score_delta_bic = best_common.get("delta_bic_coupling_minus_modulation", np.nan)
    if (
        selection_class == "blazhko_modlike"
        and str(best_common.get("blazhko_modlike_basis", "")) == "beating_and_coupling_excluded"
    ):
        # In the exclusion-based BL-MODLIKE case the IC direction can point to
        # coupling only because it has fewer parameters, despite an absolutely
        # unacceptable coupling fit.  Do not reward that opposite IC direction
        # in the BL-MODLIKE headline score.
        score_delta_aicc = np.nan
        score_delta_bic = np.nan
    if preferred_model == "modulation" and mod_row is not None:
        rank_score = _ranked_physical_score(
            selection_class,
            best_common.get("modulation_common_scatter", np.nan),
            best_scatter,
            score_delta_aicc,
            score_delta_bic,
            mod_cov,
            common_fields["common_fit_points"],
            cand,
            args_global,
        )
        mod_row["score_0_1"] = rank_score
        mod_row["headline_rank_score"] = rank_score
        mod_row["physical_ranking_eligible"] = True
    elif preferred_model == "coupling" and coup_row is not None:
        rank_score = _ranked_physical_score(
            selection_class,
            best_common.get("coupling_common_scatter", np.nan),
            best_scatter,
            score_delta_aicc,
            score_delta_bic,
            mod_cov,
            common_fields["common_fit_points"],
            cand,
            args_global,
        )
        coup_row["score_0_1"] = rank_score
        coup_row["headline_rank_score"] = rank_score
        coup_row["physical_ranking_eligible"] = True
    else:
        ambiguous_score = _ranked_physical_score(
            "ambiguous",
            np.nan,
            best_scatter,
            best_common.get("delta_aicc_coupling_minus_modulation", np.nan),
            best_common.get("delta_bic_coupling_minus_modulation", np.nan),
            mod_cov,
            common_fields["common_fit_points"],
            cand,
            args_global,
        )
        rows.append({
            "candidate_id": cand.candidate_id,
            "fB_abs": cand.fB_abs,
            "coupling_origin": chosen_origin,
            "model_group": "ambiguous",
            "model": "ambiguous_modulation_vs_coupling",
            "score_0_1": ambiguous_score,
            "frequency_matches": mod_found,
            "tested_positions": mod_tested,
            "coverage": mod_cov,
            "low_frequency_matches": low_found,
            "negative_evidence": int(
                not bool(best_common.get("modulation_fit_good", False))
                and not bool(best_common.get("coupling_fit_good", False))
            ),
            "amplitude_test_points": common_fields["common_fit_points"],
            "phase_test_points": common_fields["common_fit_points"],
            "phase_scatter_rad": np.nan,
            "amp_ratio_log_scatter": np.nan,
            "common_fit_scatter": best_scatter,
            "model_aicc": np.nan,
            "model_bic": np.nan,
            "fit_quality_passed": False,
            "physical_ranking_eligible": True,
            "headline_rank_score": ambiguous_score,
            **common_fields,
            "diagnostic": (
                f"{decision}: {best_common.get('model_selection_reason','')}; "
                f"common points={common_fields['common_fit_points']}; "
                f"k_mod/k_coup={common_fields['modulation_parameter_count']}/{common_fields['coupling_parameter_count']}; "
                f"modulation scatter={format_float(common_fields['modulation_common_scatter'],3)}; "
                f"coupling scatter={format_float(common_fields['coupling_common_scatter'],3)}; "
                f"delta AICc={format_float(common_fields['delta_aicc_coupling_minus_modulation'],4)}; "
                f"delta BIC={format_float(common_fields['delta_bic_coupling_minus_modulation'],4)}; "
                f"{side_order_diagnostic}; {anchor_diagnostic}"
            ),
            "comment": "Neither physical direction is assigned because the common comparison is underconstrained or AICc and BIC are insignificant/conflicting. The non-constant score ranks ambiguous candidates by fit quality, common-point support, multiplet coverage, and discovery support.",
        })

    # Do not promote sub-resolution close-frequency candidates to a physical
    # interpretation.  If |fB| < 1/T, the candidate may represent a long-term
    # trend, unresolved Blazhko drift, or pre-whitening residual rather than a
    # resolved beat/modulation frequency.  The frequency-grid matches are kept
    # in the full CSV, but all candidate-specific rows are capped below the
    # default report threshold.
    fb_resolution = (1.0 / args_global.baseline) if getattr(args_global, "baseline", None) else None
    if fb_resolution is not None and cand.fB_abs < fb_resolution:
        suffix = f" Candidate fB={cand.fB_abs:.6g} is below 1/T={fb_resolution:.6g}; interpretation is unresolved/trend-sensitive and capped below the default report threshold."
        for row in rows:
            if row.get("model_group") != "shared_pattern":
                row["score_0_1"] = min(float(row.get("score_0_1", 0.0) or 0.0), 0.45)
                if bool(row.get("physical_ranking_eligible", False)):
                    row["physical_ranking_eligible"] = False
                    row["selection_code"] = "UNRESOLVED"
                    row["selection_class"] = "unresolved"
                    row["selection_strength"] = "unresolved"
            row["negative_evidence"] = int(row.get("negative_evidence", 0) or 0) + 1
            row["diagnostic"] = str(row.get("diagnostic", "")) + "; unresolved fB < 1/T"
            row["comment"] = str(row.get("comment", "")) + suffix

    return rows


# argparse values used in scoring helper. Kept global to avoid passing many thresholds.
args_global: argparse.Namespace


# -----------------------------------------------------------------------------
# Assignments and output
# -----------------------------------------------------------------------------

def assign_frequency_tags(df: pd.DataFrame, matches: Sequence[FreqMatch]) -> pd.DataFrame:
    assignments: Dict[int, List[str]] = {int(i): [] for i in df["input_index"].to_numpy()}
    for m in matches:
        if m.matched and m.input_index is not None:
            assignments.setdefault(int(m.input_index), []).append(f"{m.category}:{m.model}:{m.label}")
    out = df.copy()
    out["assignments"] = ["; ".join(assignments.get(int(i), [])) for i in out["input_index"].to_numpy()]
    out["assigned"] = out["assignments"].str.len() > 0
    return out


FEATURE_FLAG_MODELS = {
    "non_sinusoidal_modulation",
    "combined_AM_FM_PM_indicator",
    "AM_like_sideband_scaling_indicator",
    "FM_PM_like_sideband_scaling_indicator",
}


def _row_evidence_tier(row: pd.Series, args: argparse.Namespace) -> int:
    """Ranking tier used only for the Markdown headline.

    2 = relation-tested rows with enough phase/complex-fit points;
    1 = model-specific but essentially frequency/amplitude-only;
    0 = deliberately grid-only bookkeeping rows.
    """
    model = str(row.get("model", ""))
    if model == "ambiguous_modulation_vs_coupling":
        return 3
    phase_pts = int(row.get("phase_test_points", 0) or 0)
    amp_pts = int(row.get("amplitude_test_points", 0) or 0)
    if (
        model.startswith("quadratic_coupling_sinusoidal_secondary_")
        or model.startswith("quadratic_coupling_nonsinusoidal_secondary_")
        or model.startswith("general_nonlinear_coupling_")
    ):
        return 2 if phase_pts >= int(getattr(args, "phase_min_points", 5) or 5) else 1
    if model == "general_combined_AM_FM_PM_modulation":
        return 2 if amp_pts >= int(getattr(args, "modfit_min_points", 8) or 8) else 1
    if model in {"equidistant_multiplet_frequency_grid", "general_quadratic_coupling_secondary_secondary"}:
        return 0
    return 1


def select_headline_row(scores_report: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if len(scores_report) == 0:
        raise ValueError("No report scores available")
    tmp = scores_report.copy()
    if "physical_ranking_eligible" in tmp.columns:
        eligible = tmp["physical_ranking_eligible"].map(
            lambda value: True if pd.isna(value) else bool(value)
        )
        if eligible.any():
            tmp = tmp[eligible].copy()
    tmp["_evidence_tier"] = tmp.apply(lambda r: _row_evidence_tier(r, args), axis=1)
    tmp["_selection_tier"] = pd.to_numeric(
        tmp.get("selection_tier", pd.Series(0, index=tmp.index)), errors="coerce"
    ).fillna(0)
    tmp["_headline_score"] = pd.to_numeric(
        tmp.get("headline_rank_score", tmp["score_0_1"]), errors="coerce"
    ).fillna(pd.to_numeric(tmp["score_0_1"], errors="coerce")).fillna(0.0)
    # The disjoint score bands enforce STRONG > LEAN > AMBIG.  The explicit
    # tier, relation evidence, and grid support only break residual ties.
    return tmp.sort_values(
        ["_headline_score", "_selection_tier", "_evidence_tier", "frequency_matches"],
        ascending=[False, False, False, False],
    ).iloc[0]


def decision_summary_line(top: pd.Series) -> str:
    """Build a truthful headline sentence for the selected report row."""
    top_group = str(top.get("model_group", ""))
    top_model = str(top.get("model", ""))
    score = float(top.get("score_0_1", 0.0) or 0.0)

    if top_group == "linear_beating":
        return (
            f"Decision summary: **`{top_model}`** is selected "
            f"(group `linear_beating`, score {score:.2f}) by the dedicated "
            "linear-beating frequency-pattern test and global side-peak gate. "
            "The common-data AICc/BIC comparison is used only for modulation "
            "versus coupling and does not enter this decision."
        )

    top_selection = str(top.get("selection_class", ""))
    top_code = str(top.get("selection_code", "")) or physical_selection_code(top_selection)
    anchor_status = str(top.get("anchor_consensus_status", "") or "")
    anchor_reason = str(top.get("anchor_consensus_reason", "") or "")
    anchor_clause = (
        f" Anchor handling: `{anchor_status}` ({anchor_reason})."
        if anchor_status and anchor_status != "none" else ""
    )
    if top_selection == "ambiguous" or top_group == "ambiguous":
        return (
            f"Decision summary: **`AMBIG`** for candidate {int(top['candidate_id'])} "
            f"(|fB|={float(top.get('fB_abs', np.nan)):.8g} d^-1): no reliable physical direction. "
            f"{top.get('model_selection_reason', '')}.{anchor_clause}"
        )
    if top_selection in {
        "modulation_strong", "modulation_lean", "blazhko_modlike",
        "coupling_lean", "coupling_strong",
    }:
        direction_text = {
            "modulation_strong": "strong modulation preference",
            "modulation_lean": "uncertain absolute fit, but modulation is relatively preferred",
            "blazhko_modlike": "beating and coupling are disfavoured, while the poor simple modulation fit points to complex or non-stationary Blazhko-like variability",
            "coupling_lean": "uncertain absolute fit, but coupling is preferred",
            "coupling_strong": "strong coupling preference",
        }[top_selection]
        return (
            f"Decision summary: **`{top_code}`** for candidate {int(top['candidate_id'])} "
            f"(|fB|={float(top.get('fB_abs', np.nan)):.8g} d^-1; score {score:.3f}): "
            f"{direction_text}. {top.get('model_selection_reason', '')}.{anchor_clause}"
        )
    return (
        f"Decision summary: **`{top_model}`** is selected "
        f"(group `{top_group}`, score {score:.2f}) by its model-specific "
        "compatibility test."
    )


def filter_report_scores(scores: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if len(scores) == 0:
        return scores
    if args.show_all_models:
        return scores.copy()
    model_scores = scores[~scores["model"].isin(FEATURE_FLAG_MODELS)].copy()
    out = model_scores[model_scores["score_0_1"] >= args.report_score_min].copy()
    if not args.show_grid_only_in_report and len(out):
        # Secondary--secondary cross-coupling is currently tested only as a
        # frequency grid. Keep it in freqdiag_model_scores.csv, but hide it
        # from the default Markdown report.
        out = out[~out["model"].eq("general_quadratic_coupling_secondary_secondary")].copy()
    # Do not force a headline model when all scores are below the report
    # threshold.  In such cases the full CSV still contains the ranked tests,
    # but the Markdown report should honestly state that no model passed.

    # If direct secondary--secondary quadratic coupling is detected, it is the
    # distinctive diagnostic of the general quadratic model.  In that situation
    # frequency-grid modulation rows and weak nested linear rows are not useful
    # as headline interpretations; they remain in freqdiag_model_scores.csv.
    if len(out) and out["model"].eq("general_quadratic_coupling_secondary_secondary").any():
        gscore = float(out.loc[out["model"].eq("general_quadratic_coupling_secondary_secondary"), "score_0_1"].max())
        if gscore >= args.report_score_min:
            keep_mask = (
                out["model"].eq("general_quadratic_coupling_secondary_secondary")
                | out["model"].str.startswith("quadratic_coupling_nonsinusoidal_secondary_")
                | out["model"].str.startswith("quadratic_coupling_sinusoidal_secondary_")
            )
            out = out[keep_mask].copy()
            # For candidate-specific primary--secondary coupling, keep only the
            # stronger close-frequency systems and hide nested sinusoidal rows
            # when a non-sinusoidal row for the same candidate is at least as good.
            if len(out):
                qnon = out[out["model"].str.startswith("quadratic_coupling_nonsinusoidal_secondary_")][["candidate_id", "coupling_origin", "score_0_1"]]
                drop_idx = []
                for idx, r in out[out["model"].str.startswith("quadratic_coupling_sinusoidal_secondary_")].iterrows():
                    same = qnon[
                        qnon["candidate_id"].eq(r["candidate_id"])
                        & qnon["coupling_origin"].eq(r.get("coupling_origin", ""))
                    ]
                    if len(same) and float(same["score_0_1"].max()) >= float(r["score_0_1"]):
                        drop_idx.append(idx)
                if drop_idx:
                    out = out.drop(index=drop_idx)

    # Hide the sinusoidal-secondary quadratic-coupling submodel whenever the
    # non-sinusoidal-secondary submodel for the same candidate is report-visible.
    # The former is a nested/inapplicable subcase once significant secondary
    # harmonics have been detected.
    if len(out):
        qnon_visible = out[out["model"].str.startswith("quadratic_coupling_nonsinusoidal_secondary_")][["candidate_id", "coupling_origin", "score_0_1"]]
        drop_idx = []
        for idx, r in out[out["model"].str.startswith("quadratic_coupling_sinusoidal_secondary_")].iterrows():
            same = qnon_visible[
                qnon_visible["candidate_id"].eq(r["candidate_id"])
                & qnon_visible["coupling_origin"].eq(r.get("coupling_origin", ""))
            ]
            if len(same):
                drop_idx.append(idx)
        if drop_idx:
            out = out.drop(index=drop_idx).copy()

    return out.sort_values(["candidate_id", "score_0_1"], ascending=[True, False])


def make_latex_summary(scores: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}",
        r"\caption{Frequency-list diagnostic summary.}",
        r"\label{tab:freqdiag_summary}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{clrrrr}",
        r"\hline\hline",
        r"Candidate & Model & Score & Freq. & Low-$f$ & Phase scatter \\",
        r"\hline",
    ]
    for _, r in scores.iterrows():
        model = str(r["model"]).replace("_", r"\_")
        phase = "--" if not np.isfinite(r.get("phase_scatter_rad", np.nan)) else f"{r['phase_scatter_rad']:.2f}"
        lines.append(f"{int(r['candidate_id'])} & {model} & {r['score_0_1']:.2f} & {int(r['frequency_matches'])} & {int(r['low_frequency_matches'])} & {phase} \\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\tablefoot{The score is a heuristic compatibility indicator, not a physical classification. Unassigned significant frequencies are not penalized by default.}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_markdown_report(
    path: Path,
    args: argparse.Namespace,
    f0: float,
    tol: float,
    harmonics: Sequence[FreqMatch],
    candidates: Sequence[Candidate],
    scores_all: pd.DataFrame,
    scores_report: pd.DataFrame,
    assignment_df: pd.DataFrame,
    phase_test_df: pd.DataFrame,
    modulation_summaries: List[Dict[str, Any]],
    coupling_summaries: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# Frequency-list diagnostic report")
    lines.append("")
    lines.append("This report checks which mathematical Fourier patterns are compatible with the input frequency list. It is not a unique physical classification of the star.")
    lines.append("")
    lines.append("## Input and tolerance")
    lines.append("")
    lines.append(f"- Script version: `{__version__}`")
    lines.append(f"- Input file: `{args.input}`")
    lines.append(f"- Adopted primary frequency: **f0 = {f0:.12g} d^-1**")
    lines.append(f"- Primary-frequency selection: `{getattr(args, 'f0_selection_method_used', 'unknown')}`")
    lines.append(f"- Frequency tolerance: **{tol:.6g} d^-1**")
    if args.baseline:
        lines.append(f"- Time span used for tolerance: T = {args.baseline:g} d, tol = {args.tol_factor:g}/T unless --tol was given")
    if bool(getattr(args, "f0_resolution_warning", False)):
        lines.append(f"- **WARNING — unresolved primary-frequency candidate:** {getattr(args, 'f0_resolution_warning_text', '')}")
    phase_disabled = bool(getattr(args, "ignore_phases", False) or getattr(args, "frequency_only", False))
    if phase_disabled:
        lines.append("- Phase use: **disabled** (`--ignore-phases`/`--frequency-only`); real amplitudes are still used for amplitude-only diagnostics")
    else:
        lines.append(f"- Phase unit: `{args.phase_unit}`; internal phase convention: sine-equivalent; input convention: `{args.phase_convention}`")
    lines.append(f"- Match selection inside tolerance: `{args.match_selection}`")
    if not bool(getattr(args, "no_linear_beating_global_gate", False)):
        side_count = int(getattr(args, "linear_beating_global_side_peak_count", 0) or 0)
        max_side_allowed = int(getattr(args, "linear_beating_max_side_peaks", 1) or 1)
        gate_state = "active" if side_count > max_side_allowed else "not triggered"
        lines.append(f"- Linear-beating global gate: {gate_state}; detected {side_count} side-peak-like frequencies near primary harmonics, allowed maximum {max_side_allowed}")
    lines.append("")

    f0_ranking = getattr(args, "f0_candidate_ranking", pd.DataFrame())
    if isinstance(f0_ranking, pd.DataFrame) and len(f0_ranking):
        lines.append("## Automatic primary-frequency ranking")
        lines.append("")
        lines.append("The automatic selector ranks complete harmonic sequences rather than adopting the single largest-amplitude peak. The score combines weighted harmonic coverage, number and continuity of detected harmonics, summed harmonic amplitude, and frequency-match precision.")
        lines.append("")
        report_top = max(1, int(getattr(args, "report_top_f0_candidates", 10)))
        table_rows = []
        for _, r in f0_ranking.head(report_top).iterrows():
            selected_mark = "yes" if bool(r.get("selected", False)) else ""
            table_rows.append([
                int(r["rank"]), selected_mark, f"{float(r['frequency']):.10g}",
                f"{float(r['score_0_1']):.3f}",
                f"{int(r['n_harmonics_matched'])}/{int(r['n_harmonics_tested'])}",
                int(r["contiguous_harmonics"]), f"{float(r['total_harmonic_amplitude']):.6g}",
                f"{float(r['fundamental_amplitude']):.6g}",
            ])
        lines.extend(markdown_table(
            ["rank", "selected", "frequency", "score", "matched/tested", "contiguous", "harmonic A", "fundamental A"],
            table_rows,
            numeric_columns={0, 2, 3, 4, 5, 6, 7},
        ))
        lines.append("")

    lines.append("## Treatment of unrelated extra frequencies")
    lines.append("")
    lines.append("Observed stars may contain additional independent modes, aliases, residual instrumental peaks, or structures outside the tested beating/modulation pattern. These are kept in an `unassigned` list and are **not** penalized by default. The diagnostics are necessary-condition tests for candidate structures, not a complete decomposition of the full spectrum.")
    if phase_disabled:
        lines.append("")
        lines.append("Because phase use is disabled, phase-dependent tests such as combination-phase scatter, complex sideband fitting, and coupling complex-ratio scatter are not used for scoring. Frequency-grid diagnostics and real-amplitude ratios/asymmetries are still evaluated.")
    lines.append("")

    lines.append("## Detected primary harmonics")
    lines.append("")
    table_rows = []
    for m in harmonics:
        k = m.label.replace(" f0", "")
        table_rows.append([k, f"{m.expected_frequency:.8f}", format_float(m.observed_frequency, 10), format_float(m.delta_frequency, 3), format_float(m.amplitude, 5)])
    lines.extend(markdown_table(
        ["k", "expected", "observed", "delta", "amplitude"],
        table_rows,
        numeric_columns={0, 1, 2, 3, 4},
    ))
    lines.append("")

    lines.append("## Candidate close-frequency offsets")
    lines.append("")
    if candidates:
        lines.append("Harmonically related spacings are retained as individual hypotheses but assigned to one modulation family. Automatic candidates are searched around every detected primary harmonic. A candidate with no observed f0-side peak has `--` in both f' columns and can support a modulation grid, but cannot act as a coupling secondary oscillator.")
        lines.append("")
        table_rows = []
        for c in candidates:
            table_rows.append([
                c.candidate_id, f"{c.fB_abs:.8f}", c.signs,
                format_float(c.right_frequency, 10), format_float(c.left_frequency, 10),
                format_float(c.representative_amplitude, 5), c.discovery_harmonics or "--",
                c.discovery_harmonic_count, c.discovery_peak_count, c.family_id,
                c.family_order, "yes" if c.family_representative else "",
            ])
        lines.extend(markdown_table(
            ["id", "|fB|", "grid sides", "right f'", "left f'", "max A", "support k", "N(k)", "N(peaks)", "family", "order", "representative"],
            table_rows,
            numeric_columns={0, 1, 3, 4, 5, 7, 8, 9, 10},
        ))
        family_notes = [c for c in candidates if "," in c.family_member_ids]
        if family_notes:
            lines.append("")
            lines.append("Family membership: " + "; ".join(f"c{c.candidate_id}: {c.family_member_orders}" for c in family_notes) + ".")
    else:
        lines.append("No close-frequency candidate was found. Use `--fb` to test a specific beat/modulation frequency.")
    lines.append("")

    lines.append("## Modulation and coupling subdiagnostics")
    lines.append("")
    if phase_disabled:
        lines.append("Phase-dependent columns are shown as `--`; amplitude-only columns remain meaningful.")
        lines.append("")
    if modulation_summaries:
        table_rows = []
        for s in modulation_summaries:
            table_rows.append([
                int(s["candidate_id"]), s.get("coverage_basis", ""),
                f"{safe_int(s.get('frequency_matches'))}/{safe_int(s.get('tested_positions'))}",
                format_float(s.get("coverage"), 3), s.get("side_orders_found", ""),
                s.get("low_orders_found", ""), format_float(s.get("median_abs_side_asymmetry_l1"), 3),
                format_float(s.get("combined_modulation_complex_scatter"), 3), safe_int(s.get("modfit_points")),
            ])
        lines.extend(markdown_table(
            ["cand", "coverage basis", "matched/tested", "coverage", "side L", "low L", "L=1 asym", "AM+PM scatter", "fit pts"],
            table_rows,
            numeric_columns={0, 2, 3, 6, 7, 8},
        ))
        lines.append("")
        lines.append("The higher-side-order envelope compares amplitudes only within the same primary harmonic and on the same grid side. It activates only for complete |L|=1,2,3 tracks at two or more distinct harmonics; therefore different harmonic coverage at L=1, 2, and 3 cannot by itself create a trend.")
        lines.append("")
        envelope_rows = []
        envelope_notes = []
        for note_index, s in enumerate(modulation_summaries, start=1):
            note_id = f"E{note_index}"
            envelope_rows.append([
                int(s["candidate_id"]),
                s.get("side_order_envelope_evidence", "insufficient"),
                safe_int(s.get("side_order_envelope_track_count")),
                safe_int(s.get("side_order_envelope_harmonic_count")),
                f"{safe_int(s.get('side_order_envelope_right_tracks'))}/{safe_int(s.get('side_order_envelope_left_tracks'))}",
                format_float(s.get("side_order_l2_l1_median_ratio"), 3),
                format_float(s.get("side_order_l3_l2_median_ratio"), 3),
                note_id,
            ])
            envelope_notes.append((note_id, str(s.get("side_order_envelope_reason", ""))))
        lines.extend(markdown_table(
            ["cand", "L-order envelope", "complete tracks", "harmonics", "right/left", "median A2/A1", "median A3/A2", "note"],
            envelope_rows,
            numeric_columns={0, 2, 3, 4, 5, 6},
        ))
        lines.append("")
        for note_id, reason in envelope_notes:
            lines.append(f"- **{note_id}.** {reason}")
        lines.append("")
        lines.append("The complementary harmonic-order diagnostic follows the raw measured amplitudes of each signed L=2 and L=3 series separately as n increases. Ratios across gaps are converted to a per-harmonic factor. A declining run is modulation-like; coupling-like evidence requires both a strong fall and a strong rise with a direction change. This lower-priority diagnostic is used only when the complete-track envelope is non-directional and has only LEAN authority: it may resolve AMBIG or reverse LEAN, but a conflict with STRONG retains the original direction and downgrades it to LEAN.")
        lines.append("")
        sequence_summary_rows = []
        sequence_summary_notes = []
        sequence_detail_rows = []
        sequence_detail_notes = []
        detail_note_index = 0
        for note_index, s in enumerate(modulation_summaries, start=1):
            note_id = f"S{note_index}"
            sequence_summary_rows.append([
                int(s["candidate_id"]),
                s.get("side_order_sequence_evidence", "insufficient"),
                safe_int(s.get("side_order_sequence_eligible_series")),
                safe_int(s.get("side_order_sequence_informative_series")),
                f"{safe_int(s.get('side_order_sequence_modulation_series'))}/{safe_int(s.get('side_order_sequence_coupling_series'))}",
                note_id,
            ])
            sequence_summary_notes.append((note_id, str(s.get("side_order_sequence_reason", ""))))
            for series in s.get("side_order_sequence_rows", []) or []:
                if safe_int(series.get("points")) <= 0:
                    continue
                detail_note_index += 1
                detail_id = f"R{detail_note_index}"
                sequence_detail_rows.append([
                    int(s["candidate_id"]),
                    series.get("series_label", ""),
                    series.get("evidence", "insufficient"),
                    safe_int(series.get("points")),
                    f"{safe_int(series.get('n_min'))}..{safe_int(series.get('n_max'))}",
                    format_float(series.get("median_amplitude_ratio_per_harmonic"), 3),
                    format_float(series.get("log_amplitude_slope_per_harmonic"), 3),
                    format_float(series.get("monotonic_step_fraction"), 3),
                    format_float(series.get("strong_rise_step_fraction"), 3),
                    safe_int(series.get("direction_change_count")),
                    detail_id,
                ])
                sequence_detail_notes.append((detail_id, str(series.get("reason", ""))))
        lines.extend(markdown_table(
            ["cand", "n-sequence evidence", "eligible series", "informative", "MOD/COUP series", "note"],
            sequence_summary_rows,
            numeric_columns={0, 2, 3, 4},
        ))
        lines.append("")
        for note_id, reason in sequence_summary_notes:
            lines.append(f"- **{note_id}.** {reason}")
        if sequence_detail_rows:
            lines.append("")
            lines.extend(markdown_table(
                ["cand", "series", "evidence", "N", "n range", "median A(n+1)/A(n)", "log slope", "non-rising fraction", "strong-rise fraction", "turns", "note"],
                sequence_detail_rows,
                numeric_columns={0, 3, 4, 5, 6, 7, 8, 9},
            ))
            lines.append("")
            for note_id, reason in sequence_detail_notes:
                lines.append(f"- **{note_id}.** {reason}")
    if coupling_summaries:
        lines.append("")
        lines.append("Each coupling row is anchored to one fixed right- or left-side secondary frequency. A higher l f' term coincident with the candidate's own or another modulation grid gives no independent presence evidence, but its measured complex amplitude is retained in the common modulation-coupling fit.")
        lines.append("")
        table_rows = []
        for s in coupling_summaries:
            table_rows.append([
                int(s["candidate_id"]), s.get("secondary_origin", ""),
                "yes" if s.get("counterpart_predicted_and_matched") else "no",
                f"{safe_int(s.get('frequency_matches'))}/{safe_int(s.get('tested_positions'))}",
                format_float(s.get("coverage"), 3), safe_int(s.get("phase_test_points")),
                format_float(s.get("phase_scatter_rad"), 3), format_float(s.get("l1_coupling_complex_scatter"), 3),
            ])
        lines.extend(markdown_table(
            ["cand", "f' side", "counterpart", "matched/tested", "coverage", "phase pts", "phase scatter", "L=1 complex"],
            table_rows,
            numeric_columns={0, 3, 4, 5, 6, 7},
        ))
        lines.append("")
        lines.append("Coupling-order notes:")
        for s in coupling_summaries:
            lines.append(
                f"- c{int(s['candidate_id'])}, {s.get('secondary_origin','')} f': "
                f"side orders={s.get('side_orders_found','--') or '--'}; usable secondary orders={s.get('secondary_harmonic_orders_found','--') or '--'}; "
                f"excluded grid-degenerate orders={s.get('excluded_secondary_harmonic_orders','--') or '--'}; "
                f"low-frequency orders={s.get('low_orders_found','--') or '--'}."
            )
    lines.append("")

    lines.append("## Additional interpretation flags")
    lines.append("")
    raw_feature_rows = scores_all[scores_all["model"].isin(FEATURE_FLAG_MODELS)].copy() if len(scores_all) else pd.DataFrame()
    raw_feature_rows_reportable = pd.DataFrame()
    if len(raw_feature_rows):
        raw_feature_rows_reportable = raw_feature_rows[raw_feature_rows["score_0_1"] >= args.report_score_min].sort_values(["candidate_id", "score_0_1"], ascending=[True, False])

    # Modulation feature flags are meaningful only as qualifiers of a modulation
    # interpretation.  If the best displayed model is beating/coupling, hide the
    # flags from the main report to avoid implying a competing modulation model.
    top_group = None
    top_model = None
    if len(scores_report):
        top_row = select_headline_row(scores_report, args)
        top_group = str(top_row.get("model_group", ""))
        top_model = str(top_row.get("model", ""))
    feature_rows = raw_feature_rows_reportable.copy()
    hidden_feature_flags = False
    if top_group not in (None, "modulation") and not args.show_all_models:
        hidden_feature_flags = len(raw_feature_rows_reportable) > 0
        feature_rows = pd.DataFrame()

    if len(feature_rows):
        lines.append("These rows are not separate competing physical models. They are qualifiers of the modulation solution, such as non-sinusoidal modulation, sideband scalings, or evidence that AM and FM/PM components coexist.")
        lines.append("")
        table_rows = []
        feature_notes = []
        for note_index, (_, r) in enumerate(feature_rows.iterrows(), start=1):
            note_id = f"F{note_index}"
            model = str(r["model"])
            table_rows.append([int(r["candidate_id"]), model_short_code(model), f"{float(r['score_0_1']):.2f}", note_id])
            feature_notes.append((note_id, model, str(r.get("diagnostic", "")), str(r.get("comment", ""))))
        lines.extend(markdown_table(
            ["cand", "flag", "strength", "note"],
            table_rows,
            numeric_columns={0, 2},
        ))
        lines.append("")
        for note_id, model, diagnostic, comment in feature_notes:
            lines.append(f"- **{note_id} — `{model}`.** Diagnostic: {diagnostic} Meaning: {comment}")
    elif hidden_feature_flags:
        lines.append(f"No modulation qualifier flag is shown here because the top-ranked non-feature model is `{top_model}` ({top_group}). The hidden qualifier rows are still written to `freqdiag_feature_flags.csv`.")
    else:
        lines.append("No additional modulation qualifier flag reached the report threshold. This statement refers only to qualifier flags, not to the modulation model-compatibility rows listed below.")
    lines.append("")

    lines.append("## Candidate model compatibility")
    lines.append("")
    if len(scores_report):
        _top = select_headline_row(scores_report, args)
        lines.append(decision_summary_line(_top))
        lines.append("")
        if len(scores_report) < len(scores_all):
            lines.append(f"Only non-feature model-compatibility rows with score >= {args.report_score_min:g} are shown here. The presence or absence of modulation qualifier flags above is independent of whether a modulation compatibility row appears below. The full table is in `freqdiag_model_scores.csv`; use `--show-all-models` to print all rows in the report.")
            lines.append("")
        lines.append("`MULTIPLET-GRID` is a neutral frequency-pattern row and never participates in physical ranking. Modulation and coupling are fitted to the same complex side peaks normalized as C_side/C_n. AICc and BIC account for their different parameter counts. If the IC-selected model nevertheless has the larger raw common-data scatter, the lower-scatter physical direction overrides that parameter-penalized result and is capped at `LEAN`. Right- and left-side f' anchors are then combined explicitly rather than cherry-picked. If both usable anchors agree, the weaker one sets the conservative modulation strength. If one anchor is direction-free (`AMBIG`) and the other favours modulation, the informative side is retained but the combined result is capped at `MOD-LEAN`; the analogous coupling case retains the supported one-sided coupling hypothesis. A direct modulation-versus-coupling conflict remains `AMBIG`. A single usable anchor is retained. The first higher-side-order test compares complete same-harmonic |L|=1,2,3 tracks: a consistently declining A1,A2,A3 envelope is modulation-like, while a robust strong rise is coupling-like. Complete tracks may independently set a `LEAN` physical direction even when no usable common modulation/coupling fit exists. If that test is non-directional, a lower-priority test follows the raw amplitudes of the individual signed L=2 and L=3 series along harmonic order n. A declining run is modulation-like; a coupling-like run must contain a strong fall, a strong rise, and a direction change. Complete tracks retain priority if the two tests conflict. The lower-priority n-sequence still requires a usable common fit: it may resolve `AMBIG` or reverse a `LEAN` result; against a `STRONG` result it retains the original direction and only downgrades its confidence to `LEAN`. Agreement does not by itself promote a result to `STRONG`. If both information criteria reach the configured delta and agree, a fit passing the shared absolute-scatter threshold is `MOD-STRONG` or `COUP-STRONG`. A poor winning fit is ordinarily `MOD-LEAN` or `COUP-LEAN`. The result becomes the more physical `BL-MODLIKE` only when the multiplet coverage is strong, the global side-peak test excludes linear beating, the simple stationary modulation fit is poor, all usable coupling alternatives are either IC-disfavoured or absolutely bad, and no higher-side-order diagnostic has selected coupling. Thus `BL-MODLIKE` does not claim literal external modulation; it denotes complex, irregular, multiperiodic, or non-stationary Blazhko-like variability left after the simpler alternatives fail. Insufficient, insignificant, or conflicting evidence remains direction-free `AMBIG`. Each spacing candidate is classified independently: in a multiperiodic Blazhko star, several detected fB/fS candidates may be simultaneously meaningful, and the headline is only a ranking convenience rather than a declaration of a unique primary variability. The score bands are disjoint (strong 0.75--1.00, leaning/BL-MODLIKE 0.55--0.74, ambiguous 0.50--0.549), so headline candidates are ranked in that order and then by fit, IC, coverage, and discovery support.")
        lines.append("")
        table_rows = []
        model_notes = []
        for note_index, (_, r) in enumerate(scores_report.iterrows(), start=1):
            note_id = f"M{note_index}"
            model = str(r["model"])
            common_points = safe_int(r.get("common_fit_points"))
            scatter_value = r.get("common_fit_scatter", np.nan)
            raw_selection_code = r.get("selection_code", "")
            selection_code = "" if pd.isna(raw_selection_code) else str(raw_selection_code)
            anchor_status = r.get("anchor_consensus_status", "--")
            anchor_status = "--" if pd.isna(anchor_status) or not str(anchor_status) else str(anchor_status)
            envelope_evidence = r.get("side_order_envelope_evidence", "--")
            envelope_evidence = "--" if pd.isna(envelope_evidence) or not str(envelope_evidence) else str(envelope_evidence)
            sequence_evidence = r.get("side_order_sequence_evidence", "--")
            sequence_evidence = "--" if pd.isna(sequence_evidence) or not str(sequence_evidence) else str(sequence_evidence)
            table_rows.append([
                int(r["candidate_id"]), model_short_code(model), selection_code or "--", f"{float(r['score_0_1']):.3f}",
                f"{int(r['frequency_matches'])}/{int(r['tested_positions'])}", common_points,
                format_float(scatter_value, 3),
                format_float(r.get("delta_aicc_coupling_minus_modulation"), 4),
                format_float(r.get("delta_bic_coupling_minus_modulation"), 4), anchor_status, envelope_evidence, sequence_evidence, note_id,
            ])
            model_notes.append((note_id, model, str(r.get("diagnostic", "")), str(r.get("comment", ""))))
        lines.extend(markdown_table(
            ["cand", "model", "decision", "score", "matched/tested", "common N", "fit scatter", "delta AICc", "delta BIC", "f' consensus", "L-track", "L=2/3 n-run", "note"],
            table_rows,
            numeric_columns={0, 3, 4, 5, 6, 7, 8},
        ))
        lines.append("")
        lines.append("Model notes:")
        for note_id, model, diagnostic, comment in model_notes:
            lines.append(f"- **{note_id} — `{model}`.** Diagnostic: {diagnostic} Interpretation: {comment}")
    else:
        non_feature = scores_all[~scores_all["model"].isin(FEATURE_FLAG_MODELS)].copy() if len(scores_all) else pd.DataFrame()
        if len(non_feature):
            best = non_feature.sort_values(["score_0_1", "frequency_matches"], ascending=[False, False]).iloc[0]
            lines.append(
                f"No model reached the report threshold {args.report_score_min:g}. The best row was "
                f"`{best['model']}` for candidate {int(best['candidate_id'])} with score {float(best['score_0_1']):.2f}."
            )
            lines.append("")
            lines.append(f"Reason: {best.get('diagnostic', '')}")
        else:
            lines.append("No model could be evaluated.")
        lines.append("")
        lines.append("See `freqdiag_model_scores.csv` for all tested rows.")
    lines.append("")

    lines.append("## Strongest unassigned frequencies")
    lines.append("")
    unassigned = assignment_df[~assignment_df["assigned"]].sort_values("amplitude", ascending=False)
    if len(unassigned) == 0:
        lines.append("No unassigned frequencies above the adopted thresholds.")
    else:
        lines.append("These frequencies were not assigned to any tested harmonic, side-peak, or combination pattern. They are retained as possible independent modes, aliases, noise, or structures outside the tested model.")
        lines.append("")
        table_rows = []
        for _, r in unassigned.head(args.report_top_extras).iterrows():
            table_rows.append([f"{r['frequency']:.8f}", f"{r['amplitude']:.5g}", format_float(r.get("phase_rad"), 4), r.get("label", "")])
        lines.extend(markdown_table(
            ["frequency", "amplitude", "phase [rad]", "label"],
            table_rows,
            numeric_columns={0, 1, 2},
        ))
    lines.append("")

    lines.append("## Unassigned close peaks around primary harmonics")
    lines.append("")
    rows = []
    for _, r in assignment_df[~assignment_df["assigned"]].iterrows():
        freq = float(r["frequency"])
        n = int(round(freq / f0)) if f0 > 0 else 0
        if 1 <= n <= args.nmax:
            delta = freq - n * f0
            if abs(delta) <= args.side_window and abs(delta) > tol:
                rows.append((n, delta, r))
    if rows:
        table_rows = []
        for n, delta, r in sorted(rows, key=lambda x: -float(x[2]["amplitude"]))[: args.report_top_extras]:
            table_rows.append([f"{n} f0", f"{float(r['frequency']):.8f}", f"{delta:.8f}", f"{float(r['amplitude']):.5g}"])
        lines.extend(markdown_table(
            ["near harmonic", "frequency", "offset", "amplitude"],
            table_rows,
            numeric_columns={0, 1, 2, 3},
        ))
    else:
        lines.append("No unassigned close peaks around the tested primary harmonics.")
    lines.append("")

    lines.append("## Combination phase diagnostics")
    lines.append("")
    if len(phase_test_df):
        if phase_disabled:
            lines.append("Phase use is disabled. The table therefore lists only the real-amplitude part of the coupling diagnostic, `A_side/(A_parent A_secondary)`. Combination phases and complex-ratio scatters are not used for scoring in this run.")
        else:
            lines.append("The right- and left-side secondary hypotheses are tested separately. For a fixed `f'=f0+s delta`, both grid sides are generated with operation sign `op=sign(L)*s`; the combination phase is `phi_side - phi_parent - op*psi_l` and the required parent order is `k=n-op*l`. Thus the opposite side is predicted from the same f' instead of being reused as another secondary. Grid-degenerate l f' rows remain in this amplitude-phase table and in the common fit, but never count as independent frequency-presence evidence.")
        lines.append("")
        table_rows = []
        for _, r in phase_test_df.head(args.report_top_phase).iterrows():
            table_rows.append([
                int(r["candidate_id"]), r.get("secondary_origin", ""), int(r["n"]), int(r["L"]),
                int(r["coupling_operation_sign"]), int(r["parent_k"]),
                "yes" if r.get("predicted_counterpart") else "",
                "yes" if r.get("secondary_is_grid_degenerate") else "", f"{float(r['side_frequency']):.8f}",
                f"{float(r['amplitude_ratio_side_over_parent_secondary']):.4g}", format_float(r.get("combination_phase_rad"), 4),
            ])
        lines.extend(markdown_table(
            ["cand", "f' side", "n", "L", "op", "parent k", "counterpart", "grid-degenerate", "frequency", "amp ratio", "Phi_comb [rad]"],
            table_rows,
            numeric_columns={0, 2, 3, 4, 5, 8, 9, 10},
        ))
    else:
        if phase_disabled:
            lines.append("No coupling amplitude-ratio rows were available for the matched frequency grid.")
        else:
            lines.append("No usable combination phase tests were available.")
    lines.append("")

    lines.append("## Output files")
    lines.append("")
    lines.append("- `freqdiag_model_scores.csv`: full model/submodel score table")
    lines.append("- `freqdiag_report_scores.csv`: non-feature model rows shown in this report")
    lines.append("- `freqdiag_feature_flags.csv`: modulation/coupling feature flags listed separately")
    lines.append("- `freqdiag_matches.csv`: all expected frequencies and matches")
    lines.append("- `freqdiag_frequency_assignments.csv`: original frequencies plus assignment tags")
    if isinstance(f0_ranking, pd.DataFrame) and len(f0_ranking):
        lines.append("- `freqdiag_f0_candidates.csv`: automatic primary-frequency candidate ranking and score components")
    if phase_disabled:
        lines.append("- `freqdiag_phase_tests.csv`: coupling amplitude-ratio diagnostics; phase columns are disabled/NaN")
    else:
        lines.append("- `freqdiag_phase_tests.csv`: amplitude-ratio and combination-phase diagnostics")
    lines.append("- `freqdiag_modulation_summary.csv`: AM/FM/PM sideband subdiagnostics")
    lines.append("- `freqdiag_side_order_sequences.csv`: separate signed L=2 and L=3 amplitude runs along harmonic order")
    lines.append("- `freqdiag_coupling_summary.csv`: coupling-grid and phase subdiagnostics")
    lines.append("- `freqdiag.json`: machine-readable summary")
    lines.append("- `freqdiag_latex_table.tex`: compact LaTeX table")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    global args_global
    args_global = args
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tol = choose_tolerance(args)
    df = read_frequency_table(input_path, args)
    if len(df) == 0:
        raise ValueError("No usable frequencies after filtering.")

    f0, f0_index, f0_ranking = determine_f0(df, args, tol)
    args.f0_candidate_ranking = f0_ranking
    if getattr(args, "baseline", None):
        f0_resolution_scale = 1.0 / float(args.baseline)
        f0_resolution_scale_label = "1/T"
    else:
        f0_resolution_scale = float(tol)
        f0_resolution_scale_label = "the adopted matching tolerance"
    args.f0_resolution_scale = f0_resolution_scale
    args.f0_resolution_warning = bool(f0 <= f0_resolution_scale)
    args.f0_resolution_warning_text = (
        f"adopted f0={f0:.6g} d^-1 is <= {f0_resolution_scale_label}="
        f"{f0_resolution_scale:.6g} d^-1. Adjacent primary harmonics are not "
        "independently resolved on this time base; verify f0 manually."
        if args.f0_resolution_warning else ""
    )
    harmonic_matches, harmonic_map = detect_harmonics(df, f0, args.nmax, tol)
    candidates = detect_candidates(df, f0, harmonic_map, args, tol)

    # Global side-peak count used to keep linear beating as a genuinely
    # isolated-peak interpretation.
    linear_side_peak_rows = find_close_harmonic_side_peaks(df, f0, args.nmax, tol, args.side_window)
    args.linear_beating_global_side_peak_count = len(linear_side_peak_rows)

    all_matches: List[FreqMatch] = []
    all_matches.extend(harmonic_matches)
    all_scores: List[Dict[str, Any]] = []
    all_phase_tests: List[pd.DataFrame] = []
    cand_records: List[Dict[str, Any]] = []
    modulation_summaries: List[Dict[str, Any]] = []
    coupling_summaries: List[Dict[str, Any]] = []
    side_order_sequence_rows: List[Dict[str, Any]] = []

    for cand in candidates:
        cand_records.append(asdict(cand))
        linear_matches, secondary_map = detect_linear_sequences(df, f0, cand, args.lmax_secondary, tol)
        effective_modulation_lmax = modulation_lmax_for_candidate(cand, args.lmax_modulation)
        secondary_exclusions = find_secondary_modulation_grid_overlaps(
            cand,
            candidates,
            secondary_map,
            args.lmax_modulation,
            tol,
        )
        low_matches = detect_low_terms(df, cand, max(effective_modulation_lmax, args.lmax_coupling), tol)
        modulation_matches = detect_modulation_grid(df, f0, cand, args.nmax, effective_modulation_lmax, tol)
        coupling_matches = detect_coupling_grid(df, f0, cand, args.nmax, args.lmax_coupling, tol)

        all_matches.extend(linear_matches)
        all_matches.extend(low_matches)
        all_matches.extend(modulation_matches)
        all_matches.extend(coupling_matches)

        modstat = modulation_stats(cand, harmonic_map, modulation_matches, low_matches, args)
        modstat["candidate_id"] = cand.candidate_id
        modulation_summaries.append(modstat)
        for sequence_row in modstat.get("side_order_sequence_rows", []) or []:
            side_order_sequence_rows.append({
                "candidate_id": cand.candidate_id,
                "fB_abs": cand.fB_abs,
                **sequence_row,
            })

        phase_df, phase_summary = phase_and_amplitude_tests(
            cand,
            harmonic_map,
            secondary_map,
            coupling_matches,
            secondary_exclusions,
        )
        if len(phase_df):
            all_phase_tests.append(phase_df)
        coupstat = coupling_stats(
            cand,
            coupling_matches,
            low_matches,
            secondary_map,
            phase_summary,
            secondary_exclusions,
        )
        coupstat["candidate_id"] = cand.candidate_id
        for hyp_summary in coupstat.get("hypotheses", []):
            coupling_summaries.append({
                "candidate_id": cand.candidate_id,
                "fB_abs": cand.fB_abs,
                **hyp_summary,
            })

        all_scores.extend(score_models_for_candidate(cand, harmonic_map, linear_matches, modulation_matches, coupling_matches, low_matches, secondary_map, modstat, coupstat))

    # General quadratic coupling among secondary oscillations.  This is a
    # multi-candidate diagnostic and must be evaluated after all close-frequency
    # candidates have been identified.  It is deliberately conservative:
    # (i) all participating close-frequency offsets must be resolved at least at
    #     the 1/T level when a baseline is available;
    # (ii) because the present implementation has no independent amplitude/phase
    #      verification for secondary--secondary terms, a pure frequency-grid
    #      match cannot receive a near-certain score.
    ss_matches: List[FreqMatch] = []
    ss_summary: Dict[str, Any] = {}
    if len(candidates) >= 2:
        # Use the strongest candidates only to avoid a combinatorial explosion
        # from very weak incidental close peaks.
        ss_candidates = sorted(candidates, key=lambda c: -(c.representative_amplitude or 0.0))[: args.max_cross_candidates]
        ss_matches = detect_secondary_secondary_quadratic_terms(df, f0, ss_candidates, args.lmax_secondary_secondary, tol)
        all_matches.extend(ss_matches)
        ss_summary = secondary_secondary_summary(ss_matches, f0)
        ss_found = int(ss_summary.get("secondary_secondary_matches", 0))
        ss_low = int(ss_summary.get("secondary_secondary_low_matches", 0))
        ss_near = int(ss_summary.get("secondary_secondary_near_harmonic_matches", 0))
        ss_tested = int(ss_summary.get("secondary_secondary_tested", 0))
        ss_cov = float(ss_summary.get("secondary_secondary_coverage", 0.0))

        # Resolution criterion requested for this diagnostic.  If no baseline was
        # supplied, fall back to the same tolerance used for frequency matching.
        ss_resolution = (1.0 / args.baseline) if args.baseline else tol
        unresolved_ids = [c.candidate_id for c in ss_candidates if c.fB_abs < ss_resolution]
        unresolved_fb = len(unresolved_ids) > 0

        ss_score = 0.0
        if ss_found >= 1:
            ss_score += 0.25
        if ss_low >= 1:
            ss_score += 0.25
        if ss_near >= 2:
            ss_score += 0.25
        elif ss_near >= 1:
            ss_score += 0.12
        if ss_found >= 4:
            ss_score += 0.15
        if ss_cov > 0.15:
            ss_score += 0.10

        # Frequency positions alone are not enough for a physical secondary--
        # secondary quadratic interpretation.  Until a dedicated amplitude/phase
        # consistency test is implemented for these cross terms, cap the score
        # below the default report threshold.  The row remains in the full CSV
        # as a weak grid-compatibility flag.
        frequency_only_cap = float(getattr(args, "secondary_secondary_frequency_only_cap", 0.49) or 0.49)
        if unresolved_fb:
            # A sub-resolution fB can be a long-term trend or prewhitening
            # residual. Keep the result in the full CSV, but do not let it become
            # a headline interpretation under the default report threshold.
            frequency_only_cap = min(frequency_only_cap, 0.45)
        ss_score = min(ss_score, frequency_only_cap)

        if unresolved_fb:
            diag_resolution = f"; unresolved fB candidates {unresolved_ids} below 1/T={ss_resolution:.6g}"
            comment_resolution = " At least one participating fB is below 1/T, so the secondary--secondary interpretation is treated as unresolved/trend-sensitive and is capped below the default report threshold."
        else:
            diag_resolution = f"; all participating fB >= 1/T={ss_resolution:.6g}"
            comment_resolution = " The score is capped below the default report threshold because this implementation currently verifies only the frequency grid for secondary--secondary terms."

        all_scores.append({
            "candidate_id": 0,
            "fB_abs": np.nan,
            "model_group": "coupling",
            "model": "general_quadratic_coupling_secondary_secondary",
            "score_0_1": min(1.0, ss_score),
            "frequency_matches": ss_found,
            "tested_positions": ss_tested,
            "coverage": ss_cov,
            "low_frequency_matches": ss_low,
            "negative_evidence": int(unresolved_fb),
            "amplitude_test_points": 0,
            "phase_test_points": 0,
            "phase_scatter_rad": np.nan,
            "amp_ratio_log_scatter": np.nan,
            "physical_ranking_eligible": False,
            "model_selection_decision": "frequency_only_not_ranked",
            "diagnostic": f"secondary--secondary terms: low={ss_low}, near harmonics={ss_near}; frequency-only score cap={frequency_only_cap:.2f}" + diag_resolution,
            "comment": "Direct quadratic coupling among secondary oscillations predicts terms such as |f'_i-f'_j| and f'_i+f'_j. These are absent from the independent-primary-coupling model, but a frequency-grid match alone is not a physical proof." + comment_resolution,
        })

    matches_df = match_to_df(all_matches)
    matches_df.to_csv(outdir / "freqdiag_matches.csv", index=False)
    match_to_df(harmonic_matches).to_csv(outdir / "freqdiag_harmonics.csv", index=False)
    if len(f0_ranking):
        f0_ranking.to_csv(outdir / "freqdiag_f0_candidates.csv", index=False)
    pd.DataFrame(cand_records).to_csv(outdir / "freqdiag_candidates.csv", index=False)
    scalar_modulation_summaries = [
        {
            key: value
            for key, value in summary_row.items()
            if key != "side_order_sequence_rows"
        }
        for summary_row in modulation_summaries
    ]
    pd.DataFrame(scalar_modulation_summaries).to_csv(
        outdir / "freqdiag_modulation_summary.csv", index=False
    )
    pd.DataFrame(side_order_sequence_rows).to_csv(
        outdir / "freqdiag_side_order_sequences.csv", index=False
    )
    pd.DataFrame(coupling_summaries).to_csv(outdir / "freqdiag_coupling_summary.csv", index=False)

    scores_df = pd.DataFrame(all_scores)
    if len(scores_df):
        scores_df = scores_df.sort_values(["candidate_id", "score_0_1"], ascending=[True, False])
    scores_df.to_csv(outdir / "freqdiag_model_scores.csv", index=False)
    if len(scores_df):
        scores_df[scores_df["model"].isin(FEATURE_FLAG_MODELS)].to_csv(outdir / "freqdiag_feature_flags.csv", index=False)
    else:
        pd.DataFrame().to_csv(outdir / "freqdiag_feature_flags.csv", index=False)

    scores_report = filter_report_scores(scores_df, args)
    scores_report.to_csv(outdir / "freqdiag_report_scores.csv", index=False)

    phase_test_df = pd.concat(all_phase_tests, ignore_index=True) if all_phase_tests else pd.DataFrame()
    phase_test_df.to_csv(outdir / "freqdiag_phase_tests.csv", index=False)

    assignment_df = assign_frequency_tags(df, all_matches)
    assignment_df.to_csv(outdir / "freqdiag_frequency_assignments.csv", index=False)

    make_latex_summary(scores_report, outdir / "freqdiag_latex_table.tex")

    f0_selection_record = None
    if len(f0_ranking):
        f0_selection_record = json.loads(
            f0_ranking[f0_ranking["selected"]].to_json(orient="records")
        )[0]

    summary = {
        "input": str(input_path),
        "f0": f0,
        "f0_input_index": f0_index,
        "f0_selection_method": getattr(args, "f0_selection_method_used", "unknown"),
        "f0_selection": f0_selection_record,
        "f0_resolution_scale": float(getattr(args, "f0_resolution_scale", tol)),
        "f0_resolution_warning": bool(getattr(args, "f0_resolution_warning", False)),
        "f0_resolution_warning_text": getattr(args, "f0_resolution_warning_text", ""),
        "tolerance": tol,
        "n_input_frequencies_used": int(len(df)),
        "n_deduplicated": int(df.attrs.get("n_deduplicated", 0)),
        "n_unassigned": int((~assignment_df["assigned"]).sum()),
        "phase_diagnostics_enabled": not bool(getattr(args, "ignore_phases", False) or getattr(args, "frequency_only", False)),
        "linear_beating_global_gate_enabled": not bool(getattr(args, "no_linear_beating_global_gate", False)),
        "linear_beating_global_side_peak_count": int(getattr(args, "linear_beating_global_side_peak_count", 0) or 0),
        "linear_beating_max_side_peaks": int(getattr(args, "linear_beating_max_side_peaks", 1) or 1),
        "candidates": cand_records,
        "modulation_summaries": modulation_summaries,
        "side_order_sequences": side_order_sequence_rows,
        "coupling_summaries": coupling_summaries,
        "scores": all_scores,
        "notes": "Unassigned significant frequencies are not penalized by default; inspect freqdiag_frequency_assignments.csv.",
    }
    (outdir / "freqdiag.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    write_markdown_report(outdir / "freqdiag_report.md", args, f0, tol, harmonic_matches, candidates, scores_df, scores_report, assignment_df, phase_test_df, modulation_summaries, coupling_summaries)

    print(f"Wrote diagnostic output to: {outdir.resolve()}")
    print(f"Adopted f0 = {f0:.12g} d^-1; tolerance = {tol:.6g} d^-1")
    print(f"f0 selection: {getattr(args, 'f0_selection_method_used', 'unknown')}")
    if bool(getattr(args, "f0_resolution_warning", False)):
        print(f"WARNING: {args.f0_resolution_warning_text}", file=sys.stderr)
    print(f"Candidate close-frequency offsets: {len(candidates)}")
    print(f"Models shown in report: {len(scores_report)} / {len(scores_df)}")
    if not bool(getattr(args, "no_linear_beating_global_gate", False)):
        print(f"Linear-beating global side peaks: {int(getattr(args, 'linear_beating_global_side_peak_count', 0) or 0)} (allowed <= {args.linear_beating_max_side_peaks})")
    print(f"Unassigned frequencies retained: {(~assignment_df['assigned']).sum()} / {len(assignment_df)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("input", help="Input frequency table, CSV/TSV/whitespace-separated")
    p.add_argument("--outdir", default="freqdiag_output", help="Output directory")
    p.add_argument("--separator", default=None, help="Input separator. Use whitespace, tab, comma, or a literal separator. Omit for autodetect")
    p.add_argument("--comment", default="#", help="Comment character for input reader")
    p.add_argument("--skiprows", type=int, default=0, help="Number of initial input lines to skip before reading")
    p.add_argument("--no-header", action="store_true", help="Input file has no header row; columns can be selected by 1-based indices")

    p.add_argument("--freq-col", default="frequency", help="Frequency column name, or 1-based column number for headerless input")
    p.add_argument("--amp-col", default="amplitude", help="Amplitude column name, or 1-based column number for headerless input")
    p.add_argument("--phase-col", default="phase", help="Phase column name, or 1-based column number for headerless input")
    p.add_argument("--snr-col", default=None, help="Optional S/N column name, or 1-based column number for headerless input")
    p.add_argument("--label-col", default=None, help="Optional label column name, or 1-based column number for headerless input")
    p.add_argument("--freq-col-index", type=int, default=None, help="1-based frequency column index; overrides --freq-col")
    p.add_argument("--amp-col-index", type=int, default=None, help="1-based amplitude column index; overrides --amp-col")
    p.add_argument("--phase-col-index", type=int, default=None, help="1-based phase column index; overrides --phase-col")
    p.add_argument("--snr-col-index", type=int, default=None, help="1-based optional S/N column index; overrides --snr-col")
    p.add_argument("--label-col-index", type=int, default=None, help="1-based optional label column index; overrides --label-col")
    p.add_argument("--phase-unit", choices=["rad", "cycles", "deg"], default="rad", help="Input phase unit")
    p.add_argument("--phase-convention", choices=["sine", "cosine", "unknown"], default="sine", help="Input phase convention; cosine phases are converted to sine-equivalent internally")
    p.add_argument("--ignore-phases", action="store_true", help="Ignore input phases. Frequency-grid tests and real-amplitude diagnostics are still computed; phase/complex-amplitude diagnostics are disabled")
    p.add_argument("--frequency-only", action="store_true", help="Alias for --ignore-phases; useful for automatic prewhitening/peak-list outputs with unreliable phases")

    p.add_argument("--f0", type=float, default=None, help="Primary frequency. If omitted, automatic selection is used")
    p.add_argument("--f0-min", type=float, default=None, help="Ignore lower frequencies when auto-selecting f0")
    p.add_argument("--f0-auto-method", choices=["harmonic-series", "strongest"], default="harmonic-series", help="Automatic f0 selector; strongest reproduces the legacy largest-amplitude rule")
    p.add_argument("--f0-auto-min-harmonics", type=int, default=2, help="Minimum matched members of f,2f,... required by the harmonic-series selector; if none qualifies, fall back to the strongest peak")
    p.add_argument("--f0-auto-max-candidates", type=int, default=200, help="Maximum number of strongest input frequencies ranked as f0 candidates; 0 tests all input frequencies")
    p.add_argument("--baseline", type=float, default=None, help="Time span T [days], used to set tolerance tol_factor/T")
    p.add_argument("--tol", type=float, default=None, help="Absolute frequency matching tolerance [d^-1]")
    p.add_argument("--tol-factor", type=float, default=1.0, help="Tolerance factor if baseline is used: tol=tol_factor/T")
    p.add_argument("--match-selection", choices=["balanced", "nearest", "strongest"], default="balanced", help="How to choose among several peaks within one tolerance window")
    p.add_argument("--match-distance-penalty", type=float, default=0.15, help="Distance penalty used by --match-selection balanced")

    p.add_argument("--min-amp", type=float, default=0.0, help="Minimum amplitude to keep from input")
    p.add_argument("--min-snr", type=float, default=None, help="Optional minimum S/N if --snr-col is present")
    p.add_argument("--max-freq", type=float, default=None, help="Maximum input frequency to keep")
    p.add_argument("--deduplicate", dest="deduplicate", action="store_true", default=True, help="Drop exact duplicate frequency/amplitude/phase rows")
    p.add_argument("--no-deduplicate", dest="deduplicate", action="store_false", help="Keep exact duplicate rows")

    p.add_argument("--nmax", type=int, default=20, help="Maximum primary harmonic order to test")
    p.add_argument("--lmax-secondary", type=int, default=5, help="Maximum secondary harmonic order l f' to test")
    p.add_argument("--lmax-modulation", type=int, default=3, help="Maximum modulation side-order L to test")
    p.add_argument("--lmax-coupling", type=int, default=3, help="Maximum coupling side-order L to test")
    p.add_argument("--lmax-secondary-secondary", type=int, default=3, help="Maximum harmonic order in secondary--secondary quadratic coupling tests")
    p.add_argument("--max-cross-candidates", type=int, default=3, help="Maximum strongest close-frequency candidates used for secondary--secondary coupling tests")

    p.add_argument("--side-window", type=float, default=0.5, help="Search window around each detected k*f0 harmonic for candidate spacings [d^-1]")
    p.add_argument("--min-fb", type=float, default=1e-6, help="Minimum |f'-f0| to consider as beat/modulation frequency")
    p.add_argument("--max-candidates", type=int, default=5, help="Maximum number of close-frequency candidates")
    p.add_argument("--min-candidate-rel-amp", type=float, default=0.01, help="Discard weakly supported automatic fB candidates whose largest discovery peak is below this fraction of the strongest candidate; candidates repeated at >=3 harmonics are retained")
    p.add_argument("--fb-min-harmonic-support", type=int, default=2, help="Minimum number of distinct k*f0 harmonics supporting an automatic spacing when no side peak is detected next to f0")
    p.add_argument("--fb", default=None, help="Manual signed fB candidates, comma/space separated. Example: '0.05,-0.025'")
    p.add_argument("--no-auto-fb", action="store_true", help="Do not automatically search side peaks around detected k*f0 harmonics; use only --fb")
    p.add_argument("--fb-cluster-tol-factor", type=float, default=2.0, help="Cluster fB candidates closer than this*tol")
    p.add_argument("--fb-family-tol-factor", type=float, default=0.5, help="Tolerance factor for grouping fB, fB/2, 2fB, ... candidates into one modulation family")
    p.add_argument("--no-collapse-fb-multiples", action="store_true", help="Do not collapse candidates that are integer multiples of a smaller fB")
    p.add_argument("--fb-collapse-max-multiple", type=int, default=6, help="Maximum integer multiple used when collapsing fB, 2fB, 3fB, ... candidates")

    p.add_argument("--phase-min-points", type=int, default=5, help="Minimum phase-test points before phase scatter can penalize coupling")
    p.add_argument("--phase-scatter-good", type=float, default=0.30, help="Good combination-phase scatter threshold [rad]")
    p.add_argument("--phase-scatter-bad", type=float, default=0.65, help="Bad combination-phase scatter threshold [rad]")
    p.add_argument("--coupling-amp-scatter-good", type=float, default=0.25, help="Good log10 amplitude-ratio scatter threshold [dex]")
    p.add_argument("--coupling-complex-scatter-good", type=float, default=0.35, help="Good normalized complex scatter threshold for quadratic-coupling complex-ratio test")
    p.add_argument("--coupling-complex-scatter-bad", type=float, default=0.75, help="Bad normalized complex scatter threshold that hard-caps physical coupling scores")
    p.add_argument("--coupling-complex-min-points", type=int, default=5, help="Minimum number of complex-ratio points required before the bad-scatter hard cap is applied")
    p.add_argument("--coupling-complex-bad-cap", type=float, default=0.49, help="Hard score cap when the coupling complex-ratio scatter is bad")
    p.add_argument("--coupling-vs-modulation-margin", type=float, default=0.10, help="Deprecated compatibility option; physical selection now uses --model-selection-delta-ic on common-data AICc/BIC")
    p.add_argument("--coupling-fit-comparison-cap", type=float, default=0.49, help="Deprecated compatibility option; use --physical-model-nonwinner-cap")
    p.add_argument("--secondary-harmonic-rel-amp-min", type=float, default=0.20, help="Higher secondary harmonics below this fraction of the l=1 secondary amplitude are treated as insignificant when choosing sinusoidal vs non-sinusoidal coupling")
    p.add_argument("--am-symmetry-limit", type=float, default=0.15, help="Median side-pair asymmetry below this is considered nearly symmetric")
    p.add_argument("--combined-asymmetry-limit", type=float, default=0.20, help="Median side-pair asymmetry above this supports combined/asymmetric modulation")
    p.add_argument("--am-scatter-limit", type=float, default=0.25, help="Scatter threshold for AM/PM normalized side-amplitude tests [dex]")
    p.add_argument("--modfit-min-points", type=int, default=8, help="Minimum number of sideband points for complex AM+FM/PM modulation fitting")
    p.add_argument("--modfit-scatter-good", type=float, default=0.35, help="Good normalized complex residual scatter for AM+FM/PM modulation fit")
    p.add_argument("--modfit-scatter-bad", type=float, default=0.75, help="Bad normalized complex residual scatter for AM+FM/PM modulation fit")
    p.add_argument("--physical-fit-min-points", type=int, default=8, help="Common minimum number of identical complex side peaks required for both modulation and coupling model selection")
    p.add_argument("--common-fit-min-per-group", type=int, default=3, help="Minimum common complex side peaks required in each signed-L group fitted by both physical models")
    p.add_argument("--side-order-envelope-min-tracks", type=int, default=2, help="Minimum complete same-harmonic |L|=1,2,3 amplitude tracks required for the cross-|L| envelope discriminator; at least two distinct harmonics are always required")
    p.add_argument("--side-order-envelope-rise-tolerance", type=float, default=1.15, help="Maximum adjacent higher-side-order amplitude ratio still treated as a non-rising modulation-like envelope")
    p.add_argument("--side-order-envelope-reversal-ratio", type=float, default=1.50, help="Minimum adjacent higher-side-order amplitude rise treated as a strong coupling-like envelope reversal")
    p.add_argument("--side-order-sequence-min-points", type=int, default=4, help="Minimum matched amplitudes required in one signed L=2 or L=3 series before its run along harmonic order n can classify modulation versus coupling")
    p.add_argument("--side-order-sequence-monotonic-fraction", type=float, default=0.75, help="Minimum fraction of observed steps in one signed L=2 or L=3 series that must avoid a significant rise for a declining run to be modulation-like")
    p.add_argument("--physical-fit-scatter-good", type=float, default=0.35, help="Common absolute-fit threshold separating STRONG from LEAN modulation/coupling preferences")
    p.add_argument("--physical-fit-scatter-bad", type=float, default=0.75, help="Common poor-fit scale used symmetrically when ranking LEAN and AMBIG candidates")
    p.add_argument("--model-selection-delta-ic", type=float, default=2.0, help="Minimum absolute delta required from both AICc and BIC for a directional STRONG/LEAN preference; delta is IC_coupling-IC_modulation")
    p.add_argument("--blazhko-modlike-min-coverage", type=float, default=0.5, help="Minimum neutral multiplet-grid coverage for promoting MOD-LEAN to BL-MODLIKE when beating is also globally gated")
    p.add_argument("--blazhko-modlike-min-matches", type=int, default=4, help="Minimum matched multiplet side peaks for promoting MOD-LEAN to BL-MODLIKE")

    p.add_argument("--report-score-min", type=float, default=0.5, help="Only show models with score >= this in Markdown report")
    p.add_argument("--physical-model-nonwinner-cap", type=float, default=0.49, help="Maximum score of non-winning physical modulation/coupling rows after the symmetric AICc/BIC comparison")
    p.add_argument("--coupling-unverified-cap", type=float, default=0.49, help="Cap candidate-specific coupling rows below this score if too few phase points are available")
    p.add_argument("--coupling-grid-only-cap", type=float, default=0.49, help="Deprecated compatibility option; the shared grid is now the neutral MULTIPLET-GRID row")
    p.add_argument("--secondary-secondary-frequency-only-cap", type=float, default=0.49, help="Cap for secondary--secondary coupling terms while only frequency-grid verification is implemented")
    p.add_argument("--show-all-models", action="store_true", help="Show all tested models in Markdown report")
    p.add_argument("--show-grid-only-in-report", action="store_true", help="Show frequency-only secondary-secondary coupling bookkeeping; MULTIPLET-GRID is always neutral and reportable")
    p.add_argument("--linear-beating-max-side-peaks", type=int, default=1, help="Maximum number of side-peak-like frequencies near primary harmonics allowed before linear beating is capped as a global interpretation")
    p.add_argument("--linear-beating-gated-cap", type=float, default=0.49, help="Cap linear-beating rows to this score when multiplet-like side structure is present")
    p.add_argument("--no-linear-beating-global-gate", action="store_true", help="Disable the global side-peak gate for linear-beating rows and restore purely candidate-level scoring")
    p.add_argument("--report-top-extras", type=int, default=30, help="Number of strongest unassigned frequencies to list in report")
    p.add_argument("--report-top-phase", type=int, default=40, help="Number of phase-test rows to list in report")
    p.add_argument("--report-top-f0-candidates", type=int, default=10, help="Number of automatic f0 candidates to list in the Markdown report")
    ns = p.parse_args()
    if getattr(ns, "frequency_only", False):
        ns.ignore_phases = True
    return ns


if __name__ == "__main__":
    run(parse_args())
