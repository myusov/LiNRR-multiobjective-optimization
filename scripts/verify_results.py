"""Compare generated manuscript tables with fixed numerical reference results."""
import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "pareto": "pareto",
    "model-validation": "validation",
    "interpretation": "interpretation",
    "extended-operation": "extended_operation",
    "transport": "transport",
}


DEFAULT_RTOL = 1e-8
DEFAULT_ATOL = 1e-10
MODEL_VALIDATION_ATOL = 1e-4


def absolute_tolerance(group, override):
    if override is not None:
        return override
    return MODEL_VALIDATION_ATOL if group == "model-validation" else DEFAULT_ATOL


# Some discrete columns are read as floats when the CSV contains missing values.
EXACT_FLOAT_COLUMNS = {
    "Condition number", "Direction sign", "Outside-range features",
    "Retained replicate count", "Stage constraint violations",
}


def compare_tables(actual, baseline, *, rtol, atol):
    """Return whether values match exactly and any concise column diagnostics."""
    if actual.shape != baseline.shape:
        return False, [f"shape {actual.shape}, expected {baseline.shape}"]
    if not actual.columns.equals(baseline.columns):
        return False, ["column names or order differ"]
    exact = True
    failures = []
    for column in baseline:
        left, right = actual[column], baseline[column]
        if left.dtype != right.dtype:
            exact = False
            failures.append(f"{column}: dtype {left.dtype}, expected {right.dtype}")
            continue
        if left.equals(right):
            continue
        exact = False
        a, b = left.to_numpy(), right.to_numpy()
        floating = pd.api.types.is_float_dtype(right)
        if floating:
            relative, absolute = ((0, 0) if column in EXACT_FLOAT_COLUMNS
                                  else (rtol, atol))
            equal = np.isclose(a, b, rtol=relative, atol=absolute, equal_nan=True)
        else:
            equal = (left.eq(right).fillna(False) | (left.isna() & right.isna())).to_numpy()
        offending = np.flatnonzero(~equal)
        if not len(offending):
            continue
        row = offending[0]
        detail = (f"{column}: {len(offending)} differences; first at data row {row + 1}: "
                  f"got {a[row]!r}, expected {b[row]!r}")
        if floating:
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any():
                maximum = np.max(np.abs(a[finite] - b[finite]))
                detail += f"; max absolute difference across finite pairs {maximum:.10g}"
        failures.append(detail)
    return exact, failures


def verify(outputs, references, groups, *, rtol=DEFAULT_RTOL, atol=None):
    """Count exact and tolerance-only matches and report failures by table."""
    exact_matches = tolerance_matches = 0
    failures = []
    for group in groups:
        folder = references / GROUPS[group]
        expected = sorted(folder.glob("*.csv"))
        if not expected:
            failures.append(f"{group}: no reference CSVs in {folder}")
            continue
        for reference in expected:
            name = f"{group}/{reference.name}"
            try:
                actual = pd.read_csv(outputs / group / reference.name, float_precision="round_trip")
                baseline = pd.read_csv(reference, float_precision="round_trip")
                exact, differences = compare_tables(
                    actual, baseline, rtol=rtol, atol=absolute_tolerance(group, atol)
                )
            except (OSError, ValueError) as error:
                failures.append(f"{name}: {error}")
                continue
            if differences:
                failures.append(f"{name}: " + "\n  ".join(differences))
            elif exact:
                exact_matches += 1
            else:
                tolerance_matches += 1
    return exact_matches, tolerance_matches, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/manuscript")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "tests/reference_results")
    parser.add_argument("--only", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--exact", action="store_true", help="Require exact numeric values.")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL, help="Relative tolerance (default: 1e-8).")
    parser.add_argument("--atol", type=float, help="Override absolute tolerance for all selected groups (defaults: 1e-4 for model validation, 1e-10 otherwise).")
    args = parser.parse_args(argv)
    if not all(value is None or (math.isfinite(value) and value >= 0)
               for value in (args.rtol, args.atol)):
        parser.error("Tolerances must be finite and nonnegative.")
    rtol, atol = (0, 0) if args.exact else (args.rtol, args.atol)
    groups = dict.fromkeys(args.only)
    print(f"Floating-point relative tolerance: {rtol:g}.")
    for group in groups:
        print(f"  {group}: atol={absolute_tolerance(group, atol):g}")
    exact, tolerated, failures = verify(
        args.output_dir, args.reference_dir, groups, rtol=rtol, atol=atol
    )
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    print(f"{exact} exact table matches, {tolerated} tolerance-only matches, {len(failures)} failures.")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
