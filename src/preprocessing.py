from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CONDITION = "Condition number"
REPLICATE = "Replicate number within condition"
FEATURE_COLUMNS = (
    "Current density (mA/cm^2)", "THF (%)", "LiBF4 concentration (M)",
    "LiTFSI concentration (M)", "BuOH concentration (M)",
    "PrOH concentration (M)", "EtOH concentration (M)",
)
FE_COLUMN = "Total FE (%)"
RATE_COLUMN = "Total rate (nmol/s/cm^2)"
TARGET_COLUMNS = (FE_COLUMN, RATE_COLUMN)
AUDIT_FARADAY_CONSTANT = 96485.33212
INPUT_TOLERANCES = {feature: 1e-10 for feature in FEATURE_COLUMNS}
INPUT_TOLERANCES["LiBF4 concentration (M)"] = 0.0005 + 1e-10
INPUT_TOLERANCES["LiTFSI concentration (M)"] = 0.0005 + 1e-10


@dataclass
class CampaignData:
    """Separate numerical model inputs from identifiers and diagnostic metadata."""

    eligible: pd.DataFrame
    retained: pd.DataFrame
    excluded: pd.DataFrame
    model_rows: pd.DataFrame
    condition_means: pd.DataFrame
    condition_counts: pd.DataFrame
    rate_audit: pd.DataFrame
    input_consistency: pd.DataFrame
    history: pd.DataFrame


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require(table: pd.DataFrame, columns, label: str) -> None:
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")


def _integers(series: pd.Series, label: str) -> pd.Series:
    values = pd.to_numeric(series, errors="raise")
    if not np.isfinite(values).all() or (values < 1).any() or (values != np.floor(values)).any():
        raise ValueError(f"{label} must contain positive integers.")
    return values.astype(int)


def _booleans(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError("IQR excluded must contain explicit True/False values.")
    return mapped.astype(bool)


def remove_outliers_within_conditions(
    data: pd.DataFrame, group_col: str, metric_cols, multiplier: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Original sequential within-condition IQR algorithm, with no input mutation.

    FE is evaluated before recorded rate by the loader. Each subsequent metric's
    quartiles are recomputed from remaining rows (linear interpolation). Values
    on a fence are retained. Groups of one and zero/undefined IQR are skipped.
    The first triggering metric and its value/fences are retained for each drop.
    """
    if not np.isfinite(multiplier) or multiplier < 0:
        raise ValueError("IQR multiplier must be finite and nonnegative.")
    _require(data, [group_col, *metric_cols], "Outlier input")
    kept, removed = [], []
    for _, group in data.groupby(group_col, dropna=False, sort=True):
        working = group.copy()
        for metric in metric_cols:
            if len(working) <= 1:
                continue
            q1 = working[metric].quantile(0.25, interpolation="linear")
            q3 = working[metric].quantile(0.75, interpolation="linear")
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
            valid = (working[metric] >= lower) & (working[metric] <= upper)
            dropped = working.loc[~valid].copy()
            if len(dropped):
                dropped["metric"], dropped["value"] = metric, dropped[metric]
                dropped["lower"], dropped["upper"] = lower, upper
                removed.append(dropped)
            working = working.loc[valid]
        kept.append(working)
    kept_table = pd.concat(kept) if kept else data.iloc[:0].copy()
    removed_table = pd.concat(removed) if removed else data.iloc[:0].copy()
    if not removed:
        for column in ("metric", "value", "lower", "upper"):
            removed_table[column] = pd.Series(index=removed_table.index, dtype=object if column == "metric" else float)
    return kept_table, removed_table


def audit_recorded_rate(data: pd.DataFrame, *, atol: float = 0.01, rtol: float = 0.001) -> pd.DataFrame:
    """Audit Faraday consistency before exclusion, without changing any observation."""
    current = data[FEATURE_COLUMNS[0]].to_numpy(dtype=float)
    fe = data[FE_COLUMN].to_numpy(dtype=float)
    recorded = data[RATE_COLUMN].to_numpy(dtype=float)
    # Preserve the notebook's multiplication order and units.
    factor = np.abs(current) * 1e-3 / (3 * AUDIT_FARADAY_CONSTANT) * 1e9 / 100.0
    calculated = fe * factor
    difference = recorded - calculated
    tolerance = atol + rtol * np.abs(calculated)
    valid = np.isfinite(current) & np.isfinite(fe) & np.isfinite(recorded) & np.isfinite(calculated)
    status = np.full(len(data), "Not assessable", dtype=object)
    status[valid] = "Within tolerance"
    status[valid & (np.abs(difference) > tolerance)] = "Discrepancy"
    relative = np.full(len(data), np.nan)
    nonzero = valid & (calculated != 0)
    relative[nonzero] = 100.0 * difference[nonzero] / np.abs(calculated[nonzero])
    result = data[[CONDITION, REPLICATE, "Exp #"]].copy()
    result["Calculated rate (nmol/s/cm^2)"] = calculated
    result["Recorded minus calculated rate (nmol/s/cm^2)"] = difference
    result["Absolute discrepancy (nmol/s/cm^2)"] = np.abs(difference)
    result["Relative discrepancy (%)"] = relative
    result["Allowed discrepancy (nmol/s/cm^2)"] = tolerance
    result["Audit status"] = status
    return result


def check_condition_inputs(data: pd.DataFrame) -> pd.DataFrame:
    """Check within-condition input spread at original reporting-precision tolerances."""
    records = []
    for identifier, group in data.groupby(CONDITION, sort=True):
        for feature in FEATURE_COLUMNS:
            values = group[feature].to_numpy(dtype=float)
            spread = np.ptp(values)
            if spread > 0:
                records.append({CONDITION: identifier, "Feature": feature,
                                "Minimum": values.min(), "Maximum": values.max(),
                                "Range": spread, "Allowed range": INPUT_TOLERANCES[feature],
                                "Within reporting precision": spread <= INPUT_TOLERANCES[feature]})
    result = pd.DataFrame(records, columns=[CONDITION, "Feature", "Minimum", "Maximum",
                                           "Range", "Allowed range", "Within reporting precision"])
    if not result.empty and not result["Within reporting precision"].all():
        bad = result.loc[~result["Within reporting precision"]]
        raise ValueError("Input differences exceed original tolerances:\n" + bad.to_string(index=False))
    return result


def load_campaign(root: str | Path | None = None, *, outlier_policy: str = "iqr") -> CampaignData:
    """Load existing repository CSVs, verify exclusion flags, and prepare model tables.

    outlier_policy='iqr' reproduces the primary analysis; 'all-successful' keeps all
    published successful campaign rows for sensitivity analysis. IQR flags are
    independently checked in BOTH cases. No validation runs are pooled here.
    """
    if outlier_policy not in {"iqr", "all-successful"}:
        raise ValueError("outlier_policy must be 'iqr' or 'all-successful'.")
    root = repository_root() if root is None else Path(root).resolve()
    directory = root / "data/campaign"
    eligible = pd.read_csv(directory / "campaign_replicate_table.csv")
    history = pd.read_csv(directory / "campaign_history.csv")
    _require(eligible, [CONDITION, REPLICATE, "Acquisition batch", "IQR excluded",
                        *FEATURE_COLUMNS, *TARGET_COLUMNS], "Campaign table")
    _require(history, [CONDITION, "Campaign order", "Acquisition batch", "Selection strategy",
                       "Search-space stage"], "Campaign history")
    for column in (CONDITION, REPLICATE, "Acquisition batch"):
        eligible[column] = _integers(eligible[column], column)
    for column in (CONDITION, "Campaign order", "Acquisition batch"):
        history[column] = _integers(history[column], column)
    if eligible.duplicated([CONDITION, REPLICATE]).any():
        raise ValueError("Duplicate condition/replicate identifiers.")
    if history[CONDITION].duplicated().any() or history["Campaign order"].duplicated().any():
        raise ValueError("Campaign history requires unique conditions and campaign-order values.")
    if set(eligible[CONDITION]) != set(history[CONDITION]):
        raise ValueError("Campaign history and replicate-table condition IDs differ.")
    batches = history.set_index(CONDITION)["Acquisition batch"]
    if not eligible[CONDITION].map(batches).equals(eligible["Acquisition batch"]):
        raise ValueError("Acquisition batch differs between history and replicate table.")
    for column in (*FEATURE_COLUMNS, *TARGET_COLUMNS):
        eligible[column] = pd.to_numeric(eligible[column], errors="raise")
    if not np.isfinite(eligible[list(FEATURE_COLUMNS) + list(TARGET_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite model inputs/outcomes: investigate rather than silently dropping rows.")
    if (eligible[FEATURE_COLUMNS[0]] <= 0).any():
        raise ValueError("Repository campaign table must store positive cathodic current magnitudes.")
    eligible["IQR excluded"] = _booleans(eligible["IQR excluded"])
    # Lexicographic BO-ID grouping reproduces the original notebook's training
    # row order (BO-1, BO-10, BO-100, ...), not numeric condition order.
    eligible["Exp #"] = "BO-" + eligible[CONDITION].astype(str)
    filtered, excluded = remove_outliers_within_conditions(eligible, "Exp #", TARGET_COLUMNS)
    recomputed = eligible.index.isin(excluded.index)
    if not np.array_equal(recomputed, eligible["IQR excluded"].to_numpy()):
        ids = eligible.loc[recomputed != eligible["IQR excluded"].to_numpy(), [CONDITION, REPLICATE]]
        raise ValueError("Stored IQR flags differ from recomputed sequential filtering:\n" + ids.to_string(index=False))
    if outlier_policy == "iqr":
        retained = filtered.copy()
    else:
        retained = pd.concat([group for _, group in eligible.groupby("Exp #", sort=True)]).copy()
    consistency = check_condition_inputs(retained)
    model_rows = retained[[CONDITION, *FEATURE_COLUMNS, *TARGET_COLUMNS]].copy()
    condition_means = model_rows.groupby(CONDITION, as_index=False, sort=True)[
        [*FEATURE_COLUMNS, *TARGET_COLUMNS]
    ].mean()
    counts = eligible.groupby(CONDITION, sort=True).agg(
        **{"Successful campaign n": (REPLICATE, "size"), "IQR exclusions": ("IQR excluded", "sum")}
    )
    counts["Retained campaign n"] = retained.groupby(CONDITION).size().reindex(counts.index, fill_value=0)
    counts = counts.reset_index()
    rate_audit = audit_recorded_rate(eligible)
    rate_audit["Retained after inclusion policy"] = eligible.index.isin(retained.index)
    # Validate the deposited pseudo-EE without replacing it or reconstructing
    # voltage from rounded FE/EE data a second time.
    if {"Average cell voltage (V)", "Pseudo-EE (%)"}.issubset(eligible.columns):
        v = eligible["Average cell voltage (V)"].to_numpy(dtype=float)
        ee = eligible["Pseudo-EE (%)"].to_numpy(dtype=float)
        if not np.isfinite(v).all() or np.any(v <= 0) or not np.isfinite(ee).all():
            raise ValueError("Invalid deposited mean voltage or pseudo-EE.")
        if not np.allclose(eligible[FE_COLUMN].to_numpy() * 1.17 / v, ee, rtol=1e-12, atol=1e-12):
            raise ValueError("Deposited pseudo-EE differs from FE * 1.17 / mean voltage.")
    return CampaignData(eligible, retained, excluded, model_rows, condition_means,
                        counts, rate_audit, consistency, history)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=repository_root())
    parser.add_argument("--outlier-policy", choices=["iqr", "all-successful"], default="iqr")
    parser.add_argument("--output-dir", type=Path, help="Optional CSV export directory, relative to repository root.")
    args = parser.parse_args(argv)
    data = load_campaign(args.repository_root, outlier_policy=args.outlier_policy)
    print(f"Successful non-baseline campaign electrolyses: {len(data.eligible)}")
    print(f"Conditions: {len(data.condition_means)}")
    print(f"Recomputed IQR exclusions: {len(data.excluded)} (stored flags verified)")
    print(f"Retained under {args.outlier_policy}: {len(data.retained)}")
    print("Rate consistency audit (diagnostic only):")
    print(data.rate_audit["Audit status"].value_counts().to_string())
    print("\nReported Pareto-condition counts:")
    print(data.condition_counts.loc[data.condition_counts[CONDITION].isin([82, 86, 89, 95])].to_string(index=False))
    if args.output_dir:
        destination = args.output_dir
        if not destination.is_absolute():
            destination = args.repository_root.resolve() / destination
        destination.mkdir(parents=True, exist_ok=True)
        for name in data.__dataclass_fields__:
            getattr(data, name).to_csv(destination / f"{name}.csv", index=False)
        print(f"\nExported tables to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
