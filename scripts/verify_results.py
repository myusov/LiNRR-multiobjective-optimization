"""Compare generated manuscript tables with fixed numerical reference results."""
import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "pareto": "pareto",
    "model-validation": "validation",
    "interpretation": "interpretation",
    "extended-operation": "extended_operation",
    "transport": "transport",
}


def verify(outputs, references, groups):
    """Return the number of exact matches and a list of actionable failures."""
    matched = 0
    failures = []
    for group in groups:
        folder = references / GROUPS[group]
        expected = sorted(folder.glob("*.csv"))
        if not expected:
            failures.append(f"{group}: no reference CSVs in {folder}")
            continue
        for reference in expected:
            generated = outputs / group / reference.name
            try:
                actual = pd.read_csv(generated, float_precision="round_trip")
                baseline = pd.read_csv(reference, float_precision="round_trip")
                pd.testing.assert_frame_equal(actual, baseline, check_exact=True)
            except (AssertionError, OSError, ValueError) as error:
                failures.append(f"{group}/{reference.name}: {error}")
            else:
                matched += 1
    return matched, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/manuscript")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "tests/reference_results")
    parser.add_argument("--only", nargs="+", choices=GROUPS, default=list(GROUPS))
    args = parser.parse_args(argv)
    matched, failures = verify(args.output_dir, args.reference_dir, dict.fromkeys(args.only))
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    print(f"{matched} exact table matches, {len(failures)} failures.")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
