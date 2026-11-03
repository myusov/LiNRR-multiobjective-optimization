from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from .preprocessing import load_campaign, repository_root, CONDITION, FE_COLUMN, RATE_COLUMN
except ImportError:
    from preprocessing import load_campaign, repository_root, CONDITION, FE_COLUMN, RATE_COLUMN

N_BOOTSTRAP = 50_000
BOOTSTRAP_SEED = 12_345
POST_CANDIDATES = ["BO-82", "BO-86", "BO-89", "BO-95"]
pareto_metrics = [FE_COLUMN, RATE_COLUMN]
POST_RATE_FACTORS = {"BO-82": 0.191, "BO-86": 0.5684}


def nondominated_mask(values):
    """Keep rows not strictly dominated; equal objective vectors both survive.

    Every column is maximized. Reject missing/nonfinite objectives rather than
    silently assigning them membership. Empty (0, m) inputs return an empty mask.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("Expected a two-dimensional objective array with at least one column.")
    if not np.isfinite(values).all():
        raise ValueError("Pareto objectives must all be finite.")
    competitor_at_least_as_good = values[None, :, :] >= values[:, None, :]
    competitor_strictly_better = values[None, :, :] > values[:, None, :]
    dominated = (competitor_at_least_as_good.all(axis=2)
                 & competitor_strictly_better.any(axis=2)).any(axis=1)
    return ~dominated


def _bootstrap_arguments(n_bootstrap, chunk_size):
    for name, value in (("n_bootstrap", n_bootstrap), ("chunk_size", chunk_size)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")


def post_campaign_bootstrap(
    data,
    n_bootstrap=N_BOOTSTRAP,
    seed=BOOTSTRAP_SEED,
    chunk_size=250,
):
    """Paired electrolysis bootstrap; no analytical-noise addition."""
    _bootstrap_arguments(n_bootstrap, chunk_size)
    required = ["Exp #", *pareto_metrics]
    work = data[required].copy()
    if work["Exp #"].isna().any() or not np.isfinite(work[pareto_metrics].to_numpy(dtype=float)).all():
        raise ValueError("Missing condition IDs or nonfinite objectives; investigate before resampling.")
    if (work[pareto_metrics].to_numpy(dtype=float) < 0).any():
        raise ValueError("Negative objectives require review; values are not silently clipped.")

    grouped = {
        exp_id: group
        for exp_id, group in work.groupby("Exp #", sort=True)
        if len(group) > 0
    }
    condition_ids = np.asarray(list(grouped))
    n_conditions = len(condition_ids)
    target_idx = np.flatnonzero(np.isin(condition_ids, POST_CANDIDATES))
    if len(target_idx) != len(POST_CANDIDATES):
        raise ValueError("At least one requested condition is missing.")
    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty((n_bootstrap, n_conditions, 2), dtype=float)

    for j, exp_id in enumerate(condition_ids):
        group = grouped[exp_id]
        electrolysis_means = group[pareto_metrics].to_numpy(dtype=float)
        
        n_electrolysis = len(group)

        selected = rng.integers(
            0, n_electrolysis, size=(n_bootstrap, n_electrolysis)
        )
        sampled_means = electrolysis_means[selected]
        
        rng.standard_normal(size=(n_bootstrap, n_electrolysis, 1))
        sampled_values = sampled_means
        bootstrap_means[:, j, :] = sampled_values.mean(axis=1)

    membership_counts = np.zeros(n_conditions, dtype=np.int64)

    for start in range(0, n_bootstrap, chunk_size):
        sample = bootstrap_means[start:start + chunk_size]
        competitor_at_least_as_good = (
            sample[:, None, :, :] >= sample[:, target_idx, None, :]
        )
        competitor_strictly_better = (
            sample[:, None, :, :] > sample[:, target_idx, None, :]
        )
        dominated = (
            competitor_at_least_as_good.all(axis=3)
            & competitor_strictly_better.any(axis=3)
        ).any(axis=2)
        membership_counts[target_idx] += (~dominated).sum(axis=0)

    summary = work.groupby("Exp #")[pareto_metrics].agg(
        ["count", "mean", "std"]
    ).reindex(condition_ids)

    results = pd.DataFrame({
        "Exp #": condition_ids,
        "n independent electrolysis": summary[(pareto_metrics[0], "count")].to_numpy(dtype=int),
                "FE mean (%)": summary[(pareto_metrics[0], "mean")].to_numpy(),
        "FE SD (%)": summary[(pareto_metrics[0], "std")].to_numpy(),
        "Rate mean (nmol/s/cm^2)": summary[(pareto_metrics[1], "mean")].to_numpy(),
        "Rate SD (nmol/s/cm^2)": summary[(pareto_metrics[1], "std")].to_numpy(),
        "Bootstrap Pareto frequency": np.where(
            np.isin(np.arange(n_conditions), target_idx), membership_counts / n_bootstrap, np.nan),
    })
    results["Mean-based Pareto"] = nondominated_mask(
        results[["FE mean (%)", "Rate mean (nmol/s/cm^2)"]].to_numpy()
    )
    return results.sort_values(
        "Bootstrap Pareto frequency", ascending=False
    ).reset_index(drop=True)


def campaign_objectives(data):
    """Adapt repository campaign rows to notebook IDs without reordering rows."""
    result = data[[CONDITION, *pareto_metrics]].copy()
    ids = result[CONDITION].to_numpy(dtype=float)
    if not np.isfinite(ids).all() or (ids < 1).any() or (ids != np.floor(ids)).any():
        raise ValueError("Condition numbers must be finite positive integers.")
    result.insert(0, "Exp #", [f"BO-{int(value)}" for value in ids])
    return result.drop(columns=CONDITION)


def load_post_validation(root=None):
    root = Path(root).resolve() if root is not None else repository_root()
    path = root / "data/validation/fraction_measurements.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required validation fraction table not found: {path}")
    fractions = pd.read_csv(path)
    needed = {"Experiment ID", "Fraction", "FE (%)"}
    if not needed.issubset(fractions.columns):
        raise ValueError(f"{path}: missing columns {sorted(needed-set(fractions.columns))}")
    rows = []
    for condition, factor in POST_RATE_FACTORS.items():
        number = int(condition.split("-")[-1])
        for run in range(1, 4):
            run_id = f"SUP-BO{number}-COMP-R{run:02d}"
            selected = fractions.loc[fractions["Experiment ID"].eq(run_id)]
            if len(selected) != 3 or set(selected["Fraction"]) != {"electrolyte", "deposits", "gas"}:
                raise ValueError(f"Expected exactly three unique fraction measurements for {run_id}.")
            values = selected.set_index("Fraction").loc[
                ["electrolyte", "deposits", "gas"], "FE (%)"
            ].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid fraction FE for {run_id}.")
            fe = values.sum()
            rows.append({"Exp #": condition,
                         "Validation run ID": f"{condition}-post-{run}",
                         "Source": "Post-campaign compartment validation",
                         FE_COLUMN: fe, RATE_COLUMN: fe * factor})
    return pd.DataFrame(rows)


def observed_front(data):
    """Return per-condition means, sample SDs, counts and nominal membership."""
    if data.empty:
        raise ValueError("No experimental rows supplied.")
    if data["Exp #"].isna().any() or not np.isfinite(data[pareto_metrics].to_numpy(float)).all():
        raise ValueError("Missing IDs or nonfinite objectives.")
    groups = data.groupby("Exp #", sort=True)[pareto_metrics].agg(["count", "mean", "std"])
    result = pd.DataFrame({
        "Exp #": groups.index,
        "n independent electrolysis": groups[(FE_COLUMN, "count")].to_numpy(),
        "FE mean (%)": groups[(FE_COLUMN, "mean")].to_numpy(),
        "FE SD (%)": groups[(FE_COLUMN, "std")].to_numpy(),
        "Rate mean (nmol/s/cm^2)": groups[(RATE_COLUMN, "mean")].to_numpy(),
        "Rate SD (nmol/s/cm^2)": groups[(RATE_COLUMN, "std")].to_numpy(),
    })
    result["Mean-based Pareto"] = nondominated_mask(result[["FE mean (%)", "Rate mean (nmol/s/cm^2)"]])
    return result


def predicted_front(predictions, fe_column="fe_mean", rate_column="rate_mean"):
    """Return a copy with predicted nondominance for a caller-supplied candidate set."""
    result = pd.DataFrame(predictions).copy()
    result["Predicted Pareto"] = nondominated_mask(result[[fe_column, rate_column]])
    return result


def run_pareto_analysis(root=None, *, n_bootstrap=N_BOOTSTRAP,
                        seed=BOOTSTRAP_SEED, chunk_size=250, output_dir=None):
    campaign = load_campaign(root)
    validation = load_post_validation(root)
    scenarios, counts = [], []
    for policy, raw in (("IQR-filtered campaign", campaign.retained),
                        ("All successful campaign", campaign.eligible)):
        objectives = campaign_objectives(raw)
        for include_validation in (False, True):
            working = objectives.copy()
            if include_validation:
                working = pd.concat([working, validation[["Exp #", *pareto_metrics]]], ignore_index=True)
            name = policy + (" + validation" if include_validation else "")
            result = post_campaign_bootstrap(working, n_bootstrap=n_bootstrap,
                       seed=seed, chunk_size=chunk_size)
            result["Dataset"] = name
            scenarios.append(result)
            for condition in POST_CANDIDATES:
                counts.append({"Dataset": name, "Exp #": condition,
                               "Campaign n": int(objectives["Exp #"].eq(condition).sum()),
                               "Added validation n": int(validation["Exp #"].eq(condition).sum()) if include_validation else 0})
    all_results = pd.concat(scenarios, ignore_index=True)
    candidates = all_results.loc[all_results["Exp #"].isin(POST_CANDIDATES)].copy()
    frequencies = candidates.pivot(index="Exp #", columns="Dataset",
                 values="Bootstrap Pareto frequency").reindex(POST_CANDIDATES) * 100
    tables = {"post_validation_runs": validation,
              "membership_all_conditions": all_results,
              "membership_candidates": candidates,
              "replicate_counts": pd.DataFrame(counts),
              "membership_percent": frequencies.reset_index()}
    if output_dir is not None:
        destination = Path(output_dir)
        if not destination.is_absolute():
            destination = (Path(root) if root is not None else repository_root()) / destination
        destination.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(destination / f"{name}.csv", index=False)
    return tables


def plot_pareto_sensitivity(tables, root=None, *, use_tex=None):
    import matplotlib.pyplot as plt
    root = Path(root).resolve() if root is not None else repository_root()
    style = root / "styles/manuscript.mplstyle"
    if not style.is_file():
        raise FileNotFoundError(f"Required manuscript style not found: {style}")
    with plt.style.context(str(style)):
        overrides = {} if use_tex is None else {"text.usetex": bool(use_tex)}
        with plt.rc_context(overrides):
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5*2/3))
            summary = tables["membership_candidates"]
            names = list(summary["Dataset"].drop_duplicates())
            for ax, name in zip(axes.flat, names):
                data = summary.loc[summary["Dataset"].eq(name)].set_index("Exp #").loc[POST_CANDIDATES]
                ax.bar(range(4), 100*data["Bootstrap Pareto frequency"], color=colors[0])
                ax.set_xticks(range(4), [x.replace("BO-", "") for x in POST_CANDIDATES])
                ax.set_xlabel("Condition number")
                ax.set_ylim(0, 105)
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.set_title(name.replace(" campaign", "").replace(" + validation", " + validation runs"))
                ax.set_ylabel(r"Bootstrap membership (\%)" if plt.rcParams["text.usetex"] else "Bootstrap membership (%)")
            fig.subplots_adjust(left=.10, right=.98, bottom=.11, top=.93, wspace=.30, hspace=.57)
    return fig, axes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--output-dir", type=Path, help="Optional CSV destination; relative paths use repository root.")
    parser.add_argument("--plot", action="store_true", help="Display a manuscript-style summary; do not save figures.")
    parser.add_argument("--no-tex", action="store_true", help="Explicitly disable style's LaTeX renderer for plotting.")
    args = parser.parse_args(argv)
    tables = run_pareto_analysis(args.root, n_bootstrap=args.n_bootstrap,
                 seed=args.seed, output_dir=args.output_dir)
    print("Paired-electrolysis bootstrap membership (%):")
    print(tables["membership_percent"].to_string(index=False))
    print("\nAdded-run rates preserve notebook approximate factors: #82=0.191, #86=0.5684.")
    if args.plot:
        plot_pareto_sensitivity(tables, args.root, use_tex=False if args.no_tex else None)
        import matplotlib.pyplot as plt
        plt.show()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
