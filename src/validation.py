from __future__ import annotations
import argparse
import builtins
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize as BoNormalize
from botorch.models.transforms.outcome import Standardize as BoStandardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
try:
    from .preprocessing import load_campaign, CONDITION, INPUT_TOLERANCES
    from .models import (FULL_FEATURE_COLS, FE_TARGET_COL, RATE_TARGET_COL,
                         CURRENT_DENSITY_COL, fit_fe_gp, predict_fe_rate,
                         calculate_nh3_rate, _feature_array, tkwargs)
except ImportError:
    from preprocessing import load_campaign, CONDITION, INPUT_TOLERANCES
    from models import (FULL_FEATURE_COLS, FE_TARGET_COL, RATE_TARGET_COL,
                        CURRENT_DENSITY_COL, fit_fe_gp, predict_fe_rate,
                        calculate_nh3_rate, _feature_array, tkwargs)


def run_validation(root=None, *, output_dir=None, verbose=True):
    """Run all validation schemes; return named pandas tables.

    Requires the campaign replicate table and history used by load_campaign.
    Uses retained campaign data only, not subsequent validation electrolyses.
    The campaign's 112 conditions define 28 nominal four-condition batches.
    No files are written unless output_dir is explicitly supplied.
    Seeds and five-fold assignments deliberately match the published analysis.
    """
    campaign = load_campaign(root)
    model_df = campaign.model_rows.copy()
    model_df_unique = campaign.condition_means.copy()
    condition_col = CONDITION
    input_atol = INPUT_TOLERANCES
    print = builtins.print if verbose else lambda *args, **kwargs: None
    # ===============================================================
    # Validation setup: shared data, acquisition batches, and stages
    # No model fitting; existing model and plotting objects are unchanged.
    # ===============================================================

    VALIDATION_SEED = 12345
    VALIDATION_MIN_TRAIN_BATCHES = 4



    # Independent copies prevent validation metadata from entering other plots.
    validation_df = model_df[
        [condition_col]
        + list(FULL_FEATURE_COLS)
        + [FE_TARGET_COL, RATE_TARGET_COL]
    ].copy()

    validation_df.insert(
        0,
        "CSV row",
        validation_df.index.to_numpy() + 2,
    )
    validation_df = validation_df.reset_index(drop=True)

    validation_condition_values = validation_df[
        condition_col
    ].to_numpy(dtype=float)

    if not (
        np.isfinite(validation_condition_values).all()
        and np.equal(
            validation_condition_values,
            np.floor(validation_condition_values),
        ).all()
    ):
        raise ValueError("Validation requires finite integer condition IDs.")

    validation_df[condition_col] = (
        validation_condition_values.astype(int)
    )

    if not validation_df[condition_col].between(1, 112).all():
        raise ValueError(
            "Found condition IDs outside 1–112. Update the campaign map "
            "before running validation."
        )

    # Four consecutive condition IDs per acquisition batch.
    # This describes the nominal acquisition order, not replicate execution dates.
    validation_df["Acquisition batch"] = (
        (validation_df[condition_col] - 1) // 4 + 1
    )
    validation_df["Search-space stage"] = np.where(
        validation_df[condition_col] <= 64,
        "Stage 1: conditions 1–64",
        "Stage 2: conditions 65–112",
    )

    # ---------- group matching inputs to prevent train/test leakage ----------
    # All rows of a condition belong to the same group.
    # Different condition IDs are joined if any recorded input rows match
    # within the documented preprocessing tolerances.
    validation_ids = np.sort(
        validation_df[condition_col].unique()
    )
    validation_id_position = {
        condition_id: position
        for position, condition_id in enumerate(validation_ids)
    }

    validation_parent = np.arange(len(validation_ids))


    def validation_find_root(position):
        while validation_parent[position] != position:
            validation_parent[position] = validation_parent[
                validation_parent[position]
            ]
            position = validation_parent[position]
        return position


    def validation_join_groups(first, second):
        first_root = validation_find_root(first)
        second_root = validation_find_root(second)

        if first_root != second_root:
            validation_parent[second_root] = first_root


    validation_inputs = validation_df[
        FULL_FEATURE_COLS
    ].to_numpy(dtype=float)

    validation_tolerances = np.array([
        input_atol[feature]
        for feature in FULL_FEATURE_COLS
    ])

    validation_row_positions = (
        validation_df[condition_col]
        .map(validation_id_position)
        .to_numpy(dtype=int)
    )

    for first in range(len(validation_df)):
        matches = np.flatnonzero(
            np.all(
                np.abs(
                    validation_inputs[first + 1:]
                    - validation_inputs[first]
                ) <= validation_tolerances,
                axis=1,
            )
        ) + first + 1

        for second in matches:
            validation_join_groups(
                validation_row_positions[first],
                validation_row_positions[second],
            )

    validation_df["Input group"] = [
        int(validation_ids[validation_find_root(position)])
        for position in validation_row_positions
    ]

    # ---------- condition-level reference observations ----------
    validation_conditions = (
        validation_df.groupby(condition_col, as_index=False)
        .agg({
            **{feature: "mean" for feature in FULL_FEATURE_COLS},
            FE_TARGET_COL: "mean",
            RATE_TARGET_COL: "mean",
            "Acquisition batch": "first",
            "Search-space stage": "first",
            "Input group": "first",
        })
        .sort_values(condition_col)
        .reset_index(drop=True)
    )

    validation_conditions["Retained replicate count"] = (
        validation_conditions[condition_col]
        .map(validation_df.groupby(condition_col).size())
        .to_numpy()
    )

    # ---------- observed Pareto membership ----------
    # This identifies observed nondominated condition means.
    validation_observed_y = validation_conditions[
        [FE_TARGET_COL, RATE_TARGET_COL]
    ].to_numpy(dtype=float)

    validation_observed_pareto_mask = np.ones(
        len(validation_conditions), dtype=bool
    )

    for i, point in enumerate(validation_observed_y):
        dominated_by_another = (
            np.all(validation_observed_y >= point, axis=1)
            & np.any(validation_observed_y > point, axis=1)
        )
        validation_observed_pareto_mask[i] = (
            not dominated_by_another.any()
        )

    validation_conditions["Observed Pareto condition"] = (
        validation_observed_pareto_mask
    )

    validation_pareto_ids = (
        validation_conditions.loc[
            validation_observed_pareto_mask,
            condition_col,
        ].tolist()
    )

    validation_df["Observed Pareto condition"] = (
        validation_df[condition_col].isin(validation_pareto_ids)
    )

    # ---------- acquisition-batch audit ----------
    validation_batch_summary = (
        validation_df.groupby(
            ["Acquisition batch", "Search-space stage"],
            as_index=False,
        )
        .agg(
            First_condition=(condition_col, "min"),
            Last_condition=(condition_col, "max"),
            N_conditions=(condition_col, "nunique"),
            N_measurements=(condition_col, "size"),
        )
    )

    validation_group_summary = (
        validation_conditions.groupby("Input group", as_index=False)
        .agg(
            N_condition_IDs=(condition_col, "size"),
            First_batch=("Acquisition batch", "min"),
            Last_batch=("Acquisition batch", "max"),
        )
    )

    # ---------- export setup information ----------


    print(
        f"Validation dataset: {len(validation_df)} measurements, "
        f"{len(validation_conditions)} conditions, "
        f"{validation_df['Input group'].nunique()} input groups."
    )
    print(
        "Observed Pareto condition IDs:",
        validation_pareto_ids,
    )
    print(
        "Initial training window:",
        VALIDATION_MIN_TRAIN_BATCHES,
        "acquisition batches.",
    )

    print("\nAcquisition-batch map:")

    print("\nInput groups containing multiple condition IDs:")

    # ===============================================================
    # Expanding-window next-batch validation
    #
    # Train on earlier acquisition batches; predict the next batch.
    # FE-only models; rate derived using held-out current density.
    # Evaluation unit: condition mean.
    #
    # Conditional on the existing filtered validation dataset.
    # ===============================================================




    # ---------- reusable condition-level scoring ----------
    def validation_score_conditions(scored_measurements):
        """Average observations and held-out predictions within each condition."""
        result = (
            scored_measurements.groupby(condition_col, as_index=False)
            .agg({
                "Acquisition batch": "first",
                "Search-space stage": "first",
                "Input group": "first",
                "Observed Pareto condition": "first",
                FE_TARGET_COL: "mean",
                RATE_TARGET_COL: "mean",
                "Held-out predicted FE (%)": "mean",
                "Held-out derived rate (nmol/s/cm^2)": "mean",
                "Outside training marginal ranges": "any",
                "Training measurements": "first",
                "Training conditions": "first",
                "Training through batch": "first",
            })
            .sort_values(condition_col)
            .reset_index(drop=True)
        )

        result["Held-out replicate count"] = (
            result[condition_col]
            .map(scored_measurements.groupby(condition_col).size())
            .to_numpy()
        )

        result["FE prediction minus observation (percentage points)"] = (
            result["Held-out predicted FE (%)"] - result[FE_TARGET_COL]
        )
        result["Rate prediction minus observation (nmol/s/cm^2)"] = (
            result["Held-out derived rate (nmol/s/cm^2)"]
            - result[RATE_TARGET_COL]
        )

        return result


    def validation_regression_metrics(condition_predictions, label):
        """Condition-weighted FE and derived-rate prediction metrics."""
        rows = []

        for response, observed_col, predicted_col, units in [
            (
                "FE",
                FE_TARGET_COL,
                "Held-out predicted FE (%)",
                "percentage points",
            ),
            (
                "Derived rate",
                RATE_TARGET_COL,
                "Held-out derived rate (nmol/s/cm^2)",
                "nmol s^-1 cm^-2",
            ),
        ]:
            observed = condition_predictions[
                observed_col
            ].to_numpy(dtype=float)
            predicted = condition_predictions[
                predicted_col
            ].to_numpy(dtype=float)

            if len(observed) == 0:
                continue

            if not (
                np.isfinite(observed).all()
                and np.isfinite(predicted).all()
            ):
                raise ValueError(f"{label}: nonfinite scoring values.")

            residual = predicted - observed
            ss_total = np.sum((observed - observed.mean())**2)

            rows.append({
                "Validation": label,
                "Response": response,
                "N conditions": len(observed),
                "RMSE": np.sqrt(np.mean(residual**2)),
                "MAE": np.mean(np.abs(residual)),
                "Mean signed error": np.mean(residual),
                "R2": (
                    1.0 - np.sum(residual**2) / ss_total
                    if len(observed) >= 2 and ss_total > 0
                    else np.nan
                ),
                "Error units": units,
            })

        return pd.DataFrame(rows)


    # ---------- forward splits ----------
    forward_measurement_parts = []
    forward_split_records = []

    forward_batches = np.sort(
        validation_df["Acquisition batch"].unique()
    )

    for batch_position in range(
        VALIDATION_MIN_TRAIN_BATCHES,
        len(forward_batches),
    ):
        test_batch = int(forward_batches[batch_position])

        test_df = validation_df.loc[
            validation_df["Acquisition batch"] == test_batch
        ].copy()

        earlier_df = validation_df.loc[
            validation_df["Acquisition batch"] < test_batch
        ].copy()

        # Prevent matching inputs under different condition IDs from
        # appearing in both training and the held-out batch.
        heldout_groups = test_df["Input group"].unique()
        overlap_mask = earlier_df["Input group"].isin(heldout_groups)

        train_df = earlier_df.loc[~overlap_mask].copy()

        if train_df.empty or test_df.empty:
            raise ValueError(
                f"Batch {test_batch}: empty training or held-out dataset."
            )

        if (
            train_df["Acquisition batch"].max()
            >= test_df["Acquisition batch"].min()
        ):
            raise RuntimeError("Future observations entered a training split.")

        if set(train_df["Input group"]) & set(test_df["Input group"]):
            raise RuntimeError("Matching inputs appear in both train and test.")

        print(
            f"Predicting batch {test_batch}/{int(forward_batches[-1])}: "
            f"train on {train_df[condition_col].nunique()} earlier conditions "
            f"({len(train_df)} measurements); "
            f"hold out {test_df[condition_col].nunique()} conditions",
            flush=True,
        )

        # Temporary model; never overwrite fe_gp_full.
        validation_fe_gp = fit_fe_gp(
            train_df,
            train_df[FE_TARGET_COL],
            FULL_FEATURE_COLS,
            seed=VALIDATION_SEED + test_batch,
        )

        predictions = predict_fe_rate(
            validation_fe_gp,
            test_df[FULL_FEATURE_COLS],
        )

        scored = test_df.copy()
        scored["Held-out predicted FE (%)"] = predictions["fe_mean"]
        scored["Held-out derived rate (nmol/s/cm^2)"] = (
            predictions["rate_mean"]
        )

        train_inputs = train_df[
            FULL_FEATURE_COLS
        ].to_numpy(dtype=float)
        test_inputs = test_df[
            FULL_FEATURE_COLS
        ].to_numpy(dtype=float)

        train_min = train_inputs.min(axis=0)
        train_max = train_inputs.max(axis=0)

        outside_by_feature = (
            (test_inputs < train_min - validation_tolerances)
            | (test_inputs > train_max + validation_tolerances)
        )

        scored["Outside training marginal ranges"] = (
            outside_by_feature.any(axis=1)
        )
        scored["Outside-range features"] = [
            "; ".join(
                feature
                for feature, outside in zip(FULL_FEATURE_COLS, row)
                if outside
            )
            for row in outside_by_feature
        ]

        scored["Training measurements"] = len(train_df)
        scored["Training conditions"] = train_df[condition_col].nunique()
        scored["Training through batch"] = int(
            train_df["Acquisition batch"].max()
        )

        forward_measurement_parts.append(scored)

        forward_split_records.append({
            "Held-out batch": test_batch,
            "Training through batch": int(
                train_df["Acquisition batch"].max()
            ),
            "Training measurements": len(train_df),
            "Training conditions": train_df[condition_col].nunique(),
            "Held-out measurements": len(test_df),
            "Held-out conditions": test_df[condition_col].nunique(),
            "Earlier measurements excluded for matching inputs": int(
                overlap_mask.sum()
            ),
            "Training condition IDs": "; ".join(
                map(str, sorted(train_df[condition_col].unique()))
            ),
            "Held-out condition IDs": "; ".join(
                map(str, sorted(test_df[condition_col].unique()))
            ),
        })

        del validation_fe_gp


    # ---------- combine and check predictions ----------
    forward_measurement_predictions = pd.concat(
        forward_measurement_parts,
        ignore_index=True,
    )

    if forward_measurement_predictions["CSV row"].duplicated().any():
        raise RuntimeError("A measurement was scored in multiple forward splits.")

    forward_prediction_columns = [
        "Held-out predicted FE (%)",
        "Held-out derived rate (nmol/s/cm^2)",
    ]
    if not np.isfinite(
        forward_measurement_predictions[
            forward_prediction_columns
        ].to_numpy()
    ).all():
        raise RuntimeError("Nonfinite forward predictions.")

    forward_condition_predictions = validation_score_conditions(
        forward_measurement_predictions
    )
    forward_split_manifest = pd.DataFrame(forward_split_records)


    # ---------- overall, stage-specific, and batch-specific metrics ----------
    forward_overall_metrics = validation_regression_metrics(
        forward_condition_predictions,
        "Expanding-window next-batch validation",
    )

    forward_stage_metrics = pd.concat([
        validation_regression_metrics(
            subset,
            f"Next-batch validation: {stage}",
        )
        for stage, subset in forward_condition_predictions.groupby(
            "Search-space stage", sort=False
        )
    ], ignore_index=True)

    forward_batch_metric_parts = []

    for batch, subset in forward_condition_predictions.groupby(
        "Acquisition batch", sort=True
    ):
        batch_metrics = validation_regression_metrics(
            subset,
            f"Next-batch validation: batch {batch}",
        )
        batch_metrics["Held-out batch"] = int(batch)
        forward_batch_metric_parts.append(batch_metrics)

    forward_batch_metrics = pd.concat(
        forward_batch_metric_parts,
        ignore_index=True,
    )


    # ---------- export ----------



    # ---------- report ----------
    print("\nOverall forward-validation performance:")

    print("\nPerformance by search-space stage:")

    print("\nFirst batch after the search-space change:")

    print("\nForward predictions at observed Pareto conditions:")

    print(
        "\nCompleted:",
        len(forward_split_manifest),
        "forward splits;",
        len(forward_condition_predictions),
        "held-out condition predictions.",
    )

    # ===============================================================
    # Fixed-origin search-space-change stress test
    #
    # Train: conditions 1–64
    # Test:  conditions 65–80
    # No model updating within the held-out block.
    #
    # FE-only surrogate; rate derived from predicted FE and input current.
    # ===============================================================



    TRANSITION_TRAIN_THROUGH = 64
    TRANSITION_TEST_START = 65
    TRANSITION_TEST_END = 80

    transition_test = validation_df.loc[
        validation_df[condition_col].between(
            TRANSITION_TEST_START,
            TRANSITION_TEST_END,
        )
    ].copy()

    transition_earlier = validation_df.loc[
        validation_df[condition_col] <= TRANSITION_TRAIN_THROUGH
    ].copy()

    # Keep matching input groups out of both sides of the split.
    transition_test_groups = transition_test["Input group"].unique()
    transition_overlap = transition_earlier["Input group"].isin(
        transition_test_groups
    )

    transition_train = transition_earlier.loc[
        ~transition_overlap
    ].copy()

    if transition_train.empty or transition_test.empty:
        raise ValueError("The transition split has an empty dataset.")

    if set(transition_train["Input group"]) & set(
        transition_test["Input group"]
    ):
        raise RuntimeError("Matching inputs appear in training and test data.")

    print(
        "Search-space-change test:\n"
        f"  Training: {transition_train[condition_col].nunique()} conditions, "
        f"{len(transition_train)} measurements\n"
        f"  Held out: {transition_test[condition_col].nunique()} conditions, "
        f"{len(transition_test)} measurements\n"
        f"  Earlier measurements excluded for matching inputs: "
        f"{int(transition_overlap.sum())}",
        flush=True,
    )

    # Temporary model; fe_gp_full remains unchanged.
    transition_fe_gp = fit_fe_gp(
        transition_train,
        transition_train[FE_TARGET_COL],
        FULL_FEATURE_COLS,
        seed=VALIDATION_SEED + 1000,
    )

    transition_predictions = predict_fe_rate(
        transition_fe_gp,
        transition_test[FULL_FEATURE_COLS],
    )

    transition_measurement_predictions = transition_test.copy()

    transition_measurement_predictions[
        "Held-out predicted FE (%)"
    ] = transition_predictions["fe_mean"]

    transition_measurement_predictions[
        "Held-out derived rate (nmol/s/cm^2)"
    ] = transition_predictions["rate_mean"]


    # ---------- training marginal-range diagnostics ----------
    transition_train_inputs = transition_train[
        FULL_FEATURE_COLS
    ].to_numpy(dtype=float)

    transition_test_inputs = transition_test[
        FULL_FEATURE_COLS
    ].to_numpy(dtype=float)

    transition_train_min = transition_train_inputs.min(axis=0)
    transition_train_max = transition_train_inputs.max(axis=0)

    transition_outside_by_feature = (
        (
            transition_test_inputs
            < transition_train_min - validation_tolerances
        )
        | (
            transition_test_inputs
            > transition_train_max + validation_tolerances
        )
    )

    transition_measurement_predictions[
        "Outside training marginal ranges"
    ] = transition_outside_by_feature.any(axis=1)

    transition_measurement_predictions["Outside-range features"] = [
        "; ".join(
            feature
            for feature, outside in zip(FULL_FEATURE_COLS, row)
            if outside
        )
        for row in transition_outside_by_feature
    ]

    transition_measurement_predictions["Training measurements"] = (
        len(transition_train)
    )
    transition_measurement_predictions["Training conditions"] = (
        transition_train[condition_col].nunique()
    )
    transition_measurement_predictions["Training through batch"] = int(
        transition_train["Acquisition batch"].max()
    )

    if not np.isfinite(
        transition_measurement_predictions[
            [
                "Held-out predicted FE (%)",
                "Held-out derived rate (nmol/s/cm^2)",
            ]
        ].to_numpy()
    ).all():
        raise RuntimeError("Nonfinite transition-test predictions.")


    # ---------- condition-level results ----------
    transition_condition_predictions = validation_score_conditions(
        transition_measurement_predictions
    )

    # Preserve the names of outside-range features at condition level.
    transition_feature_names = (
        transition_measurement_predictions.groupby(condition_col)[
            "Outside-range features"
        ]
        .agg(
            lambda entries: "; ".join(sorted({
                feature
                for entry in entries
                for feature in entry.split("; ")
                if feature
            }))
        )
    )

    transition_condition_predictions["Outside-range features"] = (
        transition_condition_predictions[condition_col]
        .map(transition_feature_names)
    )

    transition_metrics = validation_regression_metrics(
        transition_condition_predictions,
        "Fixed origin: train 1–64, test 65–80",
    )


    # ---------- descriptive metrics by marginal-range status ----------
    transition_range_metric_parts = []

    for outside, subset in transition_condition_predictions.groupby(
        "Outside training marginal ranges",
        sort=True,
    ):
        range_label = (
            "Outside at least one training marginal range"
            if outside
            else "Within all training marginal ranges"
        )

        subgroup_metrics = validation_regression_metrics(
            subset,
            f"Transition test: {range_label}",
        )
        subgroup_metrics["Marginal-range status"] = range_label
        transition_range_metric_parts.append(subgroup_metrics)

    transition_range_metrics = pd.concat(
        transition_range_metric_parts,
        ignore_index=True,
    )


    # ---------- record training ranges and split definition ----------
    transition_training_ranges = pd.DataFrame({
        "Feature": FULL_FEATURE_COLS,
        "Training minimum": transition_train_min,
        "Training maximum": transition_train_max,
        "Range comparison tolerance": validation_tolerances,
    })

    transition_split_manifest = pd.DataFrame([{
        "Validation": "Fixed-origin search-space-change stress test",
        "Training condition cutoff": TRANSITION_TRAIN_THROUGH,
        "Test first condition": TRANSITION_TEST_START,
        "Test last condition": TRANSITION_TEST_END,
        "Training conditions": transition_train[condition_col].nunique(),
        "Training measurements": len(transition_train),
        "Held-out conditions": transition_test[condition_col].nunique(),
        "Held-out measurements": len(transition_test),
        "Earlier measurements excluded for matching inputs": int(
            transition_overlap.sum()
        ),
        "Training condition IDs": "; ".join(
            map(str, sorted(transition_train[condition_col].unique()))
        ),
        "Held-out condition IDs": "; ".join(
            map(str, sorted(transition_test[condition_col].unique()))
        ),
    }])


    # ---------- export ----------



    # ---------- report ----------
    print("\nOverall search-space-change performance:")

    print("\nCondition-level predictions and errors:")

    print("\nDescriptive performance by marginal-range status:")

    print("\nTraining input ranges:")

    # This fitted model is retained under its distinct name for later
    # predictive-interval calculations. It is not used for downstream SHAP.

    # ===============================================================
    # Targeted validation at observed Pareto conditions
    #
    # LOCO: leave one condition and its matching-input group out.
    # Training may include later campaign conditions.
    # FE-only GP; rate derived from predicted FE and input current.
    # ===============================================================



    if not validation_pareto_ids:
        raise ValueError("No observed Pareto conditions were identified.")


    # ---------- explicit campaign design spaces ----------
    validation_design_spaces = {
        1: {
            "condition_range": (1, 64),
            "lower": np.array(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
            "upper": np.array(
                [10.0, 100.0, 2.0, 2.0, 0.15, 0.15, 0.15]
            ),
            "salt_sum": (0.5, 2.0),
            "donor_sum": (0.0, 0.15),
        },
        2: {
            "condition_range": (65, 112),
            "lower": np.array(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
            "upper": np.array(
                [20.0, 100.0, 2.0, 2.0, 0.15, 0.15, 0.15]
            ),
            "salt_sum": (0.8, 2.0),
            "donor_sum": (0.0, 0.15),
        },
    }

    expected_boundary_feature_order = [
        CURRENT_DENSITY_COL,
        "THF (%)",
        "LiBF4 concentration (M)",
        "LiTFSI concentration (M)",
        "BuOH concentration (M)",
        "PrOH concentration (M)",
        "EtOH concentration (M)",
    ]

    if list(FULL_FEATURE_COLS) != expected_boundary_feature_order:
        raise ValueError("Design-space arrays have an unexpected feature order.")

    boundary_current_idx = FULL_FEATURE_COLS.index(CURRENT_DENSITY_COL)
    boundary_salt_indices = [
        FULL_FEATURE_COLS.index("LiBF4 concentration (M)"),
        FULL_FEATURE_COLS.index("LiTFSI concentration (M)"),
    ]
    boundary_donor_indices = [
        FULL_FEATURE_COLS.index("BuOH concentration (M)"),
        FULL_FEATURE_COLS.index("PrOH concentration (M)"),
        FULL_FEATURE_COLS.index("EtOH concentration (M)"),
    ]


    def annotate_validation_boundaries(row):
        """Annotate a condition-mean input using its campaign-stage bounds."""
        condition_id = int(row[condition_col])
        stage_number = 1 if condition_id <= 64 else 2
        design = validation_design_spaces[stage_number]

        x = row[FULL_FEATURE_COLS].to_numpy(dtype=float).copy()
        x[boundary_current_idx] = abs(x[boundary_current_idx])

        boundaries = []
        violations = []

        for i, feature in enumerate(FULL_FEATURE_COLS):
            tolerance = validation_tolerances[i]
            lower = design["lower"][i]
            upper = design["upper"][i]

            if abs(x[i] - lower) <= tolerance:
                boundaries.append(f"{feature}: lower")
            if abs(x[i] - upper) <= tolerance:
                boundaries.append(f"{feature}: upper")

            if x[i] < lower - tolerance or x[i] > upper + tolerance:
                violations.append(feature)

        for label, indices, limits in [
            ("Total salt (M)", boundary_salt_indices, design["salt_sum"]),
            ("Total proton donor (M)", boundary_donor_indices,
             design["donor_sum"]),
        ]:
            total = x[indices].sum()
            tolerance = validation_tolerances[indices].sum()
            lower, upper = limits

            if abs(total - lower) <= tolerance:
                boundaries.append(f"{label}: lower")
            if abs(total - upper) <= tolerance:
                boundaries.append(f"{label}: upper")

            if total < lower - tolerance or total > upper + tolerance:
                violations.append(label)

        return pd.Series({
            "Design stage number": stage_number,
            "On imposed design boundary": bool(boundaries),
            "Design boundary annotations": "; ".join(boundaries),
            "Outside stage constraints": bool(violations),
            "Stage constraint violations": "; ".join(violations),
        })

    validation_boundary_manifest = pd.concat(
        [
            validation_conditions[[condition_col]].reset_index(drop=True),
            validation_conditions.apply(
                annotate_validation_boundaries, axis=1
            ).reset_index(drop=True),
        ],
        axis=1,
    )

    print("Observed Pareto conditions and imposed boundaries:")

    boundary_violations = validation_boundary_manifest.loc[
        validation_boundary_manifest["Outside stage constraints"]
    ]
    if not boundary_violations.empty:
        print(
            "\nRecorded condition means outside their stage constraints; "
            "inspect these before interpreting boundary results:"
        )


    # ---------- targeted leave-one-condition-out predictions ----------
    pareto_loco_measurement_parts = []
    pareto_loco_split_records = []

    # Retain these few models for the subsequent interval calculations.
    pareto_loco_models = {}

    for condition_id in validation_pareto_ids:
        test_df = validation_df.loc[
            validation_df[condition_col] == condition_id
        ].copy()

        heldout_groups = test_df["Input group"].unique()

        # Exclude all matching inputs, including any other condition IDs.
        train_df = validation_df.loc[
            ~validation_df["Input group"].isin(heldout_groups)
        ].copy()

        if train_df.empty or test_df.empty:
            raise ValueError(
                f"Condition {condition_id}: empty training or test dataset."
            )

        if set(train_df["Input group"]) & set(test_df["Input group"]):
            raise RuntimeError("Matching-input leakage in Pareto LOCO.")

        print(
            f"Pareto LOCO: holding out condition {condition_id}; "
            f"training on {train_df[condition_col].nunique()} conditions "
            f"({len(train_df)} measurements)",
            flush=True,
        )

        validation_fe_gp = fit_fe_gp(
            train_df,
            train_df[FE_TARGET_COL],
            FULL_FEATURE_COLS,
            seed=VALIDATION_SEED + 2000 + int(condition_id),
        )

        predictions = predict_fe_rate(
            validation_fe_gp,
            test_df[FULL_FEATURE_COLS],
        )

        scored = test_df.copy()
        scored["Held-out predicted FE (%)"] = predictions["fe_mean"]
        scored["Held-out derived rate (nmol/s/cm^2)"] = (
            predictions["rate_mean"]
        )

        train_inputs = train_df[
            FULL_FEATURE_COLS
        ].to_numpy(dtype=float)
        test_inputs = test_df[
            FULL_FEATURE_COLS
        ].to_numpy(dtype=float)

        outside_by_feature = (
            (
                test_inputs
                < train_inputs.min(axis=0) - validation_tolerances
            )
            | (
                test_inputs
                > train_inputs.max(axis=0) + validation_tolerances
            )
        )

        scored["Outside training marginal ranges"] = (
            outside_by_feature.any(axis=1)
        )
        scored["Outside-range features"] = [
            "; ".join(
                feature
                for feature, outside in zip(FULL_FEATURE_COLS, flags)
                if outside
            )
            for flags in outside_by_feature
        ]

        scored["Training measurements"] = len(train_df)
        scored["Training conditions"] = train_df[condition_col].nunique()

        # Required by the shared scoring helper.
        scored["Training through batch"] = int(
            train_df["Acquisition batch"].max()
        )

        pareto_loco_measurement_parts.append(scored)
        pareto_loco_models[int(condition_id)] = validation_fe_gp

        pareto_loco_split_records.append({
            "Held-out condition": int(condition_id),
            "Held-out input groups": "; ".join(
                map(str, sorted(heldout_groups))
            ),
            "Training measurements": len(train_df),
            "Training conditions": train_df[condition_col].nunique(),
            "Held-out measurements": len(test_df),
            "Training condition IDs": "; ".join(
                map(str, sorted(train_df[condition_col].unique()))
            ),
            "Validation type": "Retrospective Pareto LOCO",
        })


    pareto_loco_measurement_predictions = pd.concat(
        pareto_loco_measurement_parts,
        ignore_index=True,
    )

    pareto_loco_condition_predictions = validation_score_conditions(
        pareto_loco_measurement_predictions
    )

    # Avoid presenting the shared helper's batch field as a temporal cutoff.
    pareto_loco_condition_predictions = (
        pareto_loco_condition_predictions.rename(columns={
            "Training through batch": "Largest training batch (not a cutoff)"
        })
    )

    pareto_loco_condition_predictions = (
        pareto_loco_condition_predictions.merge(
            validation_boundary_manifest,
            on=condition_col,
            how="left",
            validate="one_to_one",
        )
    )

    pareto_loco_metrics = validation_regression_metrics(
        pareto_loco_condition_predictions,
        "Targeted retrospective LOCO: observed Pareto conditions",
    )

    pareto_loco_split_manifest = pd.DataFrame(pareto_loco_split_records)


    # ---------- compare LOCO and forward predictions at the same conditions ----------
    pareto_forward_predictions = forward_condition_predictions.loc[
        forward_condition_predictions[condition_col].isin(
            validation_pareto_ids
        )
    ].copy()

    pareto_forward_predictions = pareto_forward_predictions.merge(
        validation_boundary_manifest,
        on=condition_col,
        how="left",
        validate="one_to_one",
    )

    pareto_forward_metrics = validation_regression_metrics(
        pareto_forward_predictions,
        "Next-batch validation: observed Pareto conditions",
    )

    pareto_comparison_columns = [
        condition_col,
        FE_TARGET_COL,
        "Held-out predicted FE (%)",
        "FE prediction minus observation (percentage points)",
        RATE_TARGET_COL,
        "Held-out derived rate (nmol/s/cm^2)",
        "Rate prediction minus observation (nmol/s/cm^2)",
        "Outside training marginal ranges",
        "Design boundary annotations",
    ]

    pareto_validation_comparison = pd.concat(
        [
            pareto_loco_condition_predictions[
                pareto_comparison_columns
            ].assign(Validation="Retrospective LOCO"),
            pareto_forward_predictions[
                pareto_comparison_columns
            ].assign(Validation="Forward next-batch"),
        ],
        ignore_index=True,
    )

    pareto_validation_metrics = pd.concat(
        [pareto_loco_metrics, pareto_forward_metrics],
        ignore_index=True,
    )


    # ---------- export ----------


    print("\nPareto-region validation metrics:")

    print("\nLOCO versus forward predictions:")

    # ===============================================================
    # Approximate 95% predictive intervals for condition means
    #
    # Observation noise is included.
    # Replicate covariance is retained when calculating mean uncertainty.
    # Rate covariance is transformed from FE covariance using Faraday's law.
    # ===============================================================




    def condition_mean_predictive_interval(
        gp,
        condition_measurements,
        validation_name,
    ):
        """
        Predict the mean of replicate electrolysis outcomes.
        Rate is obtained by transforming the same FE distribution.
        Intervals are approximate Gaussian 95% predictive intervals.
        """
        test = condition_measurements.copy()

        if test[condition_col].nunique() != 1:
            raise ValueError("Supply measurements from one condition only.")

        n = len(test)
        if n == 0:
            raise ValueError("Cannot predict an empty condition.")

        if list(gp.manuscript_feature_cols) != list(FULL_FEATURE_COLS):
            raise ValueError("Unexpected FE-model feature order.")

        train_tensor = gp.train_inputs[0]
        test_tensor = torch.as_tensor(
            test[FULL_FEATURE_COLS].to_numpy(dtype=np.float64).copy(order="K"),
            dtype=train_tensor.dtype,
            device=train_tensor.device,
        )

        gp.eval()
        gp.likelihood.eval()

        with torch.no_grad():
            # Includes inferred observation noise, in original FE units.
            posterior = gp.posterior(
                test_tensor,
                observation_noise=True,
            )

            fe_mean_vector = (
                posterior.mean.detach().cpu().numpy().reshape(-1)
            )
            fe_covariance = (
                posterior.distribution.covariance_matrix
                .detach().cpu().numpy()
            )

        if fe_covariance.shape != (n, n):
            raise ValueError(
                f"Expected covariance shape {(n, n)}, "
                f"received {fe_covariance.shape}."
            )

        if not (
            np.isfinite(fe_mean_vector).all()
            and np.isfinite(fe_covariance).all()
        ):
            raise ValueError("Nonfinite predictive distribution.")

        # Remove negligible asymmetry from numerical evaluation.
        fe_covariance = 0.5 * (
            fe_covariance + fe_covariance.T
        )

        weights = np.full(n, 1.0 / n)

        fe_mean = float(weights @ fe_mean_vector)
        fe_variance = float(weights @ fe_covariance @ weights)

        # Multiplicative factors for FE expressed in percentage points.
        rate_factors = calculate_nh3_rate(
            np.ones(n),
            test[CURRENT_DENSITY_COL].to_numpy(dtype=float),
        )

        rate_mean_vector = rate_factors * fe_mean_vector
        rate_covariance = (
            rate_factors[:, None]
            * fe_covariance
            * rate_factors[None, :]
        )

        rate_mean = float(weights @ rate_mean_vector)
        rate_variance = float(weights @ rate_covariance @ weights)

        if fe_variance < -1e-8 or rate_variance < -1e-8:
            raise RuntimeError("Unexpected negative predictive variance.")

        fe_sd = np.sqrt(max(fe_variance, 0.0))
        rate_sd = np.sqrt(max(rate_variance, 0.0))

        observed_fe = float(test[FE_TARGET_COL].mean())
        observed_rate = float(test[RATE_TARGET_COL].mean())

        z = 1.959963984540054

        fe_lower = fe_mean - z * fe_sd
        fe_upper = fe_mean + z * fe_sd
        rate_lower = rate_mean - z * rate_sd
        rate_upper = rate_mean + z * rate_sd

        return {
            "Validation": validation_name,
            condition_col: int(test[condition_col].iloc[0]),
            "Acquisition batch": int(test["Acquisition batch"].iloc[0]),
            "Search-space stage": test["Search-space stage"].iloc[0],
            "Observed Pareto condition": bool(
                test["Observed Pareto condition"].iloc[0]
            ),
            "Held-out replicate count": n,
            "Observed mean FE (%)": observed_fe,
            "Predicted mean FE (%)": fe_mean,
            "FE mean predictive SD (percentage points)": fe_sd,
            "FE lower 95% (%)": fe_lower,
            "FE upper 95% (%)": fe_upper,
            "FE covered by 95% interval": (
                fe_lower <= observed_fe <= fe_upper
            ),
            "Observed mean rate (nmol/s/cm^2)": observed_rate,
            "Predicted mean derived rate (nmol/s/cm^2)": rate_mean,
            "Rate mean predictive SD (nmol/s/cm^2)": rate_sd,
            "Rate lower 95% (nmol/s/cm^2)": rate_lower,
            "Rate upper 95% (nmol/s/cm^2)": rate_upper,
            "Rate covered by 95% interval": (
                rate_lower <= observed_rate <= rate_upper
            ),
        }


    interval_records = []

    # ---------- 1. Pareto LOCO: reuse retained models ----------
    for condition_id in validation_pareto_ids:
        test = validation_df.loc[
            validation_df[condition_col] == condition_id
        ]

        interval_records.append(
            condition_mean_predictive_interval(
                pareto_loco_models[int(condition_id)],
                test,
                "Retrospective Pareto LOCO",
            )
        )


    # ---------- 2. Forward prediction at Pareto conditions ----------
    # Refit the same forward split using the same seed as the earlier cell.
    pareto_test_batches = sorted(
        validation_df.loc[
            validation_df[condition_col].isin(validation_pareto_ids),
            "Acquisition batch",
        ].unique()
    )

    # Retain these models if additional interval diagnostics are needed later.
    pareto_forward_interval_models = {}

    for batch in pareto_test_batches:
        batch = int(batch)

        manifest_row = forward_split_manifest.loc[
            forward_split_manifest["Held-out batch"] == batch
        ]
        if len(manifest_row) != 1:
            raise ValueError(
                f"No unique forward split is available for batch {batch}."
            )

        batch_test = validation_df.loc[
            validation_df["Acquisition batch"] == batch
        ]
        heldout_groups = batch_test["Input group"].unique()

        train = validation_df.loc[
            (validation_df["Acquisition batch"] < batch)
            & (~validation_df["Input group"].isin(heldout_groups))
        ].copy()

        # Verify that the reconstructed split matches the earlier run.
        expected_ids = {
            int(value.strip())
            for value in manifest_row.iloc[0][
                "Training condition IDs"
            ].split(";")
        }
        if set(train[condition_col].unique()) != expected_ids:
            raise RuntimeError(
                f"Batch {batch}: reconstructed training split differs "
                "from the saved forward-validation split."
            )

        print(
            f"Refitting forward batch {batch} for Pareto predictive intervals",
            flush=True,
        )

        interval_fe_gp = fit_fe_gp(
            train,
            train[FE_TARGET_COL],
            FULL_FEATURE_COLS,
            seed=VALIDATION_SEED + batch,
        )
        pareto_forward_interval_models[batch] = interval_fe_gp

        batch_pareto_ids = sorted(
            batch_test.loc[
                batch_test[condition_col].isin(validation_pareto_ids),
                condition_col,
            ].unique()
        )

        for condition_id in batch_pareto_ids:
            test = batch_test.loc[
                batch_test[condition_col] == condition_id
            ]

            interval_records.append(
                condition_mean_predictive_interval(
                    interval_fe_gp,
                    test,
                    "Forward next-batch: Pareto conditions",
                )
            )


    # ---------- 3. Fixed-origin transition test ----------
    for condition_id, test in transition_test.groupby(
        condition_col, sort=True
    ):
        interval_records.append(
            condition_mean_predictive_interval(
                transition_fe_gp,
                test,
                "Fixed origin: train 1–64, test 65–80",
            )
        )


    validation_mean_intervals = pd.DataFrame(interval_records)

    # Add design-boundary annotations from the preceding cell.
    validation_mean_intervals = validation_mean_intervals.merge(
        validation_boundary_manifest,
        on=condition_col,
        how="left",
        validate="many_to_one",
    )


    # ---------- descriptive coverage and interval widths ----------
    coverage_records = []

    for validation_name, subset in validation_mean_intervals.groupby(
        "Validation", sort=False
    ):
        for response, covered_col, lower_col, upper_col, units in [
            (
                "FE",
                "FE covered by 95% interval",
                "FE lower 95% (%)",
                "FE upper 95% (%)",
                "percentage points",
            ),
            (
                "Derived rate",
                "Rate covered by 95% interval",
                "Rate lower 95% (nmol/s/cm^2)",
                "Rate upper 95% (nmol/s/cm^2)",
                "nmol s^-1 cm^-2",
            ),
        ]:
            coverage_records.append({
                "Validation": validation_name,
                "Response": response,
                "N conditions": len(subset),
                "N covered": int(subset[covered_col].sum()),
                "Coverage fraction": float(subset[covered_col].mean()),
                "Mean interval width": float(
                    (subset[upper_col] - subset[lower_col]).mean()
                ),
                "Interval-width units": units,
            })

    validation_interval_coverage = pd.DataFrame(coverage_records)


    # ---------- export and report ----------

    print("\nPareto-condition predictive intervals:")

    print("\nDescriptive 95% predictive-interval coverage:")

    # ===============================================================
    # Held-out comparison:
    # independent rate GP versus FE GP + Faraday-derived rate
    #
    # Five-fold grouped cross-validation.
    # All replicates and matching input settings remain in one fold.
    # Plot and metric unit: one experimental condition.
    # ===============================================================

    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    CV_FOLDS = 5
    CV_SEED = 12345

    cv_df = model_df.copy()

    # ---------- construct leakage-safe groups ----------
    condition_ids = cv_df[condition_col].drop_duplicates().to_numpy()
    condition_to_position = {
        condition_id: position
        for position, condition_id in enumerate(condition_ids)
    }

    parent = np.arange(len(condition_ids))


    def find_root(position):
        while parent[position] != position:
            parent[position] = parent[parent[position]]
            position = parent[position]
        return position


    def join_groups(first, second):
        root_first = find_root(first)
        root_second = find_root(second)
        if root_first != root_second:
            parent[root_second] = root_first


    cv_inputs = cv_df[FULL_FEATURE_COLS].to_numpy(dtype=float)
    cv_condition_positions = cv_df[condition_col].map(
        condition_to_position
    ).to_numpy(dtype=int)

    cv_input_atol = np.array(
        [input_atol[feature] for feature in FULL_FEATURE_COLS],
        dtype=float,
    )
    
    for first in range(len(cv_df)):
        matching_later_rows = np.flatnonzero(
            np.all(
                np.abs(cv_inputs[first + 1:] - cv_inputs[first])
                <= cv_input_atol,
                axis=1,
            )
        ) + first + 1

        for second in matching_later_rows:
            join_groups(
                cv_condition_positions[first],
                cv_condition_positions[second],
            )

    cv_groups = np.array([
        find_root(position)
        for position in cv_condition_positions
    ])

    n_cv_groups = len(np.unique(cv_groups))
    if n_cv_groups < CV_FOLDS:
        raise ValueError(
            f"Only {n_cv_groups} independent input groups are available "
            f"for {CV_FOLDS}-fold validation."
        )

    print(
        f"Cross-validation: {len(cv_df)} measurements, "
        f"{len(condition_ids)} conditions, "
        f"{n_cv_groups} input groups, {CV_FOLDS} folds."
    )


    # ---------- independent rate GP: comparator only ----------
    def fit_rate_comparator(features, recorded_rate, seed):
        """Fit the temporary rate comparator on one training fold."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        x_array = _feature_array(features, FULL_FEATURE_COLS)
        y_array = np.asarray(
            recorded_rate, dtype=np.float64
        ).reshape(-1, 1)

        rate_gp_comparator = SingleTaskGP(
            torch.tensor(x_array, **tkwargs),
            torch.tensor(y_array, **tkwargs),
            input_transform=BoNormalize(d=len(FULL_FEATURE_COLS)),
            outcome_transform=BoStandardize(m=1),
        ).to(**tkwargs)

        comparator_mll = ExactMarginalLogLikelihood(
            rate_gp_comparator.likelihood,
            rate_gp_comparator,
        )
        fit_gpytorch_mll(comparator_mll)

        rate_gp_comparator.eval()
        rate_gp_comparator.likelihood.eval()
        return rate_gp_comparator


    # ---------- out-of-fold predictions ----------
    cv_fe_prediction = np.full(len(cv_df), np.nan)
    cv_derived_rate_prediction = np.full(len(cv_df), np.nan)
    cv_rate_gp_prediction = np.full(len(cv_df), np.nan)
    cv_fold_number = np.full(len(cv_df), -1, dtype=int)

    splitter = GroupKFold(n_splits=CV_FOLDS)

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(cv_inputs, groups=cv_groups),
        start=1,
    ):
        train_df = cv_df.iloc[train_idx]
        test_df = cv_df.iloc[test_idx]

        print(
            f"Fold {fold}/{CV_FOLDS}: "
            f"{len(train_df)} training and {len(test_df)} held-out measurements",
            flush=True,
        )

        # Fold-specific FE model; never overwrite fe_gp_full.
        cv_fe_gp = fit_fe_gp(
            train_df,
            train_df[FE_TARGET_COL],
            FULL_FEATURE_COLS,
            seed=CV_SEED + fold,
        )

        cv_predictions = predict_fe_rate(
            cv_fe_gp,
            test_df[FULL_FEATURE_COLS],
        )

        cv_fe_prediction[test_idx] = cv_predictions["fe_mean"]
        cv_derived_rate_prediction[test_idx] = cv_predictions["rate_mean"]

        # Same training rows and GP specification, with rate as the target.
        cv_rate_gp = fit_rate_comparator(
            train_df,
            train_df[RATE_TARGET_COL],
            seed=CV_SEED + fold,
        )

        with torch.no_grad():
            test_tensor = torch.tensor(
                test_df[FULL_FEATURE_COLS].to_numpy(dtype=np.float64),
                **tkwargs,
            )
            cv_rate_gp_prediction[test_idx] = (
                cv_rate_gp.posterior(test_tensor)
                .mean.cpu().numpy().reshape(-1)
            )

        cv_fold_number[test_idx] = fold

        # Comparator and fold-specific FE model are not used downstream.
        del cv_rate_gp, cv_fe_gp


    if (
        not np.isfinite(cv_fe_prediction).all()
        or not np.isfinite(cv_derived_rate_prediction).all()
        or not np.isfinite(cv_rate_gp_prediction).all()
        or (cv_fold_number < 1).any()
    ):
        raise RuntimeError("Some measurements lack valid held-out predictions.")


    # ---------- measurement-level source data ----------
    cv_measurement_results = cv_df[
        [condition_col] + FULL_FEATURE_COLS + [FE_TARGET_COL, RATE_TARGET_COL]
    ].copy()

    cv_measurement_results.insert(
        0, "CSV row", cv_measurement_results.index.to_numpy() + 2
    )
    cv_measurement_results["CV fold"] = cv_fold_number
    cv_measurement_results["Held-out FE prediction (%)"] = cv_fe_prediction
    cv_measurement_results["Held-out independent rate GP"] = (
        cv_rate_gp_prediction
    )
    cv_measurement_results["Held-out FE-derived rate"] = (
        cv_derived_rate_prediction
    )

    # All measurements for each condition must have the same held-out fold.
    if (
        cv_measurement_results.groupby(condition_col)["CV fold"]
        .nunique().gt(1).any()
    ):
        raise RuntimeError("Replicates were split across validation folds.")


    # ---------- condition-level means for plotting and metrics ----------
    cv_condition_results = (
        cv_measurement_results.groupby(condition_col, as_index=False)
        .agg({
            "CV fold": "first",
            FE_TARGET_COL: "mean",
            RATE_TARGET_COL: "mean",
            "Held-out FE prediction (%)": "mean",
            "Held-out independent rate GP": "mean",
            "Held-out FE-derived rate": "mean",
        })
    )

    cv_condition_results["Retained replicate count"] = (
        cv_condition_results[condition_col]
        .map(cv_df.groupby(condition_col).size())
        .to_numpy()
    )

    cv_metric_records = []

    for response, observed_col, predicted_col, unit in [
        (
            "Independent rate GP",
            RATE_TARGET_COL,
            "Held-out independent rate GP",
            "nmol s^-1 cm^-2",
        ),
        (
            "FE-derived rate",
            RATE_TARGET_COL,
            "Held-out FE-derived rate",
            "nmol s^-1 cm^-2",
        ),
        (
            "FE GP",
            FE_TARGET_COL,
            "Held-out FE prediction (%)",
            "percentage points",
        ),
    ]:
        observed = cv_condition_results[observed_col].to_numpy()
        predicted = cv_condition_results[predicted_col].to_numpy()

        cv_metric_records.append({
            "Response": response,
            "Evaluation unit": "Condition mean",
            "Number of conditions": len(observed),
            "RMSE": np.sqrt(mean_squared_error(observed, predicted)),
            "MAE": mean_absolute_error(observed, predicted),
            "R2": r2_score(observed, predicted),
            "RMSE and MAE unit": unit,
        })

    cv_metrics = pd.DataFrame(cv_metric_records)

    print("\nHeld-out metrics, equally weighted by condition:")
    print(cv_metrics.to_string(index=False))

    names = (
        "validation_df", "validation_batch_summary",
        "validation_group_summary", "forward_split_manifest",
        "forward_measurement_predictions", "forward_condition_predictions",
        "forward_overall_metrics", "forward_stage_metrics", "forward_batch_metrics",
        "transition_measurement_predictions", "transition_condition_predictions",
        "transition_metrics", "transition_split_manifest", "transition_training_ranges",
        "validation_boundary_manifest", "pareto_loco_split_manifest",
        "pareto_loco_measurement_predictions", "pareto_loco_condition_predictions",
        "pareto_forward_predictions", "pareto_validation_comparison",
        "pareto_validation_metrics", "validation_mean_intervals",
        "validation_interval_coverage", "cv_measurement_results",
        "cv_condition_results", "cv_metrics",
    )
    available = locals()
    tables = {name: available[name].copy() for name in names}
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(destination / f"{name}.csv", index=False)
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Repository root (autodetected by preprocessing).")
    parser.add_argument("--output-dir", type=Path, help="Optional destination for numerical CSV tables.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    tables = run_validation(args.root, output_dir=args.output_dir, verbose=not args.quiet)
    for name in ("cv_metrics", "forward_overall_metrics", "transition_metrics", "pareto_validation_metrics"):
        builtins.print("\n" + name + "\n" + tables[name].to_string(index=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
