from __future__ import annotations
import argparse
import builtins
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from .preprocessing import load_campaign, repository_root, CONDITION
    from .models import (FULL_FEATURE_COLS, CURRENT_DENSITY_COL, FE_TARGET_COL,
        RATE_TARGET_COL, fit_fe_gp, predict_fe_rate, predict_fe_mean,
        predict_rate_from_fe, calculate_nh3_rate)
except ImportError:
    from preprocessing import load_campaign, repository_root, CONDITION
    from models import (FULL_FEATURE_COLS, CURRENT_DENSITY_COL, FE_TARGET_COL,
        RATE_TARGET_COL, fit_fe_gp, predict_fe_rate, predict_fe_mean,
        predict_rate_from_fe, calculate_nh3_rate)

# Internal strict coordinates retain the foundation cell's PrOH coordinate.
COMPOSITION_FEATURE_COLS = [CURRENT_DENSITY_COL, "THF (%)", "Total salt (M)",
    "LiBF4 fraction of salt", "Total proton donor (M)",
    "BuOH fraction of donor", "PrOH fraction of donor"]
# Main-paper masking and reporting use EtOH, with PrOH reconstructed as balance.
MAINPAPER_FEATURE_COLS = COMPOSITION_FEATURE_COLS[:-1] + ["EtOH fraction of donor"]
MAINPAPER_FEATURE_LABELS = ["Current density", "THF (DG balance)", "Total salt",
    "LiBF$_4$ salt fraction", "Total proton donor", "1-BuOH donor fraction", "EtOH donor fraction"]
COMPOSITION_RAW_SALT_COLS = FULL_FEATURE_COLS[2:4]
COMPOSITION_RAW_DONOR_COLS = FULL_FEATURE_COLS[4:7]
COMPOSITION_NUMERIC_ATOL = 1e-12

def composition_input_array(values, columns):
    """Validate an input matrix and enforce the declared column order."""
    if isinstance(values, pd.DataFrame):
        array = values.loc[:, columns].to_numpy(dtype=np.float64)
    else:
        array = np.asarray(values, dtype=np.float64)

    if array.ndim == 1:
        array = array.reshape(1, -1)

    if array.ndim != 2 or array.shape[1] != len(columns):
        raise ValueError(
            f"Expected {len(columns)} columns in this order:\n"
            f"{columns}\nReceived shape {array.shape}."
        )

    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Inputs must contain finite, nonempty rows.")

    return array.copy()

def raw_to_composition(values):
    """
    Convert original GP inputs to total-and-fraction coordinates.

    Array inputs must follow FULL_FEATURE_COLS.
    At zero loading, identity fractions use the numerical convention zero.
    """
    raw = composition_input_array(values, FULL_FEATURE_COLS)
    column_index = {
        name: i for i, name in enumerate(FULL_FEATURE_COLS)
    }

    current = raw[:, column_index[CURRENT_DENSITY_COL]]
    thf = raw[:, column_index["THF (%)"]]

    salts = raw[:, [
        column_index[name] for name in COMPOSITION_RAW_SALT_COLS
    ]]
    donors = raw[:, [
        column_index[name] for name in COMPOSITION_RAW_DONOR_COLS
    ]]

    if (
        (salts < -COMPOSITION_NUMERIC_ATOL).any()
        or (donors < -COMPOSITION_NUMERIC_ATOL).any()
    ):
        raise ValueError("Negative salt or donor concentrations.")

    if (
        (thf < -COMPOSITION_NUMERIC_ATOL).any()
        or (thf > 100.0 + COMPOSITION_NUMERIC_ATOL).any()
    ):
        raise ValueError("THF percentage is outside 0–100.")

    # Remove numerical-scale negatives only.
    salts = np.maximum(salts, 0.0)
    donors = np.maximum(donors, 0.0)
    thf = np.clip(thf, 0.0, 100.0)

    total_salt = salts.sum(axis=1)
    total_donor = donors.sum(axis=1)

    libf4_fraction = np.divide(
        salts[:, 0],
        total_salt,
        out=np.zeros_like(total_salt),
        where=total_salt > 0,
    )
    buoh_fraction = np.divide(
        donors[:, 0],
        total_donor,
        out=np.zeros_like(total_donor),
        where=total_donor > 0,
    )
    proh_fraction = np.divide(
        donors[:, 1],
        total_donor,
        out=np.zeros_like(total_donor),
        where=total_donor > 0,
    )

    return np.column_stack([
        current,
        thf,
        total_salt,
        libf4_fraction,
        total_donor,
        buoh_fraction,
        proh_fraction,
    ])

def composition_to_raw(values):
    """
    Strictly reconstruct original GP inputs.
    """
    comp = composition_input_array(
        values, COMPOSITION_FEATURE_COLS
    )

    (
        current,
        thf,
        total_salt,
        libf4_fraction,
        total_donor,
        buoh_fraction,
        proh_fraction,
    ) = comp.T

    if (
        (thf < -COMPOSITION_NUMERIC_ATOL).any()
        or (thf > 100.0 + COMPOSITION_NUMERIC_ATOL).any()
    ):
        raise ValueError("THF percentage is outside 0–100.")

    if (
        (total_salt < -COMPOSITION_NUMERIC_ATOL).any()
        or (total_donor < -COMPOSITION_NUMERIC_ATOL).any()
    ):
        raise ValueError("Total concentrations must be nonnegative.")

    fractions = comp[:, [3, 5, 6]]
    if (
        (fractions < -COMPOSITION_NUMERIC_ATOL).any()
        or (fractions > 1.0 + COMPOSITION_NUMERIC_ATOL).any()
    ):
        raise ValueError("Identity fractions must lie between zero and one.")

    if (
        buoh_fraction + proh_fraction
        > 1.0 + COMPOSITION_NUMERIC_ATOL
    ).any():
        raise ValueError(
            "BuOH and PrOH donor fractions sum to more than one. "
            "A masking/projection rule must be specified separately."
        )

    # Numerical cleanup only; no substantive mixture projection.
    thf = np.clip(thf, 0.0, 100.0)
    total_salt = np.maximum(total_salt, 0.0)
    total_donor = np.maximum(total_donor, 0.0)
    libf4_fraction = np.clip(libf4_fraction, 0.0, 1.0)
    buoh_fraction = np.clip(buoh_fraction, 0.0, 1.0)
    proh_fraction = np.clip(proh_fraction, 0.0, 1.0)
    etoh_fraction = np.maximum(
        1.0 - buoh_fraction - proh_fraction,
        0.0,
    )

    raw_columns = {
        CURRENT_DENSITY_COL: current,
        "THF (%)": thf,
        "LiBF4 concentration (M)": (
            total_salt * libf4_fraction
        ),
        "LiTFSI concentration (M)": (
            total_salt * (1.0 - libf4_fraction)
        ),
        "BuOH concentration (M)": (
            total_donor * buoh_fraction
        ),
        "PrOH concentration (M)": (
            total_donor * proh_fraction
        ),
        "EtOH concentration (M)": (
            total_donor * etoh_fraction
        ),
    }

    return np.column_stack([
        raw_columns[name] for name in FULL_FEATURE_COLS
    ])

def prepare_composition_masked_inputs(values):
    """
    Normalize masked donor fractions when necessary.

    Returns:
        prepared coordinates
        flags identifying donor normalization and zero-loading cleanup
    """
    comp = composition_input_array(
        values,
        COMPOSITION_FEATURE_COLS,
    )

    fraction_columns = [3, 5, 6]
    fractions = comp[:, fraction_columns]

    if (
        (fractions < -COMPOSITION_NUMERIC_ATOL).any()
        or (fractions > 1.0 + COMPOSITION_NUMERIC_ATOL).any()
    ):
        raise ValueError("Masked identity fractions are outside 0–1.")

    # Numerical cleanup of individual fractions only.
    comp[:, fraction_columns] = np.clip(fractions, 0.0, 1.0)

    if (comp[:, 4] < -COMPOSITION_NUMERIC_ATOL).any():
        raise ValueError("Negative total donor concentration.")

    comp[:, 4] = np.maximum(comp[:, 4], 0.0)

    donor_fractions = comp[:, 5:7].copy()
    donor_sum = donor_fractions.sum(axis=1)

    donor_active = comp[:, 4] > 0
    donor_overflow = donor_active & (donor_sum > 1.0)

    # Separate substantive normalization from floating-point cleanup.
    donor_substantive_overflow = (
        donor_active
        & (donor_sum > 1.0 + COMPOSITION_NUMERIC_ATOL)
    )

    donor_fractions[donor_overflow] /= (
        donor_sum[donor_overflow, None]
    )

    zero_donor_cleanup = (
        (~donor_active)
        & np.any(
            donor_fractions > COMPOSITION_NUMERIC_ATOL,
            axis=1,
        )
    )
    donor_fractions[~donor_active] = 0.0

    comp[:, 5:7] = donor_fractions

    # At zero salt loading, salt identity is physically inactive.
    if (comp[:, 2] < -COMPOSITION_NUMERIC_ATOL).any():
        raise ValueError("Negative total salt concentration.")

    comp[:, 2] = np.maximum(comp[:, 2], 0.0)
    zero_salt_cleanup = (
        (comp[:, 2] == 0)
        & (comp[:, 3] > COMPOSITION_NUMERIC_ATOL)
    )
    comp[comp[:, 2] == 0, 3] = 0.0

    flags = {
        "Donor normalization, including numerical cleanup": donor_overflow,
        "Donor normalization above numerical tolerance": (
            donor_substantive_overflow
        ),
        "Zero-donor identity cleanup": zero_donor_cleanup,
        "Zero-salt identity cleanup": zero_salt_cleanup,
    }

    return comp, flags

def mainpaper_composition_to_raw(values):
    if isinstance(values, pd.DataFrame):
        values = values.loc[:, MAINPAPER_FEATURE_COLS].to_numpy(dtype=np.float64)
    # Same normalization convention, now applied to 1-BuOH and EtOH.
    # If their masked fractions sum above one, normalize BOTH; 1-PrOH = 0.
    prepared, flags = prepare_composition_masked_inputs(values)
    original_coordinates = prepared.copy()
    original_coordinates[:, 6] = np.where(
        prepared[:, 4] > 0,
        np.maximum(1.0 - prepared[:, 5] - prepared[:, 6], 0.0),
        0.0,
    )
    return composition_to_raw(original_coordinates), flags


def raw_to_mainpaper(values):
    """Return current, THF, total salt, salt fraction, total donor, BuOH, EtOH."""
    internal = raw_to_composition(values)
    result = internal.copy()
    result[:, 6] = np.where(result[:, 4] > 0,
        np.maximum(1.0-internal[:, 5]-internal[:, 6], 0.0), 0.0)
    return result


def composition_context(campaign):
    means = campaign.condition_means
    raw = means[FULL_FEATURE_COLS].to_numpy(dtype=np.float64)
    comp = raw_to_composition(raw)
    frame = pd.DataFrame(comp, columns=COMPOSITION_FEATURE_COLS)
    frame.insert(0, CONDITION, means[CONDITION].to_numpy().copy())
    frame["Observed mean FE (%)"] = means[FE_TARGET_COL].to_numpy()
    frame["Recorded mean rate (nmol/s/cm^2)"] = means[RATE_TARGET_COL].to_numpy()
    frame["Retained replicate count"] = means[CONDITION].map(campaign.model_rows.groupby(CONDITION).size()).to_numpy()
    frame["Salt identity defined"] = frame["Total salt (M)"] > 0
    frame["Donor identity defined"] = frame["Total proton donor (M)"] > 0
    return raw, comp, frame


def compute_shap(model, campaign, *, responses=("FE", "Rate"), verbose=True):
    """Exact seven-coordinate SHAP; equal weight per analyzed condition.

    Independent masker uses all unique condition vectors (no subsampling).
    Donor-fraction overflow is jointly normalized; zero-loading identity is zero.
    Returns numerical source tables, background, importance and masking diagnostics.
    """
    import shap
    if set(responses)-{"FE", "Rate"}:
        raise ValueError("responses must contain FE and/or Rate.")
    raw, internal, frame = composition_context(campaign)
    inputs = raw_to_mainpaper(raw)
    reconstructed, flags = mainpaper_composition_to_raw(inputs)
    np.testing.assert_allclose(reconstructed, raw, rtol=1e-10, atol=1e-12)
    observed_predictions = predict_fe_rate(model, raw)
    masker = shap.maskers.Independent(inputs, max_samples=len(inputs))
    sources, importance, diagnostics = [], [], []
    for response in responses:
        if verbose:
            builtins.print(f"Computing exact {response} SHAP at {len(inputs)} conditions...", flush=True)
        stats = {"Evaluated rows": 0, "Donor fractions normalized above tolerance": 0}
        predictor = predict_fe_mean if response == "FE" else predict_rate_from_fe
        def masked_prediction(values):
            physical, active_flags = mainpaper_composition_to_raw(values)
            stats["Evaluated rows"] += len(physical)
            stats["Donor fractions normalized above tolerance"] += int(active_flags[
                "Donor normalization above numerical tolerance"].sum())
            return predictor(model, physical)
        explainer = shap.Explainer(masked_prediction, masker, algorithm="exact",
                                   feature_names=MAINPAPER_FEATURE_LABELS)
        explanation = explainer(inputs)
        mean = predictor(model, reconstructed)
        direct = observed_predictions["fe_mean" if response == "FE" else "rate_mean"]
        np.testing.assert_allclose(mean, direct, rtol=1e-7, atol=1e-8)
        base = np.asarray(explanation.base_values).reshape(-1)
        values = np.asarray(explanation.values)
        residual = base+values.sum(axis=1)-mean
        np.testing.assert_allclose(base+values.sum(axis=1), mean, rtol=1e-6, atol=1e-6)
        absolute = np.abs(values).mean(axis=0)
        normalized = absolute/absolute.max() if absolute.max()>0 else np.zeros(7)
        for column, coordinate in enumerate(MAINPAPER_FEATURE_COLS):
            sources.append(pd.DataFrame({CONDITION:frame[CONDITION], "Response":response,
                "Coordinate":coordinate, "Input value":inputs[:,column],
                "SHAP value":values[:,column], "Base value":base, "Prediction":mean,
                "Additivity residual":residual}))
            importance.append({"Response":response, "Coordinate":coordinate,
                "Mean absolute SHAP":absolute[column], "Normalized mean absolute SHAP":normalized[column]})
        diagnostics.append({"Response":response, **stats})
    background = pd.DataFrame(inputs, columns=MAINPAPER_FEATURE_COLS)
    background.insert(0,CONDITION,frame[CONDITION])
    return {"shap_values":pd.concat(sources,ignore_index=True),
            "shap_importance":pd.DataFrame(importance),
            "shap_masking_diagnostics":pd.DataFrame(diagnostics), "shap_background":background}


def response_slices(model, campaign, *, grid_n=60):
    """Nine strict Stage-2 slices at BO-82, BO-86 and matched-current baseline."""
    if not isinstance(grid_n, int) or grid_n < 2:
        raise ValueError("grid_n must be an integer of at least two.")
    fe_gp_full = model
    model_df = campaign.model_rows
    condition_col = CONDITION
    _, composition_inputs, composition_df = composition_context(campaign)
    # ===============================================================
    # Stage-2-feasible response slices of the existing FE GP
    #
    # Reference: measured condition BO-82.
    # Rows: BO-82 FE, BO-86 rate, baseline rate.
    # Separate companion figure: latent posterior SD (observation noise excluded).
    #
    # Donor path: BuOH replaces EtOH; PrOH is fixed at zero.
    # No observed measurements are projected onto the surfaces.
    # ===============================================================




    COMPOSITION_SLICE_REFERENCE_ID = 82
    COMPOSITION_SLICE_N = grid_n

    # Explicit Stage 2 bounds in FULL_FEATURE_COLS order.
    # Current density is represented by its positive magnitude.
    slice_stage2_lower = np.array([
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])
    slice_stage2_upper = np.array([
        20.0, 100.0, 2.0, 2.0, 0.15, 0.15, 0.15
    ])
    slice_salt_limits = (0.8, 2.0)
    slice_donor_limits = (0.0, 0.15)
    slice_numeric_tolerance = 1e-10

    slice_expected_order = [
        CURRENT_DENSITY_COL,
        "THF (%)",
        "LiBF4 concentration (M)",
        "LiTFSI concentration (M)",
        "BuOH concentration (M)",
        "PrOH concentration (M)",
        "EtOH concentration (M)",
    ]

    if list(FULL_FEATURE_COLS) != slice_expected_order:
        raise ValueError("Unexpected feature order for Stage 2 bounds.")

    if (model_df[CURRENT_DENSITY_COL] < 0).any():
        raise ValueError(
            "This slice setup expects positive current-density magnitudes. "
            "Resolve the input sign convention before continuing."
        )

    slice_reference_rows = composition_df.loc[
        composition_df[condition_col] == COMPOSITION_SLICE_REFERENCE_ID
    ]

    if len(slice_reference_rows) != 1:
        raise ValueError("Expected one compositional reference row.")

    slice_reference = slice_reference_rows[
        COMPOSITION_FEATURE_COLS
    ].to_numpy(dtype=float)[0]


    def slice_stage2_feasible(raw_inputs):
        """Check box bounds and total-composition constraints."""
        raw = np.asarray(raw_inputs, dtype=float)

        valid = np.isfinite(raw).all(axis=1)
        valid &= np.all(
            raw >= slice_stage2_lower - slice_numeric_tolerance,
            axis=1,
        )
        valid &= np.all(
            raw <= slice_stage2_upper + slice_numeric_tolerance,
            axis=1,
        )

        salt_total = raw[:, 2] + raw[:, 3]
        donor_total = raw[:, 4:7].sum(axis=1)

        valid &= (
            (salt_total >= slice_salt_limits[0] - slice_numeric_tolerance)
            & (salt_total <= slice_salt_limits[1] + slice_numeric_tolerance)
            & (donor_total >= slice_donor_limits[0] - slice_numeric_tolerance)
            & (donor_total <= slice_donor_limits[1] + slice_numeric_tolerance)
        )
        return valid


    slice_reference_raw = composition_to_raw(
        slice_reference
    )

    if not slice_stage2_feasible(slice_reference_raw).all():
        raise ValueError(
            "The selected reference is outside the specified Stage 2 constraints."
        )

    # The explicit donor substitution is anchored at BO-86 only if its
    # reference donor is entirely BuOH.
    if not (
        np.isclose(slice_reference[5], 1.0, atol=1e-10)
        and np.isclose(slice_reference[6], 0.0, atol=1e-10)
    ):
        raise ValueError(
            "The donor slice assumes a pure-BuOH reference donor. "
            "Specify a new substitution path for this reference."
        )


    # ---------- distance-to-data diagnostic ----------
    slice_training_inputs = model_df[
        FULL_FEATURE_COLS
    ].to_numpy(dtype=float)

    slice_scale = slice_stage2_upper - slice_stage2_lower
    slice_training_scaled = (
        slice_training_inputs - slice_stage2_lower
    ) / slice_scale


    def slice_nearest_training_distance(raw_inputs, chunk_size=512):
        scaled = (
            np.asarray(raw_inputs) - slice_stage2_lower
        ) / slice_scale

        distances = np.empty(len(scaled))

        for start in range(0, len(scaled), chunk_size):
            chunk = scaled[start:start + chunk_size]
            difference = (
                chunk[:, None, :] - slice_training_scaled[None, :, :]
            )
            squared_distance = np.sum(difference**2, axis=2)
            distances[start:start + len(chunk)] = np.sqrt(
                squared_distance.min(axis=1)
            )

        return distances


    def evaluate_composition_slice(
        slice_name,
        x_axis,
        y_axis,
        configure_coordinates,
        x_label,
        y_label,
        reference_xy,
        reference_coordinates=None,
        reference_id=None,
    ):
        if reference_coordinates is None:
            reference_coordinates = slice_reference
        if reference_id is None:
            reference_id = COMPOSITION_SLICE_REFERENCE_ID
        x_grid, y_grid = np.meshgrid(x_axis, y_axis)

        coordinates = np.tile(
            reference_coordinates,
            (x_grid.size, 1),
        )
        configure_coordinates(
            coordinates,
            x_grid.ravel(),
            y_grid.ravel(),
        )

        # Strict reconstruction: no donor-fraction projection is used here.
        raw = composition_to_raw(coordinates)
        feasible = slice_stage2_feasible(raw)

        fe_mean = np.full(len(raw), np.nan)
        fe_sd = np.full(len(raw), np.nan)
        nearest_distance = np.full(len(raw), np.nan)

        if not feasible.any():
            raise ValueError(f"{slice_name}: no feasible grid points.")

        predictions = predict_fe_rate(
            fe_gp_full,
            raw[feasible],
        )
        fe_mean[feasible] = predictions["fe_mean"]
        fe_sd[feasible] = predictions["fe_sd"]
        nearest_distance[feasible] = slice_nearest_training_distance(
            raw[feasible]
        )

        export = pd.DataFrame(raw, columns=FULL_FEATURE_COLS)
        export.insert(0, "Slice", slice_name)
        export.insert(
            1, "Reference condition", reference_id
        )
        export["X plotted"] = x_grid.ravel()
        export["Y plotted"] = y_grid.ravel()
        export["Stage 2 feasible"] = feasible
        export["Predicted FE mean (%)"] = fe_mean
        export["FE latent posterior SD (percentage points)"] = fe_sd
        export["Nearest training-input distance (scaled Euclidean)"] = (
            nearest_distance
        )

        return {
            "name": slice_name,
            "x": x_axis,
            "y": y_axis,
            "mean": fe_mean.reshape(x_grid.shape),
            "sd": fe_sd.reshape(x_grid.shape),
            "xlabel": x_label,
            "ylabel": y_label,
            "reference_xy": reference_xy,
            "export": export,
        }



    def configure_solvent_current(coordinates, thf, current):
        coordinates[:, 0] = current
        coordinates[:, 1] = thf


    def configure_salt(coordinates, salt_fraction, total_salt):
        coordinates[:, 2] = total_salt
        coordinates[:, 3] = salt_fraction


    def configure_donor(coordinates, buoh_fraction, donor_mM):
        coordinates[:, 4] = donor_mM / 1000.0
        coordinates[:, 5] = buoh_fraction
        coordinates[:, 6] = 0.0  # EtOH = 1 - BuOH; PrOH remains zero.



    # ===============================================================
    # Reference sensitivity of feasible FE response slices
    #
    # Compare BO-86 and BO-82 using identical grids and constraints.
    # Same fitted FE GP; no refitting.
    # ===============================================================
    matched_current = float(composition_df.loc[
        composition_df[condition_col] == 86, CURRENT_DENSITY_COL
    ].iloc[0])
    baseline_raw_inputs = pd.DataFrame([{
        CURRENT_DENSITY_COL: abs(matched_current),
        "THF (%)": 100.0,
        "LiBF4 concentration (M)": 1.0,
        "LiTFSI concentration (M)": 0.0,
        "BuOH concentration (M)": 0.0,
        "PrOH concentration (M)": 0.0,
        "EtOH concentration (M)": 0.100,
    }], columns=FULL_FEATURE_COLS)
    baseline_reference_coordinates = raw_to_composition(baseline_raw_inputs)[0]
    baseline_reference_table = pd.DataFrame(
        [baseline_reference_coordinates], columns=COMPOSITION_FEATURE_COLS,
    )
    baseline_reference_table.insert(0, condition_col, "Baseline")
    RESPONSE_SLICE_REFERENCE_IDS = [82, 86, "Baseline"]

    response_slice_results = {}
    response_slice_reference_tables = []

    for reference_id in RESPONSE_SLICE_REFERENCE_IDS:
        if reference_id == "Baseline":
            reference_rows = baseline_reference_table.copy()
        else:
            reference_rows = composition_df.loc[
                composition_df[condition_col] == reference_id
            ].copy()

        if len(reference_rows) != 1:
            raise ValueError(
                f"Expected one row for reference BO-{reference_id}."
            )

        reference_coordinates = reference_rows[
            COMPOSITION_FEATURE_COLS
        ].to_numpy(dtype=float)[0]

        reference_raw = composition_to_raw(reference_coordinates)

        if not slice_stage2_feasible(reference_raw).all():
            raise ValueError(
                f"BO-{reference_id} is outside the specified Stage 2 constraints."
            )

        # Keep the donor substitution path anchored at the reference:
        # BuOH replaces EtOH, with PrOH fixed at zero.
        if not (
            reference_coordinates[4] > 0
            and -1e-10 <= reference_coordinates[5] <= 1.0 + 1e-10
            and np.isclose(reference_coordinates[6], 0.0, atol=1e-10)
        ):
            raise ValueError(
                f"BO-{reference_id} does not have a positive-loading, "
                "BuOH/EtOH donor reference with zero PrOH. Specify its donor path explicitly."
            )

        response_slice_reference_tables.append(reference_rows)

        response_slice_results[reference_id] = [
            evaluate_composition_slice(
                "Solvent-current",
                np.linspace(0.0, 100.0, COMPOSITION_SLICE_N),
                np.linspace(1.0, 20.0, COMPOSITION_SLICE_N),
                configure_solvent_current,
                r"THF (vol.\%; DG balance)",
                r"$|j|$ (mA cm$^{-2}$)",
                (
                    reference_coordinates[1],
                    abs(reference_coordinates[0]),
                ),
                reference_coordinates=reference_coordinates,
                reference_id=reference_id,
            ),
            evaluate_composition_slice(
                "Salt loading and balance",
                np.linspace(0.0, 1.0, COMPOSITION_SLICE_N),
                np.linspace(0.8, 2.0, COMPOSITION_SLICE_N),
                configure_salt,
                r"LiBF$_4$ fraction of total salt",
                "Total salt (M)",
                (
                    reference_coordinates[3],
                    reference_coordinates[2],
                ),
                reference_coordinates=reference_coordinates,
                reference_id=reference_id,
            ),
            evaluate_composition_slice(
                "1-BuOH replacing EtOH",
                np.linspace(0.0, 1.0, COMPOSITION_SLICE_N),
                np.linspace(0.0, 150.0, COMPOSITION_SLICE_N),
                configure_donor,
                "1-BuOH fraction of donor",
                "Total proton donor (mM)",
                (
                    reference_coordinates[5],
                    reference_coordinates[4] * 1000.0,
                ),
                reference_coordinates=reference_coordinates,
                reference_id=reference_id,
            ),
        ]

    response_plot_rows = []
    response_source_tables = []
    for reference_id, response in [(82, "FE"), (86, "Rate"), ("Baseline", "Rate")]:
        row_items = []
        for item in response_slice_results[reference_id]:
            table = item["export"].copy()
            table["Response"] = response
            current = table[CURRENT_DENSITY_COL].to_numpy(dtype=float)
            table["Predicted rate mean (nmol/s/cm^2)"] = calculate_nh3_rate(
                table["Predicted FE mean (%)"].to_numpy(), current,
            )
            table["Rate latent posterior SD (nmol/s/cm^2)"] = calculate_nh3_rate(
                table["FE latent posterior SD (percentage points)"].to_numpy(), current,
            )
            if response == "FE":
                mean_surface, sd_surface = item["mean"], item["sd"]
            else:
                mean_surface = table["Predicted rate mean (nmol/s/cm^2)"].to_numpy().reshape(item["mean"].shape)
                sd_surface = table["Rate latent posterior SD (nmol/s/cm^2)"].to_numpy().reshape(item["sd"].shape)
            row_items.append(dict(item, surface=mean_surface, uncertainty=sd_surface))
            response_source_tables.append(table)
        response_plot_rows.append((reference_id, response, row_items))

    return {"response_slice_grid":pd.concat(response_source_tables,ignore_index=True),
            "response_slice_references":pd.concat(response_slice_reference_tables,ignore_index=True)}


def local_sensitivity(model, campaign, *, basis="observed", fraction=0.05,
                      lower_bounds=None, upper_bounds=None):
    """Original OAT and 0-5% response curves around the measured Pareto set."""
    if not np.isfinite(fraction) or fraction <= 0:
        raise ValueError("fraction must be finite and positive.")
    fe_gp_full = model
    model_df = campaign.model_rows
    model_df_unique = campaign.condition_means
    condition_col = CONDITION
    # ===============================================================
    # Local perturbation analysis using the existing full FE GP.
    # Rate is calculated from predicted FE and each input's current density.
    # No models are fitted here.
    # ===============================================================



    feature_cols = list(FULL_FEATURE_COLS)
    target_cols = [FE_TARGET_COL, RATE_TARGET_COL]

    if list(fe_gp_full.manuscript_feature_cols) != feature_cols:
        raise ValueError(
            "The fitted FE model has an unexpected feature order."
        )

    # -----------------------------
    # User-adjustable settings
    # -----------------------------
    pareto_basis = basis   # "observed" = Fig. 2-style Pareto front; "predicted" = GP-mean Pareto front
    perturbation_fraction = fraction  # one-at-a-time step as fraction of each feature range
    enforce_sum_constraints = True

    # Set to None to use fractional steps for every variable.
    # Alternatively, uncomment/edit this dictionary to use absolute local perturbations.
    step_abs_by_feature = None
    # step_abs_by_feature = {
    #     "Current density (mA/cm^2)": 0.50,          # mA/cm^2
    #     "THF (%)": 5.0,                           # vol %
    #     "LiBF4 concentration (M)": 0.05,       # M
    #     "LiTFSI concentration (M)": 0.05,         # M
    #     "BuOH concentration (M)": 0.005,        # M = 5 mM
    #     "PrOH concentration (M)": 0.005,        # M = 5 mM
    #     "EtOH concentration (M)": 0.005,          # M = 5 mM
    # }

    # Feasible total-composition constraints used in the BO campaign.
    salt_sum_min, salt_sum_max = 0.50, 2      # LiBF4 + LiTFSI, M
    donor_sum_min, donor_sum_max = 0.00, 0.15     # BuOH + PrOH + EtOH, M

    observed_lb = model_df_unique[feature_cols].min().to_numpy(dtype=float)
    observed_ub = model_df_unique[feature_cols].max().to_numpy(dtype=float)

    bo_lb = np.asarray(observed_lb if lower_bounds is None else lower_bounds, dtype=float).reshape(-1)
    bo_ub = np.asarray(observed_ub if upper_bounds is None else upper_bounds, dtype=float).reshape(-1)

    # Guard against stale/incompatible lb/ub variables.
    if bo_lb.size != len(feature_cols) or bo_ub.size != len(feature_cols):
        bo_lb, bo_ub = observed_lb.copy(), observed_ub.copy()

    # Make sure every observed Pareto condition is inside the bounds.
    bo_lb = np.minimum(bo_lb, observed_lb)
    bo_ub = np.maximum(bo_ub, observed_ub)

    feature_ranges = np.maximum(bo_ub - bo_lb, 1e-12)

    if step_abs_by_feature is None:
        step_sizes = feature_ranges * perturbation_fraction
    else:
        step_sizes = np.array([step_abs_by_feature.get(c, feature_ranges[i] * perturbation_fraction)
                               for i, c in enumerate(feature_cols)], dtype=float)

    # -----------------------------
    # Helper functions
    # -----------------------------
    def gp_predict_mo_mean(X_np):
        """
        Return [predicted FE, Faraday-derived rate] at physical inputs.

        Column order must follow FULL_FEATURE_COLS.
        """
        predictions = predict_fe_rate(fe_gp_full, X_np)

        return np.column_stack([
            predictions["fe_mean"],
            predictions["rate_mean"],
        ])


    def pareto_mask_max(Y, tol=1e-10):
        """Pareto mask for maximization of all columns in Y."""
        Y = np.asarray(Y, dtype=float)
        keep = np.ones(Y.shape[0], dtype=bool)

        for i, y in enumerate(Y):
            if not np.all(np.isfinite(y)):
                keep[i] = False
                continue

            dominates_i = (
                np.all(Y >= y - tol, axis=1)
                & np.any(Y > y + tol, axis=1)
                & np.all(np.isfinite(Y), axis=1)
            )
            if np.any(dominates_i):
                keep[i] = False

        return keep


    def dominates_max(a, b, tol=1e-10):
        """True if a Pareto-dominates b for maximization."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return bool(np.all(a >= b - tol) and np.any(a > b + tol))


    def hypervolume_2d_max(Y, ref):
        """
        Exact 2D dominated hypervolume for maximization.
        Computes area dominated by nondominated points and bounded below by ref.
        """
        Y = np.asarray(Y, dtype=float)
        ref = np.asarray(ref, dtype=float)

        valid = np.all(np.isfinite(Y), axis=1)
        valid &= (Y[:, 0] > ref[0]) & (Y[:, 1] > ref[1])
        pts = Y[valid]

        if pts.size == 0:
            return 0.0

        pts = pts[pareto_mask_max(pts)]
        pts = pts[np.argsort(pts[:, 0])]  # ascending FE

        hv = 0.0
        prev_x = ref[0]
        for x, y in pts:
            dx = max(x - prev_x, 0.0)
            dy = max(y - ref[1], 0.0)
            hv += dx * dy
            prev_x = max(prev_x, x)

        return float(hv)


    def feasible_coordinate_range(x, feature_idx):
        """
        Bounds for one coordinate while all other coordinates are held fixed.
        Includes box bounds and optional salt/donor sum constraints.
        """
        lo = float(bo_lb[feature_idx])
        hi = float(bo_ub[feature_idx])

        if not enforce_sum_constraints:
            return lo, hi

        feature_name = feature_cols[feature_idx]

        # Match by column names to avoid relying only on index positions.
        salt_features = ["LiBF4 concentration (M)", "LiTFSI concentration (M)"]
        donor_features = ["BuOH concentration (M)", "PrOH concentration (M)", "EtOH concentration (M)"]

        if feature_name in salt_features:
            idxs = [feature_cols.index(c) for c in salt_features if c in feature_cols]
            other_sum = sum(x[j] for j in idxs if j != feature_idx)
            lo = max(lo, salt_sum_min - other_sum)
            hi = min(hi, salt_sum_max - other_sum)

        elif feature_name in donor_features:
            idxs = [feature_cols.index(c) for c in donor_features if c in feature_cols]
            other_sum = sum(x[j] for j in idxs if j != feature_idx)
            lo = max(lo, donor_sum_min - other_sum)
            hi = min(hi, donor_sum_max - other_sum)

        return lo, hi

    # -----------------------------
    # Select Pareto conditions
    # -----------------------------
    X_unique = model_df_unique[feature_cols].to_numpy(dtype=float)
    Y_unique_obs = model_df_unique[target_cols].to_numpy(dtype=float)
    Y_unique_pred = gp_predict_mo_mean(X_unique)

    if pareto_basis.lower() == "observed":
        unique_pareto_mask = pareto_mask_max(Y_unique_obs)
    elif pareto_basis.lower() == "predicted":
        unique_pareto_mask = pareto_mask_max(Y_unique_pred)
    else:
        raise ValueError("pareto_basis must be either 'observed' or 'predicted'.")

    pareto_conditions = model_df_unique.loc[unique_pareto_mask, [condition_col] + feature_cols + target_cols].copy()
    pareto_conditions = pareto_conditions.sort_values(target_cols[0]).reset_index(drop=True)

    pareto_X = pareto_conditions[feature_cols].to_numpy(dtype=float)
    pareto_Y_pred = gp_predict_mo_mean(pareto_X)

    # Check the conversion at the baseline conditions and at changed currents.
    current_idx = feature_cols.index(CURRENT_DENSITY_COL)

    conversion_check_X = np.vstack([
        pareto_X,
        pareto_X.copy(),
    ])
    conversion_check_X[len(pareto_X):, current_idx] *= 1.01

    conversion_check_Y = gp_predict_mo_mean(conversion_check_X)

    expected_check_rate = calculate_nh3_rate(
        conversion_check_Y[:, 0],
        conversion_check_X[:, current_idx],
    )

    if not np.allclose(
        conversion_check_Y[:, 1],
        expected_check_rate,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Perturbation predictions violate the FE–rate relationship."
        )


    # Explicit reference in physical FE (%) and rate (nmol s^-1 cm^-2) units.
    hv_ref_point = np.array([0.0, 0.0], dtype=float)

    baseline_hv = hypervolume_2d_max(pareto_Y_pred, hv_ref_point)

    baseline_hv_contrib = []
    for i in range(len(pareto_Y_pred)):
        y_without_i = np.delete(pareto_Y_pred, i, axis=0)
        baseline_hv_contrib.append(baseline_hv - hypervolume_2d_max(y_without_i, hv_ref_point))
    baseline_hv_contrib = np.asarray(baseline_hv_contrib, dtype=float)

    pareto_base_summary = pareto_conditions.copy()
    pareto_base_summary["Predicted FE (%)"] = pareto_Y_pred[:, 0]
    pareto_base_summary["Predicted rate (nmol/s/cm^2)"] = pareto_Y_pred[:, 1]
    pareto_base_summary["Baseline HV contribution"] = baseline_hv_contrib

    # -----------------------------
    # Local one-at-a-time perturbations
    # -----------------------------
    records = []

    for local_idx, (_, base_row) in enumerate(pareto_conditions.iterrows()):
        condition_id = base_row[condition_col]
        base_x = base_row[feature_cols].to_numpy(dtype=float)
        base_y = pareto_Y_pred[local_idx, :].copy()
        hv_without_base = hypervolume_2d_max(np.delete(pareto_Y_pred, local_idx, axis=0), hv_ref_point)

        for feat_idx, feat_name in enumerate(feature_cols):
            step = float(step_sizes[feat_idx])
            coord_lo, coord_hi = feasible_coordinate_range(base_x, feat_idx)

            for direction_label, direction_sign in [("minus", -1.0), ("plus", 1.0)]:
                target_value = base_x[feat_idx] + direction_sign * step
                pert_x = base_x.copy()
                pert_x[feat_idx] = np.clip(target_value, coord_lo, coord_hi)
                actual_delta = pert_x[feat_idx] - base_x[feat_idx]

                base_record = {
                    condition_col: condition_id,
                    "Local Pareto index": local_idx,
                    "Feature": feat_name,
                    "Direction": direction_label,
                    "Requested step": direction_sign * step,
                    "Actual delta variable": actual_delta,
                    "Base variable value": base_x[feat_idx],
                    "Perturbed variable value": pert_x[feat_idx],
                    "Coordinate feasible lower": coord_lo,
                    "Coordinate feasible upper": coord_hi,
                    "Base predicted FE (%)": base_y[0],
                    "Base predicted rate (nmol/s/cm^2)": base_y[1],
                    "Base HV contribution": baseline_hv_contrib[local_idx],
                }

                if np.isclose(actual_delta, 0.0, atol=1e-12):
                    records.append({
                        **base_record,
                        "Valid perturbation": False,
                        "Perturbed predicted FE (%)": np.nan,
                        "Perturbed predicted rate (nmol/s/cm^2)": np.nan,
                        "Delta predicted FE (%)": np.nan,
                        "Delta predicted rate (nmol/s/cm^2)": np.nan,
                        "Perturbed HV contribution": np.nan,
                        "Delta HV contribution": np.nan,
                        "Total HV if replaced": np.nan,
                        "Delta total HV if replaced": np.nan,
                        "Pareto after replacement?": False,
                        "Dominance outcome": "No feasible movement from this boundary point",
                    })
                    continue

                pert_y = gp_predict_mo_mean(pert_x.reshape(1, -1))[0]

                replaced_Y = pareto_Y_pred.copy()
                replaced_Y[local_idx, :] = pert_y

                total_hv_replaced = hypervolume_2d_max(replaced_Y, hv_ref_point)
                pert_hv_contrib = total_hv_replaced - hv_without_base
                replaced_pareto_mask = pareto_mask_max(replaced_Y)
                pareto_after_replacement = bool(replaced_pareto_mask[local_idx])

                other_Y = np.delete(pareto_Y_pred, local_idx, axis=0)
                dominated_by_other = any(dominates_max(y_other, pert_y) for y_other in other_Y)

                if dominates_max(pert_y, base_y):
                    dominance_outcome = "Perturbed dominates base"
                elif dominates_max(base_y, pert_y):
                    dominance_outcome = "Base dominates perturbed"
                elif dominated_by_other:
                    dominance_outcome = "Dominated by another original Pareto point"
                elif pareto_after_replacement:
                    dominance_outcome = "Nondominated tradeoff after replacement"
                else:
                    dominance_outcome = "Dominated after replacement"

                records.append({
                    **base_record,
                    "Valid perturbation": True,
                    "Perturbed predicted FE (%)": pert_y[0],
                    "Perturbed predicted rate (nmol/s/cm^2)": pert_y[1],
                    "Delta predicted FE (%)": pert_y[0] - base_y[0],
                    "Delta predicted rate (nmol/s/cm^2)": pert_y[1] - base_y[1],
                    "Perturbed HV contribution": pert_hv_contrib,
                    "Delta HV contribution": pert_hv_contrib - baseline_hv_contrib[local_idx],
                    "Total HV if replaced": total_hv_replaced,
                    "Delta total HV if replaced": total_hv_replaced - baseline_hv,
                    "Pareto after replacement?": pareto_after_replacement,
                    "Dominance outcome": dominance_outcome,
                })

    oat_perturbation_results = pd.DataFrame(records)

    valid_oat = oat_perturbation_results[oat_perturbation_results["Valid perturbation"]].copy()

    sensitivity_summary = (
        valid_oat
        .groupby("Feature", as_index=False)
        .agg(
            N=("Feature", "size"),
            Mean_abs_actual_delta=("Actual delta variable", lambda s: np.mean(np.abs(s))),
            Mean_abs_delta_FE=("Delta predicted FE (%)", lambda s: np.mean(np.abs(s))),
            Mean_abs_delta_rate=("Delta predicted rate (nmol/s/cm^2)", lambda s: np.mean(np.abs(s))),
            Mean_abs_delta_HV_contribution=("Delta HV contribution", lambda s: np.mean(np.abs(s))),
            Mean_delta_HV_contribution=("Delta HV contribution", "mean"),
            Mean_abs_delta_total_HV_if_replaced=("Delta total HV if replaced", lambda s: np.mean(np.abs(s))),
            Fraction_nondominated_after_replacement=("Pareto after replacement?", "mean"),
            Fraction_improves_total_HV=("Delta total HV if replaced", lambda s: np.mean(s > 0)),
        )
    )

    sensitivity_summary["Mean_abs_delta_FE_per_abs_variable_delta"] = (
        sensitivity_summary["Mean_abs_delta_FE"] / sensitivity_summary["Mean_abs_actual_delta"]
    )
    sensitivity_summary["Mean_abs_delta_rate_per_abs_variable_delta"] = (
        sensitivity_summary["Mean_abs_delta_rate"] / sensitivity_summary["Mean_abs_actual_delta"]
    )

    ranked_sensitivity = sensitivity_summary.sort_values(
        "Mean_abs_delta_total_HV_if_replaced",
        ascending=False
    ).reset_index(drop=True)
    ranked_sensitivity.insert(0, "Rank by mean |Delta total HV|", np.arange(1, len(ranked_sensitivity) + 1))

    # -----------------------------
    # Display and export
    # -----------------------------

    # -----------------------------
    # Perturbation-size response curves around Pareto conditions
    # -----------------------------
    # x-axis: normalized perturbation size, |Delta x_i| / feature range
    # y-axis: predicted change in FE, rate, or total Pareto-set HV
    # Each curve corresponds to one input variable.

    # User-adjustable settings
    normalized_perturbation_grid = np.linspace(0.0, 0.05, 26)  # 0 to 5% of each variable range
    perturbation_directions = [-1.0, 1.0]                      # average over minus and plus perturbations

    # Options:
    #   "mean_signed" = average signed Delta y over +/- perturbations
    #   "mean_abs"    = average |Delta y| over +/- perturbations
    #   "mean_loss"   = average negative component only; useful for showing movement away from ridge
    change_summary_mode = "mean_signed"

    response_curve_records = []

    for feat_idx, feat_name in enumerate(feature_cols):
        feature_range = float(feature_ranges[feat_idx])

        for eps_norm_requested in normalized_perturbation_grid:
            requested_step = eps_norm_requested * feature_range

            for local_idx, (_, base_row) in enumerate(pareto_conditions.iterrows()):
                condition_id = base_row[condition_col]
                base_x = base_row[feature_cols].to_numpy(dtype=float)
                base_y = pareto_Y_pred[local_idx, :].copy()

                for direction_sign in perturbation_directions:
                    pert_x = base_x.copy()

                    coord_lo, coord_hi = feasible_coordinate_range(base_x, feat_idx)

                    target_value = base_x[feat_idx] + direction_sign * requested_step
                    pert_x[feat_idx] = np.clip(target_value, coord_lo, coord_hi)

                    actual_delta = pert_x[feat_idx] - base_x[feat_idx]
                    actual_eps_norm = actual_delta / feature_range

                    if eps_norm_requested > 0 and np.isclose(actual_delta, 0.0, atol=1e-12):
                        continue

                    pert_y = gp_predict_mo_mean(pert_x.reshape(1, -1))[0]

                    replaced_Y = pareto_Y_pred.copy()
                    replaced_Y[local_idx, :] = pert_y

                    total_hv_replaced = hypervolume_2d_max(replaced_Y, hv_ref_point)
                    delta_total_hv = total_hv_replaced - baseline_hv

                    response_curve_records.append({
                        condition_col: condition_id,
                        "Local Pareto index": local_idx,
                        "Feature": feat_name,
                        "Direction sign": direction_sign,
                        "Requested normalized perturbation": eps_norm_requested,
                        "Actual normalized perturbation": actual_eps_norm,
                        "Abs actual normalized perturbation": abs(actual_eps_norm),
                        "Requested physical perturbation": direction_sign * requested_step,
                        "Actual physical perturbation": actual_delta,
                        "Base variable value": base_x[feat_idx],
                        "Perturbed variable value": pert_x[feat_idx],
                        "Base predicted FE (%)": base_y[0],
                        "Base predicted rate (nmol/s/cm^2)": base_y[1],
                        "Perturbed predicted FE (%)": pert_y[0],
                        "Perturbed predicted rate (nmol/s/cm^2)": pert_y[1],
                        "Delta predicted FE (%)": pert_y[0] - base_y[0],
                        "Delta predicted rate (nmol/s/cm^2)": pert_y[1] - base_y[1],
                        "Total HV if replaced": total_hv_replaced,
                        "Delta total HV if replaced": delta_total_hv,
                    })

    perturbation_response_curves_raw = pd.DataFrame(response_curve_records)


    def summarize_change(series, mode):
        """Summarize signed, absolute, or negative-only response changes."""
        s = pd.Series(series).dropna().to_numpy(dtype=float)

        if len(s) == 0:
            return np.nan

        if mode == "mean_signed":
            return np.mean(s)
        elif mode == "mean_abs":
            return np.mean(np.abs(s))
        elif mode == "mean_loss":
            return np.mean(np.minimum(s, 0.0))
        else:
            raise ValueError("change_summary_mode must be 'mean_signed', 'mean_abs', or 'mean_loss'.")


    perturbation_response_curves = (
        perturbation_response_curves_raw
        .groupby(["Feature", "Requested normalized perturbation"], as_index=False)
        .agg(
            N=("Feature", "size"),
            Mean_abs_actual_normalized_perturbation=("Abs actual normalized perturbation", "mean"),
            Mean_delta_FE=("Delta predicted FE (%)", lambda s: summarize_change(s, change_summary_mode)),
            Mean_delta_rate=("Delta predicted rate (nmol/s/cm^2)", lambda s: summarize_change(s, change_summary_mode)),
            Mean_delta_total_HV=("Delta total HV if replaced", lambda s: summarize_change(s, change_summary_mode)),
            SD_delta_FE=("Delta predicted FE (%)", "std"),
            SD_delta_rate=("Delta predicted rate (nmol/s/cm^2)", "std"),
            SD_delta_total_HV=("Delta total HV if replaced", "std"),
        )
    )
    bounds = pd.DataFrame({"Feature":feature_cols,"Lower bound":bo_lb,"Upper bound":bo_ub,
                           "OAT step":step_sizes})
    return {"local_base_summary":pareto_base_summary,
            "local_perturbations":oat_perturbation_results,
            "local_sensitivity_summary":ranked_sensitivity,
            "local_response_curves_raw":perturbation_response_curves_raw,
            "local_response_curves":perturbation_response_curves,
            "local_bounds":bounds}


def pca_analysis(campaign):
    """Original seven-input PCA: condition averages, StandardScaler, then PCA."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    rows = campaign.retained[list(FULL_FEATURE_COLS)].copy()
    rows.insert(0, "Exp #", [f"BO-{int(x)}" for x in campaign.retained[CONDITION]])
    averages = rows.groupby("Exp #",dropna=False).mean(numeric_only=True)
    averages = averages.dropna(axis=1,how="all").dropna(axis=0,how="any")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(averages)
    two = PCA(n_components=2)
    scores = two.fit_transform(scaled)
    full = PCA().fit(scaled)
    score_table = pd.DataFrame(scores,columns=["PC1","PC2"],index=averages.index)
    numbers = averages.index.to_series().str.extract(r"BO-(\d+)",expand=False).astype(float)
    score_table["Group"] = ["Random" if x<=16 else "BR-FPS" if x<=32 or 65<=x<=80 else "BO" for x in numbers]
    loadings = pd.DataFrame({"Feature":averages.columns,"PC1":two.components_[0],"PC2":two.components_[1],
                             "Scaling mean":scaler.mean_,"Scaling SD":scaler.scale_})
    variance = pd.DataFrame({"PC":np.arange(1,len(full.explained_variance_ratio_)+1),
                            "Explained variance fraction":full.explained_variance_ratio_})
    return {"pca_inputs":averages.reset_index(), "pca_scores":score_table.reset_index(),
            "pca_loadings":loadings,"pca_variance":variance}


def run_interpretation(root=None, *, analyses=("shap","slices","pca","local"),
                       model=None, grid_n=60, output_dir=None, verbose=True):
    """Run selected analyses; reuse supplied fitted model or fit the existing FE GP."""
    analyses=tuple(analyses)
    if set(analyses)-{"shap","slices","pca","local"}:
        raise ValueError("Unknown analysis; select shap, slices, pca and/or local.")
    campaign=load_campaign(root)
    tables={}
    if any(name!="pca" for name in analyses) and model is None:
        if verbose:builtins.print("Fitting existing full-data FE model (259 retained rows)...",flush=True)
        model=fit_fe_gp(campaign.model_rows,campaign.model_rows[FE_TARGET_COL],FULL_FEATURE_COLS,seed=12345)
    for name in analyses:
        if verbose:builtins.print("Running "+name+"...",flush=True)
        if name=="shap":tables.update(compute_shap(model,campaign,verbose=verbose))
        elif name=="slices":tables.update(response_slices(model,campaign,grid_n=grid_n))
        elif name=="pca":tables.update(pca_analysis(campaign))
        elif name=="local":tables.update(local_sensitivity(model,campaign))
    if output_dir is not None:
        destination=Path(output_dir)
        if not destination.is_absolute():destination=(Path(root) if root is not None else repository_root())/destination
        destination.mkdir(parents=True,exist_ok=True)
        for name,table in tables.items():table.to_csv(destination/(name+".csv"),index=False)
    return tables


def plot_shap(tables, response="FE", root=None, *, use_tex=None):
    import matplotlib.pyplot as plt
    root=Path(root).resolve() if root is not None else repository_root()
    style=root/"styles/manuscript.mplstyle"
    if not style.is_file():raise FileNotFoundError(style)
    with plt.style.context(str(style)):
        with plt.rc_context({} if use_tex is None else {"text.usetex":bool(use_tex)}):
            colors=plt.rcParams["axes.prop_cycle"].by_key()["color"]
            fig,axes=plt.subplots(2,4,figsize=(6.5,6.5*2/3))
            imp=tables["shap_importance"].loc[lambda x:x.Response.eq(response)].set_index("Coordinate").loc[MAINPAPER_FEATURE_COLS]
            labels=["Current density","THF","Total salt","LiBF4 fraction","Total donor","1-BuOH fraction","EtOH fraction"]
            axes.flat[0].barh(range(7),imp["Normalized mean absolute SHAP"],color=colors[:7])
            axes.flat[0].set_yticks(range(7),labels)
            axes.flat[0].invert_yaxis()
            axes.flat[0].set_xlabel("Normalized mean\n|SHAP value|")
            axis_labels = [r"Current density"+"\n"+r"(mA cm$^{-2}$)", "THF (vol.%)", "Total salt (M)",
                "LiBF4 fraction", "Total donor (mM)", "1-BuOH fraction", "EtOH fraction"]
            if plt.rcParams["text.usetex"]:axis_labels[1] = r"THF (vol.\%)"
            selected = tables["shap_values"].loc[lambda x:x.Response.eq(response)]
            span = max(float(selected["SHAP value"].abs().max()), 1e-12)
            for column,(axis,coordinate) in enumerate(zip(list(axes.flat)[1:],MAINPAPER_FEATURE_COLS)):
                d=tables["shap_values"].loc[lambda x:x.Response.eq(response)&x.Coordinate.eq(coordinate)]
                x=d["Input value"].to_numpy().copy()
                if column==0:x=-np.abs(x)
                if column==4:x*=1000
                axis.scatter(x,d["SHAP value"],color=colors[column],s=4,alpha=.5,edgecolors="none")
                axis.axhline(0,color="0.5",lw=.5,ls="--")
                axis.set_xlabel(axis_labels[column])
                axis.set_ylim(-span, span)
                axis.set_ylabel("FE SHAP value (pp)" if response=="FE" else "Rate SHAP value\n"+r"(nmol s$^{-1}$ cm$^{-2}$)")
            fig.subplots_adjust(left=.13,right=.99,bottom=.16,top=.97,wspace=.78,hspace=.68)
    return fig,axes


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path)
    parser.add_argument("--analyses",nargs="+",choices=["shap","slices","pca","local"],default=["shap","slices","pca","local"])
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--plot",action="store_true",help="Display FE and rate SHAP summaries without saving figures.")
    parser.add_argument("--no-tex",action="store_true",help="Explicitly disable LaTeX for plots.")
    args=parser.parse_args(argv)
    tables=run_interpretation(args.root,analyses=args.analyses,output_dir=args.output_dir)
    for name,table in tables.items():builtins.print(f"{name}: {len(table)} rows")
    if args.plot:
        if "shap_values" not in tables:parser.error("--plot requires shap in --analyses.")
        for response in ["FE","Rate"]:plot_shap(tables,response,args.root,use_tex=False if args.no_tex else None)
        import matplotlib.pyplot as plt
        plt.show()
    return 0

if __name__=="__main__":raise SystemExit(main())
