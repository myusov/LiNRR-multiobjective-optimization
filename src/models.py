from __future__ import annotations
import argparse
from pathlib import Path
import random
import json
import platform
import numpy as np
import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize as BoNormalize
from botorch.models.transforms.outcome import Standardize as BoStandardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

try:
    from .preprocessing import load_campaign, FEATURE_COLUMNS, TARGET_COLUMNS, CONDITION
except ImportError:
    from preprocessing import load_campaign, FEATURE_COLUMNS, TARGET_COLUMNS, CONDITION

FARADAY_CONSTANT = 96485.33212
ELECTRONS_PER_NH3 = 3
CURRENT_DENSITY_COL = FEATURE_COLUMNS[0]
FULL_FEATURE_COLS = list(FEATURE_COLUMNS)
FIG1_FEATURE_COLS = [CURRENT_DENSITY_COL, "BuOH concentration (M)"]
FE_TARGET_COL, RATE_TARGET_COL = TARGET_COLUMNS
tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}

def nh3_rate_factor(current_density_mA_cm2):
    """
    Multiplicative factor converting FE in percentage points
    to NH3 rate in nmol s^-1 cm^-2.
    """
    current = np.asarray(current_density_mA_cm2, dtype=np.float64)
    return np.abs(current) * 0.001 / (ELECTRONS_PER_NH3 * FARADAY_CONSTANT) * 1000000000.0 / 100.0

def calculate_nh3_rate(fe_percent, current_density_mA_cm2):
    """Calculate NH3 rate from FE (%) and current density (mA cm^-2)."""
    return np.asarray(fe_percent, dtype=np.float64) * nh3_rate_factor(current_density_mA_cm2)

def _feature_array(features, feature_cols):
    """
    Return a finite 2D array in the model's declared feature order.

    DataFrames are reordered by column name.
    Arrays must already follow feature_cols.
    A 1D array is interpreted as one input condition.
    """
    feature_cols = list(feature_cols)
    if isinstance(features, pd.DataFrame):
        array = features.loc[:, feature_cols].to_numpy(dtype=np.float64)
    else:
        array = np.asarray(features, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != len(feature_cols):
        raise ValueError(f'Expected an input array with {len(feature_cols)} columns in this order: {feature_cols}. Got shape {array.shape}.')
    if len(array) == 0:
        raise ValueError('Input features must contain at least one row.')
    if not np.isfinite(array).all():
        raise ValueError('Input features contain missing or nonfinite values.')
    # pandas 3 may expose read-only arrays. A writable copy avoids PyTorch's
    # shared-memory warning without changing any numerical input values.
    return array.copy(order="K")

def fit_fe_gp(features, fe_percent, feature_cols, seed=12345):
    """
    Fit a single FE GP.

    Training targets are FE percentage values.
    Missing rows must be handled explicitly before calling this function.
    """
    feature_cols = list(feature_cols)
    x_array = _feature_array(features, feature_cols)
    y_array = np.asarray(fe_percent, dtype=np.float64).reshape(-1, 1)
    if len(y_array) != len(x_array):
        raise ValueError('FE targets and input features have different lengths.')
    if len(y_array) < 2:
        raise ValueError('At least two training rows are required.')
    if not np.isfinite(y_array).all():
        raise ValueError('FE targets contain missing or nonfinite values.')
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    x_tensor = torch.tensor(x_array, **tkwargs)
    y_tensor = torch.tensor(y_array, **tkwargs)
    fe_gp = SingleTaskGP(x_tensor, y_tensor, input_transform=BoNormalize(d=len(feature_cols)), outcome_transform=BoStandardize(m=1)).to(**tkwargs)
    mll = ExactMarginalLogLikelihood(fe_gp.likelihood, fe_gp)
    fit_gpytorch_mll(mll)
    fe_gp.eval()
    fe_gp.likelihood.eval()
    fe_gp.manuscript_feature_cols = tuple(feature_cols)
    fe_gp.manuscript_response = "FE"
    return fe_gp

def predict_fe_rate(fe_gp, features, batch_size=4096):
    """
    Predict FE and Faraday-derived rate from one fitted FE GP.

    Returns a dictionary of 1D NumPy arrays:
        fe_mean:    FE posterior mean (%)
        fe_sd:      FE posterior standard deviation (percentage points)
        rate_mean:  derived-rate posterior mean (nmol s^-1 cm^-2)
        rate_sd:    derived-rate posterior standard deviation
                    (nmol s^-1 cm^-2)
    """
    if getattr(fe_gp, 'manuscript_response', 'FE') != 'FE':
        raise ValueError('Derived-rate prediction requires an FE model, not the rate comparator.')
    if not hasattr(fe_gp, 'manuscript_feature_cols'):
        raise ValueError('Fit this FE GP with fit_fe_gp before predicting.')
    feature_cols = list(fe_gp.manuscript_feature_cols)
    x_array = _feature_array(features, feature_cols)
    if CURRENT_DENSITY_COL not in feature_cols:
        raise ValueError('The FE model must include current density as an input.')
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer.')
    current_idx = feature_cols.index(CURRENT_DENSITY_COL)
    train_inputs = fe_gp.train_inputs[0]
    mean_parts = []
    variance_parts = []
    fe_gp.eval()
    with torch.no_grad():
        for start in range(0, len(x_array), batch_size):
            x_tensor = torch.as_tensor(x_array[start:start + batch_size], dtype=train_inputs.dtype, device=train_inputs.device)
            posterior = fe_gp.posterior(x_tensor, observation_noise=False)
            mean_parts.append(posterior.mean.detach().cpu().numpy().reshape(-1))
            variance_parts.append(posterior.variance.detach().cpu().numpy().reshape(-1))
    fe_mean = np.concatenate(mean_parts)
    fe_variance = np.concatenate(variance_parts)
    fe_sd = np.sqrt(np.maximum(fe_variance, 0.0))
    rate_factor = nh3_rate_factor(x_array[:, current_idx])
    return {'fe_mean': fe_mean, 'fe_sd': fe_sd, 'rate_mean': rate_factor * fe_mean, 'rate_sd': rate_factor * fe_sd}

def predict_fe_mean(fe_gp, features):
    """FE posterior mean; suitable for plotting and SHAP."""
    return predict_fe_rate(fe_gp, features)['fe_mean']

def predict_rate_from_fe(fe_gp, features):
    """Faraday-derived rate posterior mean; suitable for plotting and SHAP."""
    return predict_fe_rate(fe_gp, features)['rate_mean']


def fit_rate_comparator(features, recorded_rate, seed=12345, feature_cols=FULL_FEATURE_COLS):
    """Same GP architecture and fitting as FE, with recorded rate as the target.

    This is a comparator, not the manuscript's primary rate model.
    """
    model = fit_fe_gp(features, recorded_rate, feature_cols, seed=seed)
    model.manuscript_response = "Recorded rate"
    return model


def predict_rate_comparator(model, features, *, observation_noise=False):
    """Independent rate mean/SD in nmol s^-1 cm^-2, with explicit noise choice."""
    if getattr(model, "manuscript_response", None) != "Recorded rate":
        raise ValueError("Supply a model fitted with fit_rate_comparator.")
    x = _feature_array(features, model.manuscript_feature_cols)
    training = model.train_inputs[0]
    tensor = torch.as_tensor(x, dtype=training.dtype, device=training.device)
    model.eval()
    model.likelihood.eval()
    with torch.no_grad():
        posterior = model.posterior(tensor, observation_noise=observation_noise)
        mean = posterior.mean.detach().cpu().numpy().reshape(-1)
        variance = posterior.variance.detach().cpu().numpy().reshape(-1)
    return {"rate_mean": mean, "rate_sd": np.sqrt(np.maximum(variance, 0.0))}


def joint_fe_rate_posterior(fe_gp, features, *, observation_noise=False):
    """Joint FE/derived-rate means and full covariance, in original output units."""
    if getattr(fe_gp, "manuscript_response", "FE") != "FE":
        raise ValueError("Supply an FE model, not the independent rate comparator.")
    columns = list(fe_gp.manuscript_feature_cols)
    if CURRENT_DENSITY_COL not in columns:
        raise ValueError("Derived rate requires current density among the model inputs.")
    x = _feature_array(features, columns)
    train = fe_gp.train_inputs[0]
    tensor = torch.as_tensor(x, dtype=train.dtype, device=train.device)
    fe_gp.eval()
    fe_gp.likelihood.eval()
    with torch.no_grad():
        posterior = fe_gp.posterior(tensor, observation_noise=observation_noise)
        mean = posterior.mean.detach().cpu().numpy().reshape(-1)
        covariance = posterior.distribution.covariance_matrix.detach().cpu().numpy()
    covariance = .5 * (covariance + covariance.T)
    factors = nh3_rate_factor(x[:, columns.index(CURRENT_DENSITY_COL)])
    return {"fe_mean": mean, "fe_covariance": covariance,
            "rate_mean": factors * mean,
            "rate_covariance": factors[:, None] * covariance * factors[None, :]}


def condition_mean_predictive_interval(fe_gp, features):
    """Approximate 95% predictive interval for a mean of replicate outcomes."""
    posterior = joint_fe_rate_posterior(fe_gp, features, observation_noise=True)
    n = len(posterior["fe_mean"])
    weights = np.full(n, 1.0 / n)
    result = {"n": n, "observation_noise": True}
    z = 1.959963984540054
    for name in ("fe", "rate"):
        mean = float(weights @ posterior[name + "_mean"])
        variance = float(weights @ posterior[name + "_covariance"] @ weights)
        if variance < -1e-8:
            raise RuntimeError("Unexpected negative predictive variance.")
        sd = float(np.sqrt(max(variance, 0.0)))
        result.update({name + "_mean": mean, name + "_sd": sd,
                       name + "_lower_95": mean - z * sd,
                       name + "_upper_95": mean + z * sd})
    return result


def fit_manuscript_models(root=None, *, seed=12345, outlier_policy="iqr", include_rate_comparator=False):
    """Load public campaign tables and fit on individual retained electrolysis rows.

    Returns (CampaignData, FE model, optional rate comparator).
    """
    campaign = load_campaign(root, outlier_policy=outlier_policy)
    rows = campaign.model_rows
    fe = fit_fe_gp(rows, rows[FE_TARGET_COL], FULL_FEATURE_COLS, seed=seed)
    rate = (fit_rate_comparator(rows, rows[RATE_TARGET_COL], seed=seed)
            if include_rate_comparator else None)
    return campaign, fe, rate


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=root)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--outlier-policy", choices=["iqr", "all-successful"], default="iqr")
    parser.add_argument("--compare-rate", action="store_true", help="Also fit independent rate comparator.")
    parser.add_argument("--output-dir", type=Path, help="Optional output directory, relative to repository root.")
    args = parser.parse_args(argv)
    print("Fitting full-data FE GP on individual campaign electrolyses...", flush=True)
    data, fe, comparator = fit_manuscript_models(
        args.repository_root, seed=args.seed, outlier_policy=args.outlier_policy,
        include_rate_comparator=args.compare_rate)
    prediction = predict_fe_rate(fe, data.condition_means)
    table = data.condition_means[[CONDITION]].copy()
    table["Observed mean FE (%)"] = data.condition_means[FE_TARGET_COL]
    table["Observed mean rate (nmol/s/cm^2)"] = data.condition_means[RATE_TARGET_COL]
    for key, label in [("fe_mean", "Predicted FE mean (%)"),
                       ("fe_sd", "FE latent posterior SD (percentage points)"),
                       ("rate_mean", "Predicted derived rate mean (nmol/s/cm^2)"),
                       ("rate_sd", "Derived rate latent posterior SD (nmol/s/cm^2)")]:
        table[label] = prediction[key]
    if comparator is not None:
        comp = predict_rate_comparator(comparator, data.condition_means)
        table["Independent rate GP mean (nmol/s/cm^2)"] = comp["rate_mean"]
        table["Independent rate GP latent SD (nmol/s/cm^2)"] = comp["rate_sd"]
    print(f"Fitted {len(data.model_rows)} replicate rows; predicted {len(table)} condition-mean inputs.")
    print("Full-data retrospective predictions; these are NOT held-out validation results.")
    print(table.head().to_string(index=False))
    if args.output_dir:
        destination = args.output_dir if args.output_dir.is_absolute() else args.repository_root.resolve() / args.output_dir
        destination.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination / "full_data_model_predictions.csv", index=False, float_format="%.17g")
        import botorch, gpytorch, scipy
        metadata = {"seed": args.seed, "outlier_policy": args.outlier_policy,
                    "training_rows": len(data.model_rows), "feature_columns": FULL_FEATURE_COLS,
                    "analytical_variances_supplied": False,
                    "marginal_uncertainty": "latent, excluding observation noise",
                    "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                 "pandas": pd.__version__, "torch": torch.__version__,
                                 "botorch": botorch.__version__, "gpytorch": gpytorch.__version__,
                                 "scipy": scipy.__version__}}
        (destination / "model_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
        print("Exported predictions and run metadata to", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
