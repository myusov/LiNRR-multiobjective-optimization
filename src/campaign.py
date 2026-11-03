from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.distributions import Dirichlet

FEATURE_COLUMNS = (
    "Current density (mA/cm^2)", "THF (%)", "LiBF4 concentration (M)",
    "LiTFSI concentration (M)", "BuOH concentration (M)",
    "PrOH concentration (M)", "EtOH concentration (M)",
)
TARGET_COLUMNS = ("Total FE (%)", "Total rate (nmol/s/cm^2)")

def sample_initial(n_points: int, seed: int=0) -> torch.Tensor:
    """Original sampler; seeds NumPy only. Returns n_points by 7. Caller must set torch RNG for reproducibility."""
    np.random.seed(seed)
    X = []
    for _ in range(n_points):
        current = np.random.uniform(0.1, 10.0)
        thf = np.random.uniform(0.0, 100.0)
        salt_sum = np.random.uniform(0.5, 1.5)
        p = np.random.rand()
        li_bf4 = p * salt_sum
        li_tfsi = (1 - p) * salt_sum
        proton_sum = np.random.uniform(0.0, 0.15)
        d = Dirichlet(torch.ones(3))
        fractions = d.sample().numpy()
        buoh, proh, etoh = fractions * proton_sum
        X.append([current, thf, li_bf4, li_tfsi, buoh, proh, etoh])
    return torch.tensor(X, dtype=torch.double)

def sample_new_fps_on_sums_edge_penalized(X_existing: torch.Tensor, n_new: int=16, n_candidates: int=5000, seed: int=0, edge_weight: float=0.5, edge_power: float=1.0) -> torch.Tensor:
    """
    Farthest-point sampling in reduced space [current, %THF, total_salt, total_proton]
    with an edge penalty: candidates closer to any boundary are downweighted when
    we choose the next point.

    Bounds / constraints:
      current ∈ [1, 10], %THF ∈ [0, 100],
      total_salt ∈ [0.5, 2.0] (later split into LiBF4/LiTFSI),
      total_proton ∈ [0, 0.15] (later split via Dirichlet).
    Selection uses:
      score = (min^2 distance to selected set) * ((1 - edge_weight) + edge_weight * interiority^edge_power)
    where interiority = 2 * min(c, 1 - c) aggregated via min across dims (∈ [0,1]).
    """
    rng = np.random.default_rng(seed)
    lb_red = np.array([1.0, 0.0, 0.5, 0.0], dtype=float)
    ub_red = np.array([10.0, 100.0, 2.0, 0.15], dtype=float)
    span_red = ub_red - lb_red

    def sample_one_reduced():
        current = rng.uniform(1.0, 10.0)
        thf = rng.uniform(0.0, 100.0)
        salt_sum = rng.uniform(0.5, 2.0)
        proton_sum = rng.uniform(0.0, 0.15)
        return np.array([current, thf, salt_sum, proton_sum], dtype=float)
    C_red = np.vstack([sample_one_reduced() for _ in range(n_candidates)])
    C_red_norm = (C_red - lb_red) / span_red
    Xe = X_existing.detach().cpu().numpy().astype(float) if X_existing.numel() else np.empty((0, 7))
    if Xe.size:
        Xe_red = np.column_stack([Xe[:, 0], Xe[:, 1], Xe[:, 2] + Xe[:, 3], Xe[:, 4:7].sum(axis=1)])
        Xe_red_norm = (Xe_red - lb_red) / span_red
    else:
        Xe_red_norm = np.empty((0, 4))
    d2_current = np.full(C_red.shape[0], np.inf, dtype=float)
    for xb in Xe_red_norm:
        d2_current = np.minimum(d2_current, np.sum((C_red_norm - xb) ** 2, axis=1))
    margins = np.minimum(C_red_norm, 1.0 - C_red_norm)
    interiority = 2.0 * margins.min(axis=1)
    picked = []
    for _ in range(n_new):
        weights = 1.0 - edge_weight + edge_weight * interiority ** edge_power
        scores = d2_current * weights
        idx = int(np.argmax(scores))
        picked.append(idx)
        xb = C_red_norm[idx]
        d2_current = np.minimum(d2_current, np.sum((C_red_norm - xb) ** 2, axis=1))

    def expand_point(reduced_vec):
        current, thf, salt_sum, proton_sum = reduced_vec
        p_salt = rng.uniform()
        li_bf4 = p_salt * salt_sum
        li_tfsi = (1.0 - p_salt) * salt_sum
        fracs = Dirichlet(torch.ones(3)).sample().numpy()
        buoh, proh, etoh = fracs * proton_sum
        return np.array([current, thf, li_bf4, li_tfsi, buoh, proh, etoh], dtype=float)
    X_new = np.vstack([expand_point(C_red[i]) for i in picked])
    return torch.tensor(X_new, dtype=torch.double)

def sample_new_fps_on_sums_edge_penalized_stage3(X_existing: torch.Tensor, n_new: int=16, n_candidates: int=5000, seed: int=0, edge_weight: float=0.5, edge_power: float=1.0) -> torch.Tensor:
    """Original expanded-space FPS: current magnitude 1-20, THF 0-100, total salt 0.8-2 M, donor 0-0.15 M. Seeds NumPy only."""
    rng = np.random.default_rng(seed)
    lb_red = np.array([1.0, 0.0, 0.8, 0.0], dtype=float)
    ub_red = np.array([20, 100.0, 2.0, 0.15], dtype=float)
    span_red = ub_red - lb_red

    def sample_one_reduced():
        current = rng.uniform(1.0, 20.0)
        thf = rng.uniform(0.0, 100.0)
        salt_sum = rng.uniform(0.8, 2.0)
        proton_sum = rng.uniform(0.0, 0.15)
        return np.array([current, thf, salt_sum, proton_sum], dtype=float)
    C_red = np.vstack([sample_one_reduced() for _ in range(n_candidates)])
    C_red_norm = (C_red - lb_red) / span_red
    Xe = X_existing.detach().cpu().numpy().astype(float) if X_existing.numel() else np.empty((0, 7))
    if Xe.size:
        Xe_red = np.column_stack([Xe[:, 0], Xe[:, 1], Xe[:, 2] + Xe[:, 3], Xe[:, 4:7].sum(axis=1)])
        Xe_red_norm = (Xe_red - lb_red) / span_red
    else:
        Xe_red_norm = np.empty((0, 4))
    d2_current = np.full(C_red.shape[0], np.inf, dtype=float)
    for xb in Xe_red_norm:
        d2_current = np.minimum(d2_current, np.sum((C_red_norm - xb) ** 2, axis=1))
    margins = np.minimum(C_red_norm, 1.0 - C_red_norm)
    interiority = 2.0 * margins.min(axis=1)
    picked = []
    for _ in range(n_new):
        weights = 1.0 - edge_weight + edge_weight * interiority ** edge_power
        scores = d2_current * weights
        idx = int(np.argmax(scores))
        picked.append(idx)
        xb = C_red_norm[idx]
        d2_current = np.minimum(d2_current, np.sum((C_red_norm - xb) ** 2, axis=1))

    def expand_point(reduced_vec):
        current, thf, salt_sum, proton_sum = reduced_vec
        p_salt = rng.uniform()
        li_bf4 = p_salt * salt_sum
        li_tfsi = (1.0 - p_salt) * salt_sum
        fracs = Dirichlet(torch.ones(3)).sample().numpy()
        buoh, proh, etoh = fracs * proton_sum
        return np.array([current, thf, li_bf4, li_tfsi, buoh, proh, etoh], dtype=float)
    X_new = np.vstack([expand_point(C_red[i]) for i in picked])
    return torch.tensor(X_new, dtype=torch.double)

@dataclass(frozen=True)
class BOSettings:
    stage: str = "expanded"
    seed: int = 12345
    batch_size: int = 4
    mc_samples: int = 512
    num_restarts: int = 60
    raw_samples: int = 2048
    n_burnin: int = 2000
    n_thinning: int = 16
    batch_limit: int = 8
    maxiter: int = 1000
    ftol: float = 1e-4


def campaign_bounds(stage: str):
    """Return BO box bounds and minimum total salt, without changing inputs."""
    if stage not in {"initial", "expanded"}:
        raise ValueError("stage must be initial or expanded")
    lower = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    upper = [10.0 if stage == "initial" else 20.0, 100.0, 2.0, 2.0, .15, .15, .15]
    return torch.tensor([lower, upper], dtype=torch.double), .5 if stage == "initial" else .8


def load_training_snapshot(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    columns = [*FEATURE_COLUMNS, *TARGET_COLUMNS]
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"Training snapshot missing columns: {sorted(missing)}")
    table = table[columns].apply(pd.to_numeric, errors="raise")
    if len(table) < 2 or not np.isfinite(table.to_numpy()).all():
        raise ValueError("Training snapshot requires at least two complete finite rows.")
    if (table[FEATURE_COLUMNS[0]] < 0).any():
        raise ValueError("Use the original model's nonnegative current-density magnitude.")
    return table


def fit_campaign_model(training: pd.DataFrame, *, seed: int = 12345):
    """Original separate FE and rate SingleTaskGPs, fitted on replicate rows.

    BoTorch's default mean, kernel, likelihood and priors are version-dependent.    
    """
    from botorch.models import SingleTaskGP, ModelListGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    columns = [*FEATURE_COLUMNS, *TARGET_COLUMNS]
    values = training[columns].to_numpy(dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Provide at least two finite training rows.")
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X = torch.tensor(values[:, :7], dtype=torch.double, device="cpu")
    Y = torch.tensor(values[:, 7:], dtype=torch.double, device="cpu")
    models = []
    for i in range(Y.shape[-1]):
        gp = SingleTaskGP(X, Y[..., [i]], input_transform=Normalize(d=7),
                          outcome_transform=Standardize(m=1)).to(dtype=torch.double, device="cpu")
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        models.append(gp)
    model = ModelListGP(*models).to(dtype=torch.double, device="cpu")
    model.eval()
    return model, X, Y


def propose_bo_batch(training: pd.DataFrame, settings: BOSettings = BOSettings()):
    """qLogNEHVI with joint constrained optimization; return candidates/value.
    """
    from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.optim import optimize_acqf
    from botorch.optim.initializers import gen_batch_initial_conditions

    for name in ("batch_size", "mc_samples", "num_restarts", "raw_samples", "n_burnin",
                 "n_thinning", "batch_limit", "maxiter"):
        if getattr(settings, name) < 1:
            raise ValueError(f"{name} must be positive")
    bounds, salt_min = campaign_bounds(settings.stage)
    def constraint(indices, coefficients, rhs):
        return (torch.tensor(indices, dtype=torch.long),
                torch.tensor(coefficients, dtype=torch.double), torch.tensor(rhs, dtype=torch.double))
    constraints = [constraint([2, 3], [1., 1.], salt_min),
                   constraint([2, 3], [-1., -1.], -2.),
                   constraint([4, 5, 6], [-1., -1., -1.], -.15)]
    model, X, _ = fit_campaign_model(training, seed=settings.seed)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([settings.mc_samples]), seed=settings.seed)
    acquisition = qLogNoisyExpectedHypervolumeImprovement(
        model=model, ref_point=[0, 0], X_baseline=X, sampler=sampler, prune_baseline=True)
    initial = gen_batch_initial_conditions(
        acq_function=acquisition, bounds=bounds, q=settings.batch_size,
        num_restarts=settings.num_restarts, raw_samples=settings.raw_samples,
        inequality_constraints=constraints,
        options={"n_burnin": settings.n_burnin, "n_thinning": settings.n_thinning,
                 "seed": settings.seed, "batch_limit": settings.batch_limit})
    candidates, value = optimize_acqf(
        acq_function=acquisition, bounds=bounds, q=settings.batch_size,
        num_restarts=settings.num_restarts, raw_samples=settings.raw_samples,
        batch_initial_conditions=initial, inequality_constraints=constraints,
        sequential=False,
        options={"batch_limit": settings.batch_limit, "maxiter": settings.maxiter,
                 "ftol": settings.ftol, "disp": True,
                 "iprint": 2 if settings.stage == "initial" else 10})
    return pd.DataFrame(candidates.detach().cpu().numpy(), columns=FEATURE_COLUMNS), float(value.detach())


def generate_initial_design(n_points: int, *, numpy_seed: int, torch_seed: int) -> pd.DataFrame:
    """Seed BOTH generators explicitly before calling the unchanged original sampler."""
    if n_points < 1:
        raise ValueError("n_points must be positive")
    torch.manual_seed(torch_seed)
    return pd.DataFrame(sample_initial(n_points, seed=numpy_seed).numpy(), columns=FEATURE_COLUMNS)


def select_space_filling(
    existing: pd.DataFrame, *, stage: str, numpy_seed: int, torch_seed: int,
    n_new: int = 16, n_candidates: int = 5000, edge_weight: float = .5, edge_power: float = 1.,
) -> pd.DataFrame:
    """Boundary-penalized FPS with explicit NumPy/PyTorch seed handling."""
    if stage not in {"initial", "expanded"}:
        raise ValueError("stage must be initial or expanded")
    if not 1 <= n_new <= n_candidates or not 0 <= edge_weight < 1 or edge_power <= 0:
        raise ValueError("Require 1 <= n_new <= n_candidates, 0 <= edge_weight < 1, edge_power > 0.")
    values = existing[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("FPS requires a nonempty, finite existing-design table.")
    torch.manual_seed(torch_seed)
    function = (sample_new_fps_on_sums_edge_penalized if stage == "initial"
                else sample_new_fps_on_sums_edge_penalized_stage3)
    result = function(torch.tensor(values, dtype=torch.double), n_new=n_new,
                      n_candidates=n_candidates, seed=numpy_seed,
                      edge_weight=edge_weight, edge_power=edge_power)
    return pd.DataFrame(result.numpy(), columns=FEATURE_COLUMNS)


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["audit", "initial", "fps", "bo"], default="audit")
    parser.add_argument("--stage", choices=["initial", "expanded"], default="expanded")
    parser.add_argument("--snapshot", type=Path, help="Explicit historical training snapshot for BO.")
    parser.add_argument("--existing-inputs", type=Path, help="CSV with seven named features for FPS.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--torch-seed", type=int, help="Explicit seed: historical sampler did not set it.")
    parser.add_argument("--n-points", type=int, default=64)
    parser.add_argument("--output", type=Path, help="Optional CSV output path; relative to repository root.")
    args = parser.parse_args(argv)
    def resolve(path):
        return path if path.is_absolute() else root / path
    if args.mode == "audit":
        history = pd.read_csv(root / "data/campaign/campaign_history.csv")
        print("Recorded campaign conditions:", len(history))
        print(history.groupby(["Search-space stage", "Selection strategy"]).size().to_string())
        print("\nNo proposals generated. Replay requires explicit input snapshots.")
        print("The original initial/FPS routines seed NumPy only; provide --torch-seed.")
        return 0
    if args.mode in {"initial", "fps"} and args.torch_seed is None:
        parser.error("Specify --torch-seed; do not assume the historical torch RNG state.")
    if args.mode == "initial":
        result = generate_initial_design(args.n_points, numpy_seed=42 if args.seed is None else args.seed,
                                          torch_seed=args.torch_seed)
    elif args.mode == "fps":
        if args.existing_inputs is None:
            parser.error("FPS requires --existing-inputs with seven named input columns.")
        existing = pd.read_csv(resolve(args.existing_inputs))
        result = select_space_filling(existing, stage=args.stage,
                                      numpy_seed=123 if args.seed is None else args.seed,
                                      torch_seed=args.torch_seed)
    else:
        if args.snapshot is None:
            parser.error("BO requires --snapshot; the final dataset is not a historical training snapshot.")
        result, value = propose_bo_batch(load_training_snapshot(resolve(args.snapshot)),
                                         BOSettings(stage=args.stage, seed=12345 if args.seed is None else args.seed))
        print("Log acquisition value:", value)
    print(result.to_string(index=False))
    if args.output:
        destination = resolve(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False, float_format="%.17g")
        print("Saved", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
