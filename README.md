# Multi-objective optimization of lithium-mediated nitrogen reduction

This repository accompanies the study of electrolyte formulation and operating conditions for lithium-mediated ammonia synthesis. It contains the campaign data and optimization code, experimental validation data, and scripts for reproducing selected manuscript figures and numerical results.

## Setup

Clone the repository and run the following commands from its root. The analyses were developed with Python 3.11. Use an isolated environment for the project.

```text
git clone https://github.com/myusov/LiNRR-multiobjective-optimization.git
cd LiNRR-multiobjective-optimization
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS or Linux. Then install the analysis dependencies.

```text
python -m pip install numpy pandas scipy matplotlib torch botorch gpytorch shap scikit-learn pillow
```

For the notebooks, also install `jupyterlab` and `ipykernel` and select the project environment as the kernel. Exact numerical verification can depend on package versions and platform.

## Reproduce the analysis

```text
python scripts/reproduce_manuscript.py --no-tex
python scripts/verify_results.py
```

The driver writes CSV tables and PDF figures under `outputs/manuscript/`. Use a separate output directory to keep a fresh run distinct from the deposited outputs.

```text
python scripts/reproduce_manuscript.py --no-tex --output-dir outputs/reproduction
python scripts/verify_results.py --output-dir outputs/reproduction
```

Use `--only` followed by one or more group names to run a subset. The driver supports `pareto`, `model-validation`, `interpretation`, `experimental-validation`, `extended-operation`, `characterization`, and `transport`. Model validation and SHAP can take longer than the experimental analyses.

Plots use [the manuscript style](styles/manuscript.mplstyle). Omit `--no-tex` to use LaTeX rendering with the `helvet` and `sfmath` packages.

The verifier compares 59 CSVs against the fixed tables in [tests/reference_results](tests/reference_results). Its coverage includes Pareto analysis, model validation, interpretation, five-hour operation, and transport.

## Example notebooks

- [01 Quantification example](notebooks/01_quantification_example.ipynb) fits the five-hour salicylate calibrations, shows the dilution correction, and calculates ammonia recovery, Faradaic efficiency (FE), and rate.
- [02 Campaign example](notebooks/02_campaign_example.ipynb) reconstructs the measured FE–rate frontier and demonstrates random initialization, boundary-repulsive farthest-point sampling (BR-FPS), and Bayesian optimization (BO).
- [03 Manuscript walkthrough](notebooks/03_manuscript_walkthrough.ipynb) runs the reproduction driver, displays selected results, and invokes numerical verification.

## Data inventory

| Directory | Contents |
| --- | --- |
| [data/campaign](data/campaign) | Campaign history and replicate-level results for 112 conditions, including the IQR exclusion flags. |
| [data/validation](data/validation) | Validation run metadata, performance and fraction measurements, electrochemical traces, temperature measurements, isotope-control exports, and five-hour assay and recovery data. |
| [data/characterization](data/characterization) | EIS numerical exports, SEM images and panel metadata, and XPS spectra and fit exports. |
| [data/literature](data/literature) | Reference values used for the viscosity and nitrogen-transport estimates in Supplementary Fig. S32. |

## License and citation

The code is provided under the [BSD 3-Clause License](LICENSE). When using the code or data, cite the accompanying manuscript and identify the repository commit or release used.
