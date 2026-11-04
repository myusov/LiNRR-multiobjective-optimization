from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import linregress


@dataclass(frozen=True)
class Calibration:
    """Unweighted linear fit: delta absorbance = slope * concentration + intercept."""

    concentrations_uM: np.ndarray
    signals_AU: np.ndarray
    slope: float
    intercept: float
    r_squared: float
    slope_standard_error: float
    intercept_standard_error: float


@dataclass(frozen=True)
class AnalyticalSummary:
    """Dispersion across analytical measurements, not electrolysis replicates."""

    n: int
    mean: float
    sd: float
    ddof: int


@dataclass(frozen=True)
class AssayResult:
    signals_AU: np.ndarray
    assay_concentrations_uM: np.ndarray
    dilution_factors: np.ndarray
    concentrations_uM: np.ndarray
    summary: AnalyticalSummary


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty, finite one-dimensional array.")
    return result


def read_uvvis_csv(path: str | Path) -> pd.DataFrame:
    return (
        pd.read_csv(path, skiprows=1)
        .dropna(axis=1, how="all")
        .apply(pd.to_numeric, errors="coerce")
    )


def blank_corrected_signals(
    spectra: pd.DataFrame,
    *,
    high_wavelength_nm: float = 650.0,
    low_wavelength_nm: float = 475.0,
) -> np.ndarray:
    """Return (A_high - A_low) after row-wise subtraction of the water blank."""
    if spectra.shape[1] < 4 or spectra.shape[1] % 2:
        raise ValueError("Expected a blank pair followed by complete spectral pairs.")
    if not isinstance(spectra.index, pd.RangeIndex) or not spectra.index.equals(
        pd.RangeIndex(len(spectra))
    ):
        raise ValueError("Use the original CSV row order with a zero-based RangeIndex.")
    if not np.isfinite([high_wavelength_nm, low_wavelength_nm]).all():
        raise ValueError("Wavelength targets must be finite.")
    blank = spectra.iloc[:, 1]
    signals = []
    for column in range(2, spectra.shape[1], 2):
        wavelength = spectra.iloc[:, column]
        if not np.isfinite(wavelength.to_numpy(dtype=float)).any():
            raise ValueError(f"No finite wavelengths in spectral pair {column // 2}.")
        high = (wavelength - high_wavelength_nm).abs().idxmin()
        low = (wavelength - low_wavelength_nm).abs().idxmin()
        corrected = spectra.iloc[:, column + 1] - blank
        signal = corrected.iloc[high] - corrected.iloc[low]
        if not np.isfinite(signal):
            raise ValueError(f"Missing/nonfinite absorbance in spectral pair {column // 2}.")
        signals.append(signal)
    return np.asarray(signals, dtype=float)


def fit_calibration(
    concentrations_uM: Sequence[float], signals_AU: Sequence[float], *,
    style_path: str | Path | None = None,
) -> Calibration:
    """Fit unweighted ordinary least squares with a free intercept, without clipping."""
    concentration = _vector(concentrations_uM, "concentrations_uM")
    signal = _vector(signals_AU, "signals_AU")
    if len(concentration) != len(signal) or len(concentration) < 2:
        raise ValueError("Provide matching arrays with at least two standards.")
    if np.any(concentration < 0) or np.ptp(concentration) == 0:
        raise ValueError("Standard concentrations must be nonnegative and span a range.")
    fit = linregress(concentration, signal)
    if not np.isfinite(fit.slope) or fit.slope == 0:
        raise ValueError("A nonzero finite calibration slope is required.")
    calibration = Calibration(
        concentration.copy(), signal.copy(), float(fit.slope), float(fit.intercept),
        float(fit.rvalue**2), float(fit.stderr), float(fit.intercept_stderr),
    )
    # Every fitted calibration is plotted. No switch suppresses plotting.
    plot_calibration(calibration, style_path=style_path)
    return calibration


def fit_uvvis_calibration(
    path: str | Path, concentrations_uM: Sequence[float], *,
    high_wavelength_nm: float = 650.0, low_wavelength_nm: float = 475.0,
    style_path: str | Path | None = None,
) -> Calibration:
    """Fit directly from an original wide-format UV-vis CSV."""
    return fit_calibration(
        concentrations_uM,
        blank_corrected_signals(
            read_uvvis_csv(path), high_wavelength_nm=high_wavelength_nm,
            low_wavelength_nm=low_wavelength_nm,
        ), style_path=style_path,
    )


def summarize_analytical(values: Sequence[float], *, ddof: int = 0) -> AnalyticalSummary:
    array = _vector(values, "values")
    if not isinstance(ddof, (int, np.integer)) or not 0 <= ddof < len(array):
        raise ValueError("ddof must be an integer smaller than the measurement count.")
    return AnalyticalSummary(len(array), float(np.mean(array)), float(np.std(array, ddof=ddof)), ddof)


def quantify_signals(
    signals_AU: Sequence[float], calibration: Calibration,
    dilution_factors: Sequence[float], *, ddof: int = 0,
) -> AssayResult:
    """Invert the calibration and apply one TOTAL dilution factor per measurement."""
    signal = _vector(signals_AU, "signals_AU")
    dilution = _vector(dilution_factors, "dilution_factors")
    if len(signal) != len(dilution) or np.any(dilution <= 0):
        raise ValueError("Provide one positive total dilution factor per signal.")
    if not np.isfinite([calibration.slope, calibration.intercept]).all() or calibration.slope == 0:
        raise ValueError("Calibration must have finite coefficients and nonzero slope.")
    assay = (signal - calibration.intercept) / calibration.slope
    concentrations = assay * dilution
    return AssayResult(signal.copy(), assay, dilution.copy(), concentrations,
                       summarize_analytical(concentrations, ddof=ddof))


def quantify_uvvis_sample(
    path: str | Path, calibration: Calibration, dilution_factors: Sequence[float], *,
    high_wavelength_nm: float = 650.0, low_wavelength_nm: float = 475.0,
    ddof: int = 0,
) -> AssayResult:
    """Quantify a fresh original UV-vis file, without modifying its spectra."""
    return quantify_signals(
        blank_corrected_signals(
            read_uvvis_csv(path), high_wavelength_nm=high_wavelength_nm,
            low_wavelength_nm=low_wavelength_nm,
        ), calibration, dilution_factors, ddof=ddof,
    )


def calibration_from_table(
    path: str | Path, calibration_id: str, *, style_path: str | Path | None = None,
) -> Calibration:
    table = pd.read_csv(path)
    selected = table.loc[table["Calibration ID"] == calibration_id]
    if selected.empty:
        raise ValueError(f"Calibration ID not found: {calibration_id}")
    return fit_calibration(selected["Concentration (uM)"].to_numpy(),
                           selected["Blank-corrected delta absorbance"].to_numpy(),
                           style_path=style_path)


def plot_calibration(
    calibration: Calibration, *, style_path: str | Path | None = None, ax=None,
):
    import matplotlib.pyplot as plt

    # Keep the LaTeX preamble active when the figure is later drawn or saved.
    if style_path is None:
        style_path = Path(__file__).resolve().parents[1] / "styles/manuscript.mplstyle"
    style_path = Path(style_path)
    if not style_path.is_file():
        raise FileNotFoundError(f"Manuscript style file not found: {style_path}")
    plt.style.use(str(style_path))
    if ax is None:
        _, ax = plt.subplots(figsize=(3.25, 3.25))
    ax.scatter(calibration.concentrations_uM, calibration.signals_AU)
    x = np.sort(calibration.concentrations_uM)
    ax.plot(x, calibration.slope * x + calibration.intercept, linestyle="--")
    ax.set_xlabel(r"Concentration ($\mu$M)")
    ax.set_ylabel("Blank-corrected absorbance difference (AU)")
    ax.set_box_aspect(1)
    plt.show(block=False)
    return ax.figure, ax
