"""FE, rate and apparent electrical (pseudo-)energy efficiency.

FE and pseudo-EE are percentages, not fractions. The 1.17 V output-energy
reference is used at 298.15 K, as in the SI. Chemical input energies are excluded.
No hydrogen-production voltage is added to the electrical-input denominator.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

FARADAY_C_MOL = 96485.0  # preserve original notebook constant
AMMONIA_REFERENCE_V = 1.17


def pseudo_ee_percent(fe_percent, average_cell_voltage_V):
    """Return FE(%) * 1.17 / positive mean voltage magnitude.

    For varying current, supply the charge-weighted mean voltage W_el / Q,
    not the arithmetic or unweighted time mean. Both inputs can be arrays.
    """
    fe, voltage = np.broadcast_arrays(np.asarray(fe_percent, dtype=float),
                                     np.asarray(average_cell_voltage_V, dtype=float))
    if not np.isfinite(fe).all() or not np.isfinite(voltage).all() or np.any(voltage <= 0):
        raise ValueError("FE must be finite and mean voltage magnitude positive and finite.")
    return AMMONIA_REFERENCE_V * fe / voltage


def electrical_input(time_s, current_mA, cell_voltage_V):
    """Return (charge_C, electrical_energy_J) by trapezoidal CP integration.

    Supply the recorded CP interval only; no OCV time is added. Current and
    cell-voltage signs are converted to magnitudes for electrical-input accounting.
    WE potential vs a reference alone is NOT a full-cell voltage.
    """
    t = np.asarray(time_s, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Provide at least two finite, strictly increasing CP timestamps.")
    current = np.broadcast_to(np.asarray(current_mA, dtype=float), t.shape)
    voltage = np.broadcast_to(np.asarray(cell_voltage_V, dtype=float), t.shape)
    if not np.isfinite(current).all() or not np.isfinite(voltage).all():
        raise ValueError("Current and full-cell voltage must be finite.")
    dt = np.diff(t)
    amps = np.abs(current) / 1000.0
    power = amps * np.abs(voltage)
    charge = float(np.sum(0.5 * (amps[:-1] + amps[1:]) * dt))
    energy = float(np.sum(0.5 * (power[:-1] + power[1:]) * dt))
    if charge <= 0 or energy <= 0:
        raise ValueError("Integrated charge and electrical input must be positive.")
    return charge, energy


def pseudo_ee_from_ammonia_percent(ammonia_umol, electrical_energy_J):
    """Return 100 * (3 F n_NH3 * 1.17 V) / W_el, as defined in the SI."""
    amount, energy = np.broadcast_arrays(np.asarray(ammonia_umol, dtype=float),
                                        np.asarray(electrical_energy_J, dtype=float))
    if not np.isfinite(amount).all() or not np.isfinite(energy).all() or np.any(energy <= 0):
        raise ValueError("Ammonia must be finite and electrical input positive and finite.")
    return 100.0 * (3.0 * FARADAY_C_MOL) * (amount * 1e-6) * AMMONIA_REFERENCE_V / energy


@dataclass(frozen=True)
class PerformanceResult:
    fe_percent: np.ndarray
    rate_nmol_s_cm2: np.ndarray
    pseudo_ee_percent: np.ndarray


def calculate_performance(
    concentrations_uM, *, recovery_volume_mL: float, current_mA: float,
    electrode_area_cm2: float, duration_min: float, average_cell_voltage_V: float,
) -> PerformanceResult:
    """Constant-current assay conversion preserving original FE/rate arithmetic.

    Apply separately to each analytical concentration. The supplied voltage must
    be a positive mean full-cell magnitude, not a signed electrode potential.
    Background subtraction, if required, must be completed before this function.
    Only the EE definition differs from Salicylate Method Script_v2.
    """
    concentration = np.asarray(concentrations_uM, dtype=float)
    if not np.isfinite(concentration).all():
        raise ValueError("Concentrations must be finite.")
    fixed = [recovery_volume_mL, electrode_area_cm2, duration_min]
    if not np.isfinite(fixed).all() or min(fixed) <= 0 or not np.isfinite(current_mA) or current_mA == 0:
        raise ValueError("Volume, area, duration must be positive; current finite and nonzero.")
    nF = 3 * FARADAY_C_MOL
    current = abs(current_mA)
    fe = nF * recovery_volume_mL * concentration / (current / 1000 * 60 * duration_min) * 10**-7
    rate = (fe / 100) * (current / 1000 / electrode_area_cm2) / nF * 1e9
    return PerformanceResult(fe, rate, pseudo_ee_percent(fe, average_cell_voltage_V))
