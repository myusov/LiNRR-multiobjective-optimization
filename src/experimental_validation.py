import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "validation"
ID = "Experiment ID"
FE = "FE (%)"
RATE = "Rate (nmol/s/cm^2)"
ADDED = "Added NH4 (uM in final tube)"


def isotope_quantification(additions, metadata):
    """Fit total ammonium standard additions using the archived integrals."""
    rows = []
    for experiment, tubes in additions.groupby(ID):
        tubes = tubes.sort_values(ADDED)
        x = tubes[ADDED].to_numpy()
        y = tubes["NH4 integrated area"] / tubes["TMB integrated area"]
        fit = linregress(x, y)
        cov = -x.mean() * fit.stderr**2
        concentration = fit.intercept / fit.slope
        gradient = np.array([1 / fit.slope, -fit.intercept / fit.slope**2])
        variance = gradient @ np.array([[fit.intercept_stderr**2, cov],
                                        [cov, fit.stderr**2]]) @ gradient
        first = tubes.iloc[0]
        factor = (first["NMR final volume (uL)"] / first["NMR sample volume (uL)"]
                  * first["Recovered sample volume (mL)"] / 1000)
        amount = concentration * factor
        run = metadata.loc[experiment]
        area = run["Working electrode geometric area (cm^2)"]
        seconds = 60 * run["Electrolysis duration (min)"]
        charge = run["Current density magnitude (mA/cm^2)"] * area * seconds / 1000
        fe_factor = 100 * 3 * 96485 / charge / 1e6
        rows.append({ID: experiment, "Tube NH4 (uM)": concentration,
                     "Recovered NH3 (umol)": amount, FE: amount * fe_factor,
                     "FE fit SE (percentage points)": np.sqrt(variance) * factor * fe_factor,
                     RATE: amount * 1000 / (seconds * area)})
    return pd.DataFrame(rows)


def run(data_dir=DATA):
    def read(name):
        return pd.read_csv(Path(data_dir) / f"{name}.csv")

    metadata = read("validation_run_metadata").set_index(ID, verify_integrity=True)
    roles = read("experiment_roles")
    results = read("electrolysis_results")
    fractions = read("fraction_measurements")
    tables = {"isotope_nmr": isotope_quantification(
        read("isotope_controls/isotope_standard_additions"), metadata)}
    assay = results[results.Method.eq("Ammonia assay")]
    for role, key, coordinates in [
        ("Cross-platform", "cross_platform", ["Reactor", "Formulation"]),
        ("Substitution controls", "substitutions",
         ["LiBF4 (M)", "1-BuOH (mM)", "EtOH (mM)"]),
    ]:
        ids = roles.loc[roles.Role.eq(role), ID]
        suffix = (r"-ELECTROLYTE-DEPOSITS-GAS$" if key == "cross_platform" else
                  r"(?:-ELECTROLYTE-DEPOSITS|-REPORTED-AGGREGATE-FRACTION-BREAKDOWN-UNAVAILABLE)$")
        points = assay[assay[ID].isin(ids) & assay["Result ID"].str.contains(
            suffix, regex=True)].merge(
            metadata[coordinates], on=ID, validate="one_to_one")
        if len(points) != len(ids):
            raise ValueError(f"Missing or duplicate assay results for {role}")
        points[[FE, RATE]] = points[[FE, RATE]].apply(pd.to_numeric)
        tables[key + "_replicates"] = points
        tables[key + "_summary"] = points.groupby(coordinates).agg(
            n=(ID, "size"), fe_mean=(FE, "mean"), fe_sd=(FE, "std"),
            rate_mean=(RATE, "mean"), rate_sd=(RATE, "std")).reset_index()
    cross = tables["cross_platform_replicates"]
    tables["cross_platform_fractions"] = fractions.merge(
        cross[[ID, "Reactor", "Formulation"]], on=ID, validate="many_to_one")
    tables["gas_and_isotope_results"] = results[results[ID].isin(
        roles.loc[roles.Role.isin(["Gas controls", "Gas controls N2/H2", "Isotope controls"]), ID])]
    tables["temperature"] = read("temperature_measurements")
    tables["temperature_traces"] = read("electrochemical_traces").loc[
        lambda frame: frame[ID].isin(tables["temperature"][ID])]
    return tables


def plot_temperature(tables, *, use_tex=True):
    import matplotlib.pyplot as plt

    with plt.style.context(ROOT / "styles" / "manuscript.mplstyle"):
        plt.rcParams["text.usetex"] = use_tex
        figure, axis = plt.subplots(figsize=(3.25, 3.25))
        voltage = axis.twinx()
        for experiment, points in tables["temperature"].groupby(ID):
            points = points.sort_values("Elapsed time (min)")
            line, = axis.plot(points["Elapsed time (min)"], points["Temperature (C)"],
                              "o--", label=experiment)
            trace = tables["temperature_traces"].loc[lambda d: d[ID].eq(experiment)]
            trace = trace.sort_values("Aligned time (min)")
            voltage.plot(trace["Aligned time (min)"], trace["Potential (V)"],
                         color=line.get_color())
        axis.set(xlabel="Electrolysis time (min)", ylabel="Electrolyte temperature (°C)")
        voltage.set_ylabel("Cell voltage (V)")
        axis.legend(frameon=False)
        figure.tight_layout()
        return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--no-tex", action="store_true")
    args = parser.parse_args()
    tables = run()
    for name in ("isotope_nmr", "cross_platform_summary", "substitutions_summary"):
        print(tables[name].to_string(index=False))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(args.output_dir / f"{name}.csv", index=False)
    if args.plot:
        import matplotlib.pyplot as plt

        plot_temperature(tables, use_tex=not args.no_tex)
        plt.show()


if __name__ == "__main__":
    main()


