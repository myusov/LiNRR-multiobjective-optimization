"""Generate the manuscript results supported by the analysis modules."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def reproduce(group, destination, use_tex):
    import matplotlib.pyplot as plt
    import pandas as pd

    def save(name, figure):
        figure.savefig(destination / f"{name}.pdf")
        plt.close(figure)

    def export(tables):
        for name, table in tables.items():
            table.to_csv(destination / f"{name}.csv", index=False)

    if group == "pareto":
        from src import pareto

        tables = pareto.run_pareto_analysis(ROOT, output_dir=destination)
        save("S6", pareto.plot_pareto_sensitivity(tables, ROOT, use_tex=use_tex)[0])
    elif group == "model-validation":
        from src import validation

        validation.run_validation(ROOT, output_dir=destination)
    elif group == "interpretation":
        from src import interpretation

        tables = interpretation.run_interpretation(ROOT, output_dir=destination)
        for name, response in [("Fig4", "FE"), ("S22", "Rate")]:
            save(name, interpretation.plot_shap(tables, response, ROOT, use_tex=use_tex)[0])
    elif group == "experimental-validation":
        from src import experimental_validation

        tables = experimental_validation.run()
        export(tables)
        save("S12", experimental_validation.plot_temperature(tables, use_tex=use_tex))
    elif group == "extended-operation":
        from src import extended_operation

        result = extended_operation.run_extended_operation(ROOT, output_dir=destination)
        save("S13", extended_operation.plot_extended_operation(result, ROOT, use_tex=use_tex))
    elif group == "characterization":
        from src.characterization import eis, sem, xps

        data, _ = xps.run_xps(ROOT, output_dir=destination)
        for name, figure in [("S27", "summary"), ("S28", "core")]:
            save(name, xps.plot_xps(data, ROOT, figure=figure, use_tex=use_tex)[0])
        data, _ = sem.run_sem(ROOT, output_dir=destination)
        save("S29", sem.plot_sem_pairs(data, use_tex=use_tex)[0])
        save("S30", sem.plot_sem_pairs(data, spatial=True, use_tex=use_tex)[0])
        resistance, trace, summary = eis.analyze_bo82_voltage()
        export({"S11_resistance": resistance.rename_axis("Condition").reset_index(),
                "S11_cp_trace": trace, "S11_summary": pd.DataFrame([summary])})
        save("S31", eis.plot_eis_recovery_comparison(*eis.load_eis_plot_data())[0])
    elif group == "transport":
        from src import transport

        tables = transport.run_transport(ROOT, output_dir=destination)
        save("S32", transport.plot_transport(tables, ROOT, use_tex=use_tex)[0])


GROUPS = ("pareto", "model-validation", "interpretation", "experimental-validation",
          "extended-operation", "characterization", "transport")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=GROUPS, default=GROUPS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "manuscript")
    parser.add_argument("--no-tex", action="store_true")
    args = parser.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    failures = []
    for group in dict.fromkeys(args.only):
        destination = args.output_dir.resolve() / group
        destination.mkdir(parents=True, exist_ok=True)
        print(f"Running {group}", flush=True)
        try:
            with plt.style.context(ROOT / "styles" / "manuscript.mplstyle"):
                use_tex = plt.rcParams["text.usetex"] and not args.no_tex
                plt.rcParams["text.usetex"] = use_tex
                reproduce(group, destination, use_tex)
        except FileNotFoundError as error:
            failures.append(group)
            print(f"{group}: {error}", file=sys.stderr)
        finally:
            plt.close("all")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

